import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from termcolor import colored
from matplotlib import cm


####
def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                "%s: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict

####
def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
        
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


####
def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.

    Args:
        x: input array
        y: array with desired shape.

    """
    assert (
        y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)


####
def xentropy_loss(true, pred, reduction="mean"):
    """Cross entropy loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array
    
    Returns:
        cross entropy loss

    """
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss


####
def dice_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss


###
def jaccard_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32.
    Assuming of shape NxCxHxW

    """
    true = true.type(torch.float32)
    pred = pred.type(torch.float32)

    inse = torch.sum(pred * true, (0,2,3))
    l = torch.sum(pred, (0,2,3))
    r = torch.sum(true, (0,2,3))
    loss = 1.0 - (inse + smooth) / ((l + r - inse) + smooth)
    loss = torch.sum(loss)
    return loss


####
def mse_loss(true, pred):
    """Calculate mean squared error loss.

    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps 
    
    Returns:
        loss: mean squared error

    """
    loss = pred - true
    loss = (loss * loss).mean()
    return loss


####
def msge_loss(true, pred, focus):
    """Calculate the mean squared error of the gradients of 
    horizontal and vertical map predictions. Assumes 
    channel 0 is Vertical and channel 1 is Horizontal.

    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps 
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)
    
    Returns:
        loss:  mean squared error of gradients

    """

    def get_sobel_kernel(size):
        """Get sobel kernel with a given size."""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    ####
    def get_gradient_hv(hv):
        """For calculating gradient."""
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    focus = (focus[..., None]).float()  # assume input NHW
    focus = torch.cat([focus, focus], axis=-1)
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focused region
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss


class _ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, pad=True, preact=True, dilation=1):
        super().__init__()

        pad_size = int(ksize // 2) if pad else 0
        self.preact = preact

        if preact:
            self.bn = nn.BatchNorm2d(in_ch, eps=1e-5)
        else:
            self.bn = nn.BatchNorm2d(out_ch, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, padding=pad_size, bias=True, dilation=dilation)

    def forward(self, prev_feat, freeze=False):
        feat = prev_feat
        if self.training:
            with torch.set_grad_enabled(not freeze):
                if self.preact:
                    feat = self.bn(feat)
                    feat = self.relu(feat)
                    feat = self.conv(feat)
                else:
                    feat = self.conv(feat)
                    feat = self.bn(feat)
                    feat = self.relu(feat)
        else:
            if self.preact:
                feat = self.bn(feat)
                feat = self.relu(feat)
                feat = self.conv(feat)
            else:
                feat = self.conv(feat)
                feat = self.bn(feat)
                feat = self.relu(feat)

        return feat


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        unit_ch,
        ksize,
        pad=True,
        dilation=1,
    ):
        super().__init__()

        if not isinstance(unit_ch, list):
            unit_ch = [unit_ch]

        self.nr_layers = len(unit_ch)
        self.block = nn.ModuleList()

        for idx in range(self.nr_layers):
            self.block.append(
                _ConvLayer(
                    in_ch,
                    unit_ch[idx],
                    ksize,
                    pad=pad,
                    preact=False,
                    dilation=dilation
                )
            )
            in_ch = unit_ch[idx]

    def forward(self, prev_feat, freeze=False):
        feat = prev_feat
        if self.training:
            with torch.set_grad_enabled(not freeze):
                for idx in range(self.nr_layers):
                    feat = self.block[idx](feat)
        else:
            for idx in range(self.nr_layers):
                feat = self.block[idx](feat)

        return feat