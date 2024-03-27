from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Function

import numpy as np
from collections import OrderedDict
from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.utils import misc

from utils.utils import crop_op
from models.net_utils import Identity
from models.transformer_utils import Net, VisionTransformer, CONFIGS
from models.net_utils import convert_pytorch_checkpoint

class MLP(nn.Module):
    def __init__(self, d, hidden_d, nr_classes):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(d, hidden_d)
        self.out = nn.Linear(hidden_d, nr_classes)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = x.float()
        x = x.view(x.size(0), -1)
        x = self.hidden1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.out(x)
        # x = torch.sigmoid(x)
        return x

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class ResNet34_DA(nn.Module):
    def __init__(self, nr_classes, nr_domains):
        super(ResNet34_DA, self).__init__()
        self.model = models.resnet34(True) #pretrained resnet34
        num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, nr_classes)
        self.model.fc = Identity()
        self.class_classifier = nn.Linear(num_ftrs, nr_classes) # potentially add extra layers here
        self.domain_classifier = nn.Linear(num_ftrs, nr_domains) # potentially add extra layers here
    def forward(self, input_data, alpha):
        feature = self.model(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        # domain_output = self.domain_classifier(feature)
        return class_output, domain_output


class TransUNet(ModelABC):
    def __init__(
        self: ModelABC,
        num_input_channels: int = 3,
        num_types: int = 2,
        encoder: str = "R50-ViT-B_16",
        weights: str = None,
        img_size: int = 512,
        patch_size: int = 16,
    ) -> None:
        """Initialize :class:`TransUNet`."""
        
        super().__init__()
        self.num_types = num_types
        self.encoder = encoder
        
        config_vit = CONFIGS[encoder]
        config_vit.n_classes = num_types
        config_vit.n_skip = 3
        
        if encoder.find('R50') != -1:
            config_vit.patches.grid = (int(img_size / patch_size), int(img_size / patch_size))
        self.model = VisionTransformer(config_vit, img_size=img_size, num_classes=num_types).cuda()
        
        if weights is not None:
            pretrained_weights = torch.load(weights, map_location=torch.device("cuda"))
            state_dict = pretrained_weights['desc']
            state_dict = convert_pytorch_checkpoint(state_dict)
            saved_state_dict2 = state_dict.copy()
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    saved_state_dict2.pop(k)
                    saved_state_dict2[k[6:]] = v
            self.model.load_state_dict(saved_state_dict2, strict=True)
    
    def forward(  # skipcq: PYL-W0221
        self: ModelABC,
        imgs: torch.Tensor,
    ) -> dict:
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            imgs (torch.Tensor):
                Input images, the tensor is in the shape of NCHW.

        Returns:
            dict:
                A dictionary containing the inference output.
                The expected format os {decoder_name: prediction}.

        """

        imgs = imgs / 255.0  # to 0-1 range to match XY

        out, _ = self.model(imgs)
        out = crop_op(out, [184, 184])

        out_dict = OrderedDict()
        out_dict['ls'] = out

        return out_dict
    
    @staticmethod
    # skipcq: PYL-W0221  # noqa: ERA001
    def postproc(
        raw_maps: list[np.ndarray]
        ) -> np.ndarray:
        """Post-processing script for image tiles.

        Args:
            raw_maps (list(:class:`numpy.ndarray`)):
                A list of prediction outputs of each head, here [ls] (match with the output
                of `infer_batch`).

        Returns:
            :class:`numpy.ndarray` - Semantic segmentation map:
                    Pixel-wise nuclear instance segmentation prediction.

        """

        ls_map = raw_maps
        # pred_layer = TransUNet._proc_ls(ls_map)
        pred_layer = ls_map
        
        return pred_layer

    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: np.ndarray, 
        *, 
        on_gpu: bool
        ) -> dict:
        """Run inference on an input batch.

        This contains logic for forward operation as well as batch i/o
        aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (ndarray):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            on_gpu (bool):
                Whether to run inference on a GPU.

        """
        patch_imgs = batch_data

        device = misc.select_device(on_gpu=on_gpu)
        patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)  # to NCHW
        patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

        model.eval()  # infer mode

        # --------------------------------------------------------------
        with torch.inference_mode():
            pred_dict = model(patch_imgs_gpu)
            pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()],
            )

            layer_map = F.softmax(pred_dict["ls"], dim=-1)
            # keep pixel level values
        return [layer_map.cpu().numpy()]
        #     # take max value
        #     layer_map = torch.argmax(layer_map, dim=-1)#, keepdim=True)
        #     pred_dict["ls"] = layer_map
        #     pred_output = pred_dict["ls"]
        # return [pred_output.cpu().numpy().astype("uint8")]