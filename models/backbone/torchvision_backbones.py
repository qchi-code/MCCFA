import os
import torch
import torch.nn as nn
import torchvision


class TVDeeplabRes101Encoder(nn.Module):
    """
    ResNet-101 encoder used by MCCFA.

    This encoder outputs:
        1) a spatial feature map from layer3 (after channel reduction), and
        2) an image-level threshold predicted from layer4.
    """

    def __init__(self,
                 use_coco_init=False,
                 pretrained_weights=None,
                 replace_stride_with_dilation=(False, True, True)):
        super().__init__()
        self.pretrained_weights = pretrained_weights

        # Build a plain ResNet-101 backbone.
        weights = None
        try:
            if use_coco_init:
                # Fallback to torchvision pretrained weights if no custom checkpoint is given.
                weights = torchvision.models.ResNet101_Weights.DEFAULT
        except Exception:
            weights = None

        _model = torchvision.models.resnet101(
            weights=weights,
            replace_stride_with_dilation=list(replace_stride_with_dilation),
        )

        self.backbone = nn.ModuleDict()
        for name, module in _model.named_children():
            self.backbone[name] = module

        # layer3 -> spatial feature map for segmentation branch
        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        # layer4 -> image-level threshold branch
        self.reduce1d = nn.Linear(in_features=1000, out_features=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize newly added layers and optionally load external pretrained weights."""
        nn.init.kaiming_normal_(self.reduce1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.reduce1d.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.reduce1d.bias, 0.0)

        if self.pretrained_weights is not None and os.path.isfile(self.pretrained_weights):
            state_dict = torch.load(self.pretrained_weights, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {self.pretrained_weights}")
            if len(missing) > 0:
                print(f"Missing keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected keys: {unexpected}")

    def forward(self, x, low_level=False):
        """
        Args:
            x: input tensor of shape [B, 3, H, W]
            low_level: kept only for interface compatibility.

        Returns:
            feature: [B, 512, H', W'] spatial feature map from layer3.
            threshold: [B, 1] image-level threshold prediction.
        """
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["relu"](x)
        x = self.backbone["maxpool"](x)

        x = self.backbone["layer1"](x)
        x = self.backbone["layer2"](x)
        x = self.backbone["layer3"](x)
        feature = self.reduce1(x)

        x = self.backbone["layer4"](x)
        t = self.backbone["avgpool"](x)
        t = torch.flatten(t, 1)
        t = self.backbone["fc"](t)
        t = self.reduce1d(t)

        if low_level:
            return feature, None, t
        return feature, t
