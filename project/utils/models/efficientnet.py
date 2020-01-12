from collections import OrderedDict
from typing import Optional, Dict, List

import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.transforms import ToTensor

BACKBONE_TO_RETURN_BLOCKS = {
    "efficientnet-b0": [2, 4,10, 15]
}


class EfficientNetBackbone(nn.Module):
    def __init__(self, backbone: EfficientNet, return_blocks: List[int] = None):
        super().__init__()
        self.backbone = backbone
        self.return_blocks = return_blocks if return_blocks is not None else []

    def forward(self, inputs) -> OrderedDict:

        bkb = self.backbone
        result = OrderedDict()
        x = bkb._swish(bkb._bn0(bkb._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(bkb._blocks):
            drop_connect_rate = bkb._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(bkb._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self.return_blocks:
                result[idx] = x

        # omit the final convolution
        # # Head
        # x = bkb._swish(bkb._bn1(bkb._conv_head(x)))
        # result["final"] = x

        return result


def efficientnet_fpn_backbone(
        backbone_name: str = "efficientnet-b0", return_blocks: Optional[List[int]] = None,
        out_channels: int = 256
):
    efficient_net: EfficientNet = EfficientNet.from_pretrained(backbone_name)

    return_blocks = [] if return_blocks is None else return_blocks

    backbone = EfficientNetBackbone(
        efficient_net,
        return_blocks
    )

    fpn = FeaturePyramidNetwork(
        in_channels_list=[
                             efficient_net._blocks[rb]._block_args.output_filters
                             for rb in return_blocks
                         ] ,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )
    result = nn.Sequential(
        backbone, fpn
    )
    result.out_channels = out_channels
    return result


def faster_rcnn_efficientnet_fpn(
        backbone_name: str = "efficientnet-b0", return_blocks: Optional[List[int]] = None, num_classes=2, **kwargs
) -> FasterRCNN:
    return_blocks = return_blocks if return_blocks is not None else BACKBONE_TO_RETURN_BLOCKS.get(backbone_name, [])
    backbone = efficientnet_fpn_backbone(backbone_name=backbone_name, return_blocks=return_blocks)
    return FasterRCNN(
        backbone,
        num_classes,
        box_roi_pool=MultiScaleRoIAlign(
            featmap_names=return_blocks,
            output_size=7,
            sampling_ratio=2
        ),
        **kwargs
    )

img: torch.Tensor = ToTensor()(Image.new("RGB", (300, 300)))
#
# en = EfficientNet.from_pretrained("efficientnet-b0")
# # for i, b in enumerate(en._blocks):
# #     print(i, b._block_args)
#
# return_blocks = list(range(16))
#
# # efpn = EfficientNetBackbone(en, return_blocks)
# # efpn = efficientnet_fpn_backbone(return_blocks=return_blocks)
#
# # xd = efpn(img.unsqueeze(0))
# # for (i, o) in xd.items():
# #     print(i, o.shape)
#
#
# det = fasterrcnn_resnet50_fpn(
#     pretrained=False,
#     num_classes=2, #możliwe że wystarczy jedna klasa (zależy od implementacji fpn - w retinie automatycznie dodawano klasę null) #nope, nie wystarczy
#     pretrained_backbone=True,
#     # rpn_anchor_generator=anchors
#   )
# #
# det = faster_rcnn_efficientnet_fpn()

# det.eval()
# xd = det(img.unsqueeze(0))
# for (k, v) in xd.items():
#     print(k, v.shape)
