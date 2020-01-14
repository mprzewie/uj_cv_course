from collections import OrderedDict
from typing import Optional, List, Union

import numpy as np
from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

BACKBONE_TO_RETURN_BLOCKS = {'efficientnet-b0': [2, 4, 10, 15],
                             'efficientnet-b1': [4, 7, 15, 22],
                             'efficientnet-b2': [4, 7, 15, 22],
                             'efficientnet-b3': [4, 7, 17, 25],
                             'efficientnet-b4': [5, 9, 21, 31],
                             'efficientnet-b5': [7, 12, 26, 38],
                             'efficientnet-b6': [8, 14, 30, 44],
                             'efficientnet-b7': [10, 17, 37, 54]
                             }


class EfficientNetBackbone(nn.Module):
    def __init__(self, backbone: EfficientNet, return_blocks: Union[List[int], int]):
        super().__init__()
        self.backbone = backbone
        self.return_blocks = return_blocks  # if an int, on the first run indices of appropriate blocks will be calculated

    def forward(self, inputs) -> OrderedDict:

        bkb = self.backbone
        x = bkb._swish(bkb._bn0(bkb._conv_stem(inputs)))

        featuremaps = []

        # Blocks
        for idx, block in enumerate(bkb._blocks):
            drop_connect_rate = bkb._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(bkb._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            featuremaps.append(x)

        if isinstance(self.return_blocks, int):
            """a special case for checking which blocks to use. Don't use in production"""
            print("dynamic")
            ret_blocks = []
            prev_num_channels = -np.inf
            idx = len(featuremaps) - 1
            for i in range(idx, 0, -1):
                if featuremaps[i].shape[2] > prev_num_channels:
                    ret_blocks.append(i)
                    prev_num_channels = featuremaps[i].shape[2]
                if len(ret_blocks) == self.return_blocks:
                    break

            self.return_blocks = sorted(ret_blocks)

        result = OrderedDict()
        for i in self.return_blocks:
            result[i] = featuremaps[i]

        return result


def efficientnet_fpn_backbone(
        backbone_name: str = "efficientnet-b0", return_blocks: Optional[List[int]] = None,
        out_channels: int = 256
):
    efficient_net: EfficientNet = EfficientNet.from_pretrained(backbone_name)

    return_blocks = return_blocks if return_blocks is not None else BACKBONE_TO_RETURN_BLOCKS.get(backbone_name, [])

    backbone = EfficientNetBackbone(
        efficient_net,
        return_blocks
    )

    fpn = FeaturePyramidNetwork(
        in_channels_list=[
            efficient_net._blocks[rb]._block_args.output_filters
            for rb in return_blocks
        ],
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


# img: torch.Tensor = ToTensor()(Image.new("RGB", (300, 300)))
# #
# for i in range(8):
#     name = f"efficientnet-b{i}"
#     en = EfficientNet.from_pretrained(name)
#     efpn = EfficientNetBackbone(en, 4)
#     xd = efpn(img.unsqueeze(0))
#     BACKBONE_TO_RETURN_BLOCKS[name] = list(xd.keys())
#
# pprint(BACKBONE_TO_RETURN_BLOCKS)
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
# # #
det = faster_rcnn_efficientnet_fpn(backbone_name="efficientnet-b7")

# det.train()
# lb = [
#     {
#         "boxes": torch.tensor([[10,10,20,20], [30,30,40,40]]),
#         "labels": torch.tensor([1, 1])
#     }
# ]
# det.eval()
# xd = det(img.unsqueeze(0))[0]
# for (k, v) in xd.items():
#     print(k, v.shape)
