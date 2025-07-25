"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        # _cnn = models.regnet_x_400mf(pretrained=True)
        _cnn = models.regnet_x_3_2gf(pretrained=True) # Using a larger model for better performance

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code

        # Add lateral 1x1 conv layers
        self.fpn_params.update({
            "c3_lateral": nn.Conv2d(dummy_out_shapes[0][1][1], self.out_channels, 1),
            "c4_lateral": nn.Conv2d(dummy_out_shapes[1][1][1], self.out_channels, 1),
            "c5_lateral": nn.Conv2d(dummy_out_shapes[2][1][1], self.out_channels, 1)
            })
        
        # Add output 3x3 conv layers
        self.fpn_params.update({
            "p3_output": nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            "p4_output": nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            "p5_output": nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            })
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code

        # Calculate FPN features top-down. Refer to:
        # https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
        # for more useful image than the one in the FPN paper
        m5 = self.fpn_params["c5_lateral"](backbone_feats["c5"])
        fpn_feats["p5"] = self.fpn_params["p5_output"](m5)
        
        m4 = F.interpolate(m5, scale_factor=2, mode='nearest') + self.fpn_params["c4_lateral"](backbone_feats["c4"])
        fpn_feats["p4"] = self.fpn_params["p4_output"](m4)

        m3 = F.interpolate(m4, scale_factor=2, mode='nearest') + self.fpn_params["c3_lateral"](backbone_feats["c3"])
        fpn_feats["p3"] = self.fpn_params["p3_output"](m3)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }    

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code

        # Get the dimensions for easy reference
        H = feat_shape[2]
        W = feat_shape[3]

        # Use meshgrid to get all of the H and W coordinate indexes in the order they appear
        # when moving across the grid, starting from top left and moving to the right
        H_idxs, W_idxs = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype)
            )

        # Reshape and combine these coordinate indexes into (H*W, 2)
        H_idxs = H_idxs.reshape(-1, 1)
        W_idxs = W_idxs.reshape(-1, 1)
        HW = torch.hstack((H_idxs, W_idxs))

        location_coords[level_name] = (HW+0.5)*level_stride

        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code

    # Get useful shapes
    n_boxes = boxes.shape[0]

    # Get sorted list of indices of highest-scoring boxes
    _, sorted_scores_idxs = torch.sort(scores, descending=True)

    # Get box areas
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1) * (y2 - y1)

    # Define suppressed and keep tensor/list
    suppressed = torch.zeros(n_boxes, dtype=torch.int64)
    keep = []

    # NON VECTORISED
    # # Iterate through the boxes
    # for _i in range(n_boxes):
    #     i = sorted_scores_idxs[_i] # Get the index of the next highest-scoring box
    #     if suppressed[i] == 1: # If this box has already been suppressed, go to next box
    #         continue

    #     keep.append(i) # This box hasn't been suppressed, and is the next highest-scoring, so we keep it

    #     # Compare with every other box yet to be seen
    #     for _j in range(_i+1, n_boxes):
    #         j = sorted_scores_idxs[_j] # Get the index of the next highest-scoring box
        
    #         if suppressed[j] == 1: # If this box has already been suppressed, go to next box
    #             continue
            
    #         # Get intersection box coordinates and area
    #         xx1, yy1 = max(x1[i], x1[j]), max(y1[i], y1[j])
    #         xx2, yy2 = min(x2[i], x2[j]), min(y2[i], y2[j])
    #         inter_area = max(0, (xx2 - xx1)) * max(0,(yy2 - yy1))

            
    #         # Calculate IoU and suppress if over the threshold
    #         union_area = (areas[i] + areas[j]) - inter_area
    #         iou = inter_area / union_area
    #         if iou > iou_threshold:
    #             suppressed[j] = 1

    # VECTORISED
    for _i in range(n_boxes):
        i = sorted_scores_idxs[_i] # Get the index of the next highest-scoring box
        if suppressed[i] == 1: # If this box has already been suppressed, go to next box
            continue

        keep.append(i) # This box hasn't been suppressed, and is the next highest-scoring, so we keep it
        
        # Get intersection box coordinates and area between this box and every other box
        xx1_vec = torch.where(x1 > x1[i], x1, x1[i])
        yy1_vec = torch.where(y1 > y1[i], y1, y1[i])
        xx2_vec = torch.where(x2 < x2[i], x2, x2[i])
        yy2_vec = torch.where(y2 < y2[i], y2, y2[i])
        inter_area_vec = torch.clamp((xx2_vec - xx1_vec), min=0) * torch.clamp((yy2_vec - yy1_vec), min=0)

        # Calculate union area and IoU between this box and every other box
        union_area_vec = (areas[i] + areas) - inter_area_vec
        iou_vec = inter_area_vec / union_area_vec

        # Suppress the boxes where the IoU exceeds the threshold
        suppressed[iou_vec > iou_threshold] = 1

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return torch.tensor(keep, dtype=torch.long)


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
