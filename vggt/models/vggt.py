# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        """
        中文：
        前向传播VGGT模型。
        
        参数：
            images (torch.Tensor): 输入图像，形状为[S, 3, H, W]或[B, S, 3, H, W]，范围为[0, 1]。
                B: 批量大小, S: 序列长度, 3: RGB通道, H: 高度, W: 宽度
            query_points (torch.Tensor, optional): 查询点，用于跟踪，像素坐标。
                Shape: [N, 2]或[B, N, 2]，其中N是查询点的数量。
                默认：None

        返回：
            dict: 包含以下预测的词典：
                - pose_enc (torch.Tensor): 相机位姿编码，形状为[B, S, 9]（来自最后一个迭代）
                - depth (torch.Tensor): 预测的深度图，形状为[B, S, H, W, 1]
                - depth_conf (torch.Tensor): 深度图的置信度得分，形状为[B, S, H, W]
                - world_points (torch.Tensor): 每个像素的3D世界坐标，形状为[B, S, H, W, 3]
                - world_points_conf (torch.Tensor): 世界坐标的置信度得分，形状为[B, S, H, W]
                - images (torch.Tensor): 原始输入图像，用于可视化

                如果提供了query_points，还包括：
                - track (torch.Tensor): 点跟踪，形状为[B, S, N, 2]（来自最后一个迭代），像素坐标
                - vis (torch.Tensor): 跟踪点的可见性得分，形状为[B, S, N]
                - conf (torch.Tensor): 跟踪点的置信度得分，形状为[B, S, N]

        注意：
            1. 如果输入图像没有批次维度，则添加一个批次维度。
            2. 如果提供了query_points，则需要确保其形状正确。
            3. 返回的词典包含所有预测结果，包括相机位姿、深度、世界坐标等。
            4. 如果query_points为None，则不进行跟踪预测。
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        # 得到aggregated_tokens_list的形状为len(aggregated_tokens_list)=24, 每个元素的形状为[B, S, P, 2C=2048]
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

