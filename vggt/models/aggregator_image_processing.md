# Aggregator 图像处理流程

本文档概述 `aggregator.py` 中 `Aggregator.__call__` 处理输入图像张量的关键步骤，帮助理解如何从原始图像得到下游头部使用的 2048 维特征。

## 1. 输入要求与归一化
- 期望的输入形状为 `[B, S, C, H, W]`，其中 `B` 是 batch size，`S` 是序列帧数。
- 采用 ResNet 均值和方差进行归一化 (`_RESNET_MEAN`, `_RESNET_STD`)，确保图像分布与预训练的 `PatchEmbed`（默认使用 `DINOv2-ViT` 权重）一致。

## 2. 展开批次与 PatchEmbed
- 将归一化后的张量 reshape 为 `[B * S, C, H, W]` 以逐帧送入 `self.patch_embed`。
- `PatchEmbed` 负责把每帧划分为 patch 并编码成长度为 `C=1024` 的 token 序列；默认返回形状 `[B * S, P, 1024]`，`P` 为单帧 patch 数 (`H/patch_size * W/patch_size`)。

## 3. 拼接相机与寄存器 token
- `camera_token` 和 `register_token` 会根据 batch 和序列长度扩展，再与 patch token 沿序列维度拼接，形成 `[B * S, P + 1 + num_register_tokens, 1024]`。
- `self.patch_start_idx = 1 + num_register_tokens` 记录 patch token 起始位置以便后续操作。

## 4. 位置编码
- 若启用旋转位置编码（RoPE），通过 `PositionGetter` 为每个 patch 生成二维坐标，并在拼接后的序列上附加：特殊 token 的位置设为零，其余位置加一后拼接。

## 5. 交替注意力与特征聚合
- 交替执行 `frame` 和 `global` 注意力块：
  - `frame` 注意力在形状 `[B * S, P, 1024]` 的张量上操作，关注帧内关系。
  - `global` 注意力将张量 reshape 为 `[B, S * P, 1024]`，捕获跨帧上下文。
- 每轮交替后，从两种注意力的中间结果分别恢复成 `[B, S, P, 1024]`，并在通道维度拼接得到 `[B, S, P, 2048]`。
- 拼接后的 2048 维特征列表被返回给 `CameraHead`、`DPTHead`、`TrackHead` 等头部模块。

## 6. 输出
- `Aggregator` 返回 `output_list`（长度等于交替层数），其中每个条目都包含上述 2048 维特征图；同时返回 `patch_start_idx` 供下游确定 patch token 区段。

通过以上步骤，输入图像被转换成对帧内与跨帧几何一致性都敏感的高维特征，为 VGGT 的姿态、深度与跟踪任务提供输入。
