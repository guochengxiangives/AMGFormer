import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8
d_model = 64


def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class fusion_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(fusion_prenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d_prenorm(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)


class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)
        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        # 补充缺失的forward实现
        x = self.norm(x)
        x = self.conv(x)
        x = self.activation(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims * 2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='reflect')

        self.e4_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='reflect')

        self.e5_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))  # 1 8 128 128 128

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))  # 1 16 64 64 64

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))  # 1 32 32 32 32

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))  # 1 64 16 16 16

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))  # 1 128 8 8 8

        return x1, x2, x3, x4, x5


class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred


class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims * 16, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims * 8, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims * 4, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims * 2, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims * 16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims * 8, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims * 4, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims * 2, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims * 1, num_cls=num_cls)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x


# ModalityQualityAwareEnhancement
class ModalityQualityAwareEnhancement(nn.Module):
    def __init__(self, channel, num_modalities=4):
        super().__init__()
        self.channel = channel
        self.num_modalities = num_modalities

        # 轻量通道数
        mini_channel = max(channel // 32, 4)

        # 1. 质量评估网络
        self.quality_net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channel, mini_channel, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mini_channel, 2, 1),
            nn.Sigmoid()
        )

        # 2. 固定比例的下采样和上采样 - 避免动态操作
        self.downsample = nn.AdaptiveAvgPool3d(4)  # 固定到4x4x4
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # 固定scale

        # 3. 轻量注意力
        self.cross_attention = nn.Sequential(
            nn.Conv3d(channel, mini_channel, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mini_channel, 1, 1),
            nn.Sigmoid()
        )

        # 4. 融合层
        self.fusion = nn.Conv3d(channel, channel, 1)

        # 5. 门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channel, mini_channel, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mini_channel, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, modalities, mask=None):
        B, C, H, W, Z = modalities[0].shape
        device = modalities[0].device
        dtype = modalities[0].dtype
        num_valid_modalities = len(modalities)

        # 预计算质量分数
        quality_scores = []
        for i, modality in enumerate(modalities):
            if mask is None or mask[0][i]:
                scores = self.quality_net(modality)
                quality_scores.append(scores)
            else:
                quality_scores.append(torch.zeros(B, 2, 1, 1, 1, device=device, dtype=dtype))

        # 跨模态增强
        enhanced_modalities = []
        boundary_scores = []
        semantic_scores = []

        for i in range(num_valid_modalities):
            if mask is None or mask[0][i]:
                current_modality = modalities[i]

                # 初始化增强
                enhanced = current_modality

                # 计算增强特征
                enhancement = torch.zeros_like(current_modality)
                total_weight = torch.zeros(1, device=device, dtype=dtype)  # 标量权重

                for j in range(num_valid_modalities):
                    if j != i and (mask is None or mask[0][j]):
                        other_modality = modalities[j]

                        # 使用固定尺寸避免动态操作
                        if H >= 8 and W >= 8 and Z >= 8:  # 只有足够大时才下采样
                            # 下采样计算注意力
                            other_down = self.downsample(other_modality)
                            attention = self.cross_attention(other_down)

                            # 上采样回原尺寸 - 使用插值而非动态scale
                            scale_h, scale_w, scale_z = H // 4, W // 4, Z // 4
                            if scale_h > 1 or scale_w > 1 or scale_z > 1:
                                attention = F.interpolate(attention, size=(H, W, Z),
                                                          mode='trilinear', align_corners=False)
                            else:
                                attention = attention.expand(-1, -1, H, W, Z)
                        else:
                            # 小尺寸直接计算
                            attention = self.cross_attention(other_modality)

                        # 质量权重 - 取2个通道的平均，降到1个通道
                        quality_weight = quality_scores[j].mean(dim=[1, 2, 3, 4], keepdim=True)  # [B,1,1,1,1]

                        # 加权增强
                        weighted_other = other_modality * attention * quality_weight
                        enhancement = enhancement + weighted_other
                        total_weight = total_weight + quality_weight.mean()

                # 归一化和融合
                if total_weight > 1e-8:
                    enhancement = enhancement / total_weight
                    fused = self.fusion(enhancement)
                    gate = self.gate(fused)
                    enhanced = current_modality + gate * fused

                enhanced_modalities.append(enhanced)
            else:
                enhanced_modalities.append(modalities[i])

            # 收集分数
            if i < len(quality_scores):
                boundary_scores.append(quality_scores[i][:, 0:1])
                semantic_scores.append(quality_scores[i][:, 1:2])

        return enhanced_modalities, boundary_scores, semantic_scores


class QuadIntegrator_Bridge(nn.Module):
    def __init__(self, channel=256):
        super(QuadIntegrator_Bridge, self).__init__()
        # 为每个模态定义独立的卷积层序列
        self.conv_mod1 = nn.Sequential(
            nn.Conv3d(channel, channel, 1, 1, 0),
            nn.BatchNorm3d(channel),
            nn.ReLU()
        )
        self.conv_mod2 = nn.Sequential(
            nn.Conv3d(channel, channel, 1, 1, 0),
            nn.BatchNorm3d(channel),
            nn.ReLU()
        )
        self.conv_mod3 = nn.Sequential(
            nn.Conv3d(channel, channel, 1, 1, 0),
            nn.BatchNorm3d(channel),
            nn.ReLU()
        )
        self.conv_mod4 = nn.Sequential(
            nn.Conv3d(channel, channel, 1, 1, 0),
            nn.BatchNorm3d(channel),
            nn.ReLU()
        )

        # 定义融合特征的卷积层序列
        self.conv_c1 = nn.Sequential(
            nn.Conv3d(4 * channel, channel, 3, 1, 1),
            nn.BatchNorm3d(channel),
            nn.ReLU()
        )

        # 定义输出权重的卷积层
        self.conv_c2 = nn.Sequential(
            nn.Conv3d(channel, 4, 3, 1, 1),  # 输出权重的通道数变为 4
            nn.BatchNorm3d(4),
            nn.ReLU()
        )

    def fusion(self, features, weights):
        """
        融合四个模态特征
        :param features: 模态特征列表，长度为 4
        :param weights: 模态权重张量，形状为 (B, 4, H, W)
        :return: 融合后的特征
        """
        fused = 0
        for i in range(4):
            fused += weights[:, i:i+1, :, :, :] * features[i]  # 加权求和
        return fused

    def forward(self, mod1, mod2, mod3, mod4):
        # 分别提取四个模态的特征
        F1 = self.conv_mod1(mod1)
        F2 = self.conv_mod2(mod2)
        F3 = self.conv_mod3(mod3)
        F4 = self.conv_mod4(mod4)

        # 拼接模态特征
        f = torch.cat([F1, F2, F3, F4], dim=1)

        # 通过卷积层融合特征
        f = self.conv_c1(f)

        # 计算模态权重
        weights = self.conv_c2(f)  # (B, 4, H, W)

        # 对模态特征进行融合
        fused_features = self.fusion([F1, F2, F3, F4], weights)

        return fused_features


class Multi_Granular_Attention_Orchestrator(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True, dropout=0.0, k_ratios=None):
        super(Multi_Granular_Attention_Orchestrator, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

        self.k_ratios = k_ratios or [0.5, 0.67, 0.75, 0.8]

        # 可学习的 TopK 组合权重
        self.attn_weights = nn.Parameter(torch.ones(len(self.k_ratios)), requires_grad=True)

        # 特征重校准模块
        self.recalibration = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1),
            nn.BatchNorm3d(dim),
            nn.Sigmoid()
        )

        # 特征融合（enhanced + residual）
        self.fusion = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm3d(dim)
        )

    def forward(self, x):
        b, c, d, h, w = x.shape
        residual = x

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        _, _, C, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        sparse_outputs = []
        for ratio in self.k_ratios:
            k_value = int(C * ratio)
            mask = torch.zeros_like(attn, device=x.device)
            indices = torch.topk(attn, k=k_value, dim=-1, largest=True)[1]
            mask.scatter_(-1, indices, 1.)

            masked_attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
            masked_attn = self.attn_drop(masked_attn.softmax(dim=-1))

            out = masked_attn @ v
            out = rearrange(out, 'b head c (d h w) -> b (head c) d h w', head=self.num_heads, d=d, h=h, w=w)
            sparse_outputs.append(out)

        # 使用可学习的静态权重组合不同比例的 TopK 输出
        soft_weights = F.softmax(self.attn_weights, dim=0)
        weighted_sum = sum(w * o for w, o in zip(soft_weights, sparse_outputs))

        recalib = self.recalibration(weighted_sum)
        enhanced = weighted_sum * recalib

        fused = torch.cat([enhanced, residual], dim=1)
        output = self.fusion(fused)
        output = self.project_out(output)
        return output


class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.mqae_x5 = ModalityQualityAwareEnhancement(transformer_basic_dims)
        self.mqae_x4 = ModalityQualityAwareEnhancement(basic_dims * 8)
        self.mqae_x3 = ModalityQualityAwareEnhancement(basic_dims * 4)
        self.mqae_x2 = ModalityQualityAwareEnhancement(basic_dims * 2)
        self.mqae_x1 = ModalityQualityAwareEnhancement(basic_dims * 1)

        ########### IntraFormer
        self.flair_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.flair_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t1ce_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t1_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t2_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)

        self.flair_mgao = Multi_Granular_Attention_Orchestrator(
            dim=transformer_basic_dims,
            num_heads=num_heads,
            bias=True,
            dropout=0.0,  # 可以根据需要调整
            k_ratios=[0.5, 0.67, 0.75, 0.8]  # 与原始模块保持一致
        )
        self.t1ce_mgao = Multi_Granular_Attention_Orchestrator(
            dim=transformer_basic_dims,
            num_heads=num_heads,
            bias=True,
            dropout=0.0,
            k_ratios=[0.5, 0.67, 0.75, 0.8]
        )
        self.t1_mgao = Multi_Granular_Attention_Orchestrator(
            dim=transformer_basic_dims,
            num_heads=num_heads,
            bias=True,
            dropout=0.0,
            k_ratios=[0.5, 0.67, 0.75, 0.8]
        )
        self.t2_mgao = Multi_Granular_Attention_Orchestrator(
            dim=transformer_basic_dims,
            num_heads=num_heads,
            bias=True,
            dropout=0.0,
            k_ratios=[0.5, 0.67, 0.75, 0.8]
        )
        ########### IntraFormer

        ########### InterFormer
        # 用于多模态融合的增强TopK稀疏注意力
        self.multimodal_mgao = Multi_Granular_Attention_Orchestrator(
            dim=transformer_basic_dims * 4,
            num_heads=num_heads,
            bias=True,
            dropout=0.0,
            k_ratios=[0.5, 0.67, 0.75, 0.8]
        )
        self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims * num_modals, basic_dims * 16 * num_modals,
                                                kernel_size=1, padding=0)
        ########### InterFormer

        self.masker = MaskModal()
        self.qib_x1 = QuadIntegrator_Bridge(basic_dims)
        self.qib_x2 = QuadIntegrator_Bridge(basic_dims * 2)
        self.qib_x3 = QuadIntegrator_Bridge(basic_dims * 4)
        self.qib_x4 = QuadIntegrator_Bridge(basic_dims * 8)
        self.qib_x5 = QuadIntegrator_Bridge(basic_dims * 64)
        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)  #

    def forward(self, x, mask):
        flair = x[:, 0:1, :, :, :]
        t1ce = x[:, 1:2, :, :, :]
        t1 = x[:, 2:3, :, :, :]
        t2 = x[:, 3:4, :, :, :]

        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(flair)
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(t1ce)
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(t1)
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(t2)

        ########### IntraFormer
        flair_token_x5 = self.flair_encode_conv(flair_x5)
        t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5)
        t1_token_x5 = self.t1_encode_conv(t1_x5)
        t2_token_x5 = self.t2_encode_conv(t2_x5)
        flair_intra_token_x5 = self.flair_mgao(flair_token_x5)
        t1ce_intra_token_x5 = self.t1ce_mgao(t1ce_token_x5)
        t1_intra_token_x5 = self.t1_mgao(t1_token_x5)
        t2_intra_token_x5 = self.t2_mgao(t2_token_x5)

        flair_intra_x5 = flair_intra_token_x5
        t1ce_intra_x5 = t1ce_intra_token_x5
        t1_intra_x5 = t1_intra_token_x5
        t2_intra_x5 = t2_intra_token_x5

        boundary_scores = []
        semantic_scores = []

        [flair_intra_x5, t1ce_intra_x5, t1_intra_x5,
         t2_intra_x5], b_scores5, s_scores5 = self.mqae_x5(
            [flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5], mask
        )

        boundary_scores.append(b_scores5)
        semantic_scores.append(s_scores5)

        [flair_x4, t1ce_x4, t1_x4, t2_x4], b_scores4, s_scores4 = self.mqae_x4(
            [flair_x4, t1ce_x4, t1_x4, t2_x4], mask
        )

        boundary_scores.append(b_scores4)
        semantic_scores.append(s_scores4)

        [flair_x3, t1ce_x3, t1_x3, t2_x3], b_scores3, s_scores3 = self.mqae_x3(
            [flair_x3, t1ce_x3, t1_x3, t2_x3], mask
        )

        boundary_scores.append(b_scores3)
        semantic_scores.append(s_scores3)

        [flair_x2, t1ce_x2, t1_x2, t2_x2], b_scores2, s_scores2 = self.mqae_x2(
            [flair_x2, t1ce_x2, t1_x2, t2_x2], mask
        )

        boundary_scores.append(b_scores2)
        semantic_scores.append(s_scores2)

        [flair_x1, t1ce_x1, t1_x1, t2_x1], b_scores1, s_scores1 = self.mqae_x1(
            [flair_x1, t1ce_x1, t1_x1, t2_x1], mask
        )

        boundary_scores.append(b_scores1)
        semantic_scores.append(s_scores1)

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
        ########### IntraFormer
        x1_masker_input = self.qib_x1(flair_x1, t1ce_x1, t1_x1, t2_x1).unsqueeze(1).expand(-1, 4, -1, -1, -1, -1)
        x2_masker_input = self.qib_x2(flair_x2, t1ce_x2, t1_x2, t2_x2).unsqueeze(1).expand(-1, 4, -1, -1, -1, -1)
        x3_masker_input = self.qib_x3(flair_x3, t1ce_x3, t1_x3, t2_x3).unsqueeze(1).expand(-1, 4, -1, -1, -1, -1)
        x4_masker_input = self.qib_x4(flair_x4, t1ce_x4, t1_x4, t2_x4).unsqueeze(1).expand(-1, 4, -1, -1, -1, -1)
        x5_intra_masker_input = self.qib_x5(flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5).unsqueeze(
            1).expand(-1, 4, -1, -1, -1, -1)
        x1 = self.masker(x1_masker_input, mask)  # Bx4xCxHWZ
        x2 = self.masker(x2_masker_input, mask)
        x3 = self.masker(x3_masker_input, mask)
        x4 = self.masker(x4_masker_input, mask)
        x5_intra = self.masker(x5_intra_masker_input, mask)

        ########### InterFormer
        flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5 = torch.chunk(x5_intra, num_modals, dim=1)
        multimodal_token_x5 = torch.cat((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1)
        multimodal_inter_token_x5 = self.multimodal_mgao(multimodal_token_x5)
        multimodal_inter_x5 = self.multimodal_decode_conv(multimodal_inter_token_x5)
        x5_inter = multimodal_inter_x5

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        ########### InterFormer

        if self.is_training:
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), preds, boundary_scores, semantic_scores
        return fuse_pred, boundary_scores, semantic_scores


