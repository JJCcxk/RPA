import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
from performer_pytorch import SelfAttention


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]



class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        # 使用 Depthwise + Pointwise 卷积组合
        self.depthwise_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_middle)
        self.pointwise_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)
        x_1 = self.conv1(input)
        # 将 Depthwise 和 Pointwise 卷积结合
        illu_fea = self.pointwise_conv(self.depthwise_conv(x_1))
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

class PerformerAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, nb_features=256):
        super().__init__()
        self.attention = SelfAttention(dim, heads=heads, dim_head=dim_head, nb_features=nb_features)

    def forward(self, x_in):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)  # Flatten spatial dimensions
        out = self.attention(x)  # 使用 Performer Attention
        out = out.reshape(b, h, w, c)  # 恢复原始形状
        return out

class RPMSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, nb_features=256):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.performer_attention = PerformerAttention(dim, dim_head, heads, nb_features)
        self.proj = nn.Linear(dim, dim, bias=True)

        # 相对位置偏置表，用于相对位置编码
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * heads - 1) * (2 * heads - 1), 1)  # 只使用一个通道，方便扩展
        )

        # 初始化位置偏置
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x_in, illu_fea_trans=None):
        """
        x_in: [b, h, w, c] 输入特征
        illu_fea_trans: [b, h, w, c] 亮度特征，未在此实现中使用
        return: [b, h, w, c] 输出特征
        """
        b, h, w, c = x_in.shape

        # 使用 Performer Attention 计算注意力
        attn_output = self.performer_attention(x_in)

        # 获取相对位置偏置并扩展维度
        relative_position_bias = self.get_relative_position_bias(h, w)
        relative_position_bias = relative_position_bias.expand(-1, -1, -1, attn_output.size(-1))  # 扩展到 [1, 256, 256, 40]

        # 添加位置偏置到注意力输出
        attn_output = attn_output + relative_position_bias

        # 将结果投影回原始维度
        out = self.proj(attn_output)
        return out

    def get_relative_position_bias(self, h, w):
        # 创建相对位置索引表
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        relative_coords = coords_h[:, None] - coords_w[None, :]
        relative_coords = relative_coords.clamp(-self.num_heads + 1, self.num_heads - 1) + (self.num_heads - 1)

        # 获取相对位置偏置值
        relative_position_bias = self.relative_position_bias_table[relative_coords]
        relative_position_bias = relative_position_bias.unsqueeze(0)  # [1, h, w, 1]
        return relative_position_bias



class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            # 替换为 Depthwise 卷积 + Pointwise 卷积
            nn.Conv2d(dim * mult, dim * mult, kernel_size=3, padding=1, bias=False, groups=dim * mult),  # Depthwise
            GELU(),
            nn.Conv2d(dim * mult, dim, kernel_size=1, bias=False),  # Pointwise
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class LGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                RPMSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                LGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = LGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                LGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """
                                            
        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution,LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
    def forward(self, img):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img,illu_fea)

        return output_img


class RPA(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=3, num_blocks=[1,1,1]):
        super(RPA, self).__init__()
        self.stage = stage

        modules_body = [Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)

        return out

