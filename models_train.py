from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
import fvcore.nn.weight_init as weight_init
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp

from util.vgg_perceptual_loss import VGGPerceptualLoss

from util.vitdet_utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
    LayerNorm2D,
)


def rgb_to_gray(x):
    """Convert an RGB image tensor from (B, 3, H, W) to (B, 1, H, W)."""
    return (
        0.299 * x[:, 0:1]
        + 0.587 * x[:, 1:2]
        + 0.114 * x[:, 2:3]
    )


def gaussian_blur(x, kernel_size=5, sigma=1.0):
    """Apply a fixed separable Gaussian blur without trainable parameters."""
    coords = (
        torch.arange(kernel_size, device=x.device, dtype=torch.float32)
        - kernel_size // 2
    )
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel2d = (kernel[:, None] * kernel[None, :])[None, None].to(x.dtype)
    return F.conv2d(x, kernel2d, padding=kernel_size // 2)


def highpass(x, kernel_size=5, sigma=1.0):
    """Return the high-frequency residual of a single-channel image."""
    return x - gaussian_blur(x, kernel_size, sigma)


def sobel_gradients(x):
    """Return horizontal and vertical Sobel gradients for a single-channel image."""
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    return F.conv2d(x, sobel_x, padding=1), F.conv2d(x, sobel_y, padding=1)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = LayerNorm2D(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = LayerNorm2D(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = x + residual
        return x


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg16 = models.vgg16(pretrained=False)
        self.vgg16.load_state_dict(torch.load('path/vgg16.pth'))
        self.vgg16.features[:16].eval()
        for param in self.vgg16.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.vgg16(input)
        target_features = self.vgg16(target)
        loss = torch.mean(torch.abs(input_features - target_features))
        return loss


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3,B,nHead,H*W,C/nHead)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C/nHead)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))


        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class ResBottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out

class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
            )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x


class Fontify(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(
             self,
             img_size=224,
             patch_size=16,
             in_chans=3,
             embed_dim=768,
             depth=24,
             num_heads=16,
             mlp_ratio=4.,
             qkv_bias=True,
             drop_path_rate=0.,
             norm_layer=nn.LayerNorm,
             act_layer=nn.GELU,
             use_abs_pos=True,
             use_rel_pos=False,
             rel_pos_zero_init=True,
             window_size=0,
             window_block_indexes=(),
             residual_block_indexes=(),
             use_act_checkpoint=False,
             pretrain_img_size=224,
             pretrain_use_cls_token=True,
             out_feature="last_feat",
             decoder_embed_dim=128,
             loss_func="smoothl1",
             ):
        super().__init__()

        self.resize = nn.AdaptiveAvgPool2d((256, 256))
        self.discriminator = Discriminator()
        self.depth = depth
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patch_embed.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_x = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_y = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size[0] // patch_size, img_size[1] // patch_size),
            )
            #if use_act_checkpoint:
            #    block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim*4, patch_size ** 2 * self.decoder_embed_dim, bias=True)
        self.decoder_pred = nn.Sequential(
                nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1, ),
                LayerNorm2D(self.decoder_embed_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_embed_dim, 3, kernel_size=1, bias=True),
        )
        #self.decoder_pred = nn.Sequential(
        #    ResidualBlock(self.decoder_embed_dim, self.decoder_embed_dim),
        #    nn.Conv2d(self.decoder_embed_dim, 3, kernel_size=1, bias=True),
        #)
        self.loss_func = loss_func

        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.segment_token_x, std=.02)
        torch.nn.init.normal_(self.segment_token_y, std=.02)
        
        # VGG感知损失（只初始化一次，避免重复加载权重）
        self.vgg_loss = VGGPerceptualLoss()
        
        self.apply(self._init_weights)

    def get_loss_phase(self, epoch):
        """
        semantic_only_epochs 现在表示 JT 随机遮盖阶段长度。
        epoch>=B 后进入 JT/BF 同步训练阶段，edge/adv 从该阶段起重新 warmup。
        """
        B = getattr(self, 'semantic_only_epochs', 50)
        if B > 0 and epoch < B:
            return "jt_random", epoch
        return "jt_bf_sync", max(0, epoch - max(B, 0))

    def get_dynamic_loss_weights(self, epoch):
        """
        参数化的原作者式系数组合：
            total = recon + style + edge_weight * edge + adv_weight * adv
        JT-only 阶段禁用 edge/adv；JT/BF 同步阶段通用一套 loss，
        edge/adv 按配置的起点、时长和最终权重进行 warmup。
        """
        adv_warmup_epochs = getattr(self, 'adv_warmup_epochs', 8)
        edge_warmup_epochs = getattr(self, 'edge_warmup_epochs', 10)
        warmup_duration = getattr(self, 'loss_warmup_duration', 8)
        adv_weight_final = getattr(self, 'adv_weight_final', 0.4)
        edge_weight_final = getattr(self, 'edge_weight_final', 0.3)

        phase, phase_epoch = self.get_loss_phase(epoch)
        if phase == "jt_random":
            return phase, 0.0, 0.0

        if phase_epoch < adv_warmup_epochs:
            adv_weight = 0.0
        elif warmup_duration > 0 and phase_epoch < adv_warmup_epochs + warmup_duration:
            progress = (phase_epoch - adv_warmup_epochs) / warmup_duration
            adv_weight = adv_weight_final * progress
        else:
            adv_weight = adv_weight_final

        if phase_epoch < edge_warmup_epochs:
            edge_weight = 0.0
        elif warmup_duration > 0 and phase_epoch < edge_warmup_epochs + warmup_duration:
            progress = (phase_epoch - edge_warmup_epochs) / warmup_duration
            edge_weight = edge_weight_final * progress
        else:
            edge_weight = edge_weight_final

        return phase, adv_weight, edge_weight

    def improved_edge_detection(self, img):
        """
        改进的边缘检测，保护细笔画
        - 使用更温和的梯度计算
        - 添加笔画保护机制
        - 针对中文字体优化
        
        :param img: (Tensor [B, C, H, W]) 输入图像
        :return: (Tensor [B, 1, H, W]) 边缘强度图，值域[0, 1]
        """
        # 1. 转换为灰度图
        if img.shape[1] == 3:
            img_gray = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
            img_gray = img_gray.unsqueeze(1)  # [B, 1, H, W]
        else:
            img_gray = img

        # 2. 更温和的高斯模糊（减少sigma，保护细笔画）
        kernel_size = 3  # 从5减少到3
        sigma = 0.8      # 从1.4减少到0.8
        x = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size, device=img.device)
        y = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size, device=img.device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        gauss = torch.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        gauss = gauss.unsqueeze(0).unsqueeze(0)
        blurred = F.conv2d(img_gray, gauss, padding=kernel_size // 2)

        # 3. 更温和的Sobel算子（保护细笔画）
        sobel_x = torch.tensor([[-0.5, 0, 0.5],    # 从[-1,0,1]改为[-0.5,0,0.5]
                                [-1, 0, 1],
                                [-0.5, 0, 0.5]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-0.5, -1, -0.5],
                                [0, 0, 0],
                                [0.5, 1, 0.5]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)

        grad_x = F.conv2d(blurred, sobel_x, padding=1)
        grad_y = F.conv2d(blurred, sobel_y, padding=1)

        # 4. 更温和的梯度幅值计算
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # 5. 笔画保护：使用sigmoid平滑处理
        magnitude = torch.sigmoid(magnitude * 5)  # 放大后sigmoid，保护细笔画
        
        # 6. 自适应归一化（保护细笔画）
        max_val = magnitude.amax(dim=(2, 3), keepdim=True)
        magnitude = magnitude / (max_val + 1e-8)
        
        # 7. 进一步保护细笔画：降低整体强度
        magnitude = magnitude * 0.7  # 降低整体强度

        return magnitude

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == 2 * imgs.shape[3] and imgs.shape[2] % p == 0

        w = imgs.shape[3] // p
        h = w * 2
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        w = int((x.shape[1]*0.5)**.5)
        h = w * 2
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def forward_encoder(self, imgs, tgts, bool_masked_pos):
        conditioning = None
        if hasattr(self, "style_conditioner"):
            conditioning = self.style_conditioner(
                tgts, bool_masked_pos, self.patch_size
            )
        x = self.patch_embed(imgs)
        y = self.patch_embed(tgts)
        batch_size, Hp, Wp, _ = x.size()
        seq_len = Hp * Wp

        mask_token = self.mask_token.expand(batch_size, Hp, Wp, -1)
        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token).reshape(-1, Hp, Wp, 1)
        y = y * (1 - w) + mask_token * w

        # add pos embed w/o cls token
        x = x + self.segment_token_x
        y = y + self.segment_token_y
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
            y = y + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (y.shape[1], y.shape[2])
            )

        merge_idx = 2
        x = torch.cat((x, y), dim=0)  # (B*2,Hp,Wp,E)
        # apply Transformer blocks
        out = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x) # (B*2,Hp,Wp,E)
            if idx == merge_idx:
                x = (x[:x.shape[0]//2] + x[x.shape[0]//2:]) * 0.5

            if conditioning is not None and idx >= self.depth - 3:
                scale, shift = conditioning[idx - (self.depth - 3)]
                x = x * (1 + scale[:, None, None, :])
                x = x + shift[:, None, None, :]

            if self.depth == 24:
                if idx in [5, 11, 17, 23]:
                    out.append(self.norm(x))
            else:
                if idx in [2, 5, 8, 11]:
                    out.append(self.norm(x))
        return out
    # (B*2,Hp,Wp,E,4)

    def forward_decoder(self, x):
        # predictor projection (B*2,Hp,Wp,E,4)
        x = torch.cat(x, dim=-1)  # (B*2,Hp,Wp,E*4)
        x = self.decoder_embed(x)  # (B*2,Hp,Wp,E*4)——>(B*2,Hp,Wp,P^2*64)
        p = self.patch_size
        h, w = x.shape[1], x.shape[2]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.decoder_embed_dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], -1, h * p, w * p))

        x = self.decoder_pred(x) # Bx3xHxW
        return x

    def forward_loss(
        self,
        imgs,
        pred,
        tgts,
        mask,
        valid,
        epoch=0,
        no_gan=False,
        keep_loss_graph=False,
    ):
        """
        tgts: [N, 3, H, W]
        pred: [N, 3, H, W]
        mask: [N, L], 0 is keep, 1 is remove, 
        valid: [N, 3, H, W]
        epoch: 当前训练epoch，用于动态损失权重计算
        """
        mask = mask[:, :, None].repeat(1, 1, self.patch_size**2 * 3)
        mask = self.unpatchify(mask)

        # ignore if the unmasked pixels are all zeros
        imagenet_mean=torch.tensor([0.485, 0.456, 0.406]).to(tgts.device)[None, :, None, None]
        imagenet_std=torch.tensor([0.229, 0.224, 0.225]).to(tgts.device)[None, :, None, None]
        inds_ign = ((tgts * imagenet_std + imagenet_mean) * (1 - 1.*mask)).sum((1, 2, 3)) < 100*3
        if inds_ign.sum() > 0:
            valid[inds_ign] = 0.

        valid_ratio = valid.mean().detach()
        mask = mask * valid

        target = tgts
        if self.loss_func == "l1l2":
            loss = ((pred - target).abs() + (pred - target) ** 2.) * 0.5
        elif self.loss_func == "l1":
            loss = (pred - target).abs()
        elif self.loss_func == "l2":
            loss = (pred - target) ** 2.
        elif self.loss_func == "smoothl1":
            loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
        loss_l1l2 = (loss * mask).sum() / (mask.sum() + 1e-2)  # mean loss on removed patches


        transform_vgg = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

        with torch.cuda.amp.autocast(enabled=False):
            pred_img = transform_vgg(pred).float()
            target_img = transform_vgg(target).float()
            if getattr(self, "vgg_input_mode", "legacy") == "rgb":
                pred_img = pred_img * imagenet_std + imagenet_mean
                target_img = target_img * imagenet_std + imagenet_mean
            # 原作者设置：VGG loss 使用 Gram style；content 不进入总 loss。
            loss_style = self.vgg_loss(pred_img, target_img)
            loss_vgg = loss_style
            # Edge Loss

        edge_pred = self.improved_edge_detection(pred)
        edge_target = self.improved_edge_detection(target)
        loss_edge = F.l1_loss(edge_pred, edge_target)

        # Global, differentiable layout prior. Inputs are normalized; recover
        # grayscale foreground probabilities before computing projections.
        mean = imagenet_mean.to(dtype=pred.dtype)
        std = imagenet_std.to(dtype=pred.dtype)
        pred_rgb = (pred * std + mean).clamp(0, 1)
        tgt_rgb = (tgts * std + mean).clamp(0, 1)
        pred_gray_2d = 0.299 * pred_rgb[:, 0] + 0.587 * pred_rgb[:, 1] + 0.114 * pred_rgb[:, 2]
        tgt_gray_2d = 0.299 * tgt_rgb[:, 0] + 0.587 * tgt_rgb[:, 1] + 0.114 * tgt_rgb[:, 2]
        pred_fg = torch.sigmoid((0.9 - pred_gray_2d) * 10.0)
        tgt_fg = torch.sigmoid((0.9 - tgt_gray_2d) * 10.0)
        eps = pred_fg.new_tensor(1e-6)
        pred_mass = pred_fg.sum((1, 2), keepdim=True) + eps
        tgt_mass = tgt_fg.sum((1, 2), keepdim=True) + eps
        pred_row = pred_fg.sum(2, keepdim=True) / pred_mass
        tgt_row = tgt_fg.sum(2, keepdim=True) / tgt_mass
        pred_col = pred_fg.sum(1, keepdim=True) / pred_mass
        tgt_col = tgt_fg.sum(1, keepdim=True) / tgt_mass
        h, w = pred_fg.shape[-2:]
        yy = torch.linspace(-1, 1, h, device=pred.device, dtype=pred.dtype).view(1, h, 1)
        xx = torch.linspace(-1, 1, w, device=pred.device, dtype=pred.dtype).view(1, 1, w)
        pred_cx = (pred_fg * xx).sum((1, 2), keepdim=True) / pred_mass
        tgt_cx = (tgt_fg * xx).sum((1, 2), keepdim=True) / tgt_mass
        pred_cy = (pred_fg * yy).sum((1, 2), keepdim=True) / pred_mass
        tgt_cy = (tgt_fg * yy).sum((1, 2), keepdim=True) / tgt_mass
        loss_row = F.l1_loss(pred_row, tgt_row)
        loss_col = F.l1_loss(pred_col, tgt_col)
        loss_centroid = F.l1_loss(pred_cx, tgt_cx) + F.l1_loss(pred_cy, tgt_cy)
        loss_area = F.l1_loss(pred_fg.mean((1, 2)), tgt_fg.mean((1, 2)))
        structure_terms = (loss_row, loss_col, loss_centroid, loss_area)
        structure_coefficients = getattr(
            self, "structure_coefficients", (1.0, 1.0, 1.0, 1.0)
        )
        loss_structure = sum(
            coefficient * term
            for coefficient, term in zip(structure_coefficients, structure_terms)
        ) * getattr(self, "structure_common_scale", 1.0)

        phase, adv_weight, edge_weight = self.get_dynamic_loss_weights(epoch)
        structure_weight = getattr(self, 'structure_loss_weight', 0.0)
        structure_warmup_epochs = getattr(self, 'structure_warmup_epochs', 0)
        structure_warmup_duration = getattr(self, 'structure_warmup_duration', 0)
        _, phase_epoch = self.get_loss_phase(epoch)

        if phase_epoch < structure_warmup_epochs:
            structure_weight_current = 0.0
        elif structure_warmup_duration > 0 and phase_epoch < structure_warmup_epochs + structure_warmup_duration:
            progress = (phase_epoch - structure_warmup_epochs) / structure_warmup_duration
            structure_weight_current = structure_weight * progress
        else:
            structure_weight_current = structure_weight

        # Fixed high-frequency supervision for brush strokes. It uses the same
        # pixel-space masked region as reconstruction, then emphasizes strong
        # target edges without letting the target weight contribute gradients.
        detail_kernel_size = getattr(self, "detail_kernel_size", 5)
        detail_sigma = getattr(self, "detail_sigma", 1.0)
        detail_gradient_ratio = getattr(self, "detail_gradient_ratio", 0.5)
        pred_gray = rgb_to_gray(pred_rgb)
        tgt_gray = rgb_to_gray(tgt_rgb)
        pred_high = highpass(pred_gray, detail_kernel_size, detail_sigma)
        target_high = highpass(tgt_gray, detail_kernel_size, detail_sigma)
        pred_gx, pred_gy = sobel_gradients(pred_gray)
        target_gx, target_gy = sobel_gradients(tgt_gray)

        # mask has already been expanded to pixel space and multiplied by valid.
        detail_region = mask[:, :1].float()
        detail_region_ratio = detail_region.mean().detach()
        detail_region_counts = detail_region.flatten(1).sum(1)
        target_response = target_high.abs().detach()
        target_response = target_response / (
            target_response.mean(dim=(-2, -1), keepdim=True) + 1e-6
        )
        detail_weight = torch.clamp(1.0 + 2.0 * target_response, max=3.0)
        weighted_highpass_error = (
            detail_weight * detail_region * (pred_high - target_high).abs()
        )
        gradient_error = (
            detail_region
            * (
                (pred_gx - target_gx).abs()
                + (pred_gy - target_gy).abs()
            )
        )
        target_gradient_response = target_gx.abs() + target_gy.abs()

        if getattr(self, "detail_per_sample_normalize", False):
            detail_denom = detail_region_counts.clamp_min(1.0)
            loss_highpass = (
                weighted_highpass_error.flatten(1).sum(1) / detail_denom
            ).mean()
            loss_gradient = (
                gradient_error.flatten(1).sum(1) / detail_denom
            ).mean()
            target_highpass_mean = (
                (detail_region * target_high.abs()).flatten(1).sum(1)
                / detail_denom
            ).mean()
            target_gradient_mean = (
                (detail_region * target_gradient_response).flatten(1).sum(1)
                / detail_denom
            ).mean()
        else:
            detail_denom = detail_region.sum() + 1e-6
            loss_highpass = weighted_highpass_error.sum() / detail_denom
            loss_gradient = gradient_error.sum() / detail_denom
            target_highpass_mean = (
                (detail_region * target_high.abs()).sum() / detail_denom
            )
            target_gradient_mean = (
                (detail_region * target_gradient_response).sum() / detail_denom
            )

        loss_detail = loss_highpass + detail_gradient_ratio * loss_gradient

        detail_weight_target = getattr(self, "detail_loss_weight", 0.03)
        detail_warmup_epochs = getattr(self, "detail_warmup_epochs", 4)
        detail_warmup_duration = getattr(self, "detail_warmup_duration", 4)
        if phase_epoch < detail_warmup_epochs:
            detail_weight_current = 0.0
        elif (
            detail_warmup_duration > 0
            and phase_epoch < detail_warmup_epochs + detail_warmup_duration
        ):
            progress = (
                phase_epoch - detail_warmup_epochs
            ) / detail_warmup_duration
            detail_weight_current = detail_weight_target * progress
        else:
            detail_weight_current = detail_weight_target

        if hasattr(self, "stage3_update"):
            progress = min(1.0, self.stage3_update / 40.0)
            structure_weight_current = structure_weight * progress
            detail_weight_current = detail_weight_target * progress
            edge_weight = 0.3 * progress

        structure_weighted_contribution = loss_structure.detach() * (
            structure_weight_current
            if torch.is_tensor(structure_weight_current)
            else loss_structure.new_tensor(structure_weight_current)
        )
        detail_weighted_contribution = loss_detail.detach() * (
            detail_weight_current
            if torch.is_tensor(detail_weight_current)
            else loss_detail.new_tensor(detail_weight_current)
        )
        gradient_contribution = loss_gradient.detach() * (
            detail_weight_current
            if torch.is_tensor(detail_weight_current)
            else loss_gradient.new_tensor(detail_weight_current)
        ) * (
            detail_gradient_ratio
            if torch.is_tensor(detail_gradient_ratio)
            else loss_gradient.new_tensor(detail_gradient_ratio)
        )

        loss = (
            loss_l1l2
            + loss_vgg
            + edge_weight * loss_edge
            + structure_weight_current * loss_structure
            + detail_weight_current * loss_detail
        )
        if keep_loss_graph:
            self.last_loss_graph = {
                'total_weighted': loss,
                'recon_weighted': loss_l1l2,
                'style_weighted': loss_vgg,
                'edge_weighted': edge_weight * loss_edge,
                'structure_weighted': structure_weight_current * loss_structure,
                'detail_weighted': detail_weight_current * loss_detail,
                'structure': loss_structure,
                'structure_row': loss_row,
                'structure_col': loss_col,
                'structure_centroid': loss_centroid,
                'structure_area': loss_area,
                'detail': loss_detail,
                'highpass': loss_highpass,
                'gradient': loss_gradient,
                'highpass_weighted': detail_weight_current * loss_highpass,
                'gradient_weighted': (
                    detail_weight_current * detail_gradient_ratio * loss_gradient
                ),
            }

        self.last_loss_components = {
            'structure': loss_structure.detach(),
            'structure_row': loss_row.detach(), 'structure_col': loss_col.detach(),
            'structure_centroid': loss_centroid.detach(), 'structure_area': loss_area.detach(),
            'structure_weight': structure_weight_current,
            'structure_weighted': structure_weighted_contribution,
            'detail': loss_detail.detach(),
            'highpass': loss_highpass.detach(),
            'gradient': loss_gradient.detach(),
            'detail_weight': detail_weight_current,
            'detail_weighted': detail_weighted_contribution,
            'gradient_contribution': gradient_contribution,
            'detail_region_ratio': detail_region_ratio,
            'valid_ratio': valid_ratio,
            'target_highpass_mean': target_highpass_mean.detach(),
            'target_gradient_mean': target_gradient_mean.detach(),
            'structure_row_grad_ok': float(
                loss_row.requires_grad and loss_row.grad_fn is not None
            ),
            'structure_col_grad_ok': float(
                loss_col.requires_grad and loss_col.grad_fn is not None
            ),
            'structure_centroid_grad_ok': float(
                loss_centroid.requires_grad and loss_centroid.grad_fn is not None
            ),
            'structure_area_grad_ok': float(
                loss_area.requires_grad and loss_area.grad_fn is not None
            ),
            'highpass_grad_ok': float(
                loss_highpass.requires_grad and loss_highpass.grad_fn is not None
            ),
            'gradient_grad_ok': float(
                loss_gradient.requires_grad and loss_gradient.grad_fn is not None
            ),
        }

        adv_loss = pred.new_tensor(0.0)
        if not no_gan:
            # DDP static_graph requires the set of used parameters to stay fixed.
            # Keep the discriminator branch in the graph even while adv_weight is 0.
            pred_resized = self.resize(pred)
            fake_output = self.discriminator(pred_resized)
            fake_logits = fake_output.squeeze(1)
            real_labels = torch.ones_like(fake_logits)
            adv_loss = F.binary_cross_entropy_with_logits(fake_logits, real_labels)
            loss = loss + adv_weight * adv_loss

        # === 调试: 打印原作者式固定系数与本 batch 数值贡献 ===
        self._dbg_step = getattr(self, '_dbg_step', 0) + 1
        _rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        if _rank0 and self._dbg_step % 50 == 1:
            contribs = {
                "recon": loss_l1l2.detach(),
                "style": loss_style.detach(),
                "edge": loss_edge.detach() * loss_edge.new_tensor(edge_weight),
                "structure": loss_structure.detach() * loss_structure.new_tensor(
                    structure_weight_current
                ),
                "detail": loss_detail.detach() * loss_detail.new_tensor(
                    detail_weight_current
                ),
                "adv": adv_loss.detach() * adv_loss.new_tensor(adv_weight),
            }
            contrib_total = loss_l1l2.new_tensor(0.0)
            for contrib in contribs.values():
                contrib_total = contrib_total + contrib
            contrib_total = contrib_total.clamp_min(1e-12)
            actual_shares = {name: contribs[name] / contrib_total for name in contribs}

            def scalar(x):
                return float(x.detach().item())

            print(f"[loss-dbg] epoch={epoch} phase={phase} step={self._dbg_step} "
                  f"raw: recon({self.loss_func})={scalar(loss_l1l2):.4f} "
                  f"style={scalar(loss_style):.4f} edge={scalar(loss_edge):.4f} "
                  f"structure={scalar(loss_structure):.4f} "
                  f"detail={scalar(loss_detail):.4f} "
                  f"adv={scalar(adv_loss):.4f} | "
                  f"w: recon=1.000 style=1.000 "
                  f"edge={edge_weight:.3f} structure={structure_weight_current:.3f} "
                  f"detail={detail_weight_current:.3f} adv={adv_weight:.3f} | "
                  f"contrib: recon={scalar(contribs['recon']):.4f} "
                  f"style={scalar(contribs['style']):.4f} "
                  f"edge={scalar(contribs['edge']):.4f} "
                  f"structure={scalar(contribs['structure']):.4f} "
                  f"detail={scalar(contribs['detail']):.4f} "
                  f"adv={scalar(contribs['adv']):.4f} | "
                  f"share%: recon={scalar(actual_shares['recon'])*100:.1f} "
                  f"style={scalar(actual_shares['style'])*100:.1f} "
                  f"edge={scalar(actual_shares['edge'])*100:.1f} "
                  f"structure={scalar(actual_shares['structure'])*100:.1f} "
                  f"detail={scalar(actual_shares['detail'])*100:.1f} "
                  f"adv={scalar(actual_shares['adv'])*100:.1f}", flush=True)
        # === 临时调试结束 ===
        return loss, loss_l1l2, loss_vgg

    def forward(self, imgs, tgts, bool_masked_pos=None, valid=None, epoch=0, no_gan=False):
        #imgs = self.tps(imgs)
        #tgts = self.tps(tgts)
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((imgs.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(imgs.device)
        else:
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
        latent = self.forward_encoder(imgs, tgts, bool_masked_pos)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss, loss_l1l2, loss_vgg = self.forward_loss(
            imgs, pred, tgts, bool_masked_pos, valid, epoch=epoch, no_gan=no_gan
        )
        return loss, loss_l1l2, loss_vgg, self.patchify(pred), bool_masked_pos, pred



def vit_large_patch16_input896x448_win_dec64_8glb_sl1(**kwargs):
    model = Fontify(
        img_size=(896, 448), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        drop_path_rate=0.1, window_size=14, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=(list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + \
                                list(range(12, 14)), list(range(15, 17)), list(range(18, 20)), list(range(21, 23))),
        residual_block_indexes=[], use_rel_pos=True, out_feature="last_feat",
        decoder_embed_dim=64,
        loss_func="smoothl1",
        **kwargs)
    return model

def vit_base_patch16_input896x448_win_dec64_8glb_sl1(**kwargs):
    model = Fontify(
        img_size=(896, 448), patch_size=16, embed_dim=768, depth=12, num_heads=12,
        drop_path_rate=0.1, window_size=14, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=(list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + \
                                list(range(12, 14)), list(range(15, 17)), list(range(18, 20)), list(range(21, 23))),
        residual_block_indexes=[], use_rel_pos=True, out_feature="last_feat",
        decoder_embed_dim=64,
        loss_func="smoothl1",
        **kwargs)
    return model

def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)
