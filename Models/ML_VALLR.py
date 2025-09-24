import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Tuple, Optional

# =============================
# Transformer blocks (ViT style)
# =============================

class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)  # (B, N, D)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ======================================
# Spatiotemporal ViT with tubelet embed
# ======================================

class VideoViTMasked(nn.Module):
    """
    Spatiotemporal ViT with Conv3d tubelet embedding.
    - Keeps (or restores) full token sequence length so it matches VideoMAE-style N = (T/tubelet)*(H/ps)*(W/ps).
    - Set mask_ratio=0.0 to keep N fixed (e.g., 1568 for T=16, tubelet=2, H=W=224, patch=16).

    Input:  (B, C, T, H, W)
    Output: (B, N, D)
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,        # <<< important to match VideoMAE (16 frames -> T' = 8)
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        mask_ratio: float = 0.0,      # 0.0 keeps N fixed; >0.0 masks then restores with zeros
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size   = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dim    = embed_dim
        self.mask_ratio   = mask_ratio

        # Tubelet patchify: (B, C, T, H, W) -> (B, D, T', H', W')
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            bias=True
        )

        # Positional embeddings: created on first forward for the current N
        self.register_buffer("pos_embed", None, persistent=False)  # created as nn.Parameter at runtime

        # Transformer encoder
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    @torch.no_grad()
    def _rand_mask_idx(self, B: int, N: int, device: torch.device):
        K = max(1, int(round(N * (1.0 - self.mask_ratio))))
        perm = torch.argsort(torch.rand(B, N, device=device), dim=1)
        kept_idx = perm[:, :K]  # (B, K)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, kept_idx, False)
        return kept_idx, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        returns tokens: (B, N, D)
        """
        B, C, T, H, W = x.shape
        device = x.device

        # Tubelet patchify
        x = self.proj(x)                  # (B, D, T', H', W')
        Tp, Hp, Wp = x.shape[-3], x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # (B, N, D), N = Tp*Hp*Wp
        N = x.size(1)

        # Create/resize learnable positional embeddings for this N
        if (self.pos_embed is None) or (self.pos_embed.size(1) != N):
            self.pos_embed = nn.Parameter(torch.zeros(1, N, self.embed_dim, device=device))

        tokens = x + self.pos_embed  # (B, N, D)

        # Masking (optional). If masked, restore to original length with zeros.
        if self.mask_ratio > 0.0:
            kept_idx, mask = self._rand_mask_idx(B, N, device)
            batch_idx = torch.arange(B, device=device).unsqueeze(-1)
            tokens_vis = tokens[batch_idx, kept_idx]  # (B, K, D)
        else:
            tokens_vis = tokens
            kept_idx = None  # unused

        # Encode
        for blk in self.blocks:
            tokens_vis = blk(tokens_vis)
        tokens_vis = self.norm(tokens_vis)  # (B, K or N, D)

        if self.mask_ratio > 0.0:
            # Restore masked positions with zeros (keeps length == N)
            out = torch.zeros(B, N, self.embed_dim, device=device, dtype=tokens_vis.dtype)
            out[torch.arange(B, device=device).unsqueeze(-1), kept_idx] = tokens_vis
            return out  # (B, N, D)
        else:
            return tokens_vis  # (B, N, D)


# ============================
# VALLR model (pure PyTorch)
# ============================

class VALLR(nn.Module):
    """
    VALLR with spatiotemporal ViT (tubelet embedding) + temporal Conv1d downsampling + adapter + CTC head.

    Returns (logits, adapted_features):
        logits:           (B, N_down, num_classes)
        adapted_features: (B, N_down, ctc_hidden_size)
    """
    def __init__(
        self,
        adapter_dim: int,
        feature_size: int = 768,      # ViT embed dim
        ctc_hidden_size: int = 768,
        num_classes: int = 32,

        # ViT (video) config
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,        # <<< with T=16 gives T' = 8
        in_channels: int = 3,
        vit_depth: int = 8,
        vit_heads: int = 12,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.2,
        vit_attn_dropout: float = 0.0,
        vit_mask_ratio: float = 0.9,  # 0.0 keeps N fixed (e.g., 1568)
    ):
        super().__init__()

        # 1) Spatiotemporal ViT backbone producing tokens (B, N, D)
        self.backbone = VideoViTMasked(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_chans=in_channels,
            embed_dim=feature_size,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
            attn_dropout=vit_attn_dropout,
            mask_ratio=vit_mask_ratio,
        )

        # 2) Temporal downsampling over token axis N (treat tokens as a "time" sequence)
        #    Safer strides (avoid collapsing very short sequences; fine for N=1568).
        self.downsampling = nn.Sequential(
            nn.Conv1d(in_channels=feature_size, out_channels=adapter_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(adapter_dim, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(),

            nn.Conv1d(in_channels=adapter_dim, out_channels=adapter_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(adapter_dim, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(),

            nn.Conv1d(in_channels=adapter_dim, out_channels=adapter_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(adapter_dim, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=adapter_dim, out_channels=adapter_dim, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(adapter_dim, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=5, stride=6)  # Adjust kernel size and stride to control final length
        )

        # 3) Adapter to the CTC hidden size
        self.adapter = nn.Sequential(
            nn.Linear(adapter_dim, ctc_hidden_size),
            nn.ReLU(inplace=True),
        )

        # 4) CTC head
        self.ctc_head = nn.Linear(ctc_hidden_size, num_classes)

        # Init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.kaiming_normal_(module.weight, nonlinearity='relu')
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            init.ones_(module.weight)
            init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)

    def forward(self, video_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Accepts either (B, C, T, H, W) or (B, T, C, H, W).
        Returns:
            logits  : (B, N_down, num_classes)
            adapted : (B, N_down, ctc_hidden_size)
        """
        # Normalize layout to (B, C, T, H, W)
        if video_inputs.ndim != 5:
            raise ValueError(f"Expected 5D input, got shape {video_inputs.shape}")

        if video_inputs.shape[1] in (1, 3):        # (B, C, T, H, W)
            vid = video_inputs
        else:                                       # (B, T, C, H, W)
            vid = video_inputs.permute(0, 2, 1, 3, 4).contiguous()

        # 1) ViT backbone: tokens (B, N, D)
        tokens = self.backbone(vid)                 # e.g., (B, 1568, 768)

        # 2) Temporal downsampling over token axis
        x = tokens.permute(0, 2, 1)                 # (B, D, N)
        x = self.downsampling(x)                    # (B, adapter_dim, N_down)
        x = x.permute(0, 2, 1)                      # (B, N_down, adapter_dim)

        # 3) Adapter
        adapted = self.adapter(x)                   # (B, N_down, ctc_hidden_size)

        # 4) CTC logits
        logits = self.ctc_head(adapted)             # (B, N_down, num_classes)

        # print("Logits shape", logits.shape)

        return logits, adapted