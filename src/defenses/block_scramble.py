# Baseline B4: Pixel Encryption / Cut-Layer Block Scrambling (Xáo trộn khối thủ công)
import torch
import torch.nn as nn


class BlockScrambleDefense(nn.Module):
    """
    Baseline B4: Chia feature map z tại cut-layer thành các khối kích thước (block_size x block_size)
    và hoán vị ngẫu nhiên vị trí các khối hoặc các pixel trong từng khối bằng khóa bí mật.
    """
    def __init__(self, block_size=4, seed=42):
        super().__init__()
        self.block_size = block_size
        self.seed = seed
        self.perm = None

    def _init_perm(self, h, w, device):
        if self.perm is None:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed)
            num_h = h // self.block_size
            num_w = w // self.block_size
            total_blocks = num_h * num_w
            self.perm = torch.randperm(total_blocks, generator=g, device=device)

    def forward(self, z):
        # z: [B, C, H, W]
        b, c, h, w = z.shape
        bs = self.block_size
        if h % bs != 0 or w % bs != 0:
            return z

        self._init_perm(h, w, z.device)
        num_h, num_w = h // bs, w // bs

        # Tách thành các blocks
        # [B, C, num_h, bs, num_w, bs] -> [B, num_blocks, C, bs, bs]
        blocks = z.view(b, c, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).contiguous()
        blocks = blocks.view(b, num_h * num_w, c, bs, bs)

        # Hoán vị các blocks
        scrambled_blocks = blocks[:, self.perm, :, :, :]

        # Ghép lại
        scrambled = scrambled_blocks.view(b, num_h, num_w, c, bs, bs)
        scrambled = scrambled.permute(0, 3, 1, 4, 2, 5).contiguous()
        scrambled = scrambled.view(b, c, h, w)
        return scrambled
