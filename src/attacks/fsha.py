# Module Tấn công Chủ động FSHA (Feature Space Hijacking Attack - Pasquini et al., CCS 2021 & Gawron et al., 2022)
# Bề mặt tấn công chủ động điều khiển gradient kiểm chứng sự sụp đổ của B0-B6 (Novelty N1, CV2)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.resnet import ClientModel, ServerModel
from ..metrics.reconstruction import psnr_ssim, calculate_lpips
from ..metrics.distance_correlation import distance_correlation
from ..data.cifar import CIFAR10_MEAN, CIFAR10_STD


class FSHADiscriminator(nn.Module):
    """
    Discriminator / Critic đa tỷ lệ của Server độc hại trong FSHA:
    Phân biệt giữa Z_client = Defense(Client(X_private)) và Z_pilot = Pilot_Encoder(X_public).
    Thiết kế chuẩn WGAN-GP / SpectralNorm:
    - KHÔNG dùng BatchNorm2d (để bảo toàn hoàn toàn đạo hàm theo kỳ vọng và phương sai kênh gửi về Client).
    - KHÔNG dùng AdaptiveAvgPool2d ở đầu vào (để giữ nguyên độ phân giải không gian 32x32 tại cut-layer).
    - Kết hợp nhánh Pixel (1x1) và nhánh Patch/Global (3x3 strided) có Spectral Normalization.
    """
    def __init__(self, in_channels=64, hidden_dim=128, use_spectral_norm=True):
        super().__init__()
        sn = nn.utils.spectral_norm if use_spectral_norm else (lambda m: m)

        self.pixel_branch = nn.Sequential(
            sn(nn.Conv2d(in_channels, hidden_dim, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)),
        )

        self.spatial_branch = nn.Sequential(
            sn(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(hidden_dim * 2, 1, kernel_size=3, stride=1, padding=1)),
        )

    def forward_features(self, z):
        p_score = self.pixel_branch(z).mean(dim=[1, 2, 3], keepdim=False).unsqueeze(1)
        s_map = self.spatial_branch(z)
        s_score = s_map.mean(dim=[1, 2, 3], keepdim=False).unsqueeze(1)
        return 0.5 * (p_score + s_score), s_map

    def forward(self, z):
        score, _ = self.forward_features(z)
        return score


class FSHAPilotAutoEncoder(nn.Module):
    """
    Pilot AutoEncoder (f_tilde, f_tilde^-1) của Server huấn luyện trên tập dữ liệu phụ (D_public):
    - Pilot Encoder f_tilde: Ánh xạ X_public -> Z_pilot in R^(B x 64 x 32 x 32)
      với tùy chọn chiếu trực giao vào không gian con mục tiêu d_target (target_dim).
    - Pilot Decoder f_tilde^-1: Giải mã từ Z in R^(B x 64 x 32 x 32) -> X_hat in R^(B x 3 x 32 x 32)
      trực tiếp trong miền chuẩn hóa CIFAR-10 (không bị cắt cụt bởi Tanh).
    """
    def __init__(self, in_channels=3, latent_channels=64, target_dim=None, hidden_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.target_dim = target_dim

        if in_channels == 3 and latent_channels == 64:
            self.encoder = ClientModel()
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, latent_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(latent_channels),
                nn.ReLU(inplace=True),
            )

        if target_dim is not None and 0 < target_dim < latent_channels:
            init_mat = torch.randn(latent_channels, target_dim)
            q, _ = torch.linalg.qr(init_mat)
            self.v_target = nn.Parameter(q)
        else:
            self.v_target = None

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
        )

    def project_target(self, z):
        if self.v_target is not None:
            q, _ = torch.linalg.qr(self.v_target)
            p_mat = torch.matmul(q, q.t())  # Phép chiếu trực giao idempotent P^2 = P
            b, c, h, w = z.shape
            z_flat = z.permute(0, 2, 3, 1).reshape(-1, c)
            z_proj = torch.matmul(z_flat, p_mat).reshape(b, h, w, c).permute(0, 3, 1, 2)
            return z_proj
        return z

    def encode(self, x):
        z = self.encoder(x)
        return self.project_target(z)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return z, x_rec


class FSHAServerAdapter(nn.Module):
    """
    Module thích ứng phía Server độc hại (Server-side Inverse Spatial Operator) trong FSHA:
    Hỗ trợ Server đảo ngược các phép biến đổi cấu trúc không gian tại cut-layer (B4, B5)
    đồng thời truyền nguyên vẹn 100% đạo hàm đối kháng xuyên qua cut-layer về phía Client.
    """
    def __init__(self, channels=64, defense=None):
        super().__init__()
        self.defense = defense
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _invert_known_spatial_structure(self, z):
        if self.defense is None:
            return z
        cls_name = self.defense.__class__.__name__
        # Với B4 (BlockScramble): Theo nguyên lý Kerckhoffs / giải ghép mảnh Jigsaw trên Server
        if cls_name == "BlockScrambleDefense":
            if getattr(self.defense, "perm", None) is None:
                self.defense._init_perm(z.size(2), z.size(3), z.device)
            if getattr(self.defense, "perm", None) is not None:
                b, c, h, w = z.shape
                bs = self.defense.block_size
                if h % bs == 0 and w % bs == 0:
                    num_h, num_w = h // bs, w // bs
                    inv_perm = torch.argsort(self.defense.perm)
                    blocks = z.view(b, c, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).contiguous()
                    blocks = blocks.view(b, num_h * num_w, c, bs, bs)
                    unscrambled = blocks[:, inv_perm, :, :, :]
                    unscrambled = unscrambled.view(b, num_h, num_w, c, bs, bs).permute(0, 3, 1, 4, 2, 5).contiguous()
                    return unscrambled.view(b, c, h, w)
        # Với B5 (DeformableOperator): Khôi phục trường lệch lưới (Inverse Grid Warping)
        elif cls_name == "DeformableOperatorDefense":
            if getattr(self.defense, "grid_offset", None) is None:
                self.defense._init_offset(z.size(2), z.size(3), z.device)
            if getattr(self.defense, "grid_offset", None) is not None:
                b, c, h, w = z.shape
                y_grid, x_grid = torch.meshgrid(
                    torch.linspace(-1, 1, h, device=z.device),
                    torch.linspace(-1, 1, w, device=z.device),
                    indexing="ij",
                )
                base_grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
                offset = self.defense.grid_offset.permute(0, 2, 3, 1).repeat(b, 1, 1, 1)
                inv_grid = torch.clamp(base_grid - offset, -1.0, 1.0)
                return F.grid_sample(z, inv_grid, mode="bilinear", padding_mode="reflection", align_corners=True)
        return z

    def forward(self, z):
        z_aligned = self._invert_known_spatial_structure(z)
        return z_aligned * self.gain + self.bias


def compute_gradient_penalty(discriminator, z_real, z_fake, device="cuda"):
    """
    Tính Gradient Penalty (WGAN-GP, Gulrajani et al. 2017) đảm bảo điều kiện 1-Lipschitz cho Discriminator.
    """
    b = min(z_real.size(0), z_fake.size(0))
    z_r = z_real[:b].detach()
    z_f = z_fake[:b].detach()
    alpha = torch.rand(b, 1, 1, 1, device=device, dtype=z_r.dtype)
    interpolates = (alpha * z_r + (1.0 - alpha) * z_f).requires_grad_(True)
    d_inter = discriminator(interpolates)
    grads = torch.autograd.grad(
        outputs=d_inter,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(b, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def train_fsha_step(
    client,
    discriminator,
    pilot_ae,
    x_private,
    x_public,
    opt_d,
    opt_ae,
    opt_c=None,
    device="cuda",
    defense=None,
    grad_scale=1.0,
    loss_type="bce",
    gp_weight=10.0,
):
    """
    Một bước tối ưu FSHA đơn lẻ (giữ tương thích ngược 100% với run_tests.py):
    1. Huấn luyện Pilot AE tái tạo x_public.
    2. Huấn luyện Discriminator phân biệt z_client vs z_pilot.
    3. Sinh gradient độc hại đẩy z_client khớp phân phối không gian mục tiêu z_pilot.
    """
    x_private = x_private.to(device)
    x_public = x_public.to(device)

    # 1. Update Pilot AE
    if opt_ae is not None:
        opt_ae.zero_grad()
        z_pilot, x_rec_pilot = pilot_ae(x_public)
        loss_ae = F.mse_loss(x_rec_pilot, x_public)
        loss_ae.backward()
        opt_ae.step()
    else:
        with torch.no_grad():
            z_pilot, x_rec_pilot = pilot_ae(x_public)
            loss_ae = F.mse_loss(x_rec_pilot, x_public)

    # 2. Update Discriminator
    opt_d.zero_grad()
    with torch.no_grad():
        z_client = client(x_private)
        if defense is not None:
            z_client = defense(z_client)
        z_pilot_det, _ = pilot_ae(x_public)

    pred_client = discriminator(z_client)
    pred_pilot = discriminator(z_pilot_det)

    if loss_type == "wgan-gp":
        gp = compute_gradient_penalty(discriminator, z_pilot_det, z_client, device=device)
        loss_d = pred_client.mean() - pred_pilot.mean() + gp_weight * gp
    elif loss_type == "softplus":
        loss_d = F.softplus(-pred_pilot).mean() + F.softplus(pred_client).mean()
    else:
        bce = nn.BCEWithLogitsLoss()
        loss_d = bce(pred_pilot, torch.ones_like(pred_pilot)) + bce(pred_client, torch.zeros_like(pred_client))

    loss_d.backward()
    opt_d.step()

    # 3. Client adversarial gradient (nếu có cập nhật client)
    loss_c = None
    if opt_c is not None:
        opt_c.zero_grad()
        z_c_raw = client(x_private)
        z_c_adv = defense(z_c_raw) if defense is not None else z_c_raw
        pred_c_adv = discriminator(z_c_adv)

        if loss_type == "wgan-gp":
            loss_adv = -pred_c_adv.mean()
        elif loss_type == "softplus":
            loss_adv = F.softplus(-pred_c_adv).mean()
        else:
            bce = nn.BCEWithLogitsLoss()
            loss_adv = bce(pred_c_adv, torch.ones_like(pred_c_adv))

        z_reenc = pilot_ae.encode(pilot_ae.decode(z_c_adv))
        loss_subspace = F.mse_loss(z_c_adv, z_reenc)
        loss_c = grad_scale * (loss_adv + loss_subspace)
        loss_c.backward()
        opt_c.step()

    return {
        "loss_ae": float(loss_ae.item()),
        "loss_d": float(loss_d.item()),
        "loss_c": float(loss_c.item()) if loss_c is not None else 0.0,
    }


def fit_fsha_pilot(
    pilot_ae,
    pilot_server,
    pub_loader,
    test_loader,
    epochs=10,
    lr=1e-3,
    device="cuda",
    b0_ckpt=None,
    b1_dec_ckpt=None,
    save_path=None,
    mean=CIFAR10_MEAN,
    std=CIFAR10_STD,
):
    """
    PHA 1 (FITTING):
    Server độc hại xác lập không gian đặc trưng mục tiêu Z_pilot = f_tilde(X_pub)
    (khởi tạo cơ sở trực chuẩn top-d_target qua SVD trên D_pub để bảo toàn >93% năng lượng đặc trưng)
    và huấn luyện khóa giải mã f_tilde^-1 trên tập dữ liệu phụ D_pub.
    """
    if b0_ckpt is not None and os.path.isfile(b0_ckpt):
        ckpt = torch.load(b0_ckpt, map_location=device)
        if "client" in ckpt and hasattr(pilot_ae, "encoder"):
            pilot_ae.encoder.load_state_dict(ckpt["client"])
        if "server" in ckpt and pilot_server is not None:
            pilot_server.load_state_dict(ckpt["server"])

    if b1_dec_ckpt is not None and os.path.isfile(b1_dec_ckpt):
        dec_ckpt = torch.load(b1_dec_ckpt, map_location=device)
        state = dec_ckpt["decoder"] if isinstance(dec_ckpt, dict) and "decoder" in dec_ckpt else dec_ckpt
        clean_state = {k.replace("net.", ""): v for k, v in state.items()}
        pilot_ae.decoder.load_state_dict(clean_state, strict=False)

    for p in pilot_ae.encoder.parameters():
        p.requires_grad = False
    pilot_ae.encoder.eval()

    # Khởi tạo không gian con mục tiêu v_target bằng Top-d_target thành phần chính SVD trên D_pub
    if pilot_ae.v_target is not None:
        d_target = pilot_ae.v_target.size(1)
        z_samples = []
        with torch.no_grad():
            for idx, (x_p, _) in enumerate(pub_loader):
                if idx >= 6:
                    break
                z_raw = pilot_ae.encoder(x_p.to(device))
                z_sub = z_raw[:, :, ::4, ::4].permute(0, 2, 3, 1).reshape(-1, pilot_ae.latent_channels)
                z_samples.append(z_sub)
            z_mat = torch.cat(z_samples, dim=0)
            _, _, vh = torch.linalg.svd(z_mat, full_matrices=False)
            v_top = vh[:d_target, :].t().contiguous()
            pilot_ae.v_target.data.copy_(v_top)
        pilot_ae.v_target.requires_grad = False

    if pilot_server is not None:
        pilot_server.eval()
        for p in pilot_server.parameters():
            p.requires_grad = False

    opt = torch.optim.Adam(pilot_ae.decoder.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    history = []
    for ep in range(1, epochs + 1):
        pilot_ae.decoder.train()

        total_mse, total_samples = 0.0, 0
        for x_pub, _ in pub_loader:
            x_pub = x_pub.to(device, non_blocking=True)
            bs = x_pub.size(0)

            opt.zero_grad()
            with torch.no_grad():
                z_pilot = pilot_ae.encode(x_pub)
            x_rec = pilot_ae.decode(z_pilot)
            # Denoising regularization giúp Pilot Decoder bền vững với nhiễu và sai lệch nhỏ ở Pha 2
            x_rec_noisy = pilot_ae.decode(z_pilot + 0.04 * torch.randn_like(z_pilot))

            loss = F.mse_loss(x_rec, x_pub) + 0.25 * F.mse_loss(x_rec_noisy, x_pub)
            loss.backward()
            opt.step()

            total_mse += loss.item() * bs
            total_samples += bs

        sched.step()

        # Đánh giá chất lượng khóa giải mã trên tập Test
        pilot_ae.eval()
        psnr_sum, ssim_sum, correct, n_eval = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for idx, (x_t, y_t) in enumerate(test_loader):
                if idx >= 8:
                    break
                x_t, y_t = x_t.to(device), y_t.to(device)
                bs = x_t.size(0)
                z_t, x_hat = pilot_ae(x_t)
                p, s = psnr_ssim(x_t, x_hat, mean, std)
                psnr_sum += p * bs
                ssim_sum += s * bs
                if pilot_server is not None:
                    correct += (pilot_server(z_t).argmax(1) == y_t).sum().item()
                n_eval += bs

        avg_psnr = psnr_sum / max(n_eval, 1)
        avg_ssim = ssim_sum / max(n_eval, 1)
        avg_acc = correct / max(n_eval, 1)
        history.append({
            "epoch": ep,
            "train_mse": total_mse / max(total_samples, 1),
            "pilot_psnr": avg_psnr,
            "pilot_ssim": avg_ssim,
            "pilot_acc": avg_acc,
        })
        print(
            f"[Phase 1 Fitting] Epoch {ep:2d}/{epochs} | Recon MSE: {total_mse/max(total_samples, 1):.4f} | "
            f"Pilot PSNR: {avg_psnr:.2f} dB | Pilot SSIM: {avg_ssim:.4f} | Pilot Task Acc: {avg_acc*100:.2f}%",
            flush=True,
        )

    # Đóng băng toàn bộ Pilot AE sau Pha 1 để làm khóa giải mã cố định cho Pha 2
    pilot_ae.eval()
    for p in pilot_ae.parameters():
        p.requires_grad = False

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(
            {
                "pilot_ae": pilot_ae.state_dict(),
                "pilot_server": pilot_server.state_dict() if pilot_server is not None else None,
                "target_dim": pilot_ae.target_dim,
                "history": history,
            },
            save_path,
        )
        print(f"[Phase 1 Saved] Đã lưu khóa giải mã Pilot AE tại: {save_path}", flush=True)

    return history


def warmup_fsha_discriminator(
    client,
    discriminator,
    pilot_ae,
    priv_loader,
    pub_loader,
    opt_d,
    device="cuda",
    defense=None,
    server_adapter=None,
    num_batches=12,
    loss_type="softplus",
    gp_weight=10.0,
):
    """
    Khởi động (Warm-up) Discriminator vài batches trước khi truyền đạo hàm về Client
    để tránh đạo hàm ngẫu nhiên ở bước 0 làm lệch hướng Client.
    """
    client.eval()
    pilot_ae.eval()
    discriminator.train()
    if defense is not None:
        defense.eval()
    if server_adapter is not None:
        server_adapter.eval()

    pub_iter = iter(pub_loader)
    for idx, (x_priv, _) in enumerate(priv_loader):
        if idx >= num_batches:
            break
        try:
            x_pub, _ = next(pub_iter)
        except StopIteration:
            pub_iter = iter(pub_loader)
            x_pub, _ = next(pub_iter)

        x_priv = x_priv.to(device, non_blocking=True)
        x_pub = x_pub.to(device, non_blocking=True)

        opt_d.zero_grad()
        with torch.no_grad():
            z_c = client(x_priv)
            if defense is not None:
                z_c = defense(z_c)
            if server_adapter is not None:
                z_c = server_adapter(z_c)
            z_p = pilot_ae.encode(x_pub)

        pred_c = discriminator(z_c)
        pred_p = discriminator(z_p)

        if loss_type == "wgan-gp":
            gp = compute_gradient_penalty(discriminator, z_p, z_c, device=device)
            loss_d = pred_c.mean() - pred_p.mean() + gp_weight * gp
        else:
            loss_d = F.softplus(-pred_p).mean() + F.softplus(pred_c).mean()

        loss_d.backward()
        opt_d.step()


def train_fsha_hijack_epoch(
    client,
    discriminator,
    pilot_ae,
    pilot_server,
    task_server,
    priv_loader,
    pub_loader,
    opt_c,
    opt_d,
    opt_task_s,
    device="cuda",
    defense=None,
    server_adapter=None,
    opt_adapter=None,
    client_opt_is_dpsgd=False,
    grad_scale=1.0,
    loss_type="softplus",
    gp_weight=10.0,
    task_grad_weight=0.0,
):
    """
    PHA 2 (HIJACKING):
    Thực hiện 1 epoch tấn công chủ động FSHA trong quá trình huấn luyện Split Learning (tối ưu single-pass):
    1. Forward Client 1 lần duy nhất: z_c = client(x_priv), z_trans = defense(z_c).
    2. Cập nhật Discriminator D phân biệt z_trans.detach() và z_pilot.
    3. Tính hàm mục tiêu độc hại của Server trên z_d = z_trans.detach().requires_grad_(True)
       và truyền gradient giả mạo (nhân với grad_scale) ngược qua cut-layer về Client.
    """
    client.train()
    discriminator.train()
    pilot_ae.eval()
    if pilot_server is not None:
        pilot_server.eval()
    if task_server is not None:
        task_server.train()
    if defense is not None:
        defense.train()
    if server_adapter is not None:
        server_adapter.train()

    ce = nn.CrossEntropyLoss()
    pub_iter = iter(pub_loader)

    total_loss_d = 0.0
    total_loss_adv = 0.0
    total_loss_sub = 0.0
    total_task_acc = 0.0
    total_samples = 0

    is_additive_noise_defense = (
        defense is not None and defense.__class__.__name__ == "GaussianNoise"
    )

    for step_idx, (x_priv, y_priv) in enumerate(priv_loader):
        try:
            x_pub, _ = next(pub_iter)
        except StopIteration:
            pub_iter = iter(pub_loader)
            x_pub, _ = next(pub_iter)

        x_priv = x_priv.to(device, non_blocking=True)
        y_priv = y_priv.to(device, non_blocking=True)
        x_pub = x_pub.to(device, non_blocking=True)
        bs = x_priv.size(0)

        # Forward Client (1 lần duy nhất cho cả D và Hijack)
        if not client_opt_is_dpsgd:
            opt_c.zero_grad()
        z_c = client(x_priv)
        z_trans = defense(z_c) if defense is not None else z_c

        with torch.no_grad():
            z_p = pilot_ae.encode(x_pub)
            z_c_for_d = server_adapter(z_trans.detach()) if server_adapter is not None else z_trans.detach()

        # --- Bước 1: Cập nhật Discriminator trên Server ---
        opt_d.zero_grad()
        pred_c_det = discriminator(z_c_for_d)
        pred_p = discriminator(z_p)

        if loss_type == "wgan-gp":
            gp = compute_gradient_penalty(discriminator, z_p, z_c_for_d, device=device)
            loss_d = pred_c_det.mean() - pred_p.mean() + gp_weight * gp
        elif loss_type == "softplus":
            loss_d = F.softplus(-pred_p).mean() + F.softplus(pred_c_det).mean()
        else:
            bce = nn.BCEWithLogitsLoss()
            loss_d = bce(pred_p, torch.ones_like(pred_p)) + bce(pred_c_det, torch.zeros_like(pred_c_det))

        loss_d.backward()
        opt_d.step()

        # --- Bước 2: Server sinh gradient độc hại gửi về Cut-Layer ---
        if opt_adapter is not None:
            opt_adapter.zero_grad()

        z_d = z_trans.detach().requires_grad_(True)
        z_srv = server_adapter(z_d) if server_adapter is not None else z_d

        # 2a. Adversarial Critic Loss
        pred_adv = discriminator(z_srv)
        if loss_type == "wgan-gp":
            loss_adv = -pred_adv.mean()
        elif loss_type == "softplus":
            loss_adv = F.softplus(-pred_adv).mean()
        else:
            bce = nn.BCEWithLogitsLoss()
            loss_adv = bce(pred_adv, torch.ones_like(pred_adv))

        # 2b. Target Manifold Projection Loss: || z_srv - f_tilde(f_tilde^-1(z_srv)) ||^2
        z_reenc = pilot_ae.encode(pilot_ae.decode(z_srv))
        loss_subspace = F.mse_loss(z_srv, z_reenc)

        # 2c. Feature Moment Alignment
        if is_additive_noise_defense:
            loss_moment = F.mse_loss(z_srv.mean(dim=[0, 2, 3]), z_p.mean(dim=[0, 2, 3]))
        else:
            loss_moment = (
                F.mse_loss(z_srv.mean(dim=[0, 2, 3]), z_p.mean(dim=[0, 2, 3]))
                + F.mse_loss(z_srv.std(dim=[0, 2, 3]), z_p.std(dim=[0, 2, 3]))
            )

        # 2d. Pilot Coordinate Anchor Loss (khóa thứ tự kênh vào hệ tọa độ của Pilot AE)
        if pilot_server is not None:
            logits_pilot = pilot_server(z_srv)
            loss_anchor = ce(logits_pilot, y_priv)
            batch_acc = (logits_pilot.argmax(1) == y_priv).float().mean().item()
        else:
            loss_anchor = torch.tensor(0.0, device=device)
            batch_acc = 0.0

        malicious_loss = grad_scale * (
            0.25 * loss_adv + 2.5 * loss_subspace + 1.0 * loss_moment + 0.5 * loss_anchor
        )

        if task_grad_weight > 0.0 and task_server is not None:
            malicious_loss = malicious_loss + task_grad_weight * ce(task_server(z_srv), y_priv)

        malicious_loss.backward()
        if opt_adapter is not None:
            opt_adapter.step()

        grad_boundary = z_d.grad.detach()

        # --- Bước 3: Lan truyền ngược phía Client (kèm cơ chế phòng thủ nếu có) ---
        if client_opt_is_dpsgd:
            opt_c.step(x_priv, grad_boundary)
        else:
            has_extra_loss = hasattr(defense, "compute_loss") and callable(defense.compute_loss)
            if has_extra_loss:
                extra_loss = defense.compute_loss(x_priv, z_c)
                if extra_loss.requires_grad:
                    extra_loss.backward(retain_graph=True)

            if hasattr(defense, "get_orthogonality_loss") and callable(defense.get_orthogonality_loss):
                ortho_loss = defense.get_orthogonality_loss()
                if ortho_loss.requires_grad:
                    (0.1 * ortho_loss).backward(retain_graph=True)

            z_trans.backward(grad_boundary)
            torch.nn.utils.clip_grad_norm_(client.parameters(), max_norm=5.0)
            opt_c.step()

        # --- Bước 4: Cập nhật nhẹ Task Server trên z_srv.detach() (mỗi 2 bước) để đo Utility ---
        if task_server is not None and opt_task_s is not None and (step_idx % 2 == 0):
            opt_task_s.zero_grad()
            with torch.no_grad():
                z_for_task = server_adapter(z_trans.detach()) if server_adapter is not None else z_trans.detach()
            logits_task = task_server(z_for_task)
            loss_task = ce(logits_task, y_priv)
            loss_task.backward()
            opt_task_s.step()
            batch_acc = (logits_task.argmax(1) == y_priv).float().mean().item()

        total_loss_d += loss_d.item() * bs
        total_loss_adv += loss_adv.item() * bs
        total_loss_sub += loss_subspace.item() * bs
        total_task_acc += batch_acc * bs
        total_samples += bs

    n = max(total_samples, 1)
    return {
        "loss_d": total_loss_d / n,
        "loss_adv": total_loss_adv / n,
        "loss_subspace": total_loss_sub / n,
        "train_task_acc": total_task_acc / n,
    }


@torch.no_grad()
def evaluate_fsha(
    client,
    pilot_ae,
    task_server,
    test_loader,
    device="cuda",
    defense=None,
    server_adapter=None,
    lpips_fn=None,
    mean=CIFAR10_MEAN,
    std=CIFAR10_STD,
    max_dcor_batches=10,
    max_recon_batches=20,
):
    """
    Đánh giá toàn diện FSHA trên tập Test:
    - Khả năng tái tạo của Server qua khóa giải mã Pha 1: MSE, PSNR (dB), SSIM, LPIPS
    - Tiện ích tác vụ khi bị tấn công (Utility under attack): Test Accuracy (%) trên toàn tập Test
    - Rò rỉ thông tin thống kê tại cut-layer: dCor(X, Z')
    """
    client.eval()
    pilot_ae.eval()
    if task_server is not None:
        task_server.eval()
    if defense is not None:
        defense.eval()
    if server_adapter is not None:
        server_adapter.eval()

    total_mse = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    recon_samples = 0
    correct = 0
    total_samples = 0
    dcor_sum = 0.0
    dcor_batches = 0
    has_lpips = lpips_fn is not None

    for idx, (x, y) in enumerate(test_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = x.size(0)

        z_c = client(x)
        z_trans = defense(z_c) if defense is not None else z_c
        z_srv = server_adapter(z_trans) if server_adapter is not None else z_trans

        if task_server is not None:
            logits = task_server(z_srv)
            correct += (logits.argmax(1) == y).sum().item()

        if max_recon_batches is None or idx < max_recon_batches:
            x_hat = pilot_ae.decode(z_srv)
            mse_val = F.mse_loss(x_hat, x).item()
            total_mse += mse_val * bs

            b_psnr, b_ssim = psnr_ssim(x, x_hat, mean, std)
            psnr_sum += b_psnr * bs
            ssim_sum += b_ssim * bs

            if has_lpips:
                b_lpips = calculate_lpips(x, x_hat, mean, std, lpips_fn)
                if b_lpips is not None:
                    lpips_sum += b_lpips * bs

            recon_samples += bs

        if dcor_batches < max_dcor_batches:
            dcor_sum += distance_correlation(x, z_trans).item()
            dcor_batches += 1

        total_samples += bs

    n_rec = max(recon_samples, 1)
    n_tot = max(total_samples, 1)
    return {
        "mse": total_mse / n_rec,
        "psnr": psnr_sum / n_rec,
        "ssim": ssim_sum / n_rec,
        "lpips": (lpips_sum / n_rec) if has_lpips else None,
        "test_acc": (correct / n_tot) if task_server is not None else 0.0,
        "dcor": dcor_sum / max(dcor_batches, 1),
    }
