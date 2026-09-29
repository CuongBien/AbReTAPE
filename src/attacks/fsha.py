# Module Tấn công Chủ động FSHA (Feature Space Hijacking Attack - Pasquini et al., CCS 2021)
# Bề mặt tấn công căng thẳng (stress-test) kiểm chứng khả năng bẻ gãy FSHA của AR-TAPE (RQ2, CV2)
import torch
import torch.nn as nn


class FSHADiscriminator(nn.Module):
    """
    Discriminator của Server độc hại:
    Phân biệt giữa Z_client = Client(X_private) và Z_pilot = Pilot_Encoder(X_public).
    """
    def __init__(self, in_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, z):
        return self.net(z)


class FSHAPilotAutoEncoder(nn.Module):
    """
    Autoencoder của Server huấn luyện trên tập dữ liệu công khai (pilot data):
    Gồm Pilot Encoder f_tilde và Pilot Decoder g_tilde.
    """
    def __init__(self, in_channels=3, latent_channels=64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec


def train_fsha_step(client, discriminator, pilot_ae, x_private, x_public, opt_d, opt_ae, opt_c=None, device="cuda"):
    """
    Một bước tối ưu FSHA:
    1. Huấn luyện Pilot AE tái tạo x_public.
    2. Huấn luyện Discriminator phân biệt z_client vs z_pilot.
    3. Sinh gradient độc hại (WGAN/BCE loss) đẩy z_client khớp phân phối z_pilot.
    """
    bce = nn.BCEWithLogitsLoss()

    # 1. Update Pilot AE
    opt_ae.zero_grad()
    z_pilot, x_rec_pilot = pilot_ae(x_public)
    loss_ae = nn.MSELoss()(x_rec_pilot, x_public)
    loss_ae.backward()
    opt_ae.step()

    # 2. Update Discriminator
    opt_d.zero_grad()
    with torch.no_grad():
        z_client = client(x_private)
        z_pilot, _ = pilot_ae(x_public)

    pred_client = discriminator(z_client)
    pred_pilot = discriminator(z_pilot)

    loss_d = bce(pred_pilot, torch.ones_like(pred_pilot)) + bce(pred_client, torch.zeros_like(pred_client))
    loss_d.backward()
    opt_d.step()

    # 3. Client adversarial gradient (nếu có cập nhật client)
    loss_c = None
    if opt_c is not None:
        opt_c.zero_grad()
        z_c_adv = client(x_private)
        pred_c_adv = discriminator(z_c_adv)
        loss_c = bce(pred_c_adv, torch.ones_like(pred_c_adv))
        loss_c.backward()
        opt_c.step()

    return {
        "loss_ae": loss_ae.item(),
        "loss_d": loss_d.item(),
        "loss_c": loss_c.item() if loss_c is not None else 0.0
    }
