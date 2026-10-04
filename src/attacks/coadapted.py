# Module triển khai Lớp 2: Attacker Co-adapted Thụ Động (Online Shadow Inversion)
# Server tuân thủ nghiêm ngặt giao thức Split Learning (không sửa gradient),
# nhưng decoder được cập nhật đồng thời từng epoch bám theo sự tiến hóa của Client Encoder.
import copy
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from ..data.cifar import CIFAR10_MEAN, CIFAR10_STD
from ..metrics.reconstruction import psnr_ssim, calculate_lpips


def get_coadapted_data_splits(
    data_dir="./data",
    batch_size=128,
    aux_size=2500,
    test_eval_size=2000,
    num_workers=0,
    seed=42,
):
    """
    Phân chia tập dữ liệu chuẩn cho kịch bản Co-adapted Thụ Động:
    1. D_sl_train: (50000 - aux_size) mẫu dùng cho Split Learning (có Augmentation chuẩn: Crop, Flip, Normalize).
    2. D_aux: aux_size mẫu cố định dùng cho Attacker Decoder huấn luyện (KHÔNG Augmentation: chỉ Normalize).
    3. D_test_eval: test_eval_size mẫu từ tập test để đánh giá nhanh PSNR/SSIM mỗi epoch.
    4. D_test_full: toàn bộ 10000 mẫu tập test để đánh giá cuối cùng.
    """
    mean = CIFAR10_MEAN
    std = CIFAR10_STD

    # Transform cho Split Learning (có data augmentation)
    sl_train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # Transform tất định (Deterministic, không augmentation) cho tập phụ và tập test
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # Nạp dataset nền
    full_train_sl = CIFAR10(root=data_dir, train=True, download=True, transform=sl_train_tf)
    full_train_eval = CIFAR10(root=data_dir, train=True, download=True, transform=eval_tf)
    full_test = CIFAR10(root=data_dir, train=False, download=True, transform=eval_tf)

    # Chia chỉ số D_sl_train và D_aux cố định theo seed
    generator = torch.Generator().manual_seed(seed)
    total_train = len(full_train_sl)
    perm = torch.randperm(total_train, generator=generator).tolist()

    aux_indices = perm[:aux_size]
    sl_indices = perm[aux_size:]

    sl_train_subset = Subset(full_train_sl, sl_indices)
    aux_subset = Subset(full_train_eval, aux_indices)

    # Chia test subset
    test_perm = torch.randperm(len(full_test), generator=generator).tolist()
    test_eval_indices = test_perm[:test_eval_size]
    test_eval_subset = Subset(full_test, test_eval_indices)

    use_cuda = torch.cuda.is_available()
    pin = use_cuda

    sl_loader = DataLoader(
        sl_train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    aux_loader = DataLoader(
        aux_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    test_eval_loader = DataLoader(
        test_eval_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    test_full_loader = DataLoader(
        full_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )

    return sl_loader, aux_loader, test_eval_loader, test_full_loader


class CoAdaptedPassiveAttacker:
    """
    Lớp quản lý Attacker Co-adapted Thụ Động (Nhánh A - Online Shadow Inversion):
    - Decoder được cập nhật 1 epoch trên D_aux sau mỗi epoch của Split Learning.
    - Client được truy vấn ở chế độ eval() (với torch.no_grad()) để đảm bảo thống kê BN nhất quán.
    - Tuyệt đối detach() biểu diễn trung gian z, không can thiệp hay làm rò rỉ gradient về client.
    """
    def __init__(self, decoder, lr=1e-3, device="cuda"):
        self.decoder = decoder.to(device)
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.device = device
        self.total_steps = 0
        self.history = []

    def step_epoch(self, client, aux_loader, defense=None):
        """
        Thực hiện 1 epoch huấn luyện decoder trên tập phụ D_aux với Client Encoder hiện tại.
        """
        was_client_training = client.training
        was_defense_training = defense.training if defense is not None else False

        client.eval()
        if defense is not None:
            defense.eval()
        self.decoder.train()

        total_loss = 0.0
        total_samples = 0

        for x_aux, _ in aux_loader:
            x_aux = x_aux.to(self.device, non_blocking=True)
            batch_size = x_aux.size(0)

            # 1. Truy vấn biểu diễn trung gian hiện tại ở chế độ eval
            with torch.no_grad():
                z = client(x_aux)
                if defense is not None:
                    z = defense(z)

            # 2. Ngắt gradient tuyệt đối khỏi Client (Thụ động 100%)
            z_input = z.detach()

            # 3. Cập nhật trọng số của Decoder
            self.optimizer.zero_grad()
            x_hat = self.decoder(z_input)
            loss = self.criterion(x_hat, x_aux)
            loss.backward()
            self.optimizer.step()

            self.total_steps += 1
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        # Khôi phục trạng thái huấn luyện ban đầu của Client và Defense
        if was_client_training:
            client.train()
        if defense is not None and was_defense_training:
            defense.train()

        return total_loss / total_samples


@torch.no_grad()
def evaluate_coadapted_inversion(
    decoder, client, loader, device,
    defense=None, mean=CIFAR10_MEAN, std=CIFAR10_STD, lpips_fn=None
):
    """
    Đánh giá năng lực giải mã của Decoder trên tập kiểm thử (Test loader):
    Trả về: (mse, psnr, ssim, lpips)
    """
    was_client_training = client.training
    client.eval()
    if defense is not None:
        defense.eval()
    decoder.eval()

    total_mse = 0.0
    total_samples = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    has_lpips = lpips_fn is not None
    criterion = nn.MSELoss()

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        batch_size = x.size(0)

        z = client(x)
        if defense is not None:
            z = defense(z)

        x_hat = decoder(z)
        loss = criterion(x_hat, x)
        total_mse += loss.item() * batch_size

        batch_psnr, batch_ssim = psnr_ssim(x, x_hat, mean, std)
        psnr_sum += batch_psnr * batch_size
        ssim_sum += batch_ssim * batch_size

        if has_lpips:
            batch_lpips = calculate_lpips(x, x_hat, mean, std, lpips_fn)
            if batch_lpips is not None:
                lpips_sum += batch_lpips * batch_size

        total_samples += batch_size

    if was_client_training:
        client.train()

    mean_mse = total_mse / total_samples
    mean_psnr = psnr_sum / total_samples
    mean_ssim = ssim_sum / total_samples
    mean_lpips = (lpips_sum / total_samples) if has_lpips else None

    return mean_mse, mean_psnr, mean_ssim, mean_lpips


def train_fresh_matched_steps(
    decoder_ctor, client, aux_loader, test_loader, target_steps,
    device, defense=None, lr=1e-3, eval_freq_steps=None, lpips_fn=None
):
    """
    Nhánh B — Fresh Attacker với CÙNG SỐ BƯỚC GRADIENT:
    - Client và Defense đã đóng băng hoàn toàn ở trạng thái hội tụ cuối cùng (Epoch 100).
    - Khởi tạo Fresh Decoder từ đầu.
    - Huấn luyện trên D_aux lặp lại cho đến khi chạm đúng target_steps.
    """
    fresh_decoder = decoder_ctor().to(device)
    optimizer = torch.optim.Adam(fresh_decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    client.eval()
    if defense is not None:
        defense.eval()
    fresh_decoder.train()

    step_count = 0
    best_psnr = 0.0
    best_ssim = 0.0
    best_step = 0
    history = []

    epoch_idx = 0
    while step_count < target_steps:
        epoch_idx += 1
        for x_aux, _ in aux_loader:
            if step_count >= target_steps:
                break
            x_aux = x_aux.to(device, non_blocking=True)

            with torch.no_grad():
                z = client(x_aux)
                if defense is not None:
                    z = defense(z)
            z_input = z.detach()

            optimizer.zero_grad()
            x_hat = fresh_decoder(z_input)
            loss = criterion(x_hat, x_aux)
            loss.backward()
            optimizer.step()

            step_count += 1

            if eval_freq_steps and (step_count % eval_freq_steps == 0 or step_count == target_steps):
                mse, psnr, ssim, lpips = evaluate_coadapted_inversion(
                    fresh_decoder, client, test_loader, device, defense=defense, lpips_fn=lpips_fn
                )
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_ssim = ssim
                    best_step = step_count
                history.append({
                    "step": step_count,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips,
                })
                fresh_decoder.train()

    # Đánh giá cuối cùng
    final_mse, final_psnr, final_ssim, final_lpips = evaluate_coadapted_inversion(
        fresh_decoder, client, test_loader, device, defense=defense, lpips_fn=lpips_fn
    )
    if final_psnr > best_psnr:
        best_psnr = final_psnr
        best_ssim = final_ssim
        best_step = step_count

    return {
        "decoder": fresh_decoder,
        "final_psnr": final_psnr,
        "final_ssim": final_ssim,
        "final_lpips": final_lpips,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "best_step": best_step,
        "total_steps": step_count,
        "history": history,
    }


def train_fresh_epochs(
    decoder_ctor, client, aux_loader, test_loader, epochs,
    device, defense=None, lr=1e-3, eval_freq=5, lpips_fn=None
):
    """
    Nhánh C — Fresh Attacker truyền thống (30 epochs chuẩn):
    - Client và Defense đóng băng sau khi SL hội tụ.
    - Huấn luyện trên D_aux qua đúng `epochs` (mặc định 30).
    """
    fresh_decoder = decoder_ctor().to(device)
    optimizer = torch.optim.Adam(fresh_decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    client.eval()
    if defense is not None:
        defense.eval()

    best_psnr = 0.0
    best_ssim = 0.0
    best_epoch = 0
    history = []
    total_steps = 0

    for ep in range(1, epochs + 1):
        fresh_decoder.train()
        ep_loss = 0.0
        n_samples = 0

        for x_aux, _ in aux_loader:
            x_aux = x_aux.to(device, non_blocking=True)
            b_size = x_aux.size(0)

            with torch.no_grad():
                z = client(x_aux)
                if defense is not None:
                    z = defense(z)
            z_input = z.detach()

            optimizer.zero_grad()
            x_hat = fresh_decoder(z_input)
            loss = criterion(x_hat, x_aux)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item() * b_size
            n_samples += b_size
            total_steps += 1

        if ep % eval_freq == 0 or ep == epochs:
            mse, psnr, ssim, lpips = evaluate_coadapted_inversion(
                fresh_decoder, client, test_loader, device, defense=defense, lpips_fn=lpips_fn
            )
            if psnr > best_psnr:
                best_psnr = psnr
                best_ssim = ssim
                best_epoch = ep
            history.append({
                "epoch": ep,
                "step": total_steps,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
            })

    final_mse, final_psnr, final_ssim, final_lpips = evaluate_coadapted_inversion(
        fresh_decoder, client, test_loader, device, defense=defense, lpips_fn=lpips_fn
    )

    return {
        "decoder": fresh_decoder,
        "final_psnr": final_psnr,
        "final_ssim": final_ssim,
        "final_lpips": final_lpips,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "best_epoch": best_epoch,
        "total_steps": total_steps,
        "history": history,
    }
