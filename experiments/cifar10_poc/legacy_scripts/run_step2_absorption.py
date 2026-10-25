#!/usr/bin/env python3
# Bước 2: Thí nghiệm Hấp thụ (Absorption Experiment) — Chứng minh tính thất bại của phép mã hóa khả nghịch
import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ClientModel, ServerModel
from src.data import get_cifar10, CIFAR10_MEAN, CIFAR10_STD
from src.defenses import random_perm, ChannelPermute, Adapter
from src.training import train_sl_epoch, evaluate_sl, EarlyStopping
from src.attacks import (
    Decoder,
    recover_perm,
    recover_perm_from_adapter,
    recover_perm_covariance,
    match_accuracy,
    evaluate_inversion
)
from src.utils import plot_training_curves, plot_permutation_matrix


def find_step0_checkpoint(user_path=None):
    if user_path and os.path.isfile(user_path):
        return user_path
    candidates = [
        os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step0", "best_b0_vanilla.pt"),
        os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step0", "b0_vanilla.pt"),
        os.path.join(PROJECT_ROOT, "best_b0_vanilla.pt"),
        os.path.join(PROJECT_ROOT, "b0_vanilla.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]


def extract_w_ref(ref_ckpt_path, device="cpu"):
    """
    Trích xuất trọng số Conv lớp đầu tiên của Server tham chiếu (Bước 0):
    layer2.0.conv1.weight shape [128, 64, 3, 3]
    """
    if not os.path.isfile(ref_ckpt_path):
        return None

    ckpt = torch.load(ref_ckpt_path, map_location=device)
    state_dict = ckpt["server"] if "server" in ckpt else ckpt

    target_keys = ["layer2.0.conv1.weight", "server.layer2.0.conv1.weight"]
    for k in target_keys:
        if k in state_dict:
            return state_dict[k].cpu()

    for k in state_dict:
        if "layer2" in k and "conv1.weight" in k:
            return state_dict[k].cpu()

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 2: Thí nghiệm Hấp thụ ánh xạ ngược (Absorption Experiment)")
    parser.add_argument("--mode", choices=["adapter", "train_sl"], default="adapter",
                        help="'adapter': Huấn luyện Cut-Layer Adapter chứng minh hấp thụ (khuyên dùng); 'train_sl': Huấn luyện SL với Permutation")
    parser.add_argument("--epochs", type=int, default=15, help="Số epochs huấn luyện (mặc định: 15 cho adapter, 20 cho train_sl)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (mặc định: 0.01)")
    parser.add_argument("--perm-seed", type=int, default=42, help="Seed sinh hoán vị ngẫu nhiên (mặc định: 42)")
    parser.add_argument("--ref-ckpt", type=str, default=None, help="File checkpoint Step 0 (b0_vanilla.pt)")
    parser.add_argument("--init-from-ref", action="store_true", default=True, help="Nạp trọng số từ checkpoint tham chiếu")
    parser.add_argument("--no-init-from-ref", dest="init_from_ref", action="store_false", help="Không nạp trọng số, train from scratch")
    parser.add_argument("--freeze-client", action="store_true", default=True, help="Đóng băng Client để Server tự học hấp thụ")
    parser.add_argument("--train-client", dest="freeze_client", action="store_false", help="Cho phép Client cập nhật cùng Server")
    parser.add_argument("--eval-freq", type=int, default=2, help="Tần suất đánh giá test accuracy")
    parser.add_argument("--match-freq", type=int, default=2, help="Tần suất đánh giá độ khớp hoán vị")
    parser.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2,
                        help="Số luồng nạp dữ liệu (mặc định: 0 trên Windows để tránh crash IPC, 2 trên Linux)")
    parser.add_argument("--patience", type=int, default=10, help="Số epochs chờ Early Stopping (mặc định: 10, 0 để tắt)")
    parser.add_argument("--decoder-epochs", type=int, default=15, help="Số epochs huấn luyện Decoder nếu cần")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"), help="Thư mục dữ liệu")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output", "AbReTAPE_Step2"), help="Thư mục outputs")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file lưu kết quả model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn file lưu/nạp checkpoint dở dang")
    parser.add_argument("--history-file", type=str, default=None, help="Đường dẫn file lịch sử JSON")
    parser.add_argument("--plot-file", type=str, default=None, help="Đường dẫn file lưu đồ thị PNG")
    parser.add_argument("--heatmap-file", type=str, default=None, help="Đường dẫn file lưu heatmap hoán vị PNG")
    parser.add_argument("--heatmap", type=str, default=None, help="Alias cho heatmap-file")
    parser.add_argument("--resume", action="store_true", help="Khôi phục từ checkpoint nếu có")
    return parser.parse_args()


def run_adapter_mode(args, device):
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = find_step0_checkpoint(args.ref_ckpt)
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Không tìm thấy checkpoint Step 0 tại: {ckpt_path}")
        sys.exit(1)

    print("=" * 70)
    print("BƯỚC 2: CHỨNG MINH HẤP THỤ QUA CUT-LAYER ADAPTER (CONV 1x1)")
    print(f"Thiết bị: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Nạp trọng số hội tụ từ: {ckpt_path}")
    print("=" * 70)

    # 1. Nạp và đóng băng Client & Server
    client = ClientModel().to(device)
    server = ServerModel().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    client.load_state_dict(ckpt["client"] if "client" in ckpt else ckpt)
    server.load_state_dict(ckpt["server"] if "server" in ckpt else ckpt)

    client.eval()
    server.eval()
    for p in client.parameters():
        p.requires_grad = False
    for p in server.parameters():
        p.requires_grad = False

    # 2. Tạo hoán vị bí mật pi và Adapter 1x1 conv
    perm = random_perm(64, seed=args.perm_seed).to(device)
    permute = ChannelPermute(perm).to(device)
    adapter = Adapter(channels=64).to(device)

    opt_a = torch.optim.Adam(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_a = CosineAnnealingLR(opt_a, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # Thử nghiệm thống kê nhanh Covariance Attack (0 epochs)
    cov_perm_hat, cov_acc = recover_perm_covariance(client, trainloader, perm, device, num_batches=30)
    print(f"\n[PHƯƠNG PHÁP 1: COVARIANCE PROFILE (0 EPOCHS)]")
    print(f"Độ khớp khôi phục hoán vị tức thì: {cov_acc*100:.2f}%\n")

    print(f"[PHƯƠNG PHÁP 2: HUẤN LUYỆN CUT-LAYER ADAPTER (HẤP THỤ ÁNH XẠ NGƯỢC)]")
    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for x, y in trainloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt_a.zero_grad()
            with torch.no_grad():
                z_perm = permute(client(x))
            z_hat = adapter(z_perm)
            logits = server(z_hat)
            loss = criterion(logits, y)
            loss.backward()
            opt_a.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        sched_a.step()
        epoch_time = time.time() - t0

        # Kiểm tra độ khớp
        A = adapter.get_matrix()
        hat_pi = recover_perm_from_adapter(A)
        match_acc = match_accuracy(perm, hat_pi)

        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.2f}% | "
              f"Độ khớp Hoán vị khôi phục: {match_acc*100:.1f}% | Time: {epoch_time:.1f}s", flush=True)

        if match_acc >= 1.0:
            print(f"\n[EARLY STOPPING] Đạt độ khớp hoán vị hoàn hảo (100.0%) tại epoch {epoch}! Dừng sớm Adapter.")
            break
        if early_stopping.step(match_acc, epoch=epoch):
            print(f"\n[EARLY STOPPING] Dừng sớm Adapter tại epoch {epoch} do độ khớp hoán vị không cải thiện sau {args.patience} epochs!")
            break

    # Đánh giá Test Accuracy khi qua permute + adapter
    adapter.eval()
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for x_t, y_t in testloader:
            x_t, y_t = x_t.to(device, non_blocking=True), y_t.to(device, non_blocking=True)
            logits_t = server(adapter(permute(client(x_t))))
            correct_test += (logits_t.argmax(1) == y_t).sum().item()
            total_test += x_t.size(0)
    test_acc = correct_test / max(total_test, 1)
    print(f"[Adapter Result] Final Test Accuracy: {test_acc*100:.2f}% | Match Acc: {match_acc*100:.1f}%")

    # Lưu kết quả
    heatmap_file = args.heatmap_file or args.heatmap or os.path.join(args.output_dir, "adapter_heatmap.png")
    plot_permutation_matrix(adapter.get_matrix(), save_path=heatmap_file, title="Ma trận Trọng số Adapter A ≈ P_pi^T (Hấp thụ hoàn toàn)")

    out_file = args.output or os.path.join(args.output_dir, "b2_adapter.pt")
    torch.save({
        "adapter": adapter.state_dict(),
        "A": adapter.get_matrix().cpu(),
        "perm": perm.cpu(),
        "perm_hat": hat_pi,
        "match_acc": match_acc,
        "cov_match_acc": cov_acc,
        "test_acc": test_acc
    }, out_file)
    print(f"\n[DONE] Hoàn thành thí nghiệm Hấp thụ Adapter! Độ khớp hoán vị khôi phục: {match_acc*100:.1f}%")


def run_sl_mode(args, device):
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = find_step0_checkpoint(args.ref_ckpt)
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Không tìm thấy checkpoint Step 0 tại: {ckpt_path}")
        sys.exit(1)

    W_ref = extract_w_ref(ckpt_path, device="cpu")
    if W_ref is None:
        print(f"[ERROR] Không trích xuất được W_ref (layer2.0.conv1.weight) từ {ckpt_path}")
        sys.exit(1)

    print("=" * 70)
    print("BƯỚC 2: HUẤN LUYỆN SPLIT LEARNING CÓ HOÁN VỊ ĐỂ CHỨNG MINH HẤP THỤ")
    print(f"Thiết bị: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Đóng băng Client: {args.freeze_client} | Permutation Seed: {args.perm_seed}")
    print(f"Nạp trọng số từ: {ckpt_path}")
    print("=" * 70)

    trainloader, testloader = get_cifar10(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    client = ClientModel().to(device)
    server = ServerModel().to(device)

    if args.init_from_ref and os.path.isfile(ckpt_path):
        ckpt_ref = torch.load(ckpt_path, map_location=device)
        client.load_state_dict(ckpt_ref["client"] if "client" in ckpt_ref else ckpt_ref)
        server.load_state_dict(ckpt_ref["server"] if "server" in ckpt_ref else ckpt_ref)
        print("[INFO] Đã nạp thành công mô hình đã hội tụ từ Bước 0!")

    perm = random_perm(64, seed=args.perm_seed).to(device)
    defense = ChannelPermute(perm).to(device)

    if args.freeze_client:
        client.eval()
        for p in client.parameters():
            p.requires_grad = False
        opt_c = None
    else:
        opt_c = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched_s = CosineAnnealingLR(opt_s, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_match = 0.0
    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    out_file = args.output or os.path.join(args.output_dir, "b2_absorption.pt")
    last_path = args.checkpoint or os.path.join(args.output_dir, "last_checkpoint_b2.pt")
    hist_file = args.history_file or os.path.join(args.output_dir, "step2_history.json")
    plot_file = args.plot_file or os.path.join(args.output_dir, "step2_curves.png")
    heatmap_file = args.heatmap_file or args.heatmap or os.path.join(args.output_dir, "permutation_heatmap.png")

    start_epoch = 1
    if args.resume and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        server.load_state_dict(ckpt["server"])
        opt_s.load_state_dict(ckpt["opt_s"])
        sched_s.load_state_dict(ckpt["sched_s"])
        start_epoch = ckpt["epoch"] + 1
        best_match = ckpt.get("best_match", 0.0)
        history = ckpt.get("history", [])
        print(f"[RESUME] Khôi phục từ epoch {start_epoch-1} với Best Match: {best_match*100:.1f}%")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_sl_epoch(client, server, trainloader, opt_c, opt_s, criterion, device, defense=defense)
        sched_s.step()
        epoch_time = time.time() - t0

        test_loss, test_acc = None, None
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            test_loss, test_acc = evaluate_sl(client, server, testloader, device, defense=defense, criterion=criterion)

        match_acc = None
        if epoch % args.match_freq == 0 or epoch == args.epochs:
            W_hat = server.layer2[0].conv1.weight.detach().cpu()
            hat_pi = recover_perm(W_hat, W_ref)
            match_acc = match_accuracy(perm.cpu(), hat_pi)
            if match_acc > best_match:
                best_match = match_acc
            if match_acc >= 1.0:
                print(f"\n[EARLY STOPPING] Server đã hấp thụ hoàn hảo hoán vị (100.0%) tại epoch {epoch}! Dừng sớm.")
                break
            if early_stopping.step(match_acc, epoch=epoch):
                print(f"\n[EARLY STOPPING] Dừng sớm SL tại epoch {epoch} do độ khớp hoán vị không cải thiện sau {args.patience} lần đánh giá!")
                break

        entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss) if test_loss is not None else None,
            "test_acc": float(test_acc) if test_acc is not None else None,
            "match_acc": float(match_acc) if match_acc is not None else None,
            "epoch_time": float(epoch_time),
        }
        history.append(entry)

        status = f"Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%"
        if test_acc is not None:
            status += f" | Test Acc: {test_acc*100:.2f}%"
        if match_acc is not None:
            status += f" | Độ khớp Hoán vị: {match_acc*100:.1f}%"
        print(status + f" | Time: {epoch_time:.1f}s", flush=True)

        with open(hist_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        torch.save({
            "server": server.state_dict(),
            "opt_s": opt_s.state_dict(),
            "sched_s": sched_s.state_dict(),
            "epoch": epoch,
            "best_match": best_match,
            "history": history
        }, last_path)

    W_hat = server.layer2[0].conv1.weight.detach().cpu()
    hat_pi = recover_perm(W_hat, W_ref)
    final_match = match_accuracy(perm.cpu(), hat_pi)

    plot_training_curves(history, plot_file)
    plot_permutation_matrix(W_hat, W_ref, perm.cpu(), save_path=heatmap_file,
                            title="Kiểm chứng Thực nghiệm Hấp thụ: Server tự học E^-1")

    torch.save({
        "server": server.state_dict(),
        "perm": perm.cpu(),
        "perm_hat": hat_pi,
        "match_acc": final_match,
        "history": history
    }, out_file)
    print(f"\n[DONE] Hoàn thành Thí nghiệm Hấp thụ SL! Độ khớp khôi phục hoán vị: {final_match*100:.1f}%")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "adapter":
        run_adapter_mode(args, device)
    elif args.mode == "train_sl":
        run_sl_mode(args, device)


if __name__ == "__main__":
    main()
