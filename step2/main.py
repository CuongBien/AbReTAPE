# Bước 2 — Thí nghiệm Hấp thụ (Absorption Experiment): Server tự học ra phép giải mã hoán vị
import os
import sys
import time
import json
import csv
import argparse

# Thiết lập UTF-8 encoding cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP0_DIR = os.path.join(PROJECT_ROOT, "step0")

for path in [SCRIPT_DIR, PROJECT_ROOT, STEP0_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch.nn as nn
from model import ClientModel, ServerModel
from data import get_cifar10
from permute import random_perm, ChannelPermute
from train_sl import train_epoch, evaluate_sl
from recover import recover_perm, match_accuracy
from plot_absorption import plot_permutation_matrix, plot_step2_curves


def find_default_ref_ckpt():
    candidates = [
        os.path.join(STEP0_DIR, "b0_vanilla.pt"),
        os.path.join(STEP0_DIR, "best_b0_vanilla.pt"),
        os.path.join(PROJECT_ROOT, "b0_vanilla.pt"),
        os.path.join(SCRIPT_DIR, "b0_vanilla.pt"),
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

    # Tìm key tương đương nếu tên khác
    for k in state_dict:
        if "layer2" in k and "conv1.weight" in k:
            return state_dict[k].cpu()

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Bước 2: Absorption Experiment in Split Learning")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện (mặc định: 100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (mặc định: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (mặc định: 0.1)")
    parser.add_argument("--perm-seed", type=int, default=42, help="Seed sinh hoán vị kênh bí mật (mặc định: 42)")
    parser.add_argument("--eval-freq", type=int, default=5, help="Tần suất đánh giá test accuracy (mặc định: 5)")
    parser.add_argument("--match-freq", type=int, default=10, help="Tần suất đánh giá độ khớp hoán vị (mặc định: 10)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (mặc định: 2)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(STEP0_DIR, "data"), help="Thư mục chứa CIFAR-10")
    parser.add_argument("--ref-ckpt", type=str, default=find_default_ref_ckpt(), help="Đường dẫn checkpoint Bước 0 làm tham chiếu")
    parser.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "b2_absorption.pt"), help="File lưu kết quả hoán vị khôi phục")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(SCRIPT_DIR, "last_checkpoint_b2.pt"), help="File checkpoint resume")
    parser.add_argument("--history-file", type=str, default=os.path.join(SCRIPT_DIR, "step2_history.json"), help="File lưu lịch sử JSON")
    parser.add_argument("--plot-file", type=str, default=os.path.join(SCRIPT_DIR, "step2_curves.png"), help="File lưu đồ thị huấn luyện")
    parser.add_argument("--heatmap-file", type=str, default=os.path.join(SCRIPT_DIR, "permutation_heatmap.png"), help="File lưu ma trận heatmap đường chéo")
    parser.add_argument("--no-plot", action="store_true", help="Không tự động xuất đồ thị và heatmap")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint nếu có")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("==================================================", flush=True)
    print("      BƯỚC 2: THÍ NGHIỆM HẤP THỤ (ABSORPTION EXPERIMENT)", flush=True)
    print("==================================================", flush=True)
    print(f"Thiết bị           : {device}", flush=True)
    if device == "cuda":
        print(f"GPU Name           : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Số Epochs          : {args.epochs}", flush=True)
    print(f"Batch size         : {args.batch_size}", flush=True)
    print(f"Learning rate      : {args.lr}", flush=True)
    print(f"Permutation Seed   : {args.perm_seed}", flush=True)
    print(f"Model Tham chiếu   : {args.ref_ckpt}", flush=True)
    print(f"Output File        : {args.output}", flush=True)
    print("==================================================\n", flush=True)

    torch.manual_seed(0)
    trainloader, testloader = get_cifar10(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # 1. Khởi tạo hoán vị kênh cố định E(z)
    perm = random_perm(64, seed=args.perm_seed)
    permute = ChannelPermute(perm).to(device)

    # 2. Khởi tạo mô hình Client và Server
    client = ClientModel().to(device)
    server = ServerModel().to(device)

    opt_c = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    opt_s = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    sched_c = torch.optim.lr_scheduler.MultiStepLR(opt_c, milestones=[50, 75], gamma=0.1)
    sched_s = torch.optim.lr_scheduler.MultiStepLR(opt_s, milestones=[50, 75], gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    # 3. Nạp trọng số tham chiếu W_ref từ Bước 0
    W_ref = extract_w_ref(args.ref_ckpt)
    if W_ref is not None:
        print(f"[INFO] Đã nạp thành công trọng số Server tham chiếu W_ref shape: {list(W_ref.shape)}", flush=True)
    else:
        print(f"[WARN] Không tìm thấy checkpoint Bước 0 tại: {args.ref_ckpt}", flush=True)
        print("       (Lưu ý: Để đo độ khớp hoán vị chính xác, cần có b0_vanilla.pt đã train xong ở Bước 0)", flush=True)

    start_epoch = 1
    best_acc = 0.0
    final_match_acc = 0.0
    history = []

    # Khôi phục nếu có cờ --resume
    if args.resume and os.path.isfile(args.checkpoint):
        print(f"[RESUME] Nạp checkpoint từ: {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=device)
        client.load_state_dict(ckpt["client"])
        server.load_state_dict(ckpt["server"])
        opt_c.load_state_dict(ckpt["opt_c"])
        opt_s.load_state_dict(ckpt["opt_s"])
        sched_c.load_state_dict(ckpt["sched_c"])
        sched_s.load_state_dict(ckpt["sched_s"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        history = ckpt.get("history", [])
        if not history and os.path.isfile(args.history_file):
            try:
                with open(args.history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        print(f"[RESUME] Tiếp tục từ epoch {start_epoch} (Best Acc: {best_acc*100:.2f}%)", flush=True)

    print("\nBắt đầu huấn luyện Split Learning với Hoán vị Kênh...\n", flush=True)
    total_start = time.time()
    csv_file = os.path.splitext(args.history_file)[0] + ".csv"

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(client, server, permute, trainloader, opt_c, opt_s, criterion, device)
        current_lr = opt_c.param_groups[0]["lr"]
        sched_c.step()
        sched_s.step()
        epoch_time = time.time() - t0

        test_loss = None
        test_acc = None
        eval_time = 0.0
        match_acc = None
        rand_acc = None

        is_eval_epoch = (epoch % args.eval_freq == 0 or epoch == 1 or epoch == args.epochs)
        if is_eval_epoch:
            t_eval_0 = time.time()
            test_loss, test_acc = evaluate_sl(client, server, permute, testloader, device, criterion=criterion)
            eval_time = time.time() - t_eval_0
            if test_acc > best_acc:
                best_acc = test_acc

        # Đánh giá độ khớp hoán vị định kỳ
        is_match_epoch = (epoch % args.match_freq == 0 or epoch == 1 or epoch == args.epochs)
        if is_match_epoch and W_ref is not None:
            W = server.layer2[0].conv1.weight.data.cpu()
            perm_hat = recover_perm(W, W_ref)
            match_acc = match_accuracy(perm, perm_hat)
            final_match_acc = match_acc

            rand_p = torch.randperm(64).tolist()
            rand_acc = match_accuracy(perm, rand_p)

            match_str = f" | Khớp Hoán vị: {match_acc*100:.1f}% (Rand: {rand_acc*100:.1f}%)"
        else:
            match_str = ""

        if is_eval_epoch:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | LR: {current_lr:.4f} | Time: {epoch_time:.1f}s{match_str}", flush=True)
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | LR: {current_lr:.4f} | Time: {epoch_time:.1f}s", flush=True)

        epoch_entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss) if test_loss is not None else None,
            "test_acc": float(test_acc) if test_acc is not None else None,
            "match_acc": float(match_acc) if match_acc is not None else None,
            "rand_acc": float(rand_acc) if rand_acc is not None else None,
            "lr": float(current_lr),
            "epoch_time": float(epoch_time + eval_time),
        }
        history.append(epoch_entry)

        # Lưu lịch sử JSON
        os.makedirs(os.path.dirname(os.path.abspath(args.history_file)), exist_ok=True)
        with open(args.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Lưu lịch sử CSV
        try:
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "match_acc", "rand_acc", "lr", "epoch_time"])
                writer.writeheader()
                writer.writerows(history)
        except Exception:
            pass

        # Lưu checkpoint ngắt quãng sau mỗi epoch
        os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "client": client.state_dict(),
            "server": server.state_dict(),
            "opt_c": opt_c.state_dict(),
            "opt_s": opt_s.state_dict(),
            "sched_c": sched_c.state_dict(),
            "sched_s": sched_s.state_dict(),
            "best_acc": best_acc,
            "history": history,
            "perm": perm,
        }, args.checkpoint)

        # Cập nhật đồ thị và heatmap định kỳ
        if not args.no_plot and (is_match_epoch or epoch == args.epochs):
            try:
                plot_step2_curves(history, save_path=args.plot_file, show=False)
                if W_ref is not None:
                    W_curr = server.layer2[0].conv1.weight.data.cpu()
                    plot_permutation_matrix(W_curr, W_ref, perm, save_path=args.heatmap_file, show=False)
            except Exception as e:
                print(f"[WARN] Lỗi cập nhật đồ thị: {e}", flush=True)

    total_time = time.time() - total_start
    print(f"\n==================================================", flush=True)
    print(f"Huấn luyện Bước 2 hoàn thành trong {total_time/60:.2f} phút.", flush=True)
    print(f"Độ chính xác phân loại cao nhất: {best_acc*100:.2f}%", flush=True)

    # Đánh giá kết quả khôi phục hoán vị cuối cùng
    W_final = server.layer2[0].conv1.weight.data.cpu()
    perm_hat_final = None
    if W_ref is not None:
        perm_hat_final = recover_perm(W_final, W_ref)
        final_match_acc = match_accuracy(perm, perm_hat_final)
        rand_acc_final = match_accuracy(perm, torch.randperm(64).tolist())

        pass_criteria = final_match_acc > 0.80
        print("\n--- KIỂM TRA TIÊU CHÍ NGHIỆM THU (ACCEPTANCE CRITERIA) ---", flush=True)
        print(f"1. Độ khớp hoán vị thật : {final_match_acc*100:.2f}% (Tiêu chuẩn: > 80.0%) -> {'✅ ĐẠT' if pass_criteria else '❌ CHƯA ĐẠT'}", flush=True)
        print(f"2. Đối chứng ngẫu nhiên : {rand_acc_final*100:.2f}% (~ 1.56%)", flush=True)
        if pass_criteria:
            print("\n>>> KẾT LUẬN: TRỤ CỘT NOVELTY N1 ĐÃ ĐƯỢC XÁC NHẬN THỰC NGHIỆM! <<<", flush=True)
            print("    Server đã tự động học ra phép giải mã hoán vị trong các trọng số đầu của nó!", flush=True)
    else:
        print("\n[WARN] Không có file tham chiếu b0_vanilla.pt, bỏ qua tính độ khớp hoán vị.", flush=True)

    print("==================================================\n", flush=True)

    # Lưu kết quả b2_absorption.pt
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_payload = {
        "perm": perm,
        "perm_hat": torch.tensor(perm_hat_final) if perm_hat_final is not None else None,
        "match_acc": final_match_acc,
        "best_acc": best_acc,
        "client": client.state_dict(),
        "server": server.state_dict(),
        "history": history,
    }
    torch.save(save_payload, args.output)
    print(f"[DONE] Đã lưu kết quả tại: {args.output}", flush=True)
    print(f"[DONE] Đã lưu lịch sử tại: {args.history_file} & {csv_file}", flush=True)
    if not args.no_plot:
        print(f"[DONE] Đã lưu đồ thị tại: {args.plot_file}", flush=True)
        if W_ref is not None:
            print(f"[DONE] Đã lưu heatmap tại: {args.heatmap_file}\n", flush=True)


if __name__ == "__main__":
    main()
