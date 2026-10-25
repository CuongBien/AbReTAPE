# -*- coding: utf-8 -*-
"""
scripts/plot_visual_comparison.py
Tạo bảng so sánh ảnh tái tạo trực quan (Visual Reconstruction Comparison)
giữa các baseline: Ground Truth, B7, B8, và các mức lambda của AR-TAPE.
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

rows_config = [
    {
        "label": "Ảnh Gốc Chuẩn (Ground Truth)",
        "sublabel": "CIFAR-10 Test Samples",
        "path": "output/AR_TAPE_v1_Utility_Sweep/fresh_lam0.25_s42/rec_artape_v0_fresh_mlp_lam0.25_s42.png",
        "crop_top": True,
        "color": "#2c3e50"
    },
    {
        "label": "AR-TAPE v1 (λ=0.0, Không đối kháng)",
        "sublabel": "Fresh MLP SSIM: 0.7027 | Rò rỉ cấu trúc vật thể nặng",
        "path": "output/AR_TAPE_v1_Cosine_Fresh_lam0/seed_42/rec_artape_v0_fresh_mlp_lam0.0_s42.png",
        "crop_top": False,
        "color": "#c0392b"
    },
    {
        "label": "B7 LightSplit (k=512, Mode F)",
        "sublabel": "Task Acc: 88.54% | Fresh MLP SSIM: 0.4394 (Vẫn nhận diện được hình dáng)",
        "path": "output/B7_Fresh_Eval_Final/k512/rec_artape_v0_fresh_mlp_lam0.0_s42.png",
        "crop_top": False,
        "color": "#d35400"
    },
    {
        "label": "B8 Stabilized (λ=0.1, Cap=1.5586)",
        "sublabel": "Task Acc: 88.98% | Fresh Conv SSIM: 0.6431 [VỠ TRẬN TRƯỚC CONV ATTACKER!]",
        "path": "output/B8_Stabilized_Attackers_Real/fresh_conv/rec_fresh_conv_lam0.1_s42.png",
        "crop_top": False,
        "color": "#e67e22"
    },
    {
        "label": "AR-TAPE v1 (λ=0.10, Cosine)",
        "sublabel": "Task Acc: 90.37% | Fresh MLP SSIM: 0.2261 | Conv: 0.1938 (Bảo vệ mạnh)",
        "path": "output/AR_TAPE_v1_Utility_Sweep/fresh_lam0.1_s42/rec_artape_v0_fresh_mlp_lam0.1_s42.png",
        "crop_top": False,
        "color": "#16a085"
    },
    {
        "label": "★ AR-TAPE v1 (λ=0.25, Điểm Vận Hành Chính)",
        "sublabel": "Task Acc: 90.35 ± 0.24% | Fresh MLP: 0.2159 | Conv: 0.2014 (Triệt tiêu rò rỉ)",
        "path": "output/AR_TAPE_v1_Utility_Sweep/fresh_lam0.25_s42/rec_artape_v0_fresh_mlp_lam0.25_s42.png",
        "crop_top": False,
        "color": "#8e44ad"
    },
    {
        "label": "AR-TAPE v1 (λ=0.50, Phòng Thủ Biên Chặt)",
        "sublabel": "Task Acc: 89.81% | Fresh MLP: 0.2471 | Conv: 0.2249 (Khóa cứng nhiễu)",
        "path": "output/AR_TAPE_v1_Cosine_Fresh/fresh_lam0.5_s42/rec_artape_v0_fresh_mlp_lam0.5_s42.png",
        "crop_top": False,
        "color": "#27ae60"
    }
]

# Đọc và chuẩn bị từng hàng ảnh
images = []
for r in rows_config:
    im = Image.open(r["path"])
    w, h = im.size
    half_h = h // 2
    if r["crop_top"]:
        row_img = im.crop((0, 0, w, half_h))
    else:
        row_img = im.crop((0, half_h, w, h))
    images.append(row_img)

# Target width cho hình ảnh (giảm tỷ lệ vừa phải để hình gọn gàng)
target_w = 1600
scale = target_w / images[0].width
target_h = int(images[0].height * scale)

resized_rows = [im.resize((target_w, target_h), Image.Resampling.LANCZOS) for im in images]

# Tạo canvas tổng thể có lề trái để ghi chú thích
left_margin = 520
top_margin = 80
row_gap = 18
total_w = left_margin + target_w + 40
total_h = top_margin + len(resized_rows) * (target_h + row_gap) + 20

canvas = Image.new("RGB", (total_w, total_h), "#fdfefe")
draw = ImageDraw.Draw(canvas)

# Tải font hệ thống
try:
    title_font = ImageFont.truetype("arial.ttf", 26)
    label_font = ImageFont.truetype("arial.ttf", 20)
    sublabel_font = ImageFont.truetype("arial.ttf", 15)
except Exception:
    title_font = ImageFont.load_default()
    label_font = ImageFont.load_default()
    sublabel_font = ImageFont.load_default()

# Vẽ tiêu đề lớn
draw.text((left_margin, 25), "SO SÁNH THỰC TẾ ẢNH TÁI TẠO: AR-TAPE vs BASELINES (B7, B8)", fill="#1a252f", font=title_font)

y_offset = top_margin
for i, (r_cfg, r_img) in enumerate(zip(rows_config, resized_rows)):
    # Vẽ nền highlight nhẹ cho hàng Điểm Vận Hành Chính
    if "★" in r_cfg["label"]:
        draw.rectangle([10, y_offset - 6, total_w - 20, y_offset + target_h + 6], fill="#f5eef8", outline="#8e44ad", width=2)
    elif "VỠ TRẬN" in r_cfg["sublabel"]:
        draw.rectangle([10, y_offset - 6, total_w - 20, y_offset + target_h + 6], fill="#fef5e7", outline="#e67e22", width=2)

    # Ghi text nhãn bên trái
    text_y = y_offset + (target_h // 2) - 22
    draw.text((25, text_y), r_cfg["label"], fill=r_cfg["color"], font=label_font)
    draw.text((25, text_y + 26), r_cfg["sublabel"], fill="#566573", font=sublabel_font)

    # Dán ảnh hàng mẫu
    canvas.paste(r_img, (left_margin, y_offset))
    y_offset += target_h + row_gap

out_path = os.path.join(FIGURES_DIR, "visual_reconstruction_comparison.png")
canvas.save(out_path, quality=95)
print(f"Đã lưu bảng ảnh so sánh trực quan tại: {out_path}")

# Sao chép sang artifact directory
artifact_dir = r"C:\Users\ADMIN\.gemini\antigravity\brain\c0b937be-df93-44c3-bdbd-06e4f2dd8324"
if os.path.isdir(artifact_dir):
    import shutil
    art_path = os.path.join(artifact_dir, "visual_reconstruction_comparison.png")
    shutil.copyfile(out_path, art_path)
    print(f"Đã sao chép sang artifact directory tại: {art_path}")
