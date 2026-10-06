#!/bin/bash
# Kịch bản chạy thực nghiệm B8 quét tham số lambda và đánh giá hậu kiểm

set -e

# --- CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG ---
# Tự động di chuyển vào thư mục chứa file .sh này (thư mục AbReTAPE)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

# Chỉ định cứng đường dẫn đến Python của môi trường ảo (venv) bên ngoài
PYTHON_ENV="/home/dutai/CV/.venv/bin/python3"

# Kiểm tra nếu venv ngoài không tồn tại thì báo lỗi
if [ ! -f "$PYTHON_ENV" ]; then
    echo "LỖI: Không tìm thấy môi trường ảo tại $PYTHON_ENV"
    exit 1
fi

# ==============================================================================
# GIAI ĐOẠN 1: Quét lambda với B8-conv (nhẹ) để tìm điểm vận hành
# ==============================================================================
echo "--- BẮT ĐẦU QUÉT LAMBDA VỚI B8-CONV ---"
for LAMBDA in 0.1 0.5 1.0 5.0; do
    echo "Đang chạy B8-conv với lambda = $LAMBDA..."
    $PYTHON_ENV run_b8_minmax.py --adv-type conv --lam $LAMBDA --epochs 100 --batch-size 128
done

# Lưu ý: Ở đây bạn cần kiểm tra file output/AbReTAPE_Step8/b8_conv_lam_*.json
# để chọn giá trị LAMBDA_BEST mà Task Acc >= 92.7% (giảm không quá 2% so với 94.71%)
# Giả sử LAMBDA_BEST được tìm ra là 1.0 (cần cập nhật sau khi xem kết quả).
LAMBDA_BEST=1.0

# ==============================================================================
# GIAI ĐOẠN 2: Chạy B8-mlp (adversary mạnh nhất) với lambda tốt nhất
# ==============================================================================
echo "--- BẮT ĐẦU HUẤN LUYỆN B8-MLP VỚI LAMBDA=$LAMBDA_BEST ---"
$PYTHON_ENV run_b8_minmax.py --adv-type mlp --lam $LAMBDA_BEST --epochs 100 --batch-size 128

# ==============================================================================
# GIAI ĐOẠN 3: Đánh giá hậu kiểm (Post-training Attacks) trên Client B8 đã đóng băng
# ==============================================================================
echo "--- BẮT ĐẦU ĐÁNH GIÁ HẬU KIỂM B8-MLP VÀ KIỂM CHỨNG N1 ---"
# Chạy script đánh giá chuyên dụng vừa được tạo để huấn luyện Fresh Conv & Fresh MLP,
# sau đó tự động tính toán độ đo G và đưa ra kết luận theo Bảng quyết định.
$PYTHON_ENV run_b8_eval.py --checkpoint output/AbReTAPE_Step8/b8_mlp_lam_${LAMBDA_BEST}.pt --epochs 30 --batch-size 128

echo "--- GHI CHÚ BỔ SUNG ---"
echo "Để chạy Co-adapted nhẹ và FSHA, bạn có thể sử dụng các script chuyên dụng tương ứng:"
echo "Ví dụ FSHA:"
echo "python3 run_step4_fsha.py --target-client output/AbReTAPE_Step8/b8_mlp_lam_${LAMBDA_BEST}.pt"
