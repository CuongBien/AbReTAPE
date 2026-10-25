# MASTER BENCHMARK TABLE: SPLIT LEARNING CIFAR-10 RESNET-18 (CUT-1)

> **Ghi chú chuẩn hóa số liệu:**
> 1. **Acc (10k)**: Đánh giá trên toàn bộ 10,000 ảnh test CIFAR-10 tại epoch cuối cùng (Epoch 100).
> 2. **Fresh MLP / Conv**: Huấn luyện 30 epochs từ scratch trên client đóng băng, lấy số liệu tại epoch 30 cuối cùng (không cherry-pick theo test peak).
> 3. **FSHA SSIM & Task Acc**: Đánh giá tại epoch 30 cuối cùng (mức grad_scale=5.0) để đồng nhất trên toàn bộ các phương pháp.
> 4. **dCor**: Đo khoảng cách tương quan Distance Correlation giữa biểu diễn cut-layer và ảnh gốc tại epoch 100 trong huấn luyện thông thường.
> 5. **Sai số**: Dùng độ lệch chuẩn mẫu hiệu chỉnh (sample std, ddof=1) cho các cấu hình đa seed (n=3: seed 42, 7, 2024).

| Phương Pháp | Bottleneck | λ | Acc (10k) | Acc (2k) | Fresh MLP (10k) | Fresh Conv (10k) | FSHA SSIM | FSHA Task Acc | dCor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| B0 Vanilla | Full (D=65536) | None | 94.71% | N/A | N/A | 0.9955 | 0.9057 | 93.77% | 0.7192 |
| B7 Mode F (k=2048) | k=2048 | 0.0 | 92.73% | N/A | N/A (Chưa chạy riêng B7) | N/A (Chưa chạy riêng B7) | 0.0701 | 73.81% | N/A |
| B7 Mode F (k=1024) | k=1024 | 0.0 | 91.53% | N/A | 0.5461 | 0.3597 | 0.0668 | 65.54% | N/A |
| B7 Mode F (k=512) | k=512 | 0.0 | 88.54% | N/A | 0.4394 | 0.2632 | 0.0586 | 68.00% | N/A |
| B8 Un-stabilized | Full (D=65536) | 0.1 | 92.38% | 93.42% | 0.4580 | 0.7652 | N/A (Chưa chạy) | N/A (Chưa chạy) | N/A |
| B8 Stabilized | Full (D=65536) | 0.1 | 88.98% | 93.53% | 0.2621 | 0.6431 | 0.0856 | 44.34% | 0.2244 |
| AR-TAPE v1 (Cosine lam=0.0) | k=2048 | 0.0 | 91.05 ± 0.36% | 92.78 ± 0.13% | 0.7077 ± 0.0059 | 0.5627 ± 0.0065 | 0.0402 | 84.20% | 0.7219 |
| AR-TAPE v1 (Cosine lam=0.1, s42) | k=2048 | 0.1 | 90.37% | 92.69% | 0.2261 | 0.1938 | N/A (Chưa chạy) | N/A (Chưa chạy) | 0.3020 |
| AR-TAPE v1 (Cosine lam=0.25, s42) | k=2048 | 0.25 | 90.60% | 91.80% | 0.2140 | 0.2055 | N/A (Chưa chạy) | N/A (Chưa chạy) | 0.2914 |
| AR-TAPE v1 (Cosine lam=0.5) | k=2048 | 0.5 | 89.41 ± 0.50% | 90.95 ± 0.77% | 0.2425 ± 0.0144 | 0.2315 ± 0.0048 | 0.0377 | 83.44% | 0.4348 |

## NGUỒN GỐC DỮ LIỆU RAW JSON

- **B0 Vanilla**: `full_comprehensive_evaluation.json, AbReTAPE_FSHA/results_fsha_b0.json (gs=5.0, final ep 30)`
- **B7 Mode F (k=2048)**: `AbReTAPE_Step3_B7/full_comprehensive_evaluation.json, AbReTAPE_FSHA/results_fsha_b7.json (final ep 30)`
- **B7 Mode F (k=1024)**: `B7_Fresh_Eval_Final/k1024/results_artape_v0_fresh_attackers.json, AbReTAPE_FSHA/results_fsha_b7.json (final ep 30)`
- **B7 Mode F (k=512)**: `B7_Fresh_Eval_Final/k512/results_artape_v0_fresh_attackers.json, AbReTAPE_FSHA/results_fsha_b7.json (final ep 30)`
- **B8 Un-stabilized**: `AbReTAPE_Step3_B8/results_b8_conv_sweep.json, B8_Cap1.5586/*.json`
- **B8 Stabilized**: `B8_Stabilized_Attackers_Real/{fresh_mlp, fresh_conv, fsha}/*.json, AbReTAPE_Step3_B8_Stabilized/*.json`
- **AR-TAPE v1 (Cosine lam=0.0)**: `AR_TAPE_v1_Cosine_lam0, AR_TAPE_v1_Cosine_Fresh_lam0, AR_TAPE_v1_FSHA/fsha_lam0.0`
- **AR-TAPE v1 (Cosine lam=0.1, s42)**: `AR_TAPE_v1_Utility_Sweep/{lam0.1_s*, fresh_lam0.1_s*}/*.json`
- **AR-TAPE v1 (Cosine lam=0.25, s42)**: `AR_TAPE_v1_Utility_Sweep/{lam0.25_s*, fresh_lam0.25_s*}/*.json`
- **AR-TAPE v1 (Cosine lam=0.5)**: `AR_TAPE_v1_Cosine, AR_TAPE_v1_Cosine_Fresh, AR_TAPE_v1_FSHA/fsha_lam0.5`
