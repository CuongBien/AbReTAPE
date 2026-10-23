#!/usr/bin/env python3
"""
scripts/generate_master_table.py
Tự động sinh bảng Master Benchmark 100% từ các file JSON trong thư mục output/.
- Không hard-code bất kỳ số liệu đo đạc nào.
- Sử dụng độ lệch chuẩn mẫu (sample standard deviation, ddof=1) cho N=3 seeds.
- Ghi rõ nguồn file JSON cho từng dòng.
- Để 'N/A (Chưa chạy)' cho các thí nghiệm chưa được thực hiện.
"""

import os
import sys
import json
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def load_json(rel_path):
    full_path = os.path.join(OUTPUT_DIR, rel_path)
    if not os.path.isfile(full_path):
        return None
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_master_records():
    records = []

    # --------------------------------------------------------------------------
    # 1. B0 Vanilla Split Learning
    # --------------------------------------------------------------------------
    b0_eval = load_json("full_comprehensive_evaluation.json")
    b0_fsha = load_json("AbReTAPE_FSHA/results_fsha_b0.json")

    b0_row = {
        "method": "B0 Vanilla",
        "bottleneck": "Full (D=65536)",
        "lambda": "None",
        "acc_10k": f"{b0_eval['B0_Vanilla']['test_acc']*100:.2f}%" if b0_eval else "N/A",
        "acc_2k": "N/A",
        "mlp_ssim_10k": "N/A",
        "conv_ssim_10k": f"{b0_eval['B0_Vanilla']['ssim']:.4f}" if b0_eval else "N/A",
        "fsha_ssim": f"{b0_fsha[0]['ssim']:.4f}" if b0_fsha else "N/A",
        "fsha_acc": f"{b0_fsha[0]['test_acc']*100:.2f}%" if b0_fsha else "N/A",
        "dcor": f"{b0_eval['B0_Vanilla']['dcor']:.4f}" if b0_eval else "N/A",
        "json_source": "full_comprehensive_evaluation.json, AbReTAPE_FSHA/results_fsha_b0.json"
    }
    records.append(b0_row)

    # --------------------------------------------------------------------------
    # 2. B7 Mode F (k=2048)
    # --------------------------------------------------------------------------
    b7_comp = load_json("AbReTAPE_Step3_B7/full_comprehensive_evaluation.json")
    b7_fsha = load_json("AbReTAPE_FSHA/results_fsha_b7.json")

    # FSHA scenario for k=2048 in results_fsha_b7.json
    fsha_2048 = next((x for x in b7_fsha if "k2048" in x.get("scenario_id", "")), None) if b7_fsha else None

    b7_2048_row = {
        "method": "B7 Mode F (k=2048)",
        "bottleneck": "k=2048",
        "lambda": "0.0",
        "acc_10k": f"{b7_comp['B7_LightSplit_k_2048_F_s42']['test_acc']*100:.2f}%" if b7_comp else "N/A",
        "acc_2k": "N/A",
        "mlp_ssim_10k": "N/A (Chưa chạy riêng B7)",
        "conv_ssim_10k": "N/A (Chưa chạy riêng B7)",
        "fsha_ssim": f"{fsha_2048['ssim']:.4f}" if fsha_2048 else "N/A",
        "fsha_acc": f"{fsha_2048['test_acc']*100:.2f}%" if fsha_2048 else "N/A",
        "dcor": "N/A",
        "json_source": "AbReTAPE_Step3_B7/full_comprehensive_evaluation.json, AbReTAPE_FSHA/results_fsha_b7.json"
    }
    records.append(b7_2048_row)

    # --------------------------------------------------------------------------
    # 3. B7 Mode F (k=1024)
    # --------------------------------------------------------------------------
    b7_1024_fresh = load_json("B7_Fresh_Eval_Final/k1024/results_artape_v0_fresh_attackers.json")
    fsha_1024 = next((x for x in b7_fsha if "k1024_F_s42_gs5.0" in x.get("scenario_id", "")), None) if b7_fsha else None
    mlp_1024 = next((x["fresh_ssim_10k"] for x in b7_1024_fresh if x["attacker_type"] == "mlp"), None) if b7_1024_fresh else None
    conv_1024 = next((x["fresh_ssim_10k"] for x in b7_1024_fresh if x["attacker_type"] == "conv"), None) if b7_1024_fresh else None

    b7_1024_row = {
        "method": "B7 Mode F (k=1024)",
        "bottleneck": "k=1024",
        "lambda": "0.0",
        "acc_10k": f"{b7_comp['B7_LightSplit_k_1024_F_s42']['test_acc']*100:.2f}%" if b7_comp else "N/A",
        "acc_2k": "N/A",
        "mlp_ssim_10k": f"{mlp_1024:.4f}" if mlp_1024 is not None else "N/A",
        "conv_ssim_10k": f"{conv_1024:.4f}" if conv_1024 is not None else "N/A",
        "fsha_ssim": f"{fsha_1024['ssim']:.4f}" if fsha_1024 else "N/A",
        "fsha_acc": f"{fsha_1024['test_acc']*100:.2f}%" if fsha_1024 else "N/A",
        "dcor": "N/A",
        "json_source": "B7_Fresh_Eval_Final/k1024/results_artape_v0_fresh_attackers.json, AbReTAPE_FSHA/results_fsha_b7.json"
    }
    records.append(b7_1024_row)

    # --------------------------------------------------------------------------
    # 4. B7 Mode F (k=512)
    # --------------------------------------------------------------------------
    b7_512_fresh = load_json("B7_Fresh_Eval_Final/k512/results_artape_v0_fresh_attackers.json")
    fsha_512 = next((x for x in b7_fsha if "k512_F_s42_gs5.0" in x.get("scenario_id", "")), None) if b7_fsha else None
    mlp_512 = next((x["fresh_ssim_10k"] for x in b7_512_fresh if x["attacker_type"] == "mlp"), None) if b7_512_fresh else None
    conv_512 = next((x["fresh_ssim_10k"] for x in b7_512_fresh if x["attacker_type"] == "conv"), None) if b7_512_fresh else None

    b7_512_row = {
        "method": "B7 Mode F (k=512)",
        "bottleneck": "k=512",
        "lambda": "0.0",
        "acc_10k": f"{b7_comp['B7_LightSplit_k_512_F_s42']['test_acc']*100:.2f}%" if b7_comp else "N/A",
        "acc_2k": "N/A",
        "mlp_ssim_10k": f"{mlp_512:.4f}" if mlp_512 is not None else "N/A",
        "conv_ssim_10k": f"{conv_512:.4f}" if conv_512 is not None else "N/A",
        "fsha_ssim": f"{fsha_512['ssim']:.4f}" if fsha_512 else "N/A",
        "fsha_acc": f"{fsha_512['test_acc']*100:.2f}%" if fsha_512 else "N/A",
        "dcor": "N/A",
        "json_source": "B7_Fresh_Eval_Final/k512/results_artape_v0_fresh_attackers.json, AbReTAPE_FSHA/results_fsha_b7.json"
    }
    records.append(b7_512_row)

    # --------------------------------------------------------------------------
    # 5. B8 Un-stabilized (lam=0.1, s42)
    # --------------------------------------------------------------------------
    b8_sweep = load_json("AbReTAPE_Step3_B8/results_b8_conv_sweep.json")
    b8_f_conv = load_json("B8_Cap1.5586/results_b8_fresh_conv_cap1.5586.json")
    b8_f_mlp = load_json("B8_Cap1.5586/results_b8_fresh_learned_mlp_cap1.5586.json")

    b8_acc_10k = b8_sweep["b8_conv_lam0.1_s42"]["task_test_acc"] if b8_sweep else None
    b8_acc_2k = b8_sweep["b8_conv_lam0.1_s42"]["mean_task_acc_last10"] if b8_sweep else None
    b8_conv_s = next((x["fresh_conv_ssim_10k"] for x in b8_f_conv if x["lambda"] == 0.1), None) if b8_f_conv else None
    b8_mlp_s = next((x["fresh_mlp_ssim_10k"] for x in b8_f_mlp if x["lambda"] == 0.1), None) if b8_f_mlp else None

    b8_un_row = {
        "method": "B8 Un-stabilized",
        "bottleneck": "Full (D=65536)",
        "lambda": "0.1",
        "acc_10k": f"{b8_acc_10k:.2f}%" if b8_acc_10k is not None else "N/A",
        "acc_2k": f"{b8_acc_2k:.2f}%" if b8_acc_2k is not None else "N/A",
        "mlp_ssim_10k": f"{b8_mlp_s:.4f}" if b8_mlp_s is not None else "N/A",
        "conv_ssim_10k": f"{b8_conv_s:.4f}" if b8_conv_s is not None else "N/A",
        "fsha_ssim": "N/A (Chưa chạy)",
        "fsha_acc": "N/A (Chưa chạy)",
        "dcor": "N/A",
        "json_source": "AbReTAPE_Step3_B8/results_b8_conv_sweep.json, B8_Cap1.5586/*.json"
    }
    records.append(b8_un_row)

    # --------------------------------------------------------------------------
    # 6. B8 Stabilized (lam=0.1, warmup=10, clip=1.0, s42)
    # --------------------------------------------------------------------------
    b8_stab_sweep = load_json("AbReTAPE_Step3_B8_Stabilized/results_b8_conv_sweep.json")
    b8_s_acc_10k = b8_stab_sweep["b8_conv_lam0.1_s42"]["task_test_acc"] if b8_stab_sweep else None
    b8_s_acc_2k = b8_stab_sweep["b8_conv_lam0.1_s42"]["mean_task_acc_last10"] if b8_stab_sweep else None

    b8_st_row = {
        "method": "B8 Stabilized",
        "bottleneck": "Full (D=65536)",
        "lambda": "0.1",
        "acc_10k": f"{b8_s_acc_10k:.2f}%" if b8_s_acc_10k is not None else "N/A",
        "acc_2k": f"{b8_s_acc_2k:.2f}%" if b8_s_acc_2k is not None else "N/A",
        "mlp_ssim_10k": "N/A (Chưa chạy)",
        "conv_ssim_10k": "N/A (Chưa chạy)",
        "fsha_ssim": "N/A (Chưa chạy)",
        "fsha_acc": "N/A (Chưa chạy)",
        "dcor": "N/A",
        "json_source": "AbReTAPE_Step3_B8_Stabilized/results_b8_conv_sweep.json"
    }
    records.append(b8_st_row)

    # --------------------------------------------------------------------------
    # 7. AR-TAPE v1 (Cosine lam=0.0, 3 Seeds)
    # --------------------------------------------------------------------------
    seeds = [42, 7, 2024]
    lam0_accs_10k, lam0_accs_2k = [], []
    lam0_mlps, lam0_convs = [], []

    # SL task accs
    for s in seeds:
        sl_data = load_json("AR_TAPE_v1_Scheduler_Check/results_artape_v0.json") if s == 42 else load_json(f"AR_TAPE_v1_Cosine_lam0/lam0.0_s{s}/results_artape_v0.json")
        if sl_data:
            key = list(sl_data.keys())[0]
            lam0_accs_10k.append(sl_data[key]["final_full_test_acc"])
            lam0_accs_2k.append(sl_data[key]["mean_task_acc_last10_2k"])

        f_data = load_json(f"AR_TAPE_v1_Cosine_Fresh_lam0/seed_{s}/results_artape_v0_fresh_attackers.json")
        if f_data:
            lam0_mlps.append(next(x["fresh_ssim_10k"] for x in f_data if x["attacker_type"] == "mlp"))
            lam0_convs.append(next(x["fresh_ssim_10k"] for x in f_data if x["attacker_type"] == "conv"))

    # FSHA on v1 lam0.0 (seed 42)
    v1_fsha_lam0 = load_json("AR_TAPE_v1_FSHA/fsha_lam0.0/results_artape_v0_fsha.json")
    fsha_l0_gs5 = next((x for x in v1_fsha_lam0 if x.get("grad_scale") == 5.0), None) if v1_fsha_lam0 else None

    v1_lam0_row = {
        "method": "AR-TAPE v1 (Cosine lam=0.0)",
        "bottleneck": "k=2048",
        "lambda": "0.0",
        "acc_10k": f"{np.mean(lam0_accs_10k):.2f} ± {np.std(lam0_accs_10k, ddof=1):.2f}%" if len(lam0_accs_10k) > 1 else "N/A",
        "acc_2k": f"{np.mean(lam0_accs_2k):.2f} ± {np.std(lam0_accs_2k, ddof=1):.2f}%" if len(lam0_accs_2k) > 1 else "N/A",
        "mlp_ssim_10k": f"{np.mean(lam0_mlps):.4f} ± {np.std(lam0_mlps, ddof=1):.4f}" if len(lam0_mlps) > 1 else "N/A",
        "conv_ssim_10k": f"{np.mean(lam0_convs):.4f} ± {np.std(lam0_convs, ddof=1):.4f}" if len(lam0_convs) > 1 else "N/A",
        "fsha_ssim": f"{fsha_l0_gs5['final_ssim']:.4f}" if fsha_l0_gs5 else "N/A",
        "fsha_acc": f"{fsha_l0_gs5['converged_task_acc_full']:.2f}%" if fsha_l0_gs5 else "N/A",
        "dcor": f"{fsha_l0_gs5['final_dcor']:.4f}" if fsha_l0_gs5 else "N/A",
        "json_source": "AR_TAPE_v1_Cosine_lam0, AR_TAPE_v1_Cosine_Fresh_lam0, AR_TAPE_v1_FSHA/fsha_lam0.0"
    }
    records.append(v1_lam0_row)

    # --------------------------------------------------------------------------
    # 8. AR-TAPE v1 (Cosine lam=0.5, 3 Seeds)
    # --------------------------------------------------------------------------
    lam05_accs_10k, lam05_accs_2k = [], []
    lam05_mlps, lam05_convs = [], []

    for s in seeds:
        sl_data = load_json(f"AR_TAPE_v1_Cosine/lam0.5_s{s}/results_artape_v0.json")
        if sl_data:
            key = list(sl_data.keys())[0]
            lam05_accs_10k.append(sl_data[key]["final_full_test_acc"])
            lam05_accs_2k.append(sl_data[key]["mean_task_acc_last10_2k"])

        f_data = load_json(f"AR_TAPE_v1_Cosine_Fresh/fresh_lam0.5_s{s}/results_artape_v0_fresh_attackers.json")
        if f_data:
            lam05_mlps.append(next(x["fresh_ssim_10k"] for x in f_data if x["attacker_type"] == "mlp"))
            lam05_convs.append(next(x["fresh_ssim_10k"] for x in f_data if x["attacker_type"] == "conv"))

    # FSHA on v1 lam0.5 (seed 42)
    v1_fsha_lam05 = load_json("AR_TAPE_v1_FSHA/fsha_lam0.5/results_artape_v0_fsha.json")
    fsha_l05_gs5 = next((x for x in v1_fsha_lam05 if x.get("grad_scale") == 5.0), None) if v1_fsha_lam05 else None

    v1_lam05_row = {
        "method": "AR-TAPE v1 (Cosine lam=0.5)",
        "bottleneck": "k=2048",
        "lambda": "0.5",
        "acc_10k": f"{np.mean(lam05_accs_10k):.2f} ± {np.std(lam05_accs_10k, ddof=1):.2f}%" if len(lam05_accs_10k) > 1 else "N/A",
        "acc_2k": f"{np.mean(lam05_accs_2k):.2f} ± {np.std(lam05_accs_2k, ddof=1):.2f}%" if len(lam05_accs_2k) > 1 else "N/A",
        "mlp_ssim_10k": f"{np.mean(lam05_mlps):.4f} ± {np.std(lam05_mlps, ddof=1):.4f}" if len(lam05_mlps) > 1 else "N/A",
        "conv_ssim_10k": f"{np.mean(lam05_convs):.4f} ± {np.std(lam05_convs, ddof=1):.4f}" if len(lam05_convs) > 1 else "N/A",
        "fsha_ssim": f"{fsha_l05_gs5['final_ssim']:.4f}" if fsha_l05_gs5 else "N/A",
        "fsha_acc": f"{fsha_l05_gs5['converged_task_acc_full']:.2f}%" if fsha_l05_gs5 else "N/A",
        "dcor": f"{fsha_l05_gs5['final_dcor']:.4f}" if fsha_l05_gs5 else "N/A",
        "json_source": "AR_TAPE_v1_Cosine, AR_TAPE_v1_Cosine_Fresh, AR_TAPE_v1_FSHA/fsha_lam0.5"
    }
    records.append(v1_lam05_row)

    return records


def print_table(records):
    headers = [
        "Phương Pháp", "Bottleneck", "λ",
        "Acc (10k)", "Acc (2k)",
        "Fresh MLP (10k)", "Fresh Conv (10k)",
        "FSHA SSIM", "FSHA Task Acc", "dCor"
    ]
    col_keys = [
        "method", "bottleneck", "lambda",
        "acc_10k", "acc_2k",
        "mlp_ssim_10k", "conv_ssim_10k",
        "fsha_ssim", "fsha_acc", "dcor"
    ]

    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(headers))]) + " |"
    print(header_line)
    print(sep_line)
    for r in records:
        vals = [r[k] for k in col_keys]
        print("| " + " | ".join(vals) + " |")

    print("\n" + "=" * 90)
    print("NGUỒN GỐC DỮ LIỆU JSON TỪNG DÒNG:")
    print("=" * 90)
    for r in records:
        print(f"[{r['method']}]: {r['json_source']}")


if __name__ == "__main__":
    records = build_master_records()
    print_table(records)
