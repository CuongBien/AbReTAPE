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
import csv
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

    # Lấy kịch bản gs5.0 ở epoch 30 cuối cùng (để đồng nhất với toàn bảng)
    b0_fsha_sc = next((x for x in b0_fsha if x.get("grad_scale") == 5.0), b0_fsha[0]) if b0_fsha else None
    b0_fsha_last = b0_fsha_sc["history"][-1] if b0_fsha_sc and b0_fsha_sc.get("history") else {}
    b0_fsha_ssim_val = b0_fsha_last.get("test_ssim", b0_fsha_sc.get("ssim")) if b0_fsha_sc else None
    b0_fsha_acc_val = b0_fsha_last.get("test_acc", b0_fsha_sc.get("test_acc")) if b0_fsha_sc else None

    b0_row = {
        "method": "B0 Vanilla",
        "bottleneck": "Full (D=65536)",
        "lambda": "None",
        "acc_10k": f"{b0_eval['B0_Vanilla']['test_acc']*100:.2f}%" if b0_eval else "N/A",
        "acc_2k": "N/A",
        "mlp_ssim_10k": "N/A",
        "conv_ssim_10k": f"{b0_eval['B0_Vanilla']['ssim']:.4f}" if b0_eval else "N/A",
        "fsha_ssim": f"{b0_fsha_ssim_val:.4f}" if b0_fsha_ssim_val is not None else "N/A",
        "fsha_acc": f"{b0_fsha_acc_val*100:.2f}%" if b0_fsha_acc_val is not None else "N/A",
        "dcor": f"{b0_eval['B0_Vanilla']['dcor']:.4f}" if b0_eval else "N/A",
        "json_source": "full_comprehensive_evaluation.json, AbReTAPE_FSHA/results_fsha_b0.json (gs=5.0, final ep 30)"
    }
    records.append(b0_row)

    # --------------------------------------------------------------------------
    # 2. B7 Mode F (k=2048)
    # --------------------------------------------------------------------------
    b7_comp = load_json("AbReTAPE_Step3_B7/full_comprehensive_evaluation.json")
    b7_fsha = load_json("AbReTAPE_FSHA/results_fsha_b7.json")

    # FSHA scenario for k=2048 ở epoch 30 cuối cùng
    fsha_2048 = next((x for x in b7_fsha if "k2048" in x.get("scenario_id", "")), None) if b7_fsha else None
    fsha_2048_last = fsha_2048["history"][-1] if fsha_2048 and fsha_2048.get("history") else {}
    fsha_2048_ssim_val = fsha_2048_last.get("test_ssim", fsha_2048.get("ssim")) if fsha_2048 else None
    fsha_2048_acc_val = fsha_2048_last.get("test_acc", fsha_2048.get("test_acc")) if fsha_2048 else None

    b7_2048_row = {
        "method": "B7 Mode F (k=2048)",
        "bottleneck": "k=2048",
        "lambda": "0.0",
        "acc_10k": f"{b7_comp['B7_LightSplit_k_2048_F_s42']['test_acc']*100:.2f}%" if b7_comp else "N/A",
        "acc_2k": "N/A",
        "mlp_ssim_10k": "N/A (Chưa chạy riêng B7)",
        "conv_ssim_10k": "N/A (Chưa chạy riêng B7)",
        "fsha_ssim": f"{fsha_2048_ssim_val:.4f}" if fsha_2048_ssim_val is not None else "N/A",
        "fsha_acc": f"{fsha_2048_acc_val*100:.2f}%" if fsha_2048_acc_val is not None else "N/A",
        "dcor": "N/A",
        "json_source": "AbReTAPE_Step3_B7/full_comprehensive_evaluation.json, AbReTAPE_FSHA/results_fsha_b7.json (final ep 30)"
    }
    records.append(b7_2048_row)

    # --------------------------------------------------------------------------
    # 3. B7 Mode F (k=1024)
    # --------------------------------------------------------------------------
    b7_1024_fresh = load_json("B7_Fresh_Eval_Final/k1024/results_artape_v0_fresh_attackers.json")
    fsha_1024 = next((x for x in b7_fsha if "k1024_F_s42_gs5.0" in x.get("scenario_id", "")), None) if b7_fsha else None
    fsha_1024_last = fsha_1024["history"][-1] if fsha_1024 and fsha_1024.get("history") else {}
    fsha_1024_ssim_val = fsha_1024_last.get("test_ssim", fsha_1024.get("ssim")) if fsha_1024 else None
    fsha_1024_acc_val = fsha_1024_last.get("test_acc", fsha_1024.get("test_acc")) if fsha_1024 else None

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
        "fsha_ssim": f"{fsha_1024_ssim_val:.4f}" if fsha_1024_ssim_val is not None else "N/A",
        "fsha_acc": f"{fsha_1024_acc_val*100:.2f}%" if fsha_1024_acc_val is not None else "N/A",
        "dcor": "N/A",
        "json_source": "B7_Fresh_Eval_Final/k1024/results_artape_v0_fresh_attackers.json, AbReTAPE_FSHA/results_fsha_b7.json (final ep 30)"
    }
    records.append(b7_1024_row)

    # --------------------------------------------------------------------------
    # 4. B7 Mode F (k=512)
    # --------------------------------------------------------------------------
    b7_512_fresh = load_json("B7_Fresh_Eval_Final/k512/results_artape_v0_fresh_attackers.json")
    fsha_512 = next((x for x in b7_fsha if "k512_F_s42_gs5.0" in x.get("scenario_id", "")), None) if b7_fsha else None
    fsha_512_last = fsha_512["history"][-1] if fsha_512 and fsha_512.get("history") else {}
    fsha_512_ssim_val = fsha_512_last.get("test_ssim", fsha_512.get("ssim")) if fsha_512 else None
    fsha_512_acc_val = fsha_512_last.get("test_acc", fsha_512.get("test_acc")) if fsha_512 else None

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
        "fsha_ssim": f"{fsha_512_ssim_val:.4f}" if fsha_512_ssim_val is not None else "N/A",
        "fsha_acc": f"{fsha_512_acc_val*100:.2f}%" if fsha_512_acc_val is not None else "N/A",
        "dcor": "N/A",
        "json_source": "B7_Fresh_Eval_Final/k512/results_artape_v0_fresh_attackers.json, AbReTAPE_FSHA/results_fsha_b7.json (final ep 30)"
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

    # Load real fresh attackers and FSHA on B8 Stabilized
    b8_s_f_mlp = load_json("B8_Stabilized_Attackers_Real/fresh_mlp/results_b8_fresh_learned_mlp.json")
    b8_s_f_conv = load_json("B8_Stabilized_Attackers_Real/fresh_conv/results_b8_fresh_conv.json")
    b8_s_fsha = load_json("B8_Stabilized_Attackers_Real/fsha/results_fsha_b8.json")

    b8_s_mlp_ssim = b8_s_f_mlp[0]["fresh_mlp_ssim_10k"] if b8_s_f_mlp else None
    b8_s_conv_ssim = b8_s_f_conv[0]["fresh_conv_ssim_10k"] if b8_s_f_conv else None
    
    # FSHA scenario gs5.0 for fair comparison with B7 and v1 (which both use gs5.0 in table)
    b8_s_fsha_gs5 = next((x for x in b8_s_fsha if "gs5.0" in x.get("scenario_id", "")), None) if b8_s_fsha else None
    b8_s_fsha_ssim = b8_s_fsha_gs5["ssim_final"] if b8_s_fsha_gs5 and "ssim_final" in b8_s_fsha_gs5 else (b8_s_fsha_gs5["ssim"] if b8_s_fsha_gs5 else None)
    b8_s_fsha_acc = b8_s_fsha_gs5["test_acc"] * 100 if b8_s_fsha_gs5 else None
    # dCor đo tại Epoch 100 trong huấn luyện thông thường (đồng nhất với các hàng khác)
    b8_s_dcor = b8_stab_sweep["b8_conv_lam0.1_s42"]["history"][-1].get("dcor") if (b8_stab_sweep and "history" in b8_stab_sweep["b8_conv_lam0.1_s42"]) else None

    b8_st_row = {
        "method": "B8 Stabilized",
        "bottleneck": "Full (D=65536)",
        "lambda": "0.1",
        "acc_10k": f"{b8_s_acc_10k:.2f}%" if b8_s_acc_10k is not None else "N/A",
        "acc_2k": f"{b8_s_acc_2k:.2f}%" if b8_s_acc_2k is not None else "N/A",
        "mlp_ssim_10k": f"{b8_s_mlp_ssim:.4f}" if b8_s_mlp_ssim is not None else "N/A (Chưa chạy)",
        "conv_ssim_10k": f"{b8_s_conv_ssim:.4f}" if b8_s_conv_ssim is not None else "N/A (Chưa chạy)",
        "fsha_ssim": f"{b8_s_fsha_ssim:.4f}" if b8_s_fsha_ssim is not None else "N/A (Chưa chạy)",
        "fsha_acc": f"{b8_s_fsha_acc:.2f}%" if b8_s_fsha_acc is not None else "N/A (Chưa chạy)",
        "dcor": f"{b8_s_dcor:.4f}" if b8_s_dcor is not None else "N/A",
        "json_source": "B8_Stabilized_Attackers_Real/{fresh_mlp, fresh_conv, fsha}/*.json, AbReTAPE_Step3_B8_Stabilized/*.json"
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
    # 8. AR-TAPE v1 Utility Sweep (lam=0.1, Multi-Seed / s42)
    # --------------------------------------------------------------------------
    lam01_accs_10k, lam01_accs_2k = [], []
    lam01_mlps, lam01_convs = [], []
    lam01_dcors = []
    found_seeds_01 = []

    for s in [42, 7, 2024]:
        sl_data = load_json(f"AR_TAPE_v1_Utility_Sweep/lam0.1_s{s}/results_artape_v0.json")
        f_data = load_json(f"AR_TAPE_v1_Utility_Sweep/fresh_lam0.1_s{s}/results_artape_v0_fresh_attackers.json")
        if sl_data:
            key = list(sl_data.keys())[0]
            lam01_accs_10k.append(sl_data[key]["final_full_test_acc"])
            lam01_accs_2k.append(sl_data[key]["mean_task_acc_last10_2k"])
            dc = sl_data[key].get("final_dcor", sl_data[key].get("history", [{}])[-1].get("dcor"))
            if dc is not None:
                lam01_dcors.append(dc)
            found_seeds_01.append(s)
        if f_data:
            mlp_v = next((x["fresh_ssim_10k"] for x in f_data if x["attacker_type"] == "mlp"), None)
            conv_v = next((x["fresh_ssim_10k"] for x in f_data if x["attacker_type"] == "conv"), None)
            if mlp_v is not None:
                lam01_mlps.append(mlp_v)
            if conv_v is not None:
                lam01_convs.append(conv_v)

    fsha_l01 = load_json("AR_TAPE_v1_Utility_Sweep/fsha_lam0.1_s42/results_artape_v0_fsha.json")
    if not fsha_l01:
        fsha_l01 = load_json("AR_TAPE_v1_FSHA/fsha_lam0.1/results_artape_v0_fsha.json")
    fsha_l01_gs5 = next((x for x in fsha_l01 if x.get("grad_scale") == 5.0), None) if fsha_l01 else None

    if lam01_accs_10k:
        is_multi = len(lam01_accs_10k) > 1
        m_label = f"AR-TAPE v1 (Cosine lam=0.1, {len(found_seeds_01)} seeds)" if is_multi else f"AR-TAPE v1 (Cosine lam=0.1, s{found_seeds_01[0]})"
        records.append({
            "method": m_label,
            "bottleneck": "k=2048",
            "lambda": "0.1",
            "acc_10k": f"{np.mean(lam01_accs_10k):.2f} ± {np.std(lam01_accs_10k, ddof=1):.2f}%" if is_multi else f"{lam01_accs_10k[0]:.2f}%",
            "acc_2k": f"{np.mean(lam01_accs_2k):.2f} ± {np.std(lam01_accs_2k, ddof=1):.2f}%" if is_multi else f"{lam01_accs_2k[0]:.2f}%",
            "mlp_ssim_10k": f"{np.mean(lam01_mlps):.4f} ± {np.std(lam01_mlps, ddof=1):.4f}" if (is_multi and lam01_mlps) else (f"{lam01_mlps[0]:.4f}" if lam01_mlps else "N/A"),
            "conv_ssim_10k": f"{np.mean(lam01_convs):.4f} ± {np.std(lam01_convs, ddof=1):.4f}" if (is_multi and lam01_convs) else (f"{lam01_convs[0]:.4f}" if lam01_convs else "N/A"),
            "fsha_ssim": f"{fsha_l01_gs5['final_ssim']:.4f}" if fsha_l01_gs5 else "N/A (Chưa chạy)",
            "fsha_acc": f"{fsha_l01_gs5['converged_task_acc_full']:.2f}%" if fsha_l01_gs5 else "N/A (Chưa chạy)",
            "dcor": f"{np.mean(lam01_dcors):.4f} ± {np.std(lam01_dcors, ddof=1):.4f}" if (is_multi and lam01_dcors) else (f"{lam01_dcors[0]:.4f}" if lam01_dcors else "N/A"),
            "json_source": f"AR_TAPE_v1_Utility_Sweep/{{lam0.1_s*, fresh_lam0.1_s*}}/*.json"
        })

    # --------------------------------------------------------------------------
    # 9. AR-TAPE v1 Utility Sweep (lam=0.25, Multi-Seed / s42) - Sweet Spot
    # --------------------------------------------------------------------------
    lam025_accs_10k, lam025_accs_2k = [], []
    lam025_mlps, lam025_convs = [], []
    lam025_dcors = []
    found_seeds_025 = []

    for s in [42, 7, 2024]:
        sl_data = load_json(f"AR_TAPE_v1_Utility_Sweep/lam0.25_s{s}/results_artape_v0.json")
        f_data = load_json(f"AR_TAPE_v1_Utility_Sweep/fresh_lam0.25_s{s}/results_artape_v0_fresh_attackers.json")
        if sl_data:
            key = list(sl_data.keys())[0]
            lam025_accs_10k.append(sl_data[key]["final_full_test_acc"])
            lam025_accs_2k.append(sl_data[key]["mean_task_acc_last10_2k"])
            dc = sl_data[key].get("final_dcor", sl_data[key].get("history", [{}])[-1].get("dcor"))
            if dc is not None:
                lam025_dcors.append(dc)
            found_seeds_025.append(s)
        if f_data:
            mlp_v = next((x["fresh_ssim_10k"] for x in f_data if x["attacker_type"] == "mlp"), None)
            conv_v = next((x["fresh_ssim_10k"] for x in f_data if x["attacker_type"] == "conv"), None)
            if mlp_v is not None:
                lam025_mlps.append(mlp_v)
            if conv_v is not None:
                lam025_convs.append(conv_v)

    fsha_l025 = load_json("AR_TAPE_v1_Utility_Sweep/fsha_lam0.25_s42/results_artape_v0_fsha.json")
    if not fsha_l025:
        fsha_l025 = load_json("AR_TAPE_v1_FSHA/fsha_lam0.25/results_artape_v0_fsha.json")
    fsha_l025_gs5 = next((x for x in fsha_l025 if x.get("grad_scale") == 5.0), None) if fsha_l025 else None

    if lam025_accs_10k:
        is_multi = len(lam025_accs_10k) > 1
        m_label = f"AR-TAPE v1 (Cosine lam=0.25, {len(found_seeds_025)} seeds)" if is_multi else f"AR-TAPE v1 (Cosine lam=0.25, s{found_seeds_025[0]})"
        records.append({
            "method": m_label,
            "bottleneck": "k=2048",
            "lambda": "0.25",
            "acc_10k": f"{np.mean(lam025_accs_10k):.2f} ± {np.std(lam025_accs_10k, ddof=1):.2f}%" if is_multi else f"{lam025_accs_10k[0]:.2f}%",
            "acc_2k": f"{np.mean(lam025_accs_2k):.2f} ± {np.std(lam025_accs_2k, ddof=1):.2f}%" if is_multi else f"{lam025_accs_2k[0]:.2f}%",
            "mlp_ssim_10k": f"{np.mean(lam025_mlps):.4f} ± {np.std(lam025_mlps, ddof=1):.4f}" if (is_multi and lam025_mlps) else (f"{lam025_mlps[0]:.4f}" if lam025_mlps else "N/A"),
            "conv_ssim_10k": f"{np.mean(lam025_convs):.4f} ± {np.std(lam025_convs, ddof=1):.4f}" if (is_multi and lam025_convs) else (f"{lam025_convs[0]:.4f}" if lam025_convs else "N/A"),
            "fsha_ssim": f"{fsha_l025_gs5['final_ssim']:.4f}" if fsha_l025_gs5 else "N/A (Chưa chạy)",
            "fsha_acc": f"{fsha_l025_gs5['converged_task_acc_full']:.2f}%" if fsha_l025_gs5 else "N/A (Chưa chạy)",
            "dcor": f"{np.mean(lam025_dcors):.4f} ± {np.std(lam025_dcors, ddof=1):.4f}" if (is_multi and lam025_dcors) else (f"{lam025_dcors[0]:.4f}" if lam025_dcors else "N/A"),
            "json_source": f"AR_TAPE_v1_Utility_Sweep/{{lam0.25_s*, fresh_lam0.25_s*}}/*.json"
        })

    # --------------------------------------------------------------------------
    # 10. AR-TAPE v1 (Cosine lam=0.5, 3 Seeds)
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
    lines = [header_line, sep_line]
    for r in records:
        vals = [r[k] for k in col_keys]
        row_str = "| " + " | ".join(vals) + " |"
        print(row_str)
        lines.append(row_str)

    print("\n" + "=" * 90)
    print("NGUỒN GỐC DỮ LIỆU JSON TỪNG DÒNG:")
    print("=" * 90)
    for r in records:
        print(f"[{r['method']}]: {r['json_source']}")

    # Ghi file master_benchmark_table.md
    md_path = os.path.join(PROJECT_ROOT, "master_benchmark_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MASTER BENCHMARK TABLE: SPLIT LEARNING CIFAR-10 RESNET-18 (CUT-1)\n\n")
        f.write("> **Ghi chú chuẩn hóa số liệu:**\n")
        f.write("> 1. **Acc (10k)**: Đánh giá trên toàn bộ 10,000 ảnh test CIFAR-10 tại epoch cuối cùng (Epoch 100).\n")
        f.write("> 2. **Fresh MLP / Conv**: Huấn luyện 30 epochs từ scratch trên client đóng băng, lấy số liệu tại epoch 30 cuối cùng (không cherry-pick theo test peak).\n")
        f.write("> 3. **FSHA SSIM & Task Acc**: Đánh giá tại epoch 30 cuối cùng (mức grad_scale=5.0) để đồng nhất trên toàn bộ các phương pháp.\n")
        f.write("> 4. **dCor**: Đo khoảng cách tương quan Distance Correlation giữa biểu diễn cut-layer và ảnh gốc tại epoch 100 trong huấn luyện thông thường.\n")
        f.write("> 5. **Sai số**: Dùng độ lệch chuẩn mẫu hiệu chỉnh (sample std, ddof=1) cho các cấu hình đa seed (n=3: seed 42, 7, 2024).\n\n")
        f.write("\n".join(lines) + "\n\n")
        f.write("## NGUỒN GỐC DỮ LIỆU RAW JSON\n\n")
        for r in records:
            f.write(f"- **{r['method']}**: `{r['json_source']}`\n")

    # Ghi file master_benchmark_table.csv
    csv_path = os.path.join(PROJECT_ROOT, "master_benchmark_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=col_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"\nĐã lưu bảng đối chuẩn tại: {md_path} và {csv_path}\n")


if __name__ == "__main__":
    records = build_master_records()
    print_table(records)
