# Module đo lường Độ tin cậy Lâm sàng và Visual Attention Alignment
# Dành cho Tầng 2 (HAM10000 / PCAM) và phân tích Grad-CAM Alignment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix


def compute_binary_clinical_metrics(y_true, y_probs, val_y_true=None, val_y_probs=None):
    """
    Tính toán các chỉ số lâm sàng cho bài toán nhị phân (vd: PCAM - PatchCamelyon):
    - AUC-ROC
    - Sensitivity & Specificity tại ngưỡng mặc định 0.5
    - Ngưỡng tối ưu Youden's J (chọn trên tập validation nếu có) và Sens/Spec tương ứng
    - Sensitivity tại ngưỡng đạt Specificity >= 90% (Sensitivity@90% Specificity)
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    if y_probs.ndim > 1 and y_probs.shape[1] == 2:
        y_probs = y_probs[:, 1]

    auc = float(roc_auc_score(y_true, y_probs))

    # 1. Ngưỡng mặc định 0.5
    preds_05 = (y_probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds_05, labels=[0, 1]).ravel()
    sens_05 = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec_05 = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # 2. Ngưỡng Youden's J
    # Nếu có tập validation, tìm threshold trên val; nếu không, tìm trên tập hiện tại
    tune_true = val_y_true if val_y_true is not None else y_true
    tune_probs = val_y_probs if val_y_probs is not None else y_probs
    if tune_probs.ndim > 1 and tune_probs.shape[1] == 2:
        tune_probs = tune_probs[:, 1]

    fpr_v, tpr_v, thresholds_v = roc_curve(tune_true, tune_probs)
    j_scores = tpr_v - fpr_v
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds_v[best_idx])

    preds_youden = (y_probs >= best_thresh).astype(int)
    tn_y, fp_y, fn_y, tp_y = confusion_matrix(y_true, preds_youden, labels=[0, 1]).ravel()
    sens_youden = float(tp_y / (tp_y + fn_y)) if (tp_y + fn_y) > 0 else 0.0
    spec_youden = float(tn_y / (tn_y + fp_y)) if (tn_y + fp_y) > 0 else 0.0

    # 3. Sensitivity tại Specificity >= 90% (tức FPR <= 10%)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    valid_indices = np.where(fpr <= 0.10)[0]
    sens_at_90_spec = float(np.max(tpr[valid_indices])) if len(valid_indices) > 0 else 0.0

    return {
        "auc_roc": auc,
        "sensitivity_0.5": sens_05,
        "specificity_0.5": spec_05,
        "youden_threshold": best_thresh,
        "sensitivity_youden": sens_youden,
        "specificity_youden": spec_youden,
        "sensitivity_at_90_specificity": sens_at_90_spec
    }


def compute_multiclass_clinical_metrics(y_true, y_probs, num_classes=7):
    """
    Tính toán các chỉ số lâm sàng cho bài toán đa lớp (vd: HAM10000 - 7 lớp tổn thương da):
    - Macro-averaged AUC-ROC (One-vs-Rest)
    - Macro-averaged F1-score
    - Sensitivity & Specificity theo One-vs-Rest cho từng lớp
    - Macro-averaged Sensitivity@90% Specificity
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_preds = np.argmax(y_probs, axis=1)

    # 1. Macro AUC-ROC (OvR)
    try:
        macro_auc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
    except Exception:
        macro_auc = None

    # 2. Macro F1
    macro_f1 = float(f1_score(y_true, y_preds, average="macro"))

    # 3. Per-class OvR metrics
    per_class_sens_at_90_spec = []
    per_class_metrics = {}

    for c in range(num_classes):
        y_binary = (y_true == c).astype(int)
        prob_c = y_probs[:, c]
        pred_c = (y_preds == c).astype(int)

        if np.sum(y_binary) > 0:
            tn, fp, fn, tp = confusion_matrix(y_binary, pred_c, labels=[0, 1]).ravel()
            sens_c = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec_c = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            fpr, tpr, _ = roc_curve(y_binary, prob_c)
            valid_indices = np.where(fpr <= 0.10)[0]
            sens_90 = float(np.max(tpr[valid_indices])) if len(valid_indices) > 0 else 0.0
            per_class_sens_at_90_spec.append(sens_90)

            per_class_metrics[f"class_{c}"] = {
                "sensitivity": sens_c,
                "specificity": spec_c,
                "sensitivity_at_90_spec": sens_90
            }

    macro_sens_at_90_spec = float(np.mean(per_class_sens_at_90_spec)) if per_class_sens_at_90_spec else 0.0

    return {
        "macro_auc_roc": macro_auc,
        "macro_f1": macro_f1,
        "macro_sensitivity_at_90_specificity": macro_sens_at_90_spec,
        "per_class": per_class_metrics
    }


class GradCAM:
    """
    Trích xuất bản đồ kích hoạt trực quan Grad-CAM tại lớp tích chập cuối cùng (last conv layer).
    Hỗ trợ tính toán độ tương đồng Pearson và Cosine Similarity giữa 2 mô hình.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1)

        one_hot = torch.zeros_like(output)
        for i in range(output.size(0)):
            one_hot[i, class_idx[i]] = 1.0

        output.backward(gradient=one_hot, retain_graph=True)

        # Trọng số alpha: global average pooling của gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Chuẩn hóa về [0, 1] cho từng ảnh trong batch
        b, _, h, w = cam.shape
        cam_min = cam.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        cam_max = cam.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Nội suy về kích thước ảnh gốc x (B, 1, H, W)
        cam = F.interpolate(cam, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        return cam.detach()


def compute_gradcam_alignment(cam_orig, cam_defended):
    """
    Đo độ tương quan chú ý thị giác giữa mô hình gốc (Vanilla) và mô hình có mã hóa (Defended):
    - Pearson Correlation
    - Cosine Similarity
    """
    b = cam_orig.size(0)
    flat_orig = cam_orig.view(b, -1)
    flat_def = cam_defended.view(b, -1)

    # 1. Cosine similarity
    cos_sim = F.cosine_similarity(flat_orig, flat_def, dim=1).mean().item()

    # 2. Pearson correlation
    orig_centered = flat_orig - flat_orig.mean(dim=1, keepdim=True)
    def_centered = flat_def - flat_def.mean(dim=1, keepdim=True)

    numer = (orig_centered * def_centered).sum(dim=1)
    denom = torch.sqrt((orig_centered ** 2).sum(dim=1) * (def_centered ** 2).sum(dim=1) + 1e-8)
    pearson_r = (numer / denom).mean().item()

    return {
        "gradcam_cosine_similarity": float(cos_sim),
        "gradcam_pearson_correlation": float(pearson_r)
    }
