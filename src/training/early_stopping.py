# Module quản lý Early Stopping trong quá trình huấn luyện
import numpy as np


class EarlyStopping:
    """
    Bộ theo dõi Early Stopping để dừng huấn luyện sớm khi metric không cải thiện
    sau một số lượng epoch nhất định (patience).
    Hỗ trợ cả mode='max' (Accuracy, PSNR, SSIM) và mode='min' (Loss, MSE).
    """

    def __init__(self, patience=15, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.is_best = False

    def step(self, score, epoch=0):
        """
        Cập nhật điểm số đánh giá ở epoch hiện tại.
        Trả về True nếu cần dừng sớm (early stop), ngược lại False.
        """
        if self.patience is None or self.patience <= 0:
            return False

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.is_best = True
            self.counter = 0
            return False

        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.is_best = True
            self.counter = 0
        else:
            self.is_best = False
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.is_best = False
