# Cut-Layer Adapter: 1 tầng Conv 1x1 tại cut layer để kiểm chứng hiện tượng hấp thụ
import torch
import torch.nn as nn


class Adapter(nn.Module):
    """
    Cut-Layer Adapter: 1 tầng Conv 1x1 (channels -> channels, không bias).
    Được đặt ngay tại cut layer trước Server:
        z_hat = Adapter(z_perm)
    
    Khi Client và Server đã được nạp weights hội tụ từ Bước 0 và ĐÓNG BĂNG,
    Adapter là thành phần duy nhất được huấn luyện để cực tiểu hóa hàm mất mát phân loại.
    Do Server yêu cầu biểu diễn unpermuted z để phân loại chính xác,
    Adapter buộc phải học phép biến đổi nghịch đảo:
        A ≈ P_pi^T
    giúp khôi phục hoán vị kênh bí mật pi.
    """
    def __init__(self, channels=64):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, z):
        return self.conv(z)

    def get_matrix(self):
        """
        Trả về ma trận trọng số 2D shape [channels, channels].
        A[r, c] là trọng số nối từ kênh vào c của z_perm tới kênh ra r của z_hat.
        """
        return self.conv.weight.data.squeeze()
