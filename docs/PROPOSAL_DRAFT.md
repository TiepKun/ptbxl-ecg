# Bản nháp đề xuất Đề tài 10 (theo Phụ lục B)

> Nộp 1-2 trang, trước 12h00 Ngày 2. Điền các chỗ trong ngoặc vuông. Ngưỡng thành công nên chỉnh sau khi chạy baseline ở Ngày 1.

| Mục | Nội dung |
|---|---|
| **Tên đề tài** | Phân loại bất thường tim mạch từ ECG 12 chuyển đạo (PTB-XL): so sánh CNN 1D, GRU/LSTM và attention |
| **Thành viên** | [Họ tên, MSSV, vai trò A: dữ liệu và baseline] · [Họ tên, MSSV, vai trò B: CNN và attention] · [Họ tên, MSSV, vai trò C: RNN và ablation] |
| **Bài toán** | Đầu vào: bản ghi ECG 12 chuyển đạo dài 10 giây, 100 Hz (12 × 1000). Đầu ra: nhãn đa nhãn cho 5 siêu lớp chẩn đoán (NORM, MI, STTC, CD, HYP). ECG là xét nghiệm không xâm lấn rất phổ biến, nên bộ hỗ trợ sàng lọc tự động giúp giảm tải cho bác sĩ. Nhóm so sánh có kiểm soát ba họ mô hình học sâu với cùng ngân sách tham số. |
| **Dữ liệu** | PTB-XL v1.0.3, https://physionet.org/content/ptb-xl/1.0.3/ · 21.799 bản ghi từ 18.869 bệnh nhân · giấy phép CC BY 4.0 · dùng bản 100 Hz · chia theo `strat_fold` khuyến nghị của tác giả: fold 1-8 train, 9 validation, 10 test. [Đã tải về ngày .../..., số bản ghi có nhãn sau lọc: ...] |
| **Phương pháp** | (1) ResNet 1D nhỏ; (2) BiGRU/BiLSTM sau khối conv giảm mẫu; (3) CNN + self-attention (Transformer encoder 2 lớp); mỗi mô hình khoảng 0,5 triệu tham số. BCE đa nhãn, AdamW + OneCycleLR, augmentation (nhiễu, co giãn biên độ, dịch thời gian). Thư viện: PyTorch, scikit-learn, wfdb. Kỹ thuật đã học: CNN, RNN/LSTM, attention. |
| **Mô hình cơ sở** | Logistic Regression và Random Forest trên 132 đặc trưng thủ công (thống kê và năng lượng theo dải tần cho từng chuyển đạo); CNN 1D 3 tầng đơn giản. Đối chiếu thêm với benchmark công bố của Strodthoff et al. (macro-AUC khoảng 0,92-0,93 trên các tác vụ chẩn đoán). |
| **Ablation** | (a) số chuyển đạo: 12 / 6 chuyển đạo chi / chỉ chuyển đạo II; (b) augmentation bật/tắt; (c) kiến trúc CNN so với RNN so với attention cùng ngân sách tham số. Mỗi cấu hình ít nhất 3 seed, báo cáo trung bình ± độ lệch chuẩn. |
| **Tiêu chí thành công** | Macro-AUC ≥ [0,88] trên fold 10 cho mô hình tốt nhất, và vượt baseline đặc trưng thủ công ít nhất [0,05] AUC. Báo cáo cuối sẽ đối chiếu lại với tiêu chí này kể cả khi không đạt. |
| **Kế hoạch** | Xem `docs/PLAN.md`: Ngày 1 dữ liệu + baseline; Ngày 2 nộp đề xuất, bắt đầu huấn luyện; Ngày 3-4 mô hình chính; Ngày 5 ablation; Ngày 6 phân tích và báo cáo; Ngày 7 slide, README, rà soát log. A lo dữ liệu/baseline/đánh giá, B lo CNN và attention, C lo RNN, ablation và phân tích. |
| **Tài nguyên tính toán** | 1 GPU T4 (Colab/Kaggle), chia cho 3 tài khoản. Ước tính 15-20 phút mỗi lượt, khoảng 18-25 lượt, tổng 5-8 giờ GPU [cập nhật sau khi chạy thử]. |
| **Rủi ro và dự phòng** | RNN chạy chậm trên chuỗi 1000 bước: giảm còn 250 bước bằng conv stride (đã cài sẵn). Thiếu thời gian: thu hẹp về bài toán nhị phân NORM so với bất thường và bỏ nhánh LSTM. Mất cân bằng lớp: báo cáo AUC/F1 theo từng lớp, thử `pos_weight`. Tải dữ liệu lỗi: tải thủ công từ PhysioNet. Mã tham khảo (repo benchmark) chỉ dùng để đối chiếu, nhóm tự viết toàn bộ mã huấn luyện và ghi rõ trong README. |
