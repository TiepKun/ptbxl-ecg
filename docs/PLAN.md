# Phân công và kế hoạch 7 ngày

Tên thành viên đã điền trong bảng. Mỗi người **đọc kỹ và giải thích được từng dòng** trong các tệp mình phụ trách
(đề bài: thành viên không trả lời được về phần mình khai sẽ bị hạ hệ số đóng góp).

| Vai trò | Thành viên | Tệp phụ trách | Nhiệm vụ chính |
|---|---|---|---|
| A | Nguyễn Mạnh Hiếu (20261237M) | `data.py`, `prepare_data.py`, `eda.py`, `features.py`, `baseline.py`, `metrics.py`, `aggregate.py` | Tải và chuẩn bị dữ liệu, thống kê mất cân bằng nhãn, baseline thủ công, khung đánh giá, gộp kết quả |
| B | Nguyễn Khắc Tiếp (20262425M) | `models_cnn.py` (SimpleCNN, ResNet1D, CNNAttention), `train.py` | Mô hình CNN và CNN + attention, vòng huấn luyện, chọn siêu tham số |
| C | Trần Mạnh Hùng (20262423M) | `models_rnn.py` (CRNN), `ablation.py`, `saliency.py` | BiGRU/BiLSTM, toàn bộ ablation, phân tích lỗi theo lớp, saliency theo chuyển đạo, demo Gradio (tùy chọn) |

Phần báo cáo: A viết mục 3 (Dữ liệu) và baseline; B viết mục 4 (Phương pháp); C viết mục 5-6 (Thí nghiệm, phân tích). Mục 1, 2, 7 chia đều.
Mỗi người tự trình bày phần của mình khi bảo vệ. Tỉ lệ đóng góp ghi vào bảng phân công (tổng 100%).

## Cách cài đặt phần lõi (khung TODO)

Mỗi hàm/lớp cần làm có `raise NotImplementedError("TODO [A|B|C]: ...")` và mô tả yêu cầu trong docstring.
Bộ test `tests/test_<A|B|C>_*.py` là đặc tả: làm đến khi xanh. Thứ tự gợi ý để không ai bị chặn:

1. **Song song ngay:** A (`build_labels`, `features`), B (`models_cnn.py`, `EarlyStopper`, `train_one_epoch`), C (`models_rnn.py`, `build_jobs`, `lead_importance`).
2. **A ưu tiên sớm:** `split_indices`, `zscore_stats` và `metrics.py` (vì `train.run` dùng), rồi `baseline.py`, `aggregate.py`.
3. Khi test của cả ba xanh: chạy `tests/test_pipeline.py`, rồi chạy dữ liệu thật bằng `notebooks/colab_ptbxl.ipynb`.

## Việc cần bổ sung (chưa có trong repo)

- [ ] **C:** phân tích lỗi theo lớp (ví dụ ma trận nhầm lẫn đa nhãn, các bản ghi sai nhiều nhất, so sánh AUC từng lớp giữa các mô hình) bằng `runs/*/preds_test.npy`.
- [ ] **C:** đo thời gian suy luận (ms/bản ghi) của từng mô hình nếu muốn so sánh chi phí.
- [ ] **B:** nếu có thời gian, thử cân bằng lớp (`pos_weight` trong BCE) và ghi vào nhật ký như một thí nghiệm phụ.
- [ ] **A:** gói demo Gradio/Streamlit (tùy chọn cộng điểm).

## Kế hoạch 7 ngày

| Ngày | A | B | C |
|---|---|---|---|
| 1 | Tải PTB-XL, chạy `prepare_data` và `eda`, chụp bằng chứng đã tải được | Chạy thử `train.py` trên dữ liệu thật với `cnn_simple` | Đọc `ablation.py`, chuẩn bị lưới chạy, kiểm tra Colab/Kaggle |
| 1 | Chạy `baseline` để có số liệu sơ bộ, chỉnh ngưỡng thành công, điền `docs/PROPOSAL_DRAFT.md` | | |
| 2 | **Nộp đề xuất trước 12h00.** Chạy baseline 3 seed | Huấn luyện `resnet1d`, `cnn_attn` | Huấn luyện `gru`, `lstm` |
| 3-4 | Viết mục Dữ liệu, kiểm tra log | Hoàn tất nhóm `main` (đủ seed) | Hoàn tất nhóm `main` (đủ seed), chọn mô hình tốt nhất cho ablation |
| 5 | Chạy `aggregate`, đối chiếu log với số trong báo cáo | Hỗ trợ chạy `leads`, `aug` | Chạy nhóm `leads` và `aug` |
| 6 | Hình/bảng cuối, viết báo cáo mục 3 | Viết mục Phương pháp | Phân tích lỗi, saliency, viết mục Thí nghiệm và Thảo luận |
| 7 | README, `requirements.txt`, rà soát log khớp báo cáo | Slide phần của mình | Slide phần của mình, tập trình bày |

Ghi chú: ước tính mỗi lượt huấn luyện 15-20 phút trên T4, khoảng 18-25 lượt, tổng 5-8 giờ GPU (chưa đo trên PTB-XL thật). Nên chia lượt chạy ra 3 tài khoản Colab/Kaggle:
mỗi người chạy phần của mình trong `notebooks/colab_ptbxl.ipynb` (đặt `MEMBER` = A/B/C; `ablation --models ...` chia theo mô hình), log ghi vào `logs/experiments_<A|B|C>.csv`;
notebook gộp các file này thành `logs/experiments.csv` trước khi chạy `aggregate`.

## Danh sách nộp (theo đề bài)

- [ ] Repository GitHub + README chạy lại được + `requirements.txt`
- [ ] Báo cáo PDF 8-12 trang theo Phụ lục A
- [ ] Slide 8-12 trang
- [ ] Bảng phân công và tỉ lệ đóng góp
- [ ] `logs/experiments.csv` đầy đủ, khớp số trong báo cáo
- [ ] Mục "Khai báo sử dụng công cụ AI" ở cuối báo cáo
