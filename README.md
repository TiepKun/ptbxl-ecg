# Phân loại bất thường tim mạch từ ECG 12 chuyển đạo (PTB-XL)

So sánh **CNN 1D (ResNet 1D)**, **BiGRU / BiLSTM** và **CNN + self-attention (Transformer encoder nhỏ)** trên bộ dữ liệu
PTB-XL, kèm baseline cổ điển (đặc trưng thủ công + Logistic Regression / Random Forest) và các thí nghiệm ablation.
Bài tập lớn môn IT4653 - Học sâu và ứng dụng (HUST), Đề tài 10 (tự chọn).

- **Đầu vào:** bản ghi ECG 12 chuyển đạo, 10 giây, 100 Hz, tức tensor `12 × 1000`.
- **Đầu ra:** đa nhãn cho 5 siêu lớp chẩn đoán `NORM, MI, STTC, CD, HYP`.
- **Chia tập (khuyến nghị của tác giả PTB-XL):** `strat_fold` 1-8 train, 9 validation, 10 test.
- **Chỉ số chính:** macro-AUC trên test (fold 10); phụ: AUC từng lớp, macro-F1 (ngưỡng chọn trên validation).

> **Trạng thái:** pipeline đã được kiểm thử bằng `pytest` và chạy thử trên dữ liệu **giả lập** (và một thư mục PTB-XL giả
> đúng định dạng WFDB). Chưa có kết quả nào trên PTB-XL thật; mọi số liệu trong báo cáo phải do nhóm tự chạy và lấy từ
> `logs/experiments.csv`.

## 1. Cấu trúc thư mục

```
ptbxl-ecg/
├── README.md
├── requirements.txt
├── ptbxl_ecg/                 # mã nguồn chính (chạy bằng `python -m ptbxl_ecg.<module>`)
│   ├── data.py                # nhãn -> 5 siêu lớp, cache, chia fold, chuẩn hóa, augmentation
│   ├── prepare_data.py        # đọc PTB-XL thô -> data/processed/*.npy
│   ├── eda.py                 # thống kê mô tả, hình minh họa
│   ├── features.py            # đặc trưng thủ công cho baseline
│   ├── baseline.py            # Logistic Regression / Random Forest
│   ├── models.py              # SimpleCNN, ResNet1D, BiGRU/BiLSTM, CNN+attention
│   ├── train.py               # huấn luyện 1 cấu hình, ghi log
│   ├── ablation.py            # chạy lưới thí nghiệm (main / leads / aug)
│   ├── aggregate.py           # tổng hợp trung bình ± std, vẽ biểu đồ
│   ├── saliency.py            # độ quan trọng theo chuyển đạo
│   ├── metrics.py, logging_utils.py
│   └── synthetic.py           # dữ liệu giả lập để thử code (KHÔNG dùng cho kết quả thật)
├── scripts/download_data.sh   # tải PTB-XL (records100 + 2 CSV)
├── tests/test_smoke.py
├── docs/
│   ├── PLAN.md                # phân công A/B/C và kế hoạch 7 ngày
│   ├── PROPOSAL_DRAFT.md      # bản nháp đề xuất theo Phụ lục B
│   └── AI_DECLARATION.md      # mẫu khai báo sử dụng công cụ AI
├── data/                      # dữ liệu (không đưa lên git)
├── runs/                      # checkpoint, history, dự đoán từng lượt chạy (không đưa lên git)
├── logs/experiments.csv       # NHẬT KÝ THÍ NGHIỆM (nộp kèm)
└── results/                   # bảng và hình tổng hợp
```

## 2. Cài đặt

Yêu cầu Python 3.9+ (khuyến nghị 3.10+).

```bash
git clone <địa chỉ repo của nhóm> ptbxl-ecg && cd ptbxl-ecg
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# Cài PyTorch đúng với máy bạn theo https://pytorch.org, sau đó:
pip install -r requirements.txt
```

**Trên Google Colab / Kaggle** (PyTorch đã có sẵn, nhớ bật GPU T4):

```python
!git clone <địa chỉ repo của nhóm> ptbxl-ecg
%cd ptbxl-ecg
!pip install -q wfdb
```

## 3. Kiểm tra nhanh (không cần dữ liệu thật)

```bash
python -m pytest -q tests                 # 17 test, chạy trên CPU khoảng 15-20 giây
python -m ptbxl_ecg.synthetic --out-dir data/synthetic --n 1000
python -m ptbxl_ecg.train --data-dir data/synthetic --model resnet1d --epochs 3 --group dev
```

Dữ liệu giả lập rất dễ nên số AUC ở đây vô nghĩa; mục đích chỉ là kiểm tra code chạy thông.
Nhóm `--group dev` bị `aggregate.py` bỏ qua.

## 4. Dữ liệu thật

PTB-XL v1.0.3 (PhysioNet, giấy phép CC BY 4.0): https://physionet.org/content/ptb-xl/1.0.3/

```bash
bash scripts/download_data.sh data/raw          # chỉ tải records100 + 2 file CSV; có thể chạy lại để tiếp tục
python -m ptbxl_ecg.prepare_data --raw-dir data/raw --out-dir data/processed
python -m ptbxl_ecg.eda --data-dir data/processed
```

`prepare_data` đọc từng bản ghi 100 Hz bằng `wfdb`, ánh xạ `scp_codes` sang 5 siêu lớp qua `scp_statements.csv`
(giống cách benchmark gốc: mọi mã chẩn đoán có mặt trong `scp_codes` đều được tính), bỏ các bản ghi không có nhãn chẩn đoán, rồi lưu
`X.npy` (khoảng 1 GB), `Y.npy`, `folds.npy`. Trên Colab nên lưu `data/processed` lên Google Drive để khỏi phải đọc lại.

> Script tải dùng `wget` và **chưa được thử trên PhysioNet thật** (môi trường tạo repo không truy cập được). Nếu lỗi,
> tải thủ công từ trang PhysioNet ở trên rồi đặt `ptbxl_database.csv`, `scp_statements.csv`, `records100/` vào `data/raw/`.

## 5. Chạy thí nghiệm

```bash
# (1) Baseline cổ điển: đặc trưng thủ công + LR / RF
python -m ptbxl_ecg.baseline --seeds 0 1 2

# (2) Một mô hình, một seed
python -m ptbxl_ecg.train --model resnet1d --seed 0 --group main
python -m ptbxl_ecg.train --model gru --leads limb6 --no-aug --group leads

# (3) Toàn bộ lưới thí nghiệm (xem trước bằng --dry-run)
python -m ptbxl_ecg.ablation --group main  --seeds 0 1 2 --epochs 30
python -m ptbxl_ecg.ablation --group all   --seeds 0 1 2 --epochs 30 --model resnet1d   # + leads + aug

# (4) Bảng trung bình ± std và biểu đồ -> results/
python -m ptbxl_ecg.aggregate

# (5) Phân tích: chuyển đạo nào quan trọng nhất cho từng lớp
python -m ptbxl_ecg.saliency --run-dir runs/main_resnet1d_12_aug1_s0
```

Nhóm thí nghiệm:

| Nhóm | Nội dung | Ánh xạ tới đề xuất |
|---|---|---|
| `baseline` | LR / RF trên đặc trưng thủ công | mô hình cơ sở |
| `main` | `cnn_simple`, `resnet1d`, `gru`, `lstm`, `cnn_attn`; 12 chuyển đạo; có augmentation | ablation (c): kiến trúc |
| `leads` | mô hình đã chọn với `limb6` và `II` (12 chuyển đạo lấy từ `main`) | ablation (a): số chuyển đạo |
| `aug` | mô hình đã chọn với `--no-aug` | ablation (b): augmentation |

Bốn mô hình chính được đặt có số tham số xấp xỉ nhau (khoảng 0,49-0,52 triệu với 12 chuyển đạo); số tham số của mỗi lượt chạy
được ghi trong log. Muốn đổi kích thước: `--model-kwargs '{"hidden": 128}'` (xem tham số trong `models.py`).
Khi đổi, phải đổi cho **mọi** mô hình cần so sánh để giữ ngân sách công bằng.

## 6. Tái lập kết quả

- Mỗi lượt chạy đặt `--seed` cho `random`, `numpy`, `torch` và bật `cudnn.deterministic`; khuyến nghị seed `0 1 2` cho mọi cấu hình.
- Mọi lượt chạy được ghi một dòng vào `logs/experiments.csv` (thời điểm, cấu hình, seed, số tham số, AUC/F1, thời gian, mã commit git).
  Đây là **nhật ký thí nghiệm** cần nộp; **không sửa tay** file này.
- Cấu hình đầy đủ, lịch sử huấn luyện, dự đoán và checkpoint nằm trong `runs/<tên lượt chạy>/`.
- Kết quả trên GPU có thể lệch rất nhẹ giữa các máy/phiên bản PyTorch dù cùng seed; hãy báo cáo trung bình ± độ lệch chuẩn và nói rõ khác biệt nào nằm trong nhiễu.
- Chạy lại `ablation` sẽ bỏ qua các lượt đã có `test_metrics.json`, nên an toàn khi Colab bị ngắt.

## 7. Quy ước huấn luyện

AdamW (lr 1e-3, weight decay 1e-2) + OneCycleLR, BCE đa nhãn, batch 128, 30 epoch, cắt gradient 1.0, mixed precision trên GPU,
dừng sớm theo val macro-AUC (kiên nhẫn 10 epoch), lấy checkpoint có val AUC tốt nhất để đánh giá trên test.
Chuẩn hóa z-score theo từng chuyển đạo, thống kê chỉ tính trên tập train. Augmentation: nhiễu Gauss, co giãn biên độ ±20%, dịch thời gian ±0,5 giây.

## 8. Phân công và kế hoạch

Xem [`docs/PLAN.md`](docs/PLAN.md). Bản nháp đề xuất theo Phụ lục B nằm ở [`docs/PROPOSAL_DRAFT.md`](docs/PROPOSAL_DRAFT.md).

## 9. Nguồn tham khảo và phần mã tự viết

**Cần cập nhật mục này khi nộp:** ghi rõ dòng/tệp nào do nhóm tự viết, dòng nào kế thừa.

- Dữ liệu: Wagner, P. et al. *PTB-XL, a large publicly available electrocardiography dataset*, Scientific Data 2020; PhysioNet (CC BY 4.0).
- Benchmark để đối chiếu số liệu: Strodthoff, N. et al. *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL*, IEEE JBHI 2021
  (repo `helme/ecg_ptbxl_benchmarking`). **Repo này không sao chép mã của benchmark**; chỉ dùng cùng cách chia fold và cách ánh xạ 5 siêu lớp.
- Thư viện: PyTorch, scikit-learn, wfdb, pandas, scipy, matplotlib.
- Khung mã ban đầu được tạo với sự hỗ trợ của Claude (Anthropic); xem [`docs/AI_DECLARATION.md`](docs/AI_DECLARATION.md).
  Mọi thành viên phải đọc hiểu và giải thích được phần mình phụ trách.
