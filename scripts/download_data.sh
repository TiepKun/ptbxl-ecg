#!/usr/bin/env bash
# Tải PTB-XL v1.0.3 từ PhysioNet: 2 file CSV + thư mục records100 (bản 100 Hz).
# Dùng:  bash scripts/download_data.sh [thư_mục_đích]      (mặc định: data/raw)
# Lưu ý: ~43.000 file nhỏ nên mất một lúc; wget -c cho phép chạy lại để tiếp tục.
set -euo pipefail
DEST="${1:-data/raw}"
BASE="https://physionet.org/files/ptb-xl/1.0.3"
mkdir -p "$DEST"
wget -N -c -P "$DEST" "$BASE/ptbxl_database.csv" "$BASE/scp_statements.csv"
wget -r -N -c -np -nH --cut-dirs=3 -R "index.html*" -P "$DEST" "$BASE/records100/"
echo "Xong. Kiểm tra: $(find "$DEST/records100" -name '*.hea' | wc -l) file .hea (kỳ vọng 21799)."
