"""Đọc PTB-XL thô (records100 + 2 file CSV) và cache thành .npy.

    python -m ptbxl_ecg.prepare_data --raw-dir data/raw --out-dir data/processed
"""
import argparse

from .data import prepare


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    a = p.parse_args()
    prepare(a.raw_dir, a.out_dir)


if __name__ == "__main__":
    main()
