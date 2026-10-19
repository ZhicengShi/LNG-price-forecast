"""直接运行本文件，按YAML为三地区排序和筛选变量。"""
from pathlib import Path
import sys

# 共用datasets.py；本目录不设__init__.py，避免覆盖已安装的xgboost包。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selection import main


if __name__ == '__main__':
    main()
