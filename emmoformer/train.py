"""EMMoFormer 单区域训练入口，默认读取本目录 config.yaml。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emmoformer._training import train_main


if __name__ == '__main__':
    train_main()
