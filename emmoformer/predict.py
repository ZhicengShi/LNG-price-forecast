"""EMMoFormer 单区域预测入口，支持测试、验证和未来三个月预测。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import predict_main


if __name__ == '__main__':
    predict_main('EMMoFormer')
