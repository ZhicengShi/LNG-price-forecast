"""年度预算版的模型构建、数据处理、训练与预测。"""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
from collections.abc import Mapping

import numpy as np
import openpyxl
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
VERSION = "年度预算版"
TARGET = "终端非居民天然气价格（分省）"
REGIONS = {"shanghai": "上海", "zhejiang": "浙江", "jiangsu": "江苏"}
MODELS = ("lstm", "emaformer", "mofo", "emmoformer", "joint_emaformer")
MODEL_CLASSES = dict(zip(MODELS[:4], ("LSTM", "EMAformer", "MoFo", "EMMoFormer")))


class SingleBudgetModel(nn.Module):
    def __init__(self, name, network, target_index):
        super().__init__()
        self.name, self.network, self.target_index = name, network, target_index

    def forward(self, x, positions):
        first_unseen = positions + self.network.seq_len
        if self.name == "lstm":
            return self.network(x)
        if self.name == "emaformer":
            result = self.network(x, first_unseen % self.network.cycle_len)
            return result[0] if isinstance(result, tuple) else result
        if self.name == "mofo":
            return self.network(x, (first_unseen - 1) % self.network.periodic)
        return self.network(x, first_unseen)


class BudgetJointEMAformer(nn.Module):
    def __init__(self, original, indices, width):
        super().__init__()
        self.original, self.width = original, width
        self.register_buffer("channel_order", torch.tensor(indices, dtype=torch.long))

    def forward(self, x, positions):
        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError("联合模型输入通道数与保存配置不一致")
        packed = x.index_select(-1, self.channel_order)
        return self.original(packed, positions + self.original.seq_len)


def packed_indices(layout, columns):
    packed, offset = {}, 0
    for region, names in zip(layout["regions"], columns):
        required = layout["input_schema"]["common"] + layout["input_schema"][region]
        if len(names) != len(required) or set(names) != set(required):
            raise ValueError(f"{region} 的变量与联合模型布局不一致")
        for i, name in enumerate(names):
            key = name if name in layout["input_schema"]["common"] else f"{region}::{name}"
            packed.setdefault(key, offset + i)
        offset += len(names)
    return [packed[key] for key in layout["features"]]


def build_model(spec):
    name = spec["model"]
    parameters = dict(spec["parameters"])
    if name not in MODELS or spec["seq_len"] != 6:
        raise ValueError("年度预算版模型或历史窗口配置无效")
    namespace = f"{__package__}." if __package__ else ""
    module = importlib.import_module(f"{namespace}models.{name}.model")
    if name == "joint_emaformer":
        layout = spec["layout"]
        if layout["pred_len"] != 1 or parameters["seq_len"] != 6:
            raise ValueError("年度预算版须使用六个月输入并输出一个年度均价")
        model = BudgetJointEMAformer(module.JointEMAformer(layout, parameters),
                                     packed_indices(layout, spec["columns"]),
                                     sum(map(len, spec["columns"])))
    else:
        if parameters["pred_len"] != 1 or parameters["seq_len"] != 6:
            raise ValueError("年度预算版须使用六个月输入并输出一个年度均价")
        target_index = spec["columns"][0].index(TARGET)
        model = SingleBudgetModel(name, getattr(module, MODEL_CLASSES[name])(**parameters), target_index)
    increment = float(spec.get("dropout_increment", 0))
    if increment:
        for layer in model.modules():
            if isinstance(layer, nn.Dropout) and layer.p > 0:
                layer.p = min(0.35, layer.p + increment)
    return model


def read_frame(path):
    path = Path(path)
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if len(book.worksheets) != 1:
            raise ValueError(f"{path.name} 须仅包含一个工作表")
        rows = list(book.worksheets[0].values)
    finally:
        book.close()
    while rows and all(value is None for value in rows[-1]):
        rows.pop()
    if len(rows) < 3:
        raise ValueError(f"{path.name} 缺少字段、单位或月度数据")
    columns = list(rows[0])
    if len(columns) != len(set(columns)) or any(not isinstance(c, str) or not c.strip() for c in columns):
        raise ValueError(f"{path.name} 含重复或空字段名")
    if "年月" not in columns or TARGET not in columns:
        raise ValueError(f"{path.name} 缺少年月或目标价格字段")
    if rows[1][columns.index("年月")] != "YYYY-MM" or rows[1][columns.index(TARGET)] != "元/立方米":
        raise ValueError(f"{path.name} 的年月或价格单位不符合处理后表格式")
    for column in ("LNG进口量（全国）", "管道气进口量（全国）", "LNG进口量（区域）"):
        if column in columns and rows[1][columns.index(column)] != "吨":
            raise ValueError(f"{path.name} 的{column}单位须为吨")
    frame = pd.DataFrame(rows[2:], columns=columns)
    months = frame.pop("年月")
    if not all(isinstance(m, str) and re.fullmatch(r"\d{4}-\d{2}", m) for m in months):
        raise ValueError(f"{path.name} 的年月须为 YYYY-MM 文本")
    frame.index = pd.PeriodIndex(months, freq="M")
    expected = pd.period_range(frame.index[0], frame.index[-1], freq="M")
    if not frame.index.equals(expected):
        raise ValueError(f"{path.name} 的月份须连续升序，且不得重复")
    frame = frame.apply(pd.to_numeric, errors="raise")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError(f"{path.name} 含无穷值")
    return frame


def read_history(spec, data_dir, budget_year, first_month=None):
    cutoff = pd.Period(f"{budget_year - 1}-08", freq="M")
    frames = []
    for region, columns in zip(spec["regions"], spec["columns"]):
        path = data_dir[region] if isinstance(data_dir, Mapping) else Path(data_dir) / f"{REGIONS[region]}_处理后数据表.xlsx"
        frame = read_frame(path)
        if cutoff not in frame.index:
            raise ValueError(f"{REGIONS[region]}数据须覆盖 {cutoff}，当前截止 {frame.index[-1]}")
        absent = set(columns) - set(frame.columns)
        if absent:
            raise ValueError(f"{REGIONS[region]}缺少固定变量 {sorted(absent)}")
        frame = frame.loc[:cutoff]
        if first_month is not None and str(frame.index[0]) != first_month:
            raise ValueError(f"{REGIONS[region]}数据起点须为权重记录的 {first_month}")
        target = frame[TARGET].to_numpy(dtype=float)
        if not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError(f"{REGIONS[region]}截至 {cutoff} 的价格须为有限正数")
        frames.append(frame)
    if any(not f.index.equals(frames[0].index) for f in frames):
        raise ValueError("联合模型的三个地区须具有相同历史月份")
    raw = pd.concat([f[c].add_prefix(r + "::") for f, c, r in
                     zip(frames, spec["columns"], spec["regions"])], axis=1)
    targets = np.column_stack([f[TARGET].to_numpy(dtype=float) for f in frames])
    return raw, targets


def fill_inputs(raw, medians, missing):
    if missing == "linear_history":
        raw = raw.interpolate(method="linear", limit_area="inside")
    elif missing != "ffill":
        raise ValueError(f"不支持的缺失处理配置 {missing}")
    return raw.ffill().fillna(pd.Series(medians, index=raw.columns)).to_numpy(dtype=np.float64)


def prepare_training(raw, targets, spec, device):
    seq_len = spec["seq_len"]
    origins = list(range(seq_len - 1, len(raw) - 16))
    if not origins:
        raise ValueError("历史不足以形成六个月输入及完整年度标签")
    medians = raw.median().fillna(0).to_numpy()
    full = fill_inputs(raw, medians, spec["missing"])
    xmean, xscale = full.mean(0), full.std(0)
    ymean, yscale = targets.mean(0), targets.std(0)
    xscale = np.where(xscale < 1e-8, 1.0, xscale)
    yscale = np.where(yscale < 1e-8, 1.0, yscale)
    windows = np.array([fill_inputs(raw.iloc[:origin + 1], medians, spec["missing"])[-seq_len:]
                        for origin in origins])
    labels = np.stack([targets[origin + 5:origin + 17].mean(0) for origin in origins])[:, None, :]
    x = torch.as_tensor((windows - xmean) / xscale, dtype=torch.float32, device=device)
    y = torch.as_tensor((labels - ymean) / yscale, dtype=torch.float32, device=device)
    positions = torch.tensor(np.asarray(origins) - seq_len + 1, dtype=torch.long, device=device)
    scaler = dict(xmean=xmean.tolist(), xscale=xscale.tolist(), ymean=ymean.tolist(),
                  yscale=yscale.tolist(), medians=medians.tolist())
    return x, positions, y, origins, scaler


def resolve_device(choice):
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境无可用 CUDA，请使用 --device cpu")
    return choice


def checkpoint_path(weights_dir, budget_year, spec):
    group = "joint" if spec["model"] == "joint_emaformer" else spec["regions"][0]
    return Path(weights_dir) / str(budget_year) / spec["model"] / spec["profile"] / f"{group}.pt"


def load_checkpoint(path, budget_year):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cutoff = f"{budget_year - 1}-08"
    if (checkpoint.get("version") != VERSION or checkpoint.get("budget_year") != budget_year
            or checkpoint.get("cutoff_month") != cutoff):
        raise ValueError(f"权重 {path} 的版本、预算年度或数据截止月不匹配")
    return checkpoint


@torch.no_grad()
def predict_checkpoint(checkpoint, data_dir, device="cpu"):
    spec, scaler = checkpoint["spec"], checkpoint["scaler"]
    raw, _ = read_history(spec, data_dir, checkpoint["budget_year"], checkpoint["first_month"])
    seq_len = spec["seq_len"]
    values = fill_inputs(raw, scaler["medians"], spec["missing"])[-seq_len:]
    if len(values) != seq_len:
        raise ValueError("输入历史少于六个月")
    values = (values - np.asarray(scaler["xmean"])) / np.asarray(scaler["xscale"])
    x = torch.as_tensor(np.ascontiguousarray(values[None]), dtype=torch.float32, device=device)
    positions = torch.tensor([len(raw) - seq_len], dtype=torch.long, device=device)
    model = build_model(spec).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    output = model(x, positions).cpu().numpy().astype(float)
    if output.shape != (1, 1, len(spec["regions"])):
        raise ValueError(f"年度均价输出形状无效 {output.shape}")
    prices = output[0, 0] * np.asarray(scaler["yscale"]) + np.asarray(scaler["ymean"])
    if not np.isfinite(prices).all():
        raise ValueError("预测出现非有限值")
    return prices


def train_checkpoint(config, budget_year, data_dir, device="cpu", progress=None):
    spec, training = config["spec"], config["training"]
    raw, targets = read_history(spec, data_dir, budget_year)
    x, positions, labels, origins, scaler = prepare_training(raw, targets, spec, device)
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(spec).to(device)
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    optimizer = optimizers[training["optimizer"].lower()](model.parameters(), lr=training["lr"],
                                                        weight_decay=training.get("weight_decay", 0))
    epochs, batch_size = int(training["epochs"]), int(training["batch_size"])
    if epochs <= 0 or batch_size <= 0 or training["loss"].lower() != "mse":
        raise ValueError("训练轮数、批量或年度均价损失配置无效")
    scheduler_name = training.get("scheduler", "none")
    if scheduler_name not in ("none", "cosine"):
        raise ValueError(f"不支持的学习率调度 {scheduler_name}")
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if scheduler_name == "cosine" else None)
    model.train()
    for epoch in range(epochs):
        if progress is not None:
            progress(epoch, epochs)
        for indices in torch.randperm(len(origins), device=device).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            predicted = model(x[indices], positions[indices])
            if predicted.shape != labels[indices].shape:
                raise ValueError("训练输出与年度均价标签形状不一致")
            loss = (predicted - labels[indices]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("训练损失出现非有限值")
            loss.backward()
            if training.get("grad_clip", 0):
                nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    if progress is not None:
        progress(epochs, epochs)
    return dict(version=VERSION, spec=spec, training=training, scaler=scaler,
                first_month=str(raw.index[0]), cutoff_month=str(raw.index[-1]), budget_year=budget_year,
                train_origins=origins, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()})


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def forecast_rows(checkpoint, prices):
    spec, budget_year = checkpoint["spec"], checkpoint["budget_year"]
    return [dict(version=VERSION, budget_year=budget_year, cutoff_month=checkpoint["cutoff_month"],
                 target_start=f"{budget_year}-01", target_end=f"{budget_year}-12", region=region,
                 region_name=REGIONS[region], model=spec["model"], profile=spec["profile"],
                 predicted=float(price), unit="元/立方米", n_train=len(checkpoint["train_origins"]))
            for region, price in zip(spec["regions"], prices)]


def write_predictions(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_cli(training=False):
    parser = argparse.ArgumentParser(description=VERSION + ("训练并预测" if training else "预测"))
    parser.add_argument("--budget-year", type=int, default=2026)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--model", choices=("all", *MODELS), default="all")
    parser.add_argument("--profile", choices=("all", "selected", "both"), default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        device = resolve_device(args.device)
        torch.set_num_threads(2)
        if device == "cuda":
            torch.set_float32_matmul_precision("high")
        configs = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8-sig"))
        configs = [c for c in configs if (args.model == "all" or c["spec"]["model"] == args.model)
                   and (args.profile == "both" or c["spec"]["profile"] == args.profile)]
        if not configs:
            raise ValueError("所选模型及变量方案无配置")
        paths = [checkpoint_path(args.weights_dir, args.budget_year, c["spec"]) for c in configs]
        if not training:
            absent = [str(p) for p in paths if not p.is_file()]
            if absent:
                raise FileNotFoundError(f"缺少 {args.budget_year} 年度权重，请使用 train.py 训练。首个缺失文件 {absent[0]}")
        rows = []
        for config, path in zip(configs, paths):
            if training:
                checkpoint = train_checkpoint(config, args.budget_year, args.data_dir, device)
                prices = predict_checkpoint(checkpoint, args.data_dir, device)
                save_checkpoint(path, checkpoint)
            else:
                checkpoint = load_checkpoint(path, args.budget_year)
                if checkpoint["spec"] != config["spec"]:
                    raise ValueError(f"{path} 的模型配置与 configs/models.json 不一致")
                prices = predict_checkpoint(checkpoint, args.data_dir, device)
            rows.extend(forecast_rows(checkpoint, prices))
            print(f"{VERSION} {config['spec']['model']} {config['spec']['profile']} "
                  f"{','.join(config['spec']['regions'])} 完成", flush=True)
        output = args.output or ROOT / "outputs" / f"年度预算预测_{args.budget_year}.csv"
        write_predictions(output, rows)
        print(f"已保存 {len(rows)} 条年度预测至 {output.resolve()}")
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
