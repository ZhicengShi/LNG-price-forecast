# 三地区融合EMAformer

本目录保存三地区共享编码器、区域独立嵌入和预测头的EMAformer。单份权重同时输出上海、浙江、江苏未来3个月价格。全变量与组合变量分别有配置、预处理参数和权重；主结果种子为42，公开预测与指标CSV不展示seed列。

## 数据与模型

数据路径和变量清单读取上一级config.yaml，只使用三地区处理后XLSX。data.py复用datasets.py读表，负责三地区拼接和对应预处理。全变量输入中的共同指标只保存一份，其余字段按地区分组；组合输入为上海4列、浙江3列、江苏5列，均包含历史目标价格。组合与XGBoost筛选结果一致。

模型以6个月历史预测未来3个月，d_model=256，2层4头共享编码器，前馈维度512，周期24；区域预测头各3层。窗口归一化开启，通道嵌入关闭。输出在标准化空间为：

```text
z_hat[r,h] = z[r,last] + 0.25 * f[r,h](history) * window_std[r]
price_hat[r,h] = z_hat[r,h] * training_y_scale[r] + training_y_mean[r]
```

历史目标通道在输入中的位置由保存的布局确定。末层以零权重初始化，训练学习价格修正量。原始模型中的其他随机初始化顺序保留，支持核对参考训练结果。

输入采用linear_history：每个预测起点只对截至当时的历史做内部线性插值，末端前向填充，开头缺失使用对应训练历史中位数。标准化仅拟合该次训练历史，目标标签不插补。公共配置中的单区域ffill策略用于单区域模型；融合模型的missing由自己的配置指定。

## 固定训练与测试

从experiment_reproduction目录运行：

```powershell
python -B joint_emaformer/train.py
python -B joint_emaformer/train.py --profile selected
python -B joint_emaformer/train.py --config joint_emaformer/configs/all.yaml
```

默认分别运行全变量和组合变量。训练采用AdamW、MAPE损失，24轮、batch=64、梯度裁剪0.5，学习率0.00013681810651684312。参数在configs/all.yaml和selected.yaml内；training.lr_schedule=null时，根据parameters.lr和training.epochs生成固定学习率。显式填写列表时按列表重放，用于复现搜索确定的变动学习率轨迹。--epochs可覆盖轮数；--seed、--device、--data-config、--output-dir可覆盖运行设置。

配套权重训练截至2024-07（115个月），参数选择也截至该月；测试为2024-08至2026-05，每地区20个起点、每个预测步20组、三步合计60组。训练输出保存best.pt、配置、预处理参数、训练日志、训练时间、来源文件信息、逐条测试预测和分月指标。结果位于outputs/train，运行不覆盖weights中的配套权重。

本目录融合模型与单区域模型的训练截止月份不同：单区域权重拟合截至2022-11并使用验证集选轮数，融合权重拟合截至2024-07。汇总表用于核对各自实验结果，解释模型差异时需同时说明训练协议。

training.cutoff可以填写YYYY-MM或latest。latest使用源表全部已观测历史；当训练后没有完整测试标签时，训练结果说明无可用测试范围。窗口长度使用本目录parameters.seq_len，公共window.seq_len仍控制单区域模型。

## 预测

```powershell
python -B joint_emaformer/predict.py
python -B joint_emaformer/predict.py --mode future --device cpu
python -B joint_emaformer/predict.py --profile selected --mode future
python -B joint_emaformer/predict.py --checkpoint joint_emaformer/outputs/train/实际运行目录/selected/best.pt --mode future
```

默认加载weights/all.pt和selected.pt复现测试。指定checkpoint后，方案、模型结构、变量顺序和预处理均由权重读取；可用--data-config指定更新的数据配置。复现原指标请使用配套处理后表，程序检查时间范围与变量布局，verify.py核对预测与参考数值。future允许更新或追加完整已观测历史，使用保存的预处理，不重新拟合；最近观测月不能早于训练或参数选择截止月。

当前三表截至2026-05，future输出2026-06至2026-08，每方案9行，包含地区、月份、预测步、价格、model_training_cutoff和latest_observed_month。未来实际值未知，future输出价格与元数据。模型更新通过train.py重新训练。

HTTP服务复用本目录的计算函数，另行组合未来价格区间、历史评价和版本状态。服务启动、上传及调用说明见[统一接口README](../service/README.md)，本目录CLI原有价格CSV格式保持原样。

## 全历史训练并预测未来三个月

以下业务流程适用于本目录融合模型的all（全变量）和selected（组合变量）两种方案，命令均从experiment_reproduction目录执行。建议保留原实验配置，使用副本开展新实验；--config指定模型或搜索配置，--data-config指定公共配置。YAML中的相对路径以该YAML所在目录为准，复制配置到其他目录时应同步检查data_config、数据路径和输出路径。

以组合变量为例，在joint_emaformer/configs/selected.yaml中仅将training.cutoff改为latest，其他训练字段保留：

```yaml
training:
  cutoff: latest
```

在公共config.yaml中确认data指向更新后的三地区处理表，period.start为希望使用的历史首月。latest使用从该首月至三表一致的最新观测月的全部历史；三地区月份须连续且一致，目标价格须为正且不可缺失。训练会重新拟合这一历史范围的预处理参数和模型权重，变量组合与结构参数沿用当前配置，不会自动执行特征筛选或参数搜索。

```powershell
python -B joint_emaformer/train.py --profile selected
```

训练结束后，根据TRAIN_COMPLETE打印的实际输出目录，加载本次新生成的best.pt预测：

```powershell
python -B joint_emaformer/predict.py --checkpoint "joint_emaformer/outputs/train/实际运行目录/selected/best.pt" --mode future
```

指定--checkpoint后，方案由权重确定，无需再传--profile或--config；使用公共配置副本或新数据配置时，可在预测命令中增加--data-config。预测默认使用的weights/selected.pt仍是配套历史权重，因此使用新训练结果时应明确填写新best.pt路径。

当前三表为2015-01至2026-05，共137个月。使用全部历史训练后，预测输出2026-06、2026-07、2026-08，三地区合计9行，保存为outputs/predict/实际运行目录/selected/future_predictions.csv，并生成metadata.json。全变量方案对应修改configs/all.yaml、训练时使用--profile all，再加载新输出的all/best.pt。

全历史用于构造训练窗口；最终预测时，当前配置以最近6个月观测为输入，一次输出未来3个月。未来实际价格未知，不计算此次未来预测的MAPE或精度得分，可另行展示已有回测指标。全部数据用于训练后，训练目录会以evaluation.json说明无可用的完整3个月测试标签。

## 留出最近三个月测试

例如三表均更新至2026-05，要求使用截至2026-02的全部历史训练，留出2026年3、4、5月比较真实价格和预测价格。三张数据表仍保留到2026-05，数据路径在公共config.yaml的data中设置，无需手动删除测试月份。

在对应模型配置configs/selected.yaml中，仅将训练截止月设置为：

```yaml
training:
  cutoff: "2026-02"
```

公共config.yaml中的period.test_end设置为2026-05。沿用现有参数时，其余边界可以保留现有值：

```yaml
period:
  start: "2015-01"
  train_end: "2022-11"
  validation_end: "2024-07"
  test_end: "2026-05"
```

融合模型本次最终拟合权重的截止月由模型配置中的training.cutoff决定。公共period.train_end仍用于全变量有效列判断、参数搜索初筛等，period.validation_end记录选参截止月并参与测试起始位置判断；它们不能替代training.cutoff。此场景下validation_end应不晚于2026-02。

```powershell
python -B joint_emaformer/train.py --profile selected
```

程序仅使用2015-01至2026-02拟合中位数、标准化和模型权重，自动以2026-02为历史截止月，一次预测2026-03、2026-04、2026-05，并与保留的真实价格比较。当前seq_len=6时，这次预测的输入为2025-09至2026-02；3至5月的解释变量和价格不进入这次预测输入。无需额外使用--mode future完成这项测试。

本次训练目录outputs/train/实际运行目录/selected包含：

| 文件 | 内容 |
| --- | --- |
| best.pt | 截至2026-02训练的模型权重及配套预处理参数 |
| test_predictions.csv | 三地区3、4、5月的真实价格、预测价格、预测步及历史截止月，共9行 |
| test_metrics.csv / test_metrics.json | 各地区第1、2、3个月的MAPE与精度得分，以及平均MAPE和平均精度得分 |
| config.json / preprocessing.json | 本次配置及拟合的预处理参数 |
| history.csv / training_audit.json | 训练损失、开始和结束时间、数据版本及训练截止月份等记录 |

全变量方案同样将configs/all.yaml中的training.cutoff设为2026-02，再运行python -B joint_emaformer/train.py --profile all。两份配置都已调整时，也可不传--profile，一次训练两种方案。

本例每地区只有1个预测起点，各预测步各1组预测与真实值，三步合计3组。每一步的MAPE就是该月的绝对百分比误差，平均MAPE为三个预测月等权平均，精度得分为100-MAPE（MAPE以百分数表示）。这与使用多个起点的滚动回测评分范围不同，应分别报告。数据继续更新时，将training.cutoff设为最后一个数据月之前3个月，period.test_end设为最后一个数据月，同时确保选参截止月不晚于训练截止月。

## 使用最新开发数据重新搜索参数

需要为新历史重新选参时，先在保留测试月份之前划分训练与验证。例如，在公共config.yaml中设置：

```yaml
period:
  start: "2015-01"
  train_end: "2024-06"
  validation_end: "2026-02"
  test_end: "2026-05"
```

| 阶段 | 数据范围 | 用途 |
| --- | --- | --- |
| 参数搜索初筛训练 | 2015-01至2024-06 | 拟合候选模型 |
| 参数搜索初筛验证 | 2024-07至2026-02 | 比较参数和选择训练轮数，共20个月 |
| 多折复核 | 2026-02及以前的开发历史 | 在多个时间区间和种子下复核前列候选 |
| 选参后最终训练 | 2015-01至2026-02 | 按选定参数重新拟合最终权重和预处理 |
| 最终测试 | 2026-03至2026-05 | 与留出月份真实价格比较 |

configs/search.yaml控制搜索数量和范围。目前默认每方案64个候选，前4名进入3折复核，每折验证20个月，每折使用42、7、2026三个种子；这些设置可沿用，也可以按计算预算调整。search_space可配置历史窗口seq_len、编码维度d_model、注意力头数n_heads、编码层数e_layers、前馈维度倍数、预测头层数、Dropout、学习率、批大小、优化器、损失函数、最大轮数和早停耐心等，initial_candidates设置优先评估的候选。需保留足够历史以构造全部训练窗口。

```powershell
# 分别搜索全变量和组合变量
python -B joint_emaformer/tune.py --profile all selected

# 或仅搜索组合变量
python -B joint_emaformer/tune.py --profile selected
```

搜索读取数据至period.validation_end，即本例的2026-02；3、4、5月不参与候选模型拟合、验证选轮数或参数排名。模型按三地区等权平均验证MAPE进行比较，选出的是本次搜索候选中的最佳方案。tune.py沿用公共配置中的变量清单，不自动重新运行XGBoost筛选。

每个方案的搜索输出目录均包含final_config.yaml，保存选出的模型结构、训练轮数和学习率轨迹，training.cutoff自动设为period.validation_end，即2026-02。随后明确使用生成的配置训练：

```powershell
python -B joint_emaformer/train.py --config "joint_emaformer/outputs/search/实际运行目录/selected/final_config.yaml"
python -B joint_emaformer/train.py --config "joint_emaformer/outputs/search/实际运行目录/all/final_config.yaml"
```

实际运行目录应替换为SEARCH_COMPLETE打印的路径。此时训练结果位于各final_config.yaml旁的training_outputs/实际运行目录/变量方案中；训练结束自动生成本例3、4、5月的test_predictions.csv及分月指标。直接运行默认train.py仍读取随包配置，要应用搜索结果应通过--config指定新final_config.yaml。更新数据或改变搜索设置后应开启新搜索；--run-dir用于恢复配置、数据版本和代码均一致的同一次搜索。

若最近三个月测试效果不理想，应回到2026-02及以前的验证历史比较候选参数，并检查各地区、各预测步的误差。根据3至5月成绩反复修改参数后，该段已参与模型选择，不能再称为独立测试；独立验收应另留后续未参与选择的月份。完成评估并确定参数后，可按前述全历史流程将training.cutoff改为latest，用截至2026-05的全部历史重新训练，再预测2026-06至2026-08。

## 参数搜索

```powershell
python -B joint_emaformer/tune.py
python -B joint_emaformer/tune.py --profile selected
python -B joint_emaformer/tune.py --run-dir joint_emaformer/outputs/search/实际运行目录
```

configs/search.yaml设置初始候选、搜索空间、搜索种子和复核范围。每方案64个不同候选，先按训练截止2022-11、验证截止2024-07的三地区等权平均MAPE筛选，再对前4名做3折、3种子复核。复核折均位于开发历史内，每折验证20个月，种子42、7、2026。目标为地区、历史折和种子等权平均MAPE，测试标签不参与选择。

最终轮数取最后一折三个最佳轮数的中位数，并冻结其学习率轨迹。最终配置写入搜索输出下各方案的final_config.yaml，可以传给train.py训练；预测新训练权重时指定实际best.pt。搜索结果包括完整候选、逐轮验证结果及复核记录。全变量和组合变量分别搜索，本次最终结构参数相同，输入布局和权重分别保存。

run-dir支持恢复；恢复时核对配置、协议版本、设备与来源文件信息。不同协议版本应建立新运行目录，旧完成结果仍可查看。重新运行不会覆盖随包提供的最终配置或权重。

## 滚动回测

```powershell
python -B joint_emaformer/backtest.py
python -B joint_emaformer/backtest.py --profile selected
python -B joint_emaformer/backtest.py --start 2024-08 --end 2026-03
python -B joint_emaformer/backtest.py --run-dir joint_emaformer/outputs/backtest/实际运行目录
```

configs/backtest.yaml指定两方案配置、种子和起点范围。默认主种子42，每方案126个连续起点，合计252次从头训练。初始历史9个月，历史截止月2015-09至2026-02，预测首月2015-10至2026-03，最后目标月2026-05。每地区、每方案378组预测与真实值。training_seeds改为[42,7,2026]可完整复核三种子，共756次训练。

每个起点使用截至当时的数据重新拟合中位数、标准化和网络权重，固定24轮与学习率，不按回测误差重新选参数。数据、配置和代码版本保存在protocol.json；每起点写审计与预测，结束时保存末起点last.pt。恢复时只认9行预测和审计均完整的起点，缺少末起点权重时重跑该点。

变量与参数由截至2024-07的开发数据确定。更早起点属于固定事后组合与参数的回顾性评价；填充、标准化与网络训练仍只用该起点可见历史。2024-08起的20个起点另行汇总。该历史测试范围此前已查看，不称为全新盲测。

## 分析与核验

```powershell
python -B joint_emaformer/analysis.py
python -B joint_emaformer/analysis.py --profile selected
python -B joint_emaformer/analysis.py --run-dir joint_emaformer/outputs/backtest/实际运行目录
python -B joint_emaformer/verify.py --device cpu
```

analysis默认读取reference/backtest的主种子记录；新回测传入实际run-dir。输出分月指标、分年指标、选参截止后指标、MAPE/得分/MAE/RMSE的95%时间块Bootstrap区间、块长度敏感性、两方案成对差异及历史残差预测区间。年度按预测首月的年份划分。表格对每个月分别报告MAPE和100-MAPE的精度得分，并给出三步等权平均。

默认连续12个起点为一个时间块，重抽样5000次，保留三地区和三个预测步的相关性；块长敏感性为6、12、18个月。重抽样不重新训练模型，得到的是指标估计的单项置信区间。预测价格波动区间单独使用已成熟的过去绝对残差构建，至少24个、最近最多60个，并报告实际覆盖率。它与指标置信区间含义不同。

reference保存两种方案、三个种子的固定测试与回测明细，以及搜索记录。weights只提供主种子42的两份固定训练权重。verify重载两份权重核对360个主种子测试预测，CPU容差0.0001元/立方米，CUDA容差0.000001元/立方米；源参数张量保持不变。CPU与CUDA随机数和浮点实现不同，训练复现的一致性核对以当前CUDA环境为准。

所有运行只使用本复现目录、配置指定的数据和已安装的依赖。reference中的原始来源路径仅作追溯，运行不读取原项目的实验目录。
