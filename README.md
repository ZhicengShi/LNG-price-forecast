# 三地区天然气价格实验复现

年度预算版已接入同一 HTTP 服务，接口前缀为 `/annual-budget`，支持年度均价预测、模型查询、异步训练、历史结果查询与 CSV 下载。运行 `python -B -m service.app` 后，在 `/docs` 中查看“年度预算版”分组。请求和数据更新流程见 [年度预算版接口说明](service/年度预算版接口说明.md)。

本目录包含上海、浙江、江苏三地区的LSTM、EMAformer、MoFo和EMMoFormer单区域实验，以及三地区融合EMAformer。每个单区域模型文件夹包含模型定义、YAML配置、训练入口、预测入口及三地区×两种方案的6份配套权重，共24份；融合模型包含全变量、组合变量各1份同时覆盖三地区的权重，合计26份。数据输入仅使用data中的三张处理后XLSX。EMMoFormer仅包含最终版本的代码、配置、权重、预处理参数和预测结果。

## 文件组织

```text
config.yaml                 数据路径、时间划分、变量清单、运行设备和种子
datasets.py                 共用读取、窗口、缺失处理和标准化
metrics.py                  共用分月MAPE、分月得分及平均指标
runtime.py / training.py    共用预测流程及LSTM、EMAformer、MoFo训练流程
data/                       三地区处理后数据表
lstm/                       model.py、train.py、predict.py、config.yaml、weights/
emaformer/                  同上，另有模型层文件
mofo/                       同上，另有模型核心文件
emmoformer/                 模型及层定义、YAML、训练、预测、6份权重和预处理参数
joint_emaformer/             三地区融合模型、训练、预测、搜索、回测与分析
service/                    统一HTTP接口、未来价格区间、评价及版本信息
deployment/                 固定依赖、启动脚本和本机部署验证
reference/                  当前逐条预测、6张指标表、权重索引和核验记录
verify.py                   重载26份权重并生成6张指标表
tests/                      时间隔离、周期位置和指标口径检查
xgboost/                    变量排序、组合筛选、YAML及三地区参考结果
```

每个模型的weights/shanghai、weights/zhejiang、weights/jiangsu中分别保存all.pt和selected.pt。权重包含模型参数、变量顺序、中位数、标准化参数和时间边界。当前24份单区域权重均使用6个月输入；组合变量为上海4列、浙江3列、江苏5列。EMMoFormer另附每份权重对应的all_preprocessing.json或selected_preprocessing.json。来源和训练范围在reference/weight_index.json中，仅供追溯，程序不读取原项目或临时实验目录。

## 安装和快速核验

在Python 3.11环境中，于本目录执行：

```powershell
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
python -B verify.py --device auto
python -B -m unittest discover -s tests -p "test_*.py" -v
```

依赖按实际环境锁定，requirements.txt包含deployment/requirements-common.txt中的接口依赖闭包。原实验为PyTorch 2.14.0+cu130、NVIDIA RTX 5060 Ti。本次使用既有算法环境核验，没有重做全部依赖的全新安装，不含离线wheel包。CPU可以预测；对齐原GPU实验应使用对应CUDA环境。

verify检查数据版本、变量顺序、预处理和1800个主种子测试预测值，其中单区域模型1440个、融合模型360个。在outputs/verification生成三地区×两种方案的6张指标CSV，每张包含四个单区域模型与一个融合模型。reference中的指标表从配套实验逐条预测计算，作为核对基准。

融合模型的训练、预测、参数搜索、逐月重训回测和区间分析，见[joint_emaformer/README.md](joint_emaformer/README.md)。可直接运行python -B joint_emaformer/train.py、predict.py、tune.py、backtest.py、analysis.py对应入口，命令均以前面的joint_emaformer/目录为前缀。两个方案的最终配置在joint_emaformer/configs/all.yaml和selected.yaml。

融合权重训练截至2024-07，单区域权重训练截至2022-11并通过验证集选轮数；汇总表保留各自协议的结果，解释模型差异时需同时说明训练范围。

## 使用最新数据训练与验收

三地区融合模型已支持以下流程，命令均在本目录执行，具体配置和输出见融合模型README：

- [全历史训练并预测未来三个月](joint_emaformer/README.md#全历史训练并预测未来三个月)：将对应方案的training.cutoff设为latest，重新训练，再通过--checkpoint加载本次新权重执行--mode future。
- [留出最近三个月测试](joint_emaformer/README.md#留出最近三个月测试)：例如三表截至2026-05，将training.cutoff设为2026-02、公共period.test_end设为2026-05，运行训练后自动得到3、4、5月的真实价格与预测价格及分月指标。
- [使用最新开发数据重新搜索参数](joint_emaformer/README.md#使用最新开发数据重新搜索参数)：在2026-02及以前划分训练和验证区间，运行tune.py，再用生成的final_config.yaml训练与测试；保留的3、4、5月不用于搜索。

以上适用于全变量all和组合变量selected两种融合方案。四个单区域模型仍按各自既定训练、验证、测试划分训练。建议保留原实验配置用于复现；使用配置副本开展业务训练时，通过--config或--data-config指定，并按副本所在目录检查相对路径。

## 接口部署与试运行

在具备服务依赖的Python环境中执行：

```powershell
python -B -m service.app --device cpu
```

默认地址为http://127.0.0.1:8000，当前服务版本为1.0。GET /models查询全部模型版本及数据兼容性，POST /predictions提交{"profile":"selected"}或{"profile":"all"}，一次返回三地区未来3个月价格、经验区间、历史MAPE与精度得分，以及模型/数据版本和时间信息。支持三表上传、显式选择数据版本、自动调参、按tuning_id训练及模型热切换，无需为这些操作重启服务；已有持久化默认关系不会因新上传或配置表变更自动改变。随包旧权重的训练执行时间未记录，返回null；API新训练权重会记录实际完成时间。全部22项业务操作见[完整接口说明](service/API_DESIGN_README.md)。

接口复用三地区融合模型计算；当前HTTP服务不登记单区域模型。区间使用相同参数和变量方案最近24条已成熟滚动回测残差，名义水平95%，并返回校准样本数和历史覆盖率。默认登记权重训练截至2024-07，数据截至2026-05；未来预测月份始终相对于数据末月。实际使用前需更新数据、选择并登记模型版本。完整字段和调用方式见[接口说明](service/API_DESIGN_README.md)，安装与维护见[部署说明](deployment/README.md)。

```powershell
python -B deployment/validate.py
```

该命令启动实际HTTP服务，完成测试、接口与CLI数值核对、上传、自动调参到训练的小预算流程、热切换、并发调用及重启恢复，并停止自己创建的进程、清理临时工作目录。既有本机技术验证结果（当时程序版本为3.0.0）见[验证记录](reference/service_v3_validation/report.md)，20份权重元数据迁移及参数保留检查见[迁移核验](reference/version_migration_verification.json)，全部权重预测复现见[数值核验](reference/reproduction_v3_verification.json)，命令行小预算运行和恢复检查见[命令行核验](reference/cli_v3_verification.json)；目标机器部署、隔离环境安装和真实业务月精度试运行需在对应环境与月份完成。

## XGBoost变量筛选

在本目录执行，默认分别筛选上海、浙江、江苏：

```powershell
python -B xgboost/run.py
python -B xgboost/run.py --region shanghai
python -B xgboost/run.py --check-reference
```

入口为xgboost/run.py，算法在selection.py，筛选配置在xgboost/config.yaml。数据路径、训练与验证截止月和缺失策略读取公共config.yaml；XGBoost窗口由history_months独立设置（默认12个月），单区域神经网络窗口由公共window.seq_len设置（默认6个月）；candidate_features配置候选字段，树参数、排序折数、置换次数、种子和前向选择范围均可配置。--config指定筛选YAML，--data-config指定公共YAML，--output-dir指定输出根目录。YAML内部路径相对于该YAML，显式命令行路径相对于启动目录。

原实验使用12个月历史预测未来3个月。训练区间2015-01至2022-11内进行3折扩展窗口排序，每折验证15个月；同一变量的12个滞后一起按3个月时间块置换，每折20次，主排序指标为置换后的MAPE增加百分点，同时提供total_gain占比。组合候选包含置换排序和增益排序的全部Top-K，以及在置换排序前8个外生变量中最多5步的前向选择，并比较depth2、depth3两种参数。历史目标始终保留。按2022-12至2024-07独立验证集上三个种子平均预测的MAPE确定组合；2024-08之后的测试数据不参与排序或选择。

每个预测月独立拟合XGBoost，使用原始数值。输入前向填充、起始缺失使用对应训练历史中位数；早期排序折中训练期全空的列保留NaN由XGBoost处理。中位数不借用验证期或测试期。重要性可以为负，相关变量可能互相替代；其含义是当前XGBoost的预测贡献，不是因果效应或神经网络重要性。跨折标准差描述历史折的变化，不是置信区间。最终组合经过验证集比较，因此不一定等于重要性前几名；只保证在本次搜索候选中最优。

运行结果保存在xgboost/outputs/selection_年月日_时分，包含三地区各一份XLSX，每份有“变量重要性”“组合比较”“最终变量”三个工作表。组合比较按第1、2、3个月分别列MAPE和精度得分，并给出平均MAPE与平均精度得分，全部属于XGBoost验证指标；得分=100-MAPE。种子保存在运行元数据，表格不显示种子列。selection.json保留完整排序、逐折结果、全部组合、参数、来源文件信息和运行时间；run.json保存配置快照。

selected_features.yaml以features.selected格式输出变量清单，历史目标放在首列，其余按处理表顺序排列。核对后将对应地区清单写入公共config.yaml的features.selected，再训练配套模型；筛选程序不会自动改写公共配置或已有权重。只运行一个地区时，输出清单仅包含该地区，更新时保留其余地区配置。

xgboost/reference保存与原实验核对后的三地区参考表、变量清单及原筛选记录。--check-reference要求原数据版本，用于核对全部排序、组合及验证MAPE；更新数据或修改参数后运行时可省略该选项。参考中的最终组合分别为上海4列、浙江3列、江苏5列（均含历史价格），与本目录配套神经网络权重的输入一致。模块运行只读取本复现目录及指定的数据，不访问原项目的临时实验目录。

## 训练

```powershell
python -B lstm/train.py
python -B emaformer/train.py
python -B mofo/train.py
python -B emmoformer/train.py
```

每个入口直接读取自己的config.yaml，默认运行三地区的全变量和组合变量共6组实验。也可指定一组：

```powershell
python -B lstm/train.py --region shanghai --profile selected
python -B emaformer/train.py --region jiangsu --profile all --device cuda
python -B emmoformer/train.py --region shanghai --profile selected --device cuda
```

训练保存best.pt、config.json、preprocessing.json、history.csv、training_audit.json、input_snapshot.csv、test_predictions.csv、test_metrics.json及test_metrics.csv。结果写入模型目录下outputs/train/时间目录/地区/变量方案，配套weights不被覆盖。同一分钟重复运行以序号区分。

LSTM、EMAformer和MoFo的模型YAML通过parameters设置结构，training设置regions、profiles、epochs、batch_size、lr、patience和output_dir。EMMoFormer通过experiments.<region>.<profile>.parameters与training分别设置每组实验的模型参数和训练参数，顶层training设置运行地区、变量方案和输出目录。--epochs覆盖本次运行各组实验的最大轮数，可用于短流程检查；--config指定其他模型配置，--data-config指定其他公共配置。直接在IDE运行也会读取默认配置。命令行显式路径按启动目录解释；YAML内部相对路径按对应YAML目录解释。

LSTM、EMAformer和MoFo训练使用Adam、MSE及验证MAPE早停。EMMoFormer使用各地区、各方案对应的优化器、损失函数及学习率计划，具体见下表和自身YAML，训练实现位于emmoformer/_training.py。四个单区域模型均不合并验证集拟合权重；缺失处理和标准化仅拟合训练历史。原单区域脚本的DataLoader及tqdm.auto包装保留，更换依赖或迭代包装可能改变随机序列。新训练记录实际开始/结束时间、耗时、参数和来源文件信息；原权重未补造历史训练时间。

## EMMoFormer模型与最终配置

EMMoFormer是单区域模型。输入先按配置进行窗口归一化，并叠加时间位置和月份相位嵌入；时间编码器在每个变量的历史序列内提取依赖，时间聚合与历史线性投影共同生成变量token。变量token再叠加通道嵌入、相位嵌入及通道与相位联合嵌入，经通道编码器交互后，由目标变量token及可选的目标历史进入预测头。网络输出经过对应窗口统计量还原，生成未来3个月价格。模型定义在emmoformer/model.py，共用层在_layers.py；与三地区融合模型分别训练和保存权重。

|地区|变量方案|隐层维度|注意力头数|时间/通道层数|预测头层数|窗口归一化|优化器/损失|学习率|学习率计划|
|---|---|---|---|---|---|---|---|---|
|上海|全变量|64|4|1/1|1|revin|Adam/MAE|0.0003|无|
|上海|组合变量|64|4|1/1|1|revin，标准差下限0.05|Adam/MAPE|0.001|无|
|浙江|全变量|128|2|2/2|3|revin|Adam/MSE|0.0008260586854801185|无|
|浙江|组合变量|32|4|1/1|1|revin，标准差下限0.05|Adam/MAPE|0.0003|无|
|江苏|全变量|128|8|1/2|2|center|AdamW/MAPE|0.0010673964450514098|余弦|
|江苏|组合变量|32|4|1/2|1|center|AdamW/MAPE|0.00034106524299288|无|

六组配置均为6个月输入、3个月输出，batch_size=16、最多160轮、验证早停耐心20、梯度裁剪1.0、种子42。AdamW权重衰减为0.001，其余为0；完整嵌入、归一化、预测头及训练参数以emmoformer/config.yaml为准。每组独立训练，使用训练集拟合预处理参数，并以验证集MAPE保存最佳轮数权重。当前EMMoFormer配套结果为固定权重逐月测试；第6节的滚动重训回测结果对应三地区融合模型。

## 历史测试与未来预测

```powershell
python -B lstm/predict.py
python -B emaformer/predict.py
python -B mofo/predict.py
python -B emmoformer/predict.py
```

默认prediction.mode为test，加载本模型weights中的6份配套权重，输出逐条预测和分月指标。可在模型YAML设置地区、方案、mode、checkpoint_dir、batch_size和output_dir。结果写入模型目录下outputs/predict/模式_时间目录/地区/变量方案。

```powershell
python -B lstm/predict.py --mode future --region shanghai --profile selected
python -B emaformer/predict.py --mode future
python -B mofo/predict.py --mode future
python -B emmoformer/predict.py --mode future
```

future输出future_predictions.csv及metadata.json，包含地区、方案、模型、月份、步长、预测价格、训练截止月、验证截止月、最新观测月及数据版本。价格单位为元/立方米；未来实际值未知，不计算MAPE或精度得分。当前数据对应预测2026-06至2026-08。

使用新训练的权重时，明确指定实际生成的文件：

```powershell
python -B lstm/predict.py --checkpoint lstm/outputs/train/实际运行目录/shanghai/selected/best.pt --mode test
```

checkpoint指定后，地区和方案从权重读取，无需再指定region/profile。预测始终使用权重中的变量顺序、预处理和模型参数，修改训练YAML不会改写已有权重。mode也支持val，检查独立验证分区。

## 指标表

每个地区和方案分别成表，每行一个模型，列顺序为：

|模型|第1个月MAPE（%）|第1个月精度得分（%）|第2个月MAPE（%）|第2个月精度得分（%）|第3个月MAPE（%）|第3个月精度得分（%）|平均MAPE（%）|平均精度得分（%）|
|---|---|---|---|---|---|---|---|---|

各月得分=100−当月MAPE；平均MAPE为三步等权平均，平均精度得分为三个得分的平均，等于100−平均MAPE。负得分不截断，结果表不展示seed；种子保留在配置及权重中。

原报告使用CUDA，CPU算子有小量浮点差异。verify对CPU的LSTM采用0.0002元/立方米价格绝对容差，其他CPU模型为0.0001元/立方米，CUDA为0.000001元/立方米；实际差异记录在verification.json，不把CPU数值改写成原GPU数值。

当前24份单区域权重均使用6个月历史预测未来3个月，组合变量为原验证选择结果。当前核验见reference/verification.json；既有LSTM、EMAformer和MoFo训练历史保存于training_history.csv和retrained_history.csv，窗口更新记录见reference/window6_update.json。原12个月输入的复现实验审计保留为历史记录，不代表当前单区域权重。CPU训练的dropout随机实现与CUDA不同，不保证重训得到相同权重。

manifest.json记录代码、配套数据、权重及参考结果的文件清单，不进行文件摘要校验。模型和服务运行配置YAML可编辑，清单见manifest的editable_configs；如果改变实验区间或变量组合，应训练配套新权重，原参考指标仅对应原实验。既有LSTM、EMAformer和MoFo模块迁移至含空格的独立目录后，从其他工作目录重载18份权重并进行未来预测的历史核验见reference/relocation_verification.json。融合模块核验记录见joint_emaformer/reference/reproduction_check.json。

## 数据与时间口径

公共config.yaml的data支持绝对或相对路径。三表均为2015-01至2026-05、137个月，一个工作表，第1行为年月及模板24列，第2行为单位。年月使用YYYY-MM文本，进口量为吨，目标为元/立方米。历史目标价格始终输入，目标标签必须为正且不可缺失。

训练2015-01至2022-11共95个月，验证2022-12至2024-07共20个月，测试2024-08至2026-05共22个月。输入6个月、直接输出3个月，标签不跨分区；评价窗口可以读取届时已观测的前期历史。测试有20个起点，各步20组配对、三步60组，属于固定权重逐月测试，不是逐月重训回测。

复现原指标请使用随包处理后数据表及配套权重；历史测试仍检查时间范围、字段与预处理，预测数值由verify.py与参考结果核对。future允许追加完整已观测历史，至少覆盖权重验证选模截止月；程序按权重首月截取，保持前向填充及周期行号一致。请提供从权重首月开始的完整已观测历史，不要只提供最后6行；未来解释变量无需提供。不会在预测时重新拟合预处理。

features.selected列出组合变量；features.all默认按表格顺序选择有效候选，排除停用及训练期全空列。全变量输入数为上海23、浙江20、江苏21，组合变量为4、3、5。浙江停用区域LNG进口量、区域天然气产量；江苏停用工业增加值、GDP、非居民消费量。

LSTM默认32维单层；EMAformer为512维、8头2层、前馈2048、cycle=24；MoFo为本项目既有月度预测版本，24维、4头1层、periodic=6及通道融合。周期分块长度与6个月输入窗口匹配。EMAformer相位为预测首月全局行号模周期，MoFo为最后历史月行号模周期。

## 当前组合变量与输入窗口

四个单区域模型及融合模型均使用最近6个月预测未来3个月。selected采用XGBoost验证比较选出的原组合：上海4列、浙江3列、江苏5列，完整字段及顺序见config.yaml。各单区域模型均配套全变量、组合变量的6个月输入权重。融合模型使用原组合变量，配套固定测试、滚动回测及区间校准证据。最新六张指标表与年度结果见[实验结果](reference/实验结果.md)。
