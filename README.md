# 三地区天然气价格实验复现

本目录包含上海、浙江、江苏三地区的LSTM、EMAformer和MoFo单区域实验，以及三地区融合EMAformer。每个单区域模型文件夹包含模型定义、YAML配置、训练入口、预测入口及三地区×两种方案的6份配套权重，共18份；融合模型包含全变量、组合变量各1份同时覆盖三地区的权重，合计20份。数据输入仅使用data中的三张处理后XLSX。

## 文件组织

```text
config.yaml                 数据路径、时间划分、变量清单、运行设备和种子
datasets.py                 共用读取、窗口、缺失处理和标准化
metrics.py                  共用分月MAPE、分月得分及平均指标
runtime.py / training.py    三个独立入口共用的预测和训练流程
data/                       三地区处理后数据表
lstm/                       model.py、train.py、predict.py、config.yaml、weights/
emaformer/                  同上，另有模型层文件
mofo/                       同上，另有模型核心文件
joint_emaformer/             三地区融合模型、训练、预测、搜索、回测与分析
service/                    统一HTTP接口、未来价格区间、评价及版本信息
deployment/                 固定依赖、启动脚本和本机部署验证
reference/                  原实验逐条预测、6张指标表、权重索引和核验记录
verify.py                   重载20份权重并生成6张指标表
tests/                      时间隔离、周期位置和指标口径检查
xgboost/                    变量排序、组合筛选、YAML及三地区参考结果
```

每个模型的weights/shanghai、weights/zhejiang、weights/jiangsu中分别保存all.pt和selected.pt。权重包含模型参数、变量顺序、中位数、标准化参数和时间边界。原参数张量保留，元数据整理成统一格式；来源和哈希在reference/weight_index.json中，仅供追溯，程序不读取原项目或临时实验目录。

## 安装和快速核验

在Python 3.11环境中，于本目录执行：

```powershell
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
python -B verify.py --device auto
python -B -m unittest discover -s tests -p "test_*.py" -v
```

依赖按实际环境锁定，requirements.txt包含deployment/requirements-common.txt中的接口依赖闭包。原实验为PyTorch 2.14.0+cu130、NVIDIA RTX 5060 Ti。本次使用既有算法环境核验，没有重做全部依赖的全新安装，不含离线wheel包。CPU可以预测；对齐原GPU实验应使用对应CUDA环境。

verify检查数据版本、变量顺序、预处理和1440个主种子测试预测值，在outputs/verification生成三地区×两种方案的6张指标CSV，每张包含三个单区域模型与一个融合模型。reference中的指标表从配套实验逐条预测计算，作为核对基准。

融合模型的训练、预测、参数搜索、逐月重训回测和区间分析，见[joint_emaformer/README.md](joint_emaformer/README.md)。可直接运行python -B joint_emaformer/train.py、predict.py、tune.py、backtest.py、analysis.py对应入口，命令均以前面的joint_emaformer/目录为前缀。两个方案的最终配置在joint_emaformer/configs/all.yaml和selected.yaml。

融合权重训练截至2024-07，单区域权重训练截至2022-11并通过验证集选轮数；汇总表保留各自协议的结果，解释模型差异时需同时说明训练范围。

## 使用最新数据训练与验收

三地区融合模型已支持以下流程，命令均在本目录执行，具体配置和输出见融合模型README：

- [全历史训练并预测未来三个月](joint_emaformer/README.md#全历史训练并预测未来三个月)：将对应方案的training.cutoff设为latest，重新训练，再通过--checkpoint加载本次新权重执行--mode future。
- [留出最近三个月测试](joint_emaformer/README.md#留出最近三个月测试)：例如三表截至2026-05，将training.cutoff设为2026-02、公共period.test_end设为2026-05，运行训练后自动得到3、4、5月的真实价格与预测价格及分月指标。
- [使用最新开发数据重新搜索参数](joint_emaformer/README.md#使用最新开发数据重新搜索参数)：在2026-02及以前划分训练和验证区间，运行tune.py，再用生成的final_config.yaml训练与测试；保留的3、4、5月不用于搜索。

以上适用于全变量all和组合变量selected两种融合方案。三个单区域模型仍按各自既定训练、验证、测试划分训练。建议保留原实验配置用于复现；使用配置副本开展业务训练时，通过--config或--data-config指定，并按副本所在目录检查相对路径。

## 接口部署与试运行

在具备服务依赖的Python环境中执行：

```powershell
python -B -m service.app --device cpu
```

默认地址为http://127.0.0.1:8000，当前服务版本为2.0.0。GET /models查询全部模型版本及数据兼容性，POST /predictions提交{"profile":"selected"}或{"profile":"all"}，一次返回三地区未来3个月价格、经验区间、历史MAPE与精度得分，以及模型/数据版本和时间信息。支持三表上传、显式选择数据版本、自动调参、按tuning_id训练及模型热切换，无需为这些操作重启服务；已有持久化默认关系不会因新上传或配置表变更自动改变。随包旧权重的训练执行时间未记录，返回null；API新训练权重会记录实际完成时间。全部22项业务操作见[完整接口说明](service/API_DESIGN_README.md)。

接口复用现有模型计算；区间使用相同参数和变量方案最近24条已成熟滚动回测残差，名义水平95%，并返回校准样本数和历史覆盖率。默认登记权重训练截至2024-07，数据截至2026-05；未来预测月份始终相对于数据末月。甲方实际使用前需更新数据、选择并登记模型版本。完整字段和调用方式见[接口说明](service/README.md)，安装与维护见[部署说明](deployment/README.md)。

```powershell
python -B deployment/validate.py
```

该命令启动实际HTTP服务，完成测试、接口与CLI数值核对、上传、自动调参到训练的小预算流程、热切换、并发调用及重启恢复，并停止自己创建的进程、清理临时工作目录。随包2.0.0本机技术验证结果见[验证记录](reference/service_v2_validation/report.md)；甲方机器部署、隔离环境安装和真实业务月精度试运行需在对应环境与月份完成。

## XGBoost变量筛选

在本目录执行，默认分别筛选上海、浙江、江苏：

```powershell
python -B xgboost/run.py
python -B xgboost/run.py --region shanghai
python -B xgboost/run.py --check-reference
```

入口为xgboost/run.py，算法在selection.py，筛选配置在xgboost/config.yaml。数据路径、训练与验证截止月、历史窗口和缺失策略读取公共config.yaml；candidate_features配置候选字段，树参数、排序折数、置换次数、种子和前向选择范围均可配置。--config指定筛选YAML，--data-config指定公共YAML，--output-dir指定输出根目录。YAML内部路径相对于该YAML，显式命令行路径相对于启动目录。

原实验使用12个月历史预测未来3个月。训练区间2015-01至2022-11内进行3折扩展窗口排序，每折验证15个月；同一变量的12个滞后一起按3个月时间块置换，每折20次，主排序指标为置换后的MAPE增加百分点，同时提供total_gain占比。组合候选包含置换排序和增益排序的全部Top-K，以及在置换排序前8个外生变量中最多5步的前向选择，并比较depth2、depth3两种参数。历史目标始终保留。按2022-12至2024-07独立验证集上三个种子平均预测的MAPE确定组合；2024-08之后的测试数据不参与排序或选择。

每个预测月独立拟合XGBoost，使用原始数值。输入前向填充、起始缺失使用对应训练历史中位数；早期排序折中训练期全空的列保留NaN由XGBoost处理。中位数不借用验证期或测试期。重要性可以为负，相关变量可能互相替代；其含义是当前XGBoost的预测贡献，不是因果效应或神经网络重要性。跨折标准差描述历史折的变化，不是置信区间。最终组合经过验证集比较，因此不一定等于重要性前几名；只保证在本次搜索候选中最优。

运行结果保存在xgboost/outputs/selection_年月日_时分，包含三地区各一份XLSX，每份有“变量重要性”“组合比较”“最终变量”三个工作表。组合比较按第1、2、3个月分别列MAPE和精度得分，并给出平均MAPE与平均精度得分，全部属于XGBoost验证指标；得分=100-MAPE。种子保存在运行元数据，表格不显示种子列。selection.json保留完整排序、逐折结果、全部组合、参数、数据哈希和运行时间；run.json保存配置快照。

selected_features.yaml以features.selected格式输出变量清单，历史目标放在首列，其余按处理表顺序排列。核对后将对应地区清单写入公共config.yaml的features.selected，再训练配套模型；筛选程序不会自动改写公共配置或已有权重。只运行一个地区时，输出清单仅包含该地区，更新时保留其余地区配置。

xgboost/reference保存与原实验核对后的三地区参考表、变量清单及原筛选记录。--check-reference要求原数据版本，用于核对全部排序、组合及验证MAPE；更新数据或修改参数后运行时可省略该选项。参考中的最终组合分别为上海4列、浙江3列、江苏5列（均含历史价格），与本目录配套神经网络权重的输入一致。模块运行只读取本复现目录及指定的数据，不访问原项目的临时实验目录。

## 训练

```powershell
python -B lstm/train.py
python -B emaformer/train.py
python -B mofo/train.py
```

每个入口直接读取自己的config.yaml，默认运行三地区的全变量和组合变量共6组实验。也可指定一组：

```powershell
python -B lstm/train.py --region shanghai --profile selected
python -B emaformer/train.py --region jiangsu --profile all --device cuda
```

训练保存best.pt、config.json、preprocessing.json、history.csv、training_audit.json、input_snapshot.csv、test_predictions.csv、test_metrics.json及test_metrics.csv。结果写入模型目录下outputs/train/时间目录/地区/变量方案，配套weights不被覆盖。同一分钟重复运行以序号区分。

模型YAML的parameters设置结构，training设置regions、profiles、epochs、batch_size、lr、patience和output_dir。--epochs可用于短流程检查，--config指定其他模型配置，--data-config指定其他公共配置。直接在IDE运行也会读取默认配置。命令行显式路径按启动目录解释；YAML内部相对路径按对应YAML目录解释。

训练使用Adam、MSE、验证MAPE早停，不合并验证集拟合权重。缺失处理和标准化仅拟合训练历史。原单区域脚本的DataLoader及tqdm.auto包装保留，更换依赖或迭代包装可能改变随机序列。新训练记录实际开始/结束时间、耗时、参数和数据哈希；原权重未补造历史训练时间。

## 历史测试与未来预测

```powershell
python -B lstm/predict.py
python -B emaformer/predict.py
python -B mofo/predict.py
```

默认prediction.mode为test，加载本模型weights中的6份配套权重，输出逐条预测和分月指标。可在模型YAML设置地区、方案、mode、checkpoint_dir、batch_size和output_dir。结果写入模型目录下outputs/predict/模式_时间目录/地区/变量方案。

```powershell
python -B lstm/predict.py --mode future --region shanghai --profile selected
python -B emaformer/predict.py --mode future
python -B mofo/predict.py --mode future
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

原报告使用CUDA，CPU算子有小量浮点差异。verify对CPU采用0.0001元/立方米价格绝对容差，对CUDA采用0.000001元/立方米；实际差异记录在verification.json，不把CPU数值改写成原GPU数值。

本次18组实验均按原配置从头重训，最佳轮数、验证MAPE及最佳权重参数张量逐项一致；580行训练历史的最大数值差为8.89×10⁻¹⁶。记录见reference/reproduction_audit.json，原历史与本次重训历史分别保存于training_history.csv和retrained_history.csv。这是当前CUDA环境下的核对，CPU训练的dropout随机实现不同，不保证重训得到相同权重。

manifest.json用于校验代码、配套数据、权重及参考结果。模型和服务运行配置YAML可编辑，清单见manifest的editable_configs，不纳入不可变文件哈希；如果改变实验区间或变量组合，应训练配套新权重，原参考指标仅对应原实验。单区域模块迁移至含空格的独立目录后，从其他工作目录重载18份权重并进行未来预测已通过，见reference/relocation_verification.json。融合模块核验记录见joint_emaformer/reference/reproduction_check.json。

## 数据与时间口径

公共config.yaml的data支持绝对或相对路径。三表均为2015-01至2026-05、137个月，一个工作表，第1行为年月及模板24列，第2行为单位。年月使用YYYY-MM文本，进口量为吨，目标为元/立方米。历史目标价格始终输入，目标标签必须为正且不可缺失。

训练2015-01至2022-11共95个月，验证2022-12至2024-07共20个月，测试2024-08至2026-05共22个月。输入12个月、直接输出3个月，标签不跨分区；评价窗口可以读取届时已观测的前期历史。测试有20个起点，各步20组配对、三步60组，属于固定权重逐月测试，不是逐月重训回测。

历史复现要求数据SHA256与权重一致。future允许追加完整已观测历史，至少覆盖权重验证选模截止月；程序按权重首月截取，保持前向填充及周期行号一致。不要只提供最后12行，也不需要未来解释变量。不会在预测时重新拟合预处理。

features.selected列出组合变量；features.all默认按表格顺序选择有效候选，排除停用及训练期全空列。全变量输入数为上海23、浙江20、江苏21，组合变量为4、3、5。浙江停用区域LNG进口量、区域天然气产量；江苏停用工业增加值、GDP、非居民消费量。

LSTM默认32维单层；EMAformer为512维、8头2层、前馈2048、cycle=24；MoFo为本项目既有月度预测版本，24维、4头1层、periodic=12及通道融合。EMAformer相位为预测首月全局行号模周期，MoFo为最后历史月行号模周期。
