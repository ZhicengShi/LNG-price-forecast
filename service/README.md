# 三地区天然气价格统一接口

当前服务版本为2.0.0，支持数据上传、自动调参、按方案训练、多版本选择和模型热切换。全部22项业务操作、请求字段和返回字段见[完整接口说明](API_DESIGN_README.md)。本文件提供启动命令、常用接口及预测响应字段。

从experiment_reproduction目录执行。接口复用joint_emaformer中的模型、预处理及forecast_future计算，默认在CPU加载全变量、组合变量两份配套权重。每次请求选择一个方案，同时返回上海、浙江、江苏未来3个月，共9条预测。

## 启动与调用

安装、内网访问、停止和验证命令见[部署说明](../deployment/README.md)。Windows已具备依赖时直接运行：

```powershell
python -B -m service.app
```

默认地址http://127.0.0.1:8000，交互文档为/docs，接口结构为/openapi.json。交互文档页面的脚本默认来自公共CDN；不联网时可用下面的HTTP命令直接调用API，/openapi.json不依赖外网。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/models
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predictions -ContentType 'application/json' -Body '{"profile":"selected"}'
```

请求示例：

```json
{"profile": "selected"}
```

profile取all或selected，未指定模型时默认selected。model_version可选择任意已登记版本，省略profile时从模型记录推断方案。data_version可指定上传或配置导入的数据版本，省略时使用持久化默认数据。首次启动从公共config.yaml导入三地区处理表；以后上传、选择数据和热切换模型均无需重启。修改设备或后台策略配置后重启生效，路径相对于所属YAML解析。

### 自动调参与训练

先上传三张处理表或选用已有数据版本，再调用POST /tuning-jobs，最简请求为：

```json
{"profile": "selected"}
```

系统按数据末月自动留出最近3个月测试，之前20个月用于初筛验证，搜索范围和预算读取服务端搜索配置。通过GET /tuning-jobs/{tuning_id}查看进度，只有status=succeeded后才使用该方案训练：

```json
{"tuning_id": "tune_22222222222222222222222222222222", "mode": "holdout"}
```

示例编号须替换为真实返回值。训练自动继承调参任务的变量方案、数据版本、结构和冻结训练计划。mode=holdout留出最近3个月检验；mode=full_history使用全部历史训练，用于后续未来预测。训练完成后返回model_version，可直接在POST /predictions中选择；不自动替换默认模型。

### Linux CUDA启动与调用

适用于已按[Linux部署步骤](../deployment/README.md#linux部署cuda)安装好Python 3.11、项目CUDA依赖和NVIDIA驱动的服务器。将下面目录替换为实际部署位置，从包根目录执行；现有模型权重和三地区处理表可直接随项目复制，Linux路径区分大小写。

```bash
cd /path/to/experiment_reproduction
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python -B -m service.app --config service/config.yaml --device cuda --host 127.0.0.1 --port 8000
```

CUDA_VISIBLE_DEVICES=0选择编号为0的GPU；--device cuda覆盖服务配置中的默认CPU设置。显式指定cuda时，GPU不可用会报错，不会自动改用CPU。服务前台运行，按Ctrl+C停止。启动就绪后，在服务器另一个终端执行：

```bash
curl -fsS http://127.0.0.1:8000/models
curl -fsS -X POST http://127.0.0.1:8000/predictions \
  -H 'Content-Type: application/json' \
  -d '{"profile":"selected"}'
```

/models返回的models数组中各模型的device应为cuda，预测响应中model_status.device也应为cuda。若需供内网其他机器调用，启动前设置export LNG_API_KEY='替换为实际密钥'，将--host改为0.0.0.0；客户端使用服务器实际IP，并给/models及预测请求增加-H "X-API-Key: $LNG_API_KEY"。新终端须设置同一密钥变量。接口字段和Windows版本一致。

## 常用接口

| 方法与路径 | 内容 |
| --- | --- |
| GET /models | 两方案的模型版本、变量、训练时间信息及默认数据状态 |
| POST /datasets、GET /datasets | 上传三表、查询数据版本 |
| POST /tuning-jobs、GET /tuning-jobs/{tuning_id} | 自动调参、查询进度 |
| POST /trainings、GET /trainings/{training_id} | 按方案训练、查询结果 |
| POST /models/{model_version}/activate | 热加载并切换默认模型 |
| GET /templates/shanghai | 下载上海完整处理表样例；也支持zhejiang、jiangsu |
| POST /predictions | 执行预测并返回统一四部分结果 |
| GET /predictions/{prediction_id} | 查询完整预测记录，服务重启后仍保留 |
| GET /predictions/{prediction_id}/csv | 下载9行价格与区间明细，不含seed列 |

## 返回字段的通用约定

JSON中的字段名称区分大小写。string为文本，integer为整数，number为数值，boolean为true/false，object为对象，array为数组；null表示未记录或当前不可用，不应当作0。说明中的[]表示数组元素，例如predictions[]表示predictions中的一条价格记录。

地区编码统一为shanghai（上海）、zhejiang（浙江）、jiangsu（江苏）；变量方案统一为all（全变量）、selected（组合变量）。月份文本格式为YYYY-MM；执行时间采用带时区的ISO 8601格式，当前服务生成的+00:00为UTC，北京时间需加8小时。价格及价格区间单位为元/立方米。名称带_percent的值已经是百分数，例如2.5表示2.5%，无需再乘100；confidence=0.95表示95%的名义水平。

设置LNG_API_KEY时，业务接口须携带X-API-Key请求头，自动文档入口无需密钥；未设置时不启用该校验。以下字段表说明成功响应，通用失败响应见[错误响应](#错误响应)。

## 各接口返回说明

### GET /models

成功返回200、application/json，查询全部已登记模型及指定数据的兼容性，不执行价格预测。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `models` | array[object] | 全部模型版本的状态列表，一个方案可以有多个版本。包含[模型状态字段](#模型状态字段)，另有source、load_state、is_default、parameters、training_plan、compatibility和evaluation等，详见完整接口说明的ModelRecord。 |

此接口中的loaded_at等字段属于当前服务进程；查询旧预测记录时，同名字段属于当时生成那次预测的进程。

### GET /templates/{region}

region为路径参数，取shanghai、zhejiang或jiangsu。成功返回200及对应地区完整处理表的XLSX二进制文件，不返回JSON。

响应头Content-Type为application/vnd.openxmlformats-officedocument.spreadsheetml.sheet；Content-Disposition提供下载文件名。工作簿只有一个工作表，第1行为字段名，第2行为单位，从第3行开始为月度数据。以下是当前三地区样例共有的25列，顺序与表格一致：

| 字段 | 单位或格式 | 含义 |
| --- | --- | --- |
| `年月` | YYYY-MM | 每行数据所属月份，须按月连续、升序、不重复。 |
| `JKM现货价格` | 美元/百万英热单位 | JKM天然气现货价格指标。 |
| `LNG进口价格` | 元/吨 | LNG进口价格指标。 |
| `TTF期货价格` | 美元/百万英热单位 | TTF天然气期货价格指标。 |
| `Henry Hub期货价格` | 美元/百万英热单位 | Henry Hub天然气期货价格指标。 |
| `Brent原油价格` | 美元/桶 | Brent原油价格指标。 |
| `管道气进口价格` | 元/吨 | 管道天然气进口价格指标。 |
| `美元/人民币汇率` | 人民币元/美元 | 1美元对应的人民币金额。 |
| `LNG进口量（全国）` | 吨 | 全国LNG进口量。 |
| `管道气进口量（全国）` | 吨 | 全国管道天然气进口量。 |
| `LNG进口量（区域）` | 吨 | 当前地区的LNG进口量字段。 |
| `天然气产量（全国总量）` | 亿立方米 | 全国天然气产量。 |
| `天然气产量（区域）` | 亿立方米 | 当前地区的天然气产量字段。 |
| `LNG产量（全国）` | 万吨 | 全国LNG产量。 |
| `门站价基准价格（分省）` | 元/立方米 | 当前地区门站基准价格指标。 |
| `中国液化天然气出厂价格指数` | 元/吨 | 全国LNG出厂价格指标，单位按当前处理表保存。 |
| `工业增加值（分省）` | % | 地区工业增加值相关指标，按处理表口径保存。 |
| `制造业PMI增速（全国）` | % | 全国制造业PMI增速字段的处理值。 |
| `GDP增速（分省）` | % | 当前地区GDP增速字段的处理值。 |
| `平均气温（分省）` | ℃ | 当前地区的月度平均气温指标。 |
| `采暖度日数HDD（分省）` | ℃·日 | 当前地区的采暖度日数指标。 |
| `非居民天然气消费量（分省）` | 亿立方米 | 当前地区非居民天然气消费量字段。 |
| `柴油价格（0#柴油）` | 元/升 | 0号柴油价格指标。 |
| `动力煤价格` | 元/吨 | 动力煤价格指标。 |
| `终端非居民天然气价格（分省）` | 元/立方米 | 历史目标价格，始终作为模型输入，且须为正、不缺失。 |

这是处理表格式样例，保留的字段不一定全部进入当前模型；实际使用的变量及顺序见input_features。未取得的非目标字段可保持空值，是否能够使用由当前权重及预处理要求判断。下载样例不会修改配置中的数据，也不触发训练或预测。

### POST /predictions

请求体为application/json，必须提供JSON对象；{}表示采用默认组合变量方案。

| 请求字段 | 类型 | 是否必填 | 含义 |
| --- | --- | --- | --- |
| `profile` | string | 否 | 未指定模型版本时默认selected；指定版本时可省略，由该版本推断方案。 |
| `model_version` | string或null | 否，默认null | 选择已登记权重版本；省略时使用该方案默认版本。填写profile时须与指定模型一致。 |
| `data_version` | string或null | 否，默认null | 指定数据快照；省略时使用当前默认数据版本。 |

成功返回200、application/json。未提供Idempotency-Key时每次调用生成新的prediction_id并保存响应；同一幂等键和相同请求复用首次结果。数据来自显式指定或当前默认的数据版本。返回字段见[统一返回结构](#统一返回结构)及其后各嵌套字段表。

### GET /predictions/{prediction_id}

prediction_id为POST /predictions实际返回的编号，格式为pred_后接32位十六进制字符。应从实际响应复制编号。

成功返回200、application/json，所有返回字段与POST /predictions一致，详见[统一返回结构](#统一返回结构)。返回的是保存时的价格、区间、历史评价和模型状态，generated_at仍为原预测生成时间；本次GET不重新预测，也不因模型、数据或区间设置更新而重新计算旧记录。编号无效或记录不存在时返回404。

### GET /predictions/{prediction_id}/csv

prediction_id的要求与上面的查询接口相同。成功返回200、text/csv，并通过Content-Disposition提供文件名pred_编号.csv。文件为带BOM的UTF-8编码；正常三地区三步预测有9行数据，另有1行表头。

| CSV列名 | 内容类型 | 含义 |
| --- | --- | --- |
| `region` | 文本 | 地区编码。 |
| `region_name` | 文本 | 地区中文名。 |
| `target_month` | YYYY-MM文本 | 本行预测对应的目标月份。 |
| `horizon` | 整数 | 预测步，1、2、3分别为未来第1、2、3个月。 |
| `prediction` | 数值 | 预测价格，单位见unit。 |
| `unit` | 文本 | 价格单位，当前为元/立方米。 |
| `lower` | 数值或空单元格 | 价格区间下限；区间不可用时为空。 |
| `upper` | 数值或空单元格 | 价格区间上限；区间不可用时为空。 |
| `interval_available` | True/False文本 | 本行价格区间是否可用。 |
| `model_version` | 文本 | 生成本条预测时使用的模型权重版本。 |
| `model_training_cutoff` | YYYY-MM文本 | 该权重拟合时使用的最后一个历史月份。 |
| `latest_observed_month` | YYYY-MM文本 | 生成本次预测时，输入表中的最新观测月份。 |

该文件导出已有结果，不重新预测。公开CSV不包含seed列，区间不可用时lower和upper留空，不填0。完整区间原因、校准数量、历史精度和其他版本信息通过对应JSON查询接口获取。

## 统一返回结构

POST /predictions与GET /predictions/{prediction_id}共用下面的JSON结构；这些字段都是返回字段，不是请求参数。

### 顶层响应字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `prediction_id` | string | 本次完整预测的唯一编号，用于后续查询或CSV下载；不表示地区或预测步。 |
| `generated_at` | string | 本次预测结果生成时间，带时区的ISO 8601时间戳。 |
| `predictions` | array[object] | 点预测价格明细，正常为三地区×三个预测步，共9条。 |
| `prediction_intervals` | object | 价格区间的计算设置、校准来源及9条区间明细。 |
| `evaluation` | object | 对应历史协议下的分地区、分步MAPE和精度得分，带评分范围与样本数。 |
| `model_status` | object | 本次使用的模型版本、设备、时间信息、数据版本和输入变量。 |

### 预测价格字段

下面是predictions[]每条记录的字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `region` | string | 地区编码，例如shanghai。 |
| `region_name` | string | 地区中文名，例如上海。 |
| `target_month` | string | 本条价格预测对应的目标月份，例如2026-06。 |
| `horizon` | integer | 相对最新观测月的预测步；1为下个月，2为下两个月，3为下三个月。 |
| `prediction` | number | 模型预测价格，经目标标准化参数还原后的原始价格数值。 |
| `unit` | string | 价格单位，当前为元/立方米。 |

### 价格区间设置字段

下面是prediction_intervals对象的字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `method` | string | 计算方法标识。matured_absolute_residual_order_statistic表示对真实价格已取得的历史绝对误差排序，用误差阈值确定区间半宽。 |
| `confidence` | number | 名义覆盖水平，当前为0.95，即95%；不是历史实测覆盖率。 |
| `calibration_basis` | string | 校准来源口径。same_parameters_features_rolling_refit_protocol表示相同结构、变量及训练计划的逐月重训回测误差。 |
| `calibration_version` | string或null | 校准历史预测明细文件的SHA256指纹，用于追溯证据文件；未登记该证据时为null。 |
| `min_residuals` | integer | 计算区间所需的最少已成熟残差数，当前为24。 |
| `residual_window` | integer | 最多选取的最近残差数量，当前为24。 |
| `interpretation` | string | 区间含义的文字说明，可供页面直接展示。 |
| `items` | array[object] | 各地区、各预测步的区间明细，正常共9条，字段见下表。 |

### 价格区间明细字段

下面是prediction_intervals.items[]每条记录的字段；通过region和horizon与predictions中的价格对应。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `region` | string | 本条区间所属地区编码。 |
| `target_month` | string | 本条区间对应的价格目标月份。 |
| `horizon` | integer | 预测步，1、2或3；不同预测步分别校准。 |
| `available` | boolean | 区间是否具备计算条件；true表示可用，false时查看reason。 |
| `lower` | number或null | 价格区间下限，元/立方米；不可用时为null。 |
| `upper` | number或null | 价格区间上限，元/立方米；不可用时为null。 |
| `calibration_count` | integer | 为本条区间实际选取的已成熟历史误差数量；达到当前默认条件时为24，不足时仍返回实际数量。 |
| `latest_calibration_target_month` | string或null | 所选校准误差中最新一条对应的真实价格月份；没有可用残差时为null。 |
| `reason` | string或null | 区间不可用原因，例如残差不足、没有匹配证据或历史数据发生修订；可用时通常为null。 |
| `historical_coverage_percent` | number或null | 该地区、该预测步过去可评价价格区间的实际覆盖率，按包含真实价格的区间数÷可评价区间数×100计算；无可评价区间时为null。 |
| `historical_coverage_count` | integer | 计算上述覆盖率的历史区间数量；无可评价区间时为0，与本次calibration_count含义不同。 |

例如calibration_count=24表示本次区间使用24条误差；historical_coverage_count=102且historical_coverage_percent约95.10，表示过去102个可评价区间中有97个覆盖真实价格。这不等于本次未来价格严格有95.10%的概率落入区间。

### 历史评价字段

下面是evaluation对象的字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `recent_backtest` | object | 最近若干个完整已成熟起点的滚动重训回测指标，属于相同参数和变量方案的训练协议参考。 |
| `registered_model_test` | object | 当前登记权重的固定权重历史测试指标，证据文件须与该权重匹配。 |
| `interpretation` | string | 指标公式、评价协议及适用范围的文字说明。 |

### 单类评价结果字段

evaluation.recent_backtest与evaluation.registered_model_test具有相同的字段结构：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `available` | boolean | 这类历史评价是否可用；false时通常返回空regions数组。 |
| `reason` | string或null | 评价不可用原因，例如没有登记证据、没有成熟完整起点或历史参考不匹配；正常可用时为null。 |
| `basis` | string | 评价口径。same_parameters_features_rolling_refit_protocol为逐月重训协议；registered_checkpoint_fixed_weight_holdout为当前权重固定测试。 |
| `version` | string或null | 该类历史预测明细文件的SHA256指纹；未登记证据时为null。 |
| `first_forecast_month` | string或null | 纳入评分的最早预测起点的第1个月目标月份。 |
| `last_forecast_month` | string或null | 纳入评分的最后一个预测起点的第1个月目标月份。 |
| `last_target_month` | string或null | 纳入评分的全部预测步中最晚的目标月份，通常比last_forecast_month晚2个月。 |
| `regions` | array[object] | 上海、浙江、江苏各自的分步与平均指标；不可用时为空数组。 |

例如最后一个起点预测2026年3、4、5月时，last_forecast_month为2026-03，last_target_month为2026-05；它们分别描述最后一个起点的首月与完整三步评分范围的末月。

### 地区评价字段

下面是上述两类评价结果中regions[]每条记录的字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `region` | string | 地区编码。 |
| `region_name` | string | 地区中文名。 |
| `origins` | integer | 本地区实际参与评分的完整预测起点数量。 |
| `by_horizon` | array[object] | 第1、2、3个月各自的指标，共3条，字段见下表。 |
| `average_mape_percent` | number | 三个预测步MAPE的等权平均，单位为%。 |
| `average_accuracy_score_percent` | number | 三个预测步精度得分的等权平均，等于100-average_mape_percent，单位为%。 |

### 分预测步评价字段

下面是by_horizon[]每条记录的字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `horizon` | integer | 对应第1、2或3个月的预测步。 |
| `observations` | integer | 该地区、该预测步实际参与评分的预测与真实值配对数量。 |
| `mape_percent` | number | 平均绝对百分比误差：mean(abs(预测值-真实值)/真实值)×100，数值越小表示该历史范围平均误差越小。 |
| `accuracy_score_percent` | number | 预测精度得分，等于100-mape_percent；例如MAPE为2.5时得分为97.5。 |

当前默认最多取最近20个完整起点。有20个时，每地区origins=20、各步observations=20，三步合计60组配对；不足20个时返回实际可用数量，无完整起点时返回available=false。指标属于声明的历史范围，不为尚未取得真实值的未来月份计算误差；精度得分为负时不截断。

### 模型状态字段

GET /models的models[]与预测响应model_status共用以下全部字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `model_name` | string | 模型名称，当前为三地区融合EMAformer。 |
| `service_version` | string | 生成这份状态时的接口程序版本。 |
| `model_version` | string | 权重版本标识，由方案与权重SHA256前16位组成；权重不变时版本不变。 |
| `checkpoint_sha256` | string | 当前权重文件完整的SHA256指纹，用于核对模型文件。 |
| `profile` | string | 当前变量方案，all或selected。 |
| `device` | string | 该次推理使用的实际设备，例如cpu或cuda。 |
| `registered_at` | string | 该权重版本首次在当前服务存储目录登记的时间；保留存储目录时，重启后保持。 |
| `loaded_at` | string | 生成这份状态的服务进程加载该模型的时间，重启后新预测会记录新的加载时间。 |
| `trained_at` | string或null | 权重中记录的训练执行时间；优先取完成时间，只有开始记录时取开始时间，未记录时为null。 |
| `training_time_status` | string | recorded表示trained_at有记录，not_recorded表示原始训练执行时间未记录。 |
| `model_training_cutoff` | string | 该权重拟合时使用的最后一个数据月份，当前随包权重为2024-07。 |
| `parameter_selection_cutoff` | string | 变量与模型参数选择所使用的开发数据截止月份，当前随包权重为2024-07。 |
| `latest_observed_month` | string | 本次输入处理表中的最新观测月份；未来三个目标月从其后开始。 |
| `data_imported_at` | string | 该文件内容版本首次被服务读取、校验并登记的时间，不是原始数据发布日或数据所属月份。 |
| `data_version` | string | 三地区文件内容共同确定的只读数据版本；可在预测、训练或调参请求中选择。 |
| `data_sha256` | object | 三地区输入文件各自的完整SHA256指纹；键为shanghai、zhejiang、jiangsu，值均为字符串。 |
| `historical_reference_matches` | boolean | 当前数据在校准、评价参考历史段内的数值是否一致；追加新月份可以保持true。false时旧区间与旧历史评价不可用，点预测仍可执行。 |
| `seq_len` | integer | 模型每次读取的历史窗口月数，当前随包融合权重为6。 |
| `pred_len` | integer | 每次直接输出的未来月数，当前为3。 |
| `input_features` | object | 各地区实际输入字段及其顺序，包含历史目标价格；键为shanghai、zhejiang、jiangsu，值为字段名字符串数组。 |

训练截止月份、最新观测月份描述数据范围，trained_at等时间戳描述实际执行或登记时间。原始训练执行时间缺失时保持null，不用文件修改时间补造。文件重新保存可能改变SHA256和data_version，historical_reference_matches则按参考月份内的数值核对。

## 错误响应

| HTTP状态 | 常见原因 | 返回形式 |
| --- | --- | --- |
| 401 | 已启用API密钥但未提供正确X-API-Key。 | UNAUTHORIZED。 |
| 404 | 数据、模型、任务、预测记录或路径不存在。 | NOT_FOUND。 |
| 405 | 请求方法不匹配，例如用GET访问模型启用接口。 | METHOD_NOT_ALLOWED。 |
| 409 | 默认版本已改变、幂等键冲突或任务状态不允许。 | VERSION_CONFLICT或INVALID_STATE。 |
| 413、415 | 文件过大或不是有效XLSX。 | FILE_TOO_LARGE或UNSUPPORTED_FORMAT。 |
| 422 | 请求字段、数据、边界或模型兼容性校验失败。 | INVALID_REQUEST、INVALID_DATA等。 |
| 429 | 后台任务队列已满。 | QUEUE_FULL。 |
| 500、503 | 服务内部异常或模型加载失败。 | INTERNAL_ERROR或MODEL_LOAD_FAILED。 |

错误体统一如下，响应头X-Request-ID与request_id一致：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `error` | string | 稳定机器错误编码。 |
| `message` | string | 可展示的错误摘要。 |
| `request_id` | string | 服务日志中的请求定位编号。 |
| `details` | array[Issue] | 具体字段、地区、月份及修复建议；无详情时为空数组。 |

例如，提供的API密钥无效时：

```json
{
  "error": "UNAUTHORIZED",
  "message": "API密钥无效",
  "request_id": "req_example",
  "details": []
}
```

查询不存在的预测记录时：

```json
{
  "error": "NOT_FOUND",
  "message": "指定的记录不存在",
  "request_id": "req_example",
  "details": []
}
```

## 自动文档入口

/docs与/redoc返回HTML接口文档页面，不返回业务预测数据。/openapi.json返回机器可读的OpenAPI结构，主要包含：openapi（规范版本）、info（接口标题和版本）、paths（路径、请求方法及参数）、components（请求/响应类型定义和API密钥规则）。前面的业务字段表与当前代码的响应模型对应；完整可调用样例见[全变量响应](../reference/service_v2_validation/all_response.json)、[组合变量响应](../reference/service_v2_validation/selected_response.json)。示例文件中的prediction_id仅用于展示，应以自己实际POST返回的编号查询对应服务存储目录中的记录。

## 未来价格区间

默认名义水平95%，每地区、每预测步独立校准。仅使用目标月份不晚于最新观测月的已成熟绝对残差，使用最近24个，至少需要24个。对n个残差排序，半宽取第min(ceil((n+1)×0.95),n)个值；当前n=24时取这24个残差的最大值，价格区间为本次预测值±半宽。

修改service/config.yaml后需重启服务，再通过POST /predictions生成新结果。已保存的预测记录保留生成时的区间设置，GET查询旧prediction_id不会重新计算。历史记录可能采用不同窗口，当前设置以实际响应中的residual_window为准；随包当前验证样例使用24条窗口。

当前校准来源是随包相同变量布局、网络参数、训练轮数和学习率计划的逐月重训回测。服务会检查这些条件、证据文件SHA256、目标真实值及输入布局。它属于训练协议的历史误差参考，固定当前权重的未来覆盖率不作保证；早于2024-07选参截止月的回测起点是固定事后参数的回顾性评价。历史覆盖率按每个历史起点当时已成熟的残差逐点计算，并返回实际覆盖数量。这些价格区间与MAPE的Bootstrap置信区间不同。

历史参考段内的三表数值须与登记的参考数据一致，允许在其后追加新月份。若修改历史值，点预测仍可执行，旧区间和旧评价返回available=false及原因；应重新准备该数据版本对应的回测证据。校准样本不足或新权重没有匹配证据时，上下限为null，禁止用0或MAPE区间代替。实时新增预测先保留记录；其真实值成熟后的在线误差归集和周期性重新回测由运维流程执行，本版本不自动将在线结果混入随包历史校准表。

## 数据与模型版本

三表使用现有单工作表模板：第1行为年月及24个字段，第2行为单位，后续为连续YYYY-MM月度数据。全国指标三地一致，目标价格为正且不缺失。配置中的处理表须保留权重首月2015-01至三地一致最新观测月的完整历史；预处理始终使用权重保存的参数。当前样例截至2026-05，预测目标为2026-06至2026-08，月份不由系统日期代替。

service/config.yaml控制设备、线程、端口、存储位置、月份上限和区间设置，models.yaml登记权重与配套证据，内部路径按所属YAML解析。LNG_SERVICE_CONFIG指定其他服务配置，LNG_SERVICE_STORAGE可覆盖运行存储目录，LNG_SERVICE_DEVICE可覆盖设备。公开输出不暴露服务端文件路径。

model_version由方案与权重SHA256前16位组成，权重不变时版本不变。API训练完成后自动计算并登记版本，无需用户填写路径或SHA256。GET /models列出全部版本，POST /predictions可以直接指定版本；POST /models/{model_version}/activate用于热切换默认模型。models.yaml保留为启动导入配置；新训练权重没有匹配回测时，返回点预测及版本信息，价格区间不可用，不能沿用其他结构或训练计划的旧证据。

运行产物位于outputs/service：data_versions保存数据快照，trainings和tuning_jobs保存任务及模型产物，predictions保存完整预测，registry.sqlite3保存版本、任务、默认关系和幂等结果，logs/events.jsonl保存事件。相同文件内容复用首次数据版本；上传不会自动修改默认数据。启动时发现配置表变更只导入新版本，已持久化的默认关系保持，须显式选择或设为默认。按甲方数据管理要求备份整个运行目录；已有预测保留生成时的价格、区间和版本信息。
