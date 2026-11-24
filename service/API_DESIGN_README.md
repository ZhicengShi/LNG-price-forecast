# 天然气价格预测服务 API 接口说明书

年度预算版通过 `/annual-budget` 接入本服务，复用数据版本、API 密钥、幂等请求和任务队列。年度模型、预测、训练、历史结果和下载接口见 [年度预算版接口说明](年度预算版接口说明.md)，交互文档位于 `/docs` 的“年度预算版”分组。

| 文档属性 | 内容 |
| --- | --- |
| 文档版本 | 1.0 |
| 适用程序版本 | 1.0 |
| 发布日期 | 2026-10-19 |
| 适用系统 | 上海、浙江、江苏三地区天然气价格预测服务 |
| 适用对象 | 系统集成开发人员、接口测试人员及部署运维人员 |
| 交互协议 | HTTP，JSON及文件传输 |
| 接口范围 | 数据管理、模型管理、训练任务、调参任务、预测及结果查询，共22项业务操作 |

本文定义接口地址、请求参数、响应字段、状态码、任务生命周期及数据约束。环境安装、启动命令及试运行方式见[部署说明](../deployment/README.md)。运行实例提供`/docs`、`/redoc`交互文档和`/openapi.json`接口结构。

阅读导航：[通用约定](#2-通用约定) · [接口目录](#3-接口目录) · [训练](#6-训练接口) · [调参](#7-调参接口) · [预测](#8-预测接口) · [返回结构](#9-预测与评价的公共返回结构) · [错误处理](#10-统一错误响应与冲突处理) · [数据约定](#11-处理表与变量约定) · [运行管理](#12-运行管理与集成验证)。

## 1. 服务概述

### 1.1 功能与适用范围

服务采用三地区融合EMAformer，同时处理上海、浙江、江苏的月度数据，每次预测未来3个月的地区价格，共返回9条预测。支持全变量`all`与组合变量`selected`两种方案，分别管理模型版本及默认模型。

接口接收三地区处理后的XLSX数据表，支持版本登记、训练、自动调参、模型启用、价格预测及历史结果查询。LSTM、单区域EMAformer、MoFo及完整滚动回测使用项目提供的命令行入口，调用方式见各模块说明。

### 1.2 业务调用流程

1. 下载地区数据样例，准备三地区处理后的XLSX表。
2. 上传三张表，取得`data_version`，并检查数据范围及模型兼容性。
3. 使用兼容的已登记模型执行预测，或根据业务需要提交训练、调参任务。
4. 调参任务成功后，使用`tuning_id`引用推荐参数及冻结训练计划，提交训练任务。
5. 训练成功后取得`model_version`，检查留出测试结果，选择该版本执行预测。
6. 需要更新默认数据或默认模型时，调用相应启用接口；单次指定版本不改变默认关系。
7. 通过预测编号查询完整响应或下载CSV，保留预测时使用的数据、模型与评价信息。

### 1.3 时间范围与评价口径

以下以数据覆盖2015-01至2026-05、保留测试3个月及初筛验证20个月为例：

| 操作 | 使用范围 | 结果含义 |
| --- | --- | --- |
| 留出3个月训练 | 拟合截至2026-02，测试2026-03至2026-05 | 一个固定新权重的三步留出测试，各地区各步1组配对 |
| 调参默认时间划分 | 初筛训练截至2024-06，验证2024-07至2026-02，保留测试2026-03至2026-05 | 根据验证历史选参，不读取保留测试标签选参 |
| 全历史训练 | 拟合截至2026-05 | 生成候选生产权重，本次没有留出测试标签 |
| 使用截至2026-05的观测预测 | 输入各权重要求的最近历史窗口 | 预测2026-06、2026-07、2026-08，目标月份由数据末月确定 |

历史留出评价与未来业务评价分别记录。已用于人工调整方案的历史测试成绩，应在模型选择记录中说明其使用情况；后续业务预测须在真实价格取得后另行评价。

## 2. 通用约定

### 2.1 类型、编码和默认值

| 项目 | 约定 |
| --- | --- |
| JSON字段 | 使用英文小写及下划线命名；未定义的请求字段返回422 |
| 类型 | `string`文本、`integer`整数、`number`有限数值、`boolean`布尔、`object`对象、`array`数组；`null`表示未记录或不可用，不表示0 |
| 地区 | `shanghai`上海、`zhejiang`浙江、`jiangsu`江苏；三地一次联合处理 |
| 变量方案 | `all`全变量、`selected`组合变量；预测及训练的默认方案为selected，显式指定模型或调参任务时按对应记录推断；同时提供的方案须一致 |
| 模型 | 接口业务模型为三地区融合EMAformer；预测长度固定3个月 |
| 月份 | 严格`YYYY-MM`，例如`2026-02`，并检查真实年月合法性 |
| 时间戳 | 使用带时区的ISO 8601格式；服务返回UTC时间，调用方可转换为业务所在地时间 |
| 百分数 | 名称带`_percent`的数值已乘100；2.5表示2.5%。`confidence=0.95`表示95% |
| 价格 | 元/立方米；JSON保留计算精度，页面显示位数不改变计算和导出结果 |
| 请求 | 除文件上传外使用`application/json`；POST预测和训练至少提交`{}`对象 |
| 默认数据 | 省略`data_version`时使用当前默认数据；引用tuning_id训练时优先沿用该调参任务绑定的数据版本；任务接受后固定解析结果 |
| 默认模型 | 每个变量方案各有一个默认模型。显式选版本不改变全局默认模型 |
| 标识 | 数据版本为`data_`加32位随机十六进制字符，已登记的24位编号继续有效；预测、训练、调参分别为`pred_`、`train_`、`tune_`加32位随机十六进制字符 |

模型版本使用独立编号：随包模型为`all-bundled-v1`与`selected-bundled-v1`，接口训练生成的编号为`变量方案-训练任务32位编号`。每次成功训练登记一个新版本。通过服务器配置导入模型时，`model_version`须与权重元数据一致，并登记配套评价证据。

数据、模型、训练任务、调参任务及预测记录列表一次返回符合查询条件的全部记录，默认按创建时间降序，同时间按编号降序。模型列表使用`models`数组，其余记录列表使用`items`数组；空列表正常返回200。

### 2.2 请求头

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `Content-Type` | 有请求体时 | JSON使用`application/json`；文件上传使用`multipart/form-data`，boundary由HTTP客户端生成 |
| `X-API-Key` | 服务启用密钥时必填 | 值须与服务器环境变量`LNG_API_KEY`一致；`/docs`、`/redoc`、`/openapi.json`无需该请求头 |
| `Idempotency-Key` | 否 | POST业务操作的幂等键，1至200个字符；建议调用方为每次独立操作生成唯一值，重试时复用 |

同一服务内，相同`Idempotency-Key`与相同业务请求在保留期内复用首次已保存的处理结果。相同幂等键用于不同请求时返回409。上传请求按三份文件内容、`name`与`note`比较，不按文件名或multipart边界比较。未指定的默认数据、默认模型和配置以首次处理结果为准，重试不重新解析默认值。

幂等记录默认保留24小时，可通过服务端配置调整。原请求仍在处理或处理状态尚未确认时，同键重试可返回409；调用方应先查询已有任务或记录。未提供幂等键的POST按独立操作处理。

响应头`X-Request-ID`由服务生成，用于日志定位。错误响应体中的`request_id`与该响应头一致。

### 2.3 服务地址与请求规则

默认服务地址为`http://127.0.0.1:8000`，实际主机与端口由部署配置确定。本文接口路径均相对于服务地址，未配置额外业务路径前缀。`/docs`与`/redoc`用于交互调试，程序集成直接调用业务接口。

GET请求不包含请求体。JSON类型的POST须提交对象，未使用可选字段时提交`{}`。响应结构按各接口定义返回，不额外包装统一的`code`或`data`字段。文件下载接口返回文件内容。

路径参数必须提供。未定义的请求体字段、查询字段以及上传中的未知或重复表单字段返回422。列表接口返回符合条件的全部记录，不接收分页参数。

### 2.4 计算设备

训练、调参和预测的设备由服务启动配置统一确定。CPU使用`--device cpu`，GPU使用`--device cuda`，`--device auto`在CUDA可用时选择GPU，否则选择CPU。显式要求CUDA但设备不可用时，服务启动失败。

Windows PowerShell启动示例，从项目根目录执行：

```powershell
python -B -m service.app --device cpu --host 127.0.0.1 --port 8000
```

```powershell
python -B -m service.app --device cuda --host 127.0.0.1 --port 8000
```

也可配置`service/config.yaml`中的`device`。启动参数优先于设备环境变量，设备环境变量优先于YAML设置。任务接受后设备随私有配置固定，切换服务设备不改变已提交任务的设备。Linux及CUDA环境安装步骤见[部署说明](../deployment/README.md#linux部署cuda)。

## 3. 接口目录

| 方法与路径 | 成功状态码 | 功能 |
| --- | --- | --- |
| `GET /templates/{region}` | 200 | 下载地区处理表样例 |
| `POST /datasets` | 201或200 | 上传并校验三地区处理表 |
| `GET /datasets` | 200 | 查询数据版本列表 |
| `GET /datasets/{data_version}` | 200 | 查询数据版本详情与模型兼容性 |
| `POST /datasets/{data_version}/activate` | 200 | 切换默认数据版本 |
| `GET /models` | 200 | 查询模型版本列表 |
| `GET /models/{model_version}` | 200 | 查询模型参数、训练范围及评价 |
| `POST /models/{model_version}/activate` | 200 | 加载并启用指定模型版本 |
| `POST /trainings` | 202 | 提交训练任务 |
| `GET /trainings` | 200 | 查询训练任务列表 |
| `GET /trainings/{training_id}` | 200 | 查询训练状态及结果 |
| `GET /trainings/{training_id}/csv` | 200 | 下载训练留出测试明细 |
| `POST /trainings/{training_id}/cancel` | 202或200 | 请求取消训练任务 |
| `POST /tuning-jobs` | 202 | 提交自动调参任务 |
| `GET /tuning-jobs` | 200 | 查询调参任务列表 |
| `GET /tuning-jobs/{tuning_id}` | 200 | 查询调参状态及进度 |
| `GET /tuning-jobs/{tuning_id}/results` | 200 | 查询推荐配置及候选排名 |
| `POST /tuning-jobs/{tuning_id}/cancel` | 202或200 | 请求取消调参任务 |
| `POST /predictions` | 200 | 生成并保存三地区预测结果 |
| `GET /predictions` | 200 | 查询预测历史列表 |
| `GET /predictions/{prediction_id}` | 200 | 查询已保存的完整预测响应 |
| `GET /predictions/{prediction_id}/csv` | 200 | 下载预测价格及区间明细 |

示例中的资源编号用于说明请求格式，调用时须替换为当前服务返回的真实编号。各接口的异常状态码见第10节。

## 4. 数据接口

### 4.1 GET /templates/{region}

`region`为`shanghai`、`zhejiang`或`jiangsu`。无请求体，成功200，返回对应XLSX二进制文件；`Content-Type`为`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`，`Content-Disposition`包含下载文件名，不返回JSON。

样例为随包的完整处理表，用于核对列名、单位和数据组织方式。三张上传文件均须为处理后表。字段及单位见[处理表约定](#11-处理表与变量约定)。

### 4.2 POST /datasets

请求类型为`multipart/form-data`。三地区文件须在同一次请求中提交，全部校验通过后登记为一个数据版本。

| 请求字段 | 类型 | 必填及默认 | 含义 |
| --- | --- | --- | --- |
| `shanghai_file` | file | 必填 | 上海处理后XLSX |
| `zhejiang_file` | file | 必填 | 浙江处理后XLSX |
| `jiangsu_file` | file | 必填 | 江苏处理后XLSX |
| `name` | string | 可选，自动生成 | 数据版本显示名称，最多100字符 |
| `note` | string | 可选，空文本 | 更新说明，最多500字符，例如“补充至2026年6月” |

默认限制为每文件20 MiB、三文件合计60 MiB、每表最多2400个月、每个工作簿解压后总内容最多200 MiB，由服务端配置控制。上传文件须具有`.xlsx`扩展名且内部结构有效，文件名主体可自定义；地区由表单字段识别。原始文件名仅用于展示，存储路径由服务生成。

新版本成功201，完全相同的三份文件复用已有版本时200：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `created` | boolean | 是否新建了数据内容版本 |
| `dataset` | DataVersion | 数据版本完整信息，见下表 |

**DataVersion对象：**

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `data_version` | string | 本次三地区数据快照的独立版本编号；相同文件内容复用已有版本 |
| `name` | string | 数据版本显示名称 |
| `note` | string | 更新说明 |
| `created_at` | string | 此内容版本首次登记时间；重复上传不改写首次时间与原名称说明 |
| `source` | string | `configured`从服务器配置表导入（包括随包表），或`upload`上传数据 |
| `is_default` | boolean | 是否为当前全局默认数据版本 |
| `status` | string | 已登记版本为`valid`；校验失败的上传不登记成可用版本 |
| `first_month` | string | 三表共同首月 |
| `latest_observed_month` | string | 三表共同末月 |
| `months` | integer | 每地区月度数据行数，不含字段行和单位行 |
| `regions` | array[DataRegion] | 三地区各自的表格统计 |
| `warnings` | array[Issue] | 不阻止入库的提示，例如非目标字段为空 |
| `compatible_models` | array[ModelCompatibility] | 已登记模型在本数据上的点预测兼容性；与是否具有历史评价证据分开判断 |

**DataRegion对象：**

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `region` | string | 地区编码 |
| `region_name` | string | 地区中文名 |
| `filename` | string | 用户原始文件名，仅展示，不是服务器路径 |
| `months` | integer | 数据行数 |
| `first_month` | string | 首月 |
| `latest_observed_month` | string | 末月 |
| `missing_counts` | object | 业务字段名到空值数量的映射，覆盖24个业务字段 |
| `all_empty_columns` | array[string] | 整个上传历史均为空的列名；不是根据它直接决定训练输入 |

**Issue对象：**

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `code` | string | 稳定的提示或错误编码 |
| `region` | string或null | 涉及地区；三表共同问题可为null |
| `month` | string或null | 涉及月份；非单月问题为null |
| `field` | string或null | 业务列名或请求字段位置 |
| `message` | string | 可直接展示的中文说明 |
| `suggestion` | string或null | 建议的修复方式 |

**ModelCompatibility对象：**

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `model_version` | string | 被检查的模型版本 |
| `profile` | string | 模型变量方案 |
| `can_predict` | boolean | 数据范围、输入列和数值能否满足该权重推理要求 |
| `reasons` | array[Issue] | 不兼容的具体原因，兼容时为空 |

校验内容包括：单工作表、25列顺序与单位行、月份合法且连续升序无重复、三地月份一致、全国共同指标一致、数值有限，以及目标价格完整且为正。非目标字段允许空值，其插补参数在后续训练中仅使用训练历史拟合。文件校验通过后，仍须按所选模型检查输入兼容性及评价证据的适用范围。

校验失败返回422和具体Issue；文件过大返回413，格式不支持返回415。失败时不生成可选数据版本，不修改默认数据或原始三表。服务直接比较三份文件的内容，相同内容重新上传返回原有版本。

Linux调用示例，命令从项目根目录执行；`curl -F`自行生成boundary，不另写multipart的Content-Type：

```bash
curl -fsS -X POST http://127.0.0.1:8000/datasets \
  -H "X-API-Key: $LNG_API_KEY" \
  -F 'shanghai_file=@data/上海_处理后数据表.xlsx' \
  -F 'zhejiang_file=@data/浙江_处理后数据表.xlsx' \
  -F 'jiangsu_file=@data/江苏_处理后数据表.xlsx' \
  -F 'name=三地区截至2026年5月' \
  -F 'note=处理后表上传示例'
```

### 4.3 GET /datasets

无请求体，可选查询参数如下：

| 查询字段 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `source` | string | 不过滤 | 按来源过滤：configured配置导入或upload上传 |
| `is_default` | boolean | 不过滤 | true仅返回当前默认数据，false返回其他版本 |

成功200返回`items: array[DataVersion]`，包含符合条件的全部数据版本。模型兼容性按查询时的登记清单计算，查询不修改原文件。

### 4.4 GET /datasets/{data_version}

无请求体，成功200直接返回DataVersion。版本不存在时返回404。用于查询三地数据范围、缺失情况及模型兼容性。

### 4.5 POST /datasets/{data_version}/activate

将指定版本设为全局默认数据。后续省略数据版本的请求使用此默认值；已提交任务和已保存结果继续使用原绑定版本。单次请求可直接传入`data_version`，无需先修改全局默认值。

| 请求字段 | 类型 | 必填及默认 | 含义 |
| --- | --- | --- | --- |
| `expected_current_data_version` | string或null | 可选 | 当前默认版本的前置条件；与实际默认版本不一致时返回409 |

无附加条件可提交`{}`。成功200：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `data_version` | string | 新默认数据版本 |
| `previous_data_version` | string或null | 原默认版本 |
| `changed` | boolean | 是否实际改变默认版本；重复设为同版本为false |
| `activated_at` | string | 最近一次实际切换时间；重复启用同一版本时保留原值 |
| `compatible_models` | array[ModelCompatibility] | 对各已登记权重的兼容性 |

切换前检查默认组合变量模型能否使用该数据；不兼容时返回409，保留原默认关系。需要同时更换数据与模型时，使用模型启用接口统一提交。切换仅更新服务登记信息，配置中的原始数据文件保持原样。

## 5. 模型管理接口

### 5.1 GET /models

返回所有登记版本的列表，保留顶层`models`数组。

| 查询字段 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `profile` | string | 不过滤 | `all`或`selected` |
| `is_default` | boolean | 不过滤 | 是否为该方案默认模型 |
| `data_version` | string | 当前默认数据 | 按这份数据计算推理兼容性及评价可用性 |

成功200，返回`models: array[ModelRecord]`，包含符合条件的全部模型版本。兼容性及评价可用性按查询选定的数据计算；查询操作不改变全局默认关系。

### 5.2 GET /models/{model_version}

可选查询参数`data_version`，默认当前默认数据；无请求体。成功200返回ModelRecord，不存在404。

ModelRecord包含[ModelStatus全部字段](#93-模型状态modelstatus)，并增加：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `display_name` | string | 自动生成的中文名称，如“组合变量｜训练截至2026-05｜2026-10-18 训练” |
| `source` | string | `bundled`为交付包预置权重；`configured`为维护人员通过服务配置导入的其他权重；`training`为训练接口生成的权重 |
| `training_id` | string或null | 产生该权重的训练任务；随包版本为null |
| `tuning_id` | string或null | 参数来源调参任务；使用随包配置为null |
| `training_data_version` | string或null | 实际拟合权重使用的数据版本；旧权重无该登记编号时为null，训练范围与来源记录继续保留 |
| `is_default` | boolean | 是否为该方案当前默认权重 |
| `load_state` | string | 模型在服务中的加载状态：`not_loaded`、`loading`、`ready`或`error`，各状态含义见下表 |
| `parameters` | ModelParameters | 28项实际模型与训练设置，见调参字段表 |
| `training_plan` | TrainingPlan | 实际训练轮数及学习率轨迹 |
| `compatibility` | ModelCompatibility | 当前所选数据与该模型的点预测兼容性 |
| `evaluation` | Evaluation | 对应历史评价对象，见公共指标结构 |

**load_state模型加载状态：**

| 状态值 | 中文含义 | 说明 |
| --- | --- | --- |
| `not_loaded` | 未加载 | 模型已登记，但权重尚未加载到CPU内存或GPU显存；需要使用时先加载 |
| `loading` | 加载中 | 正在读取权重、构建模型，并加载到指定的CPU或GPU设备 |
| `ready` | 已加载 | 模型加载成功，可供推理调用；实际预测仍须通过所选数据的兼容性检查 |
| `error` | 加载失败 | 权重读取、模型或预处理参数检查、设备内存分配等步骤失败；排查后可重新尝试加载 |

模型登记保存权重位置、版本及参数信息；模型加载读取权重并占用CPU内存或GPU显存。登记成功后可保持`not_loaded`，预测时再加载，减少设备内存占用。该过程使用已有权重。加载失败时返回错误，本次预测结果不保存。

模型版本身份及训练范围不随查询数据改变；`latest_observed_month`、`data_version`、`data_imported_at`和历史参考匹配状态描述本次查询选择的数据。模型列表不强制把全部权重都加载到GPU。

### 5.3 POST /models/{model_version}/activate

将指定版本热加载并设为其变量方案的默认模型。请求体：

| 请求字段 | 类型 | 必填及默认 | 含义 |
| --- | --- | --- | --- |
| `data_version` | string或null | 可选，当前默认 | 与新模型一起使用的数据；显式填写时同时将它设为默认数据 |
| `expected_current_model_version` | string或null | 可选 | 预期的该方案旧默认模型，已变化则409 |
| `expected_current_data_version` | string或null | 可选 | 当前默认数据版本的前置条件；提供后即检查，不一致时返回409 |

成功200：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `model_version` | string | 新默认权重版本 |
| `profile` | string | 变更的变量方案 |
| `previous_model_version` | string或null | 该方案原默认权重 |
| `data_version` | string | 切换后默认数据版本 |
| `previous_data_version` | string或null | 切换前默认数据版本 |
| `changed` | boolean | 是否实际发生默认关系变化 |
| `activated_at` | string | 最近一次实际切换时间 |
| `warnings` | array[Issue] | 启用成功后的提示；当前可能返回`NO_CALIBRATION`，表示缺少匹配的价格区间校准证据 |

服务先检查权重、预处理参数、输入布局和选定数据，再加载并执行推理验证；全部成功后提交默认关系变更。显式同时更换模型和数据时统一提交，校验或加载失败返回422或503并保留原默认关系。

切换后必须保证默认预测方案selected对应的模型与默认数据仍可用；否则409并保留旧关系。需要更换到旧组合模型无法使用的数据时，可先一次启用兼容的新selected模型和新数据，再调整all方案，避免默认预测入口失效。

启用无需重启服务，加载期间可能产生等待。每次预测固定其模型和数据引用，后续切换不改变该次计算。缺少历史测试或校准证据时仍可启用有效的点预测，相关指标和区间通过`available=false`及`reason`说明。另一变量方案的兼容性可通过模型查询接口确认。

回退时调用本接口启用已登记的旧版本。单次预测可直接指定`model_version`，无需改变全局默认模型。

## 6. 训练接口

### 6.1 POST /trainings

一次任务使用一个变量方案，训练一份三地区融合权重。全变量与组合变量须分别提交。训练按已确定的参数、轮数及学习率轨迹执行；结构搜索及早停选轮数由调参任务完成。

| 请求字段 | 类型 | 必填及默认 | 含义 |
| --- | --- | --- | --- |
| `profile` | string | 可选 | `all`或`selected`；指定tuning_id时从该调参任务推断，未指定时默认selected；同时明确填写时须与调参任务一致 |
| `data_version` | string或null | 可选 | 指定tuning_id时默认沿用该调参任务绑定的数据版本，否则使用当前默认数据；接受任务后固定 |
| `name` | string或null | 可选，自动生成 | 任务显示名称，最多100字符 |
| `mode` | string | 可选，`holdout` | `holdout`自动留出、`full_history`全部历史、`custom`指定边界 |
| `holdout_months` | integer | 仅holdout可用，默认3 | 末尾留出的测试月份数，范围3至2400；还须满足可用历史长度要求 |
| `training_cutoff` | string | custom必填，其他模式不传 | 指定训练截止月，YYYY-MM |
| `test_end_month` | string | custom可选，默认数据末月 | 指定测试末月，不晚于数据末月；须在训练之后至少保留3个月 |
| `tuning_id` | string或null | 可选，null | 使用成功调参任务的推荐结构和冻结训练计划；省略则用对应方案的随包最终配置 |
| `epochs` | integer | 可选，随配置，当前24 | 覆盖训练轮数，范围1至500；引用tuning_id时不可同时提供 |
| `learning_rate` | number | 可选，随配置 | 覆盖为固定学习率，须大于0且不超过0.01；引用tuning_id时不可同时提供 |

设备由服务启动配置确定，变量清单来自对应方案配置或调参结果。请求字段以上表为准，训练过程沿用既定特征选择结果。

**日期与模式规则：**

| mode | 训练截止月 | 测试范围 | 可提供的日期字段 |
| --- | --- | --- | --- |
| `holdout` | 数据末月减holdout_months | 截止月次月至数据末月 | holdout_months |
| `full_history` | 数据末月 | 无留出测试 | 无 |
| `custom` | training_cutoff | 截止月次月至test_end_month | training_cutoff、test_end_month |

模式专属字段交叉传入返回422。实际历史首月取方案配置起始月与数据首月中的较晚者。训练范围须至少包含`seq_len + 3`个月，并满足输入变量有效性及参数选择截止要求；留出测试须位于最终选参截止月之后。

引用`tuning_id`时，调参任务须为`succeeded`，变量方案一致，训练截止月不早于其选参截止月。训练数据在全部选参历史段的数值和单位须与调参时一致，允许其后追加新月份。若修订选参历史，须重新调参。该模式完整使用冻结的轮数和逐轮学习率，不能同时提供`epochs`或`learning_rate`覆盖计划。

按推荐方案自动训练时，提交`tuning_id`及所需`mode`即可。服务从调参记录取得变量方案、绑定数据、模型参数及冻结训练计划；`mode`省略时为`holdout`。显式提供`data_version`时优先使用该版本并校验历史一致性；否则依次采用调参绑定版本、当前默认数据。接受后固定解析结果，后续默认关系变更不影响任务。

未引用调参任务时，仅传`epochs`将截取源学习率轨迹或重复最后学习率以满足新轮数；传`learning_rate`将生成覆盖实际轮数的固定轨迹；两者均省略时沿用源计划。解析后的参数及轨迹通过任务详情返回。

简单留出测试：

```json
{
  "profile": "selected",
  "data_version": "data_11111111111111111111111111111111",
  "mode": "holdout",
  "holdout_months": 3
}
```

指定截至2026-02，用3、4、5月检验：

```json
{
  "profile": "selected",
  "mode": "custom",
  "training_cutoff": "2026-02",
  "test_end_month": "2026-05",
  "epochs": 24
}
```

按调参结果自动训练并留出最近3个月检验：

```json
{
  "tuning_id": "tune_22222222222222222222222222222222",
  "mode": "holdout"
}
```

若所选调参任务绑定2015-01至2026-05的数据，正式训练会使用2015-01至2026-02的全部历史重新拟合权重和预处理，再预测2026-03、2026-04、2026-05并与真实价格比较。调参时的初筛训练/验证划分只用于选参数，不会要求正式训练仍停在初筛训练截止月。数据版本、结构、变量顺序和冻结训练计划随任务保存。

按调参结果使用全部历史训练：

```json
{
  "mode": "full_history",
  "tuning_id": "tune_22222222222222222222222222222222"
}
```

上述相同数据采用full_history时，正式训练使用2015-01至2026-05的全部历史，生成用于未来预测的新模型版本；本次没有留出测试，不生成本次测试MAPE。训练完成后，再由预测接口选择该模型版本执行未来3个月预测。

提交时检查数据、参数及日期边界。成功返回202及TrainingTask，初始`status=queued`、`stage=queued`、`result=null`。响应已包含解析后的实际训练与测试月份。202表示任务已接受，执行结果通过任务查询接口获取。

### 6.2 公共任务字段

TrainingTask与TuningTask均包含以下字段，各自补充字段分别见6.3与7.3。提交响应、列表元素及详情响应使用相同任务结构。

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `kind` | string | `training`或`tuning` |
| `name` | string | 中文任务名称 |
| `profile` | string | 变量方案 |
| `model_name` | string | 三地区融合EMAformer |
| `data_version` | string | 接受时绑定的数据版本，后续切换默认数据不影响它 |
| `device` | string | 服务器给本任务安排的实际设备，如cpu、cuda |
| `status` | string | 见任务状态表 |
| `stage` | string | 当前或最后执行阶段，见下表；终态由status判断 |
| `submitted_at` | string | 接受时间 |
| `started_at` | string或null | 开始执行时间，排队时null |
| `finished_at` | string或null | 进入终态时间，运行时null |
| `elapsed_seconds` | number | 实际执行耗时，不含排队；尚未开始为0 |
| `queue_position` | integer或null | 排队时从1开始的位置；非排队状态为null |
| `progress` | TaskProgress | 当前阶段进度 |
| `warnings` | array[Issue] | 可展示的提示 |
| `error` | TaskError或null | 失败、中断等原因；正常时null |
| `status_url` | string | 本任务GET详情相对URL，不是服务器文件路径 |

TaskProgress：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `unit` | string | 当前工作单位：`epoch`轮、`fit`一次拟合、`origin`预测起点或`step`其他阶段步骤 |
| `completed_units` | integer | 当前阶段已完成单位数；调参已结束的失败拟合也计入进度，但不参与优选 |
| `total_units` | integer或null | 当前阶段总数，尚未确定时null |
| `phase_percent` | number或null | 当前阶段完成比例0至100；不是整个任务的耗时百分比，切换阶段可重新从0开始 |
| `current_epoch` | integer或null | 正在执行的候选或训练的当前轮次；没有轮次概念时null |
| `max_epochs` | integer或null | 本轮训练上限或固定轮数 |

TaskError：`code:string`为错误编码，`message:string`为中文说明，`retryable:boolean`表示排除问题后是否适合重新提交任务，详见10.3。

| status | 中文状态 | 说明 |
| --- | --- | --- |
| `queued` | 排队中 | 已接受但未开始 |
| `running` | 执行中 | 结合stage展示具体阶段 |
| `cancel_requested` | 正在停止 | 已收到停止请求，等待安全检查点 |
| `succeeded` | 已完成 | 全部必需阶段成功，结果已持久化 |
| `failed` | 失败 | 执行异常或有效候选不足 |
| `cancelled` | 已取消 | 未执行完的任务已停止 |
| `interrupted` | 已中断 | 服务/工作进程异常或达到时限等，不能当成功 |

| stage | 中文含义 |
| --- | --- |
| `queued` | 等待执行 |
| `preparing` | 加载冻结配置及数据，准备训练资源 |
| `training` | 拟合正式训练权重 |
| `evaluating` | 计算留出测试结果 |
| `registering` | 保存并登记模型及评价信息 |
| `screening` | 对调参候选进行初筛 |
| `confirming` | 对入选候选进行多折复核 |
| `finalizing` | 汇总排名及冻结推荐训练计划 |
| `interrupted` | 历史中断记录可能保留的阶段标识 |

任务持久化后返回202，由独立工作进程执行；客户端断开不影响任务。当前单机同时执行一个训练或调参任务，其余排队，默认队列容量为20。运行状态和阶段分别读取`status`与`stage`，阶段百分比不能作为整体耗时进度。

服务启动时保留排队任务，并核查运行任务的工作进程。无法继续执行的任务标为`interrupted`，完成记录继续可查。API未提供断点续训操作；需要重新执行时另行提交，命令行搜索的恢复能力见相应模块说明。

### 6.3 TrainingTask补充字段

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `training_id` | string | 训练任务唯一编号 |
| `request` | TrainingRequest | 原始合法业务请求，字段与6.1一致，不含HTTP头；解析后的默认值见同对象其他实际配置字段 |
| `resolved_split` | DateSplit | 实际训练、验证与测试边界 |
| `parameters` | ModelParameters | 实际使用的28项设置 |
| `training_plan` | TrainingPlan | 实际固定训练计划 |
| `tuning_id` | string或null | 推荐配置来源；未使用为null |
| `result` | TrainingResult或null | 成功且登记完成后返回，否则null |

DateSplit：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `first_month` | string | 实际使用历史首月 |
| `training_cutoff` | string | 训练任务的权重拟合截止月；调参任务中表示初筛拟合截止月 |
| `validation_start` | string或null | 本任务用于选择参数的验证首月；固定计划训练为null |
| `validation_end` | string或null | 验证末月；固定计划训练为null |
| `test_start` | string或null | 留出测试首月，无测试时null |
| `test_end` | string或null | 留出测试末月，无测试时null |
| `latest_observed_month` | string | 所选完整数据版本末月 |
| `pred_len` | integer | 固定3 |

TrainingPlan：`epochs:integer`为实际固定训练轮数；`lr_schedule:array[number]`为逐轮正学习率，长度等于epochs。使用搜索结果时按冻结轨迹重放。ModelParameters中的max_epochs、patience用于搜索阶段，不替代此处的最终固定计划。

TrainingResult：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `model_version` | string | 已成功登记的新权重版本，后续用它推理或设为默认 |
| `display_name` | string | 模型中文显示名称 |
| `registered_at` | string | 权重登记时间 |
| `trained_at` | string | 此次实际训练完成时间 |
| `model_training_cutoff` | string | 实际拟合截止月 |
| `parameter_selection_cutoff` | string | 当前配置及变量方案的选参截止月份，独立于最终权重拟合截止月 |
| `epochs_completed` | integer | 完成的训练轮数 |
| `evaluation` | Evaluation | 留出测试及匹配的滚动参考；具体结构见第9节 |
| `test_csv_url` | string或null | 存在留出测试时的下载URL，否则null |
| `model_url` | string | 该版本GET模型详情URL |

训练成功后保存权重、预处理参数、实际配置、训练历史和审计记录，并登记新模型版本。全局默认模型须通过启用接口另行设置。无完整测试标签时，evaluation对应项`available=false`并给出原因。留出测试采用固定新权重，逐起点读取历史输入；完整滚动回测则在各起点重新拟合权重，两类结果分别报告。

有H个月完整留出标签时，完整三步测试起点数为H-3+1，每地区每步各有这么多组配对；三步合计乘3。标签和未来输入不能进入权重、插补或标准化的拟合。某个后续测试起点可以使用当时已经观测到的前一个测试月输入，这仍是固定权重的历史滚动输入评价。

### 6.4 GET /trainings

无请求体，可选查询参数如下：

| 查询字段 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `profile` | string | 不过滤 | all或selected |
| `status` | string | 不过滤 | 按6.2节的任务状态过滤 |
| `data_version` | string | 不过滤 | 训练绑定的数据版本 |
| `tuning_id` | string | 不过滤 | 训练引用的调参任务编号 |

成功200返回`items: array[TrainingTask]`，包含符合条件的全部训练任务；没有符合条件的记录时返回空数组。

### 6.5 GET /trainings/{training_id}

无请求体，成功200返回TrainingTask，不存在404。运行中result为null；终态失败通过error说明，GET本身仍为200，表示成功查询到了一个失败任务。

### 6.6 GET /trainings/{training_id}/csv

无请求体，成功200返回带BOM的UTF-8 CSV，`Content-Disposition`文件名为`<training_id>.csv`。任务未成功或没有测试明细时返回409及具体原因。

| CSV列 | 类型 | 含义 |
| --- | --- | --- |
| `training_id` | 文本 | 训练任务编号 |
| `model_version` | 文本 | 本次测试对应的新权重 |
| `data_version` | 文本 | 测试使用的数据版本 |
| `profile` | 文本 | 变量方案 |
| `region` | 文本 | 地区编码 |
| `region_name` | 文本 | 地区中文名 |
| `history_cutoff` | YYYY-MM | 此预测起点可用历史截止月，不代表模型每起点重新拟合 |
| `first_forecast_month` | YYYY-MM | 此起点预测第1个月的目标月 |
| `target_month` | YYYY-MM | 本条目标月 |
| `horizon` | 整数 | 第1、2或3个月 |
| `actual` | 数值 | 真实价格，元/立方米 |
| `prediction` | 数值 | 预测价格，元/立方米 |
| `absolute_percentage_error_percent` | 数值 | 本条绝对百分比误差；MAPE是这些误差的平均值 |

### 6.7 POST /trainings/{training_id}/cancel

请求体为`{}`。成功响应字段如下：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `training_id` | string | 请求取消的训练任务编号 |
| `status` | string | 处理后的任务状态，通常为cancel_requested或cancelled |
| `requested_at` | string | 取消请求登记时间；已取消任务重复调用时保留原值 |
| `message` | string | 取消处理结果说明 |

排队任务直接转为`cancelled`，返回202；运行任务先转为`cancel_requested`，返回202，在下一安全检查点停止。停止可能需要等待当前计算步骤结束。已为`cancelled`时重复调用返回200；`succeeded`、`failed`或`interrupted`任务返回409。取消操作保留已有任务及结果记录。

## 7. 调参接口

### 7.1 POST /tuning-jobs

调参任务输出推荐参数、候选排名及冻结训练计划。服务复用[tune.py](../joint_emaformer/tune.py)的候选生成和训练算法，搜索范围读取[服务端搜索配置](../joint_emaformer/configs/search.yaml)。正式权重由后续训练任务生成；保留测试成绩不参与本次搜索排名。

| 请求字段 | 类型 | 必填及默认 | 含义 |
| --- | --- | --- | --- |
| `profile` | string | 可选，selected | all或selected，一次只搜索一个变量方案 |
| `data_version` | string或null | 可选，当前默认 | 接受时绑定的数据版本 |
| `name` | string或null | 可选，自动生成 | 任务显示名称，最多100字符 |
| `preset` | string | 可选，standard | quick快速检查、standard标准、thorough较大预算；见下表 |
| `split_mode` | string | 可选，auto | auto自动边界，custom手动边界 |
| `holdout_months` | integer | 仅auto，默认3 | 末尾保留的测试月份数，范围3至2400 |
| `validation_months` | integer | 仅auto，默认20 | 初筛验证标签区间长度，范围3至2400个月 |
| `train_end_month` | string | custom必填 | 初筛训练截止月 |
| `validation_end_month` | string | custom必填 | 开发期末月，即本次参数选择截止月 |
| `test_end_month` | string | custom可选，数据末月 | 保留测试末月；须在开发期后至少3个月 |
| `search_trials` | integer | 可选，随preset | 包括初始候选在内的候选总数，范围1至256 |
| `finalists` | integer | 可选，随preset | 初筛入选复核的候选数，范围1至16，且不大于search_trials |
| `validation_folds` | integer | 可选，随preset | 开发期内复核折数，范围1至5 |
| `fold_months` | integer | 可选，随preset | 每折验证标签区间长度，范围3至60个月 |
| `include_initial_candidates` | boolean | 可选，true | 是否优先评估现有8组初始配置 |
| `search_space` | object或null | 可选，随包范围 | 覆盖搜索范围中的指定字段；未给字段沿用默认范围 |
| `max_duration_minutes` | integer或null | 可选，服务端配置 | 执行时限，范围1至10080分钟；最终解析值为null时不设时限。自开始执行计时，达到时限后在安全检查点中断并保留已完成记录 |

preset预算如下；standard读取服务端search_config指定的搜索配置，quick和thorough为内置预算，搜索范围仍读取同一配置：

| preset | 初筛候选数 | 入选复核数 | 折数 | 每折月份 | 每折重复次数 | 计划拟合次数 |
| --- | --- | --- | --- | --- | --- | --- |
| `quick` | 16 | 2 | 2 | 12 | 3 | 28 |
| `standard` | 64 | 4 | 3 | 20 | 3 | 100 |
| `thorough` | 128 | 8 | 3 | 20 | 3 | 200 |

计划拟合次数为`search_trials + finalists × validation_folds × replications_per_fold`。例如标准预算为64次初筛加36次复核，共100次拟合。内部重复使用固定随机性设置，失败拟合也计入预算。全变量与组合变量须分别提交。

auto模式：`validation_end=数据末月-holdout_months`，`train_end=validation_end-validation_months`，测试从开发期次月开始。custom模式须满足训练末月早于验证末月、验证末月早于测试首月，且各评分段至少能构造完整3个月标签。每个复核折都位于开发期内，折的训练范围仅截至其验证首月之前；检查最长窗口及最早复核训练历史是否足够。模式专属字段混传返回422。

**自动日期划分与独立任务配置：**

日常自动调参只需选择数据版本和变量方案，其余请求字段可以省略。搜索范围、候选数、复核折数及训练设置读取服务端搜索配置，日期根据本次数据版本的实际末月计算，不要求用户手动填写参数或计算日期。留出测试默认3个月、初筛验证默认20个月，由服务端策略统一管理；下表示例采用这组默认规则。

| 调参阶段 | 数据截至2026-05时的范围 | 数据更新至2026-06时的范围 | 用途 |
| --- | --- | --- | --- |
| 初筛训练 | 2015-01至2024-06 | 2015-01至2024-07 | 拟合各候选模型 |
| 初筛验证 | 2024-07至2026-02 | 2024-08至2026-03 | 选择参数和训练轮数 |
| 保留测试 | 2026-03至2026-05 | 2026-04至2026-06 | 不参与选参，供正式训练后检验效果 |

表中假设历史首月为2015-01，实际首月以本次采用的数据范围和服务端策略为准。复核仍按搜索配置，在开发期内自动构造各折，不只重复同一初筛验证段。数据不足以覆盖最长候选窗口和全部复核折时，提交前返回具体原因，不读取测试月份补足历史。

服务接受任务时生成独立配置，固定本次数据版本、变量清单、搜索范围、实际日期边界及服务程序版本，并保存为12.1节的`effective_config.json`。执行始终使用该配置；排队期间上传数据、切换默认版本或修改服务端配置不影响已提交任务。此过程由服务完成，项目公共YAML继续用于原实验复现。

命令行tune.py的初筛训练末月和开发期末月继续读取公共配置中的period.train_end与period.validation_end，以便复现原实验。自动接口的执行适配层使用本任务计算出的月份边界和数据快照调用同一算法，独立记录配置。参数搜索、插补、标准化、早停和候选排名只使用开发期；原表中已经存在的保留测试真实值也不参与选参。

拟合、插补、标准化、早停和候选排名仅使用截至`validation_end`的开发历史。全变量有效列由开发期训练历史确定，并冻结为所有候选共用布局。组合变量使用既定清单，不重新运行XGBoost；复核折早于该组合选择时间时，其结果属于固定事后组合下的回顾验证。

组合变量方案的开发期末月不得早于配置记录的变量选择截止月。较早月份可用于开发期内的回顾复核，选参后的留出测试须单独报告。

最简请求：

```json
{
  "profile": "selected",
  "data_version": "data_11111111111111111111111111111111"
}
```

接受后取得`tuning_id`，通过查询接口跟踪任务。任务成功后，按6.1节示例引用该编号提交训练，无需重复提供技术参数。实际时间范围、搜索预算、推荐参数、冻结训练计划及评价结果均随记录保存。

限定部分结构与训练范围：

```json
{
  "profile": "selected",
  "preset": "standard",
  "split_mode": "custom",
  "train_end_month": "2024-06",
  "validation_end_month": "2026-02",
  "test_end_month": "2026-05",
  "search_trials": 32,
  "search_space": {
    "seq_len": [6, 12],
    "d_model": [128, 256],
    "n_heads": [2, 4],
    "e_layers": [1, 2],
    "lr_log10": [-5.0, -3.0]
  }
}
```

成功202返回TuningTask；校验失败返回422，不进入执行队列。达到执行时限后，状态为`interrupted`、错误编码为`TIME_LIMIT`。已完成候选仍可查询，但最终推荐不可用；训练接口仅接受成功调参任务的编号。

### 7.2 搜索范围与ModelParameters

`search_space`除`lr_log10`外均接收非空候选数组，元素类型见下表；未知键返回422。`lr_log10`接收两个递增有限数，实际学习率按对应的10次幂区间对数均匀抽样。返回的ModelParameters共28项，每项为已确定的单值；其中`lr:number`替代`lr_log10`，其余27项名称与下表一致。

| 字段 | 元素类型 | 默认候选或范围 | 含义 |
| --- | --- | --- | --- |
| `seq_len` | integer | 6、12、18、24、36 | 历史窗口月数 |
| `d_model` | integer | 32、64、128、256、512 | 编码隐藏维度 |
| `n_heads` | integer | 1、2、4、8 | 注意力头数 |
| `e_layers` | integer | 1、2、3、4 | 编码层数 |
| `ff_multiplier` | integer | 2、4、8 | 前馈宽度=d_model乘该倍数 |
| `activation` | string | gelu、relu | 激活函数 |
| `norm_first` | boolean | false、true | 是否前置层归一化 |
| `window_norm` | boolean | true、false | 是否使用窗口内归一化 |
| `channel_embedding` | boolean | true、false | 是否使用变量通道嵌入 |
| `cycle` | integer | 0、12、24、48 | 周期嵌入长度，0关闭 |
| `dropout` | number | 0、0.05、0.1、0.2 | 编码网络Dropout |
| `embedding_dropout` | number | 0、0.05、0.1、0.2 | 输入嵌入Dropout |
| `attention_dropout` | number | 0、0.05、0.1、0.2 | 注意力Dropout |
| `head_dropout` | number | 0、0.05、0.1、0.2 | 预测头Dropout |
| `head_layers` | integer | 1、2、3 | 预测头线性层数 |
| `head_multiplier` | integer | 1、2 | 预测头隐藏层宽度的逐层扩展倍数 |
| `output_mode` | string | level、delta | 价格水平输出或最后价格加修正量 |
| `residual_scale` | number | 0.1、0.25、0.5、1 | delta模式修正量比例 |
| `missing` | string | linear_history、ffill | 仅历史范围插值或前向填充，剩余缺失用训练历史中位数 |
| `loss` | string | mse、smooth_l1、mae、mape | 训练损失 |
| `optimizer` | string | Adam、AdamW | 优化器 |
| `batch_size` | integer | 8、16、32、64 | 批大小 |
| `lr_log10` | number区间 | -5至-2.5 | 实际lr约0.00001至0.003162；结果对象返回lr单值 |
| `weight_decay` | number | 0、0.000001、0.00001、0.0001、0.001、0.01 | 权重衰减 |
| `grad_clip` | number | 0、0.5、1、5 | 梯度裁剪阈值，0关闭 |
| `scheduler` | string | none、cosine、plateau | 学习率调整策略 |
| `max_epochs` | integer | 24、48、96 | 搜索训练上限；原初始候选另包含4轮 |
| `patience` | integer | 6、10、15 | 验证未改善的早停耐心；原初始候选另包含4 |

候选须满足隐藏维度能被头数整除、Dropout位于[0,1)、学习率为正及服务端规模限制。`level`模式的`residual_scale`规范为1；单层预测头的`head_multiplier`规范为1。规范后等效的参数组合按同一候选处理。

默认优先纳入8组初始候选，其中包含4轮配置。覆盖搜索字段后，服务剔除不符合该字段约束的初始候选；其余字段保留预设值。不同有效候选不足`search_trials`时返回422。候选总数、网络规模及历史长度均在提交时检查。

### 7.3 TuningTask补充字段

除公共任务字段外：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `tuning_id` | string | 调参任务编号 |
| `request` | TuningRequest | 7.1的原始合法业务请求；解析后的默认预算和边界见resolved_search与resolved_split |
| `resolved_split` | DateSplit | 实际开发、保留测试边界 |
| `resolved_search` | TuningPlan | 实际搜索预算、范围和复核折 |
| `best_so_far` | Candidate或null | 当前最好已完成候选；不能当作最终推荐 |
| `result_available` | boolean | 最终推荐是否已形成；只有succeeded后为true |
| `results_url` | string | 结果及排名GET相对URL |

TuningPlan：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `search_trials` | integer | 初筛候选总数 |
| `finalists` | integer | 复核名额 |
| `validation_folds` | integer | 折数 |
| `fold_months` | integer | 每折验证月份数 |
| `replications_per_fold` | integer | 每折内部随机性复核次数，当前3；不公开种子字段 |
| `include_initial_candidates` | boolean | 是否纳入有效初始候选 |
| `initial_candidates_count` | integer | 实际纳入的预设数量 |
| `search_space` | object | 实际搜索范围，键及数组类型与7.2一致 |
| `max_duration_minutes` | integer或null | 实际执行时限 |
| `planned_fits` | integer | 初筛加复核的计划拟合次数，不是epoch数 |
| `folds` | array[ValidationFold] | 各折实际时间边界 |

ValidationFold含`fold:integer`（从1开始）、`training_cutoff:string`、`validation_start:string`、`validation_end:string`。没有随机打乱月份；有20个月验证标签时每地区每步最多18个完整三步起点。

### 7.4 GET /tuning-jobs

无请求体，可选查询参数如下：

| 查询字段 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `profile` | string | 不过滤 | all或selected |
| `status` | string | 不过滤 | 按6.2节的任务状态过滤 |
| `data_version` | string | 不过滤 | 调参绑定的数据版本 |

成功200返回`items: array[TuningTask]`，包含符合条件的全部调参任务；没有符合条件的记录时返回空数组。

### 7.5 GET /tuning-jobs/{tuning_id}

无请求体，成功200返回TuningTask，编号不存在时返回404。调用方可轮询此接口获取进度。个别候选失败会记录在候选详情中；有效初筛或完整复核候选不足时，任务标为`failed`并返回原因。

### 7.6 GET /tuning-jobs/{tuning_id}/results

可选查询`phase=screening|confirmation`，默认confirmation。无请求体，200返回以下对象，包含指定阶段的全部候选；任务尚未成功时仍可查已完成候选，最终推荐字段为null。

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `tuning_id` | string | 调参任务编号 |
| `status` | string | 当前任务状态 |
| `result_available` | boolean | 是否已有最终推荐 |
| `profile` | string | 变量方案 |
| `data_version` | string | 数据版本 |
| `selection_cutoff` | string | 开发期末月 |
| `objective` | string | 固定`mean_validation_mape_percent`，越低越好 |
| `recommended_trial` | integer或null | 最终选中的候选编号，任务成功后才有值 |
| `recommended_parameters` | ModelParameters或null | 推荐的完整28项设置 |
| `frozen_training_plan` | TrainingPlan或null | 最终冻结的轮数和学习率轨迹 |
| `selection_rule` | string | 选择规则文字说明，见下文 |
| `phase` | string | 返回候选属于初筛或复核，不将两者分数混排 |
| `items` | array[Candidate] | 指定阶段的候选排名 |

Candidate：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `trial` | integer | 本任务内候选编号 |
| `rank` | integer或null | 同阶段已完成有效候选排名；失败或未完成为null |
| `phase` | string | screening或confirmation |
| `status` | string | complete、running、failed或partial |
| `parameters` | ModelParameters | 本组完整参数 |
| `score_mape_percent` | number或null | 本阶段三地区等权验证MAPE；复核还对折与内部重复等权平均 |
| `regions` | array[RegionMetric] | 分地区、分预测步验证指标；复核对各次拟合的指标等权平均 |
| `fits_completed` | integer | 已结束拟合次数 |
| `fits_planned` | integer | 本候选本阶段计划拟合次数 |
| `best_epoch` | integer或null | 初筛最佳轮数；多折复核不以一个数替代全部轮数，返回null |
| `elapsed_seconds` | number | 本阶段累计执行秒数 |
| `error` | TaskError或null | 失败说明 |

候选状态`complete`表示本阶段所需拟合全部完成；`running`表示正在拟合；`failed`表示拟合失败；`partial`表示中断或取消后仅保留部分拟合结果。

复核RegionMetric的`origins`与各步`observations`为各次验证预测的累计配对数，包含重叠历史及重复训练，不表示独立样本数。MAPE按各次完整拟合的分步指标等权汇总，展示时应注明“折与重复等权验证”。仅完成全部复核的候选参与最终推荐。

选择规则：初筛按三地区等权平均验证MAPE排序；复核对地区、折及内部重复等权汇总，并列时依次比较最差地区MAPE、候选编号。最终轮数取最后一折三次最佳轮数的中位数，并冻结对应学习率轨迹。推荐结果表示本次搜索预算内的优选配置。正式训练须调用`POST /trainings`并引用`tuning_id`。

### 7.7 POST /tuning-jobs/{tuning_id}/cancel

请求体为`{}`。响应字段与6.7节相同，任务编号字段替换为`tuning_id:string`。排队或运行任务的有效取消请求返回202，已取消任务重复调用返回200，其他终态返回409。停止后保留已完成候选及错误记录，最终推荐保持不可用。

## 8. 预测接口

### 8.1 POST /predictions

JSON请求，至少提供`{}`。请求字段如下：

| 请求字段 | 类型 | 必填及默认 | 含义 |
| --- | --- | --- | --- |
| `data_version` | string或null | 可选，当前默认数据 | 使用已登记的数据快照，不接收本地/服务器路径 |
| `model_version` | string或null | 可选 | 显式选一个已登记模型；未加载时按受控缓存策略加载 |
| `profile` | string | 可选，未选模型时selected | 未传model_version时选择该方案默认模型；传了model_version则从版本元数据推断方案 |

同时提供`profile`与`model_version`时须一致，否则返回422。仅指定模型版本时，变量方案从该模型元数据推断。显式指定数据或模型版本仅作用于本次预测。

最简调用：

```json
{}
```

指定数据与模型版本：

```json
{
  "data_version": "data_11111111111111111111111111111111",
  "model_version": "selected-bundled-v1"
}
```

成功200返回PredictionResponse，完整字段见第9节。每次独立请求生成新`prediction_id`，保存完整响应，并返回三地区共9条价格。有效期内同一幂等请求复用首次已保存结果。

服务固定本次模型、数据及区间配置，检查数据覆盖权重所需历史、最新观测月不早于训练及选参截止月，以及输入布局与保存的预处理一致性。输出为非有限值或非正价格时返回422、`INVALID_OUTPUT`。点预测有效但区间证据不足时，仍返回价格，并以`available=false`及`reason`说明区间不可用原因。

### 8.2 GET /predictions

查询预测历史摘要，无请求体。

| 查询字段 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `profile` | string | 不过滤 | all或selected |
| `model_version` | string | 不过滤 | 某个权重版本 |
| `data_version` | string | 不过滤 | 某个数据版本 |
| `generated_from` | string | 不过滤 | 带时区生成时间下界，含该时刻 |
| `generated_to` | string | 不过滤 | 带时区生成时间上界，不含该时刻 |

成功200返回`items: array[PredictionSummary]`，包含符合条件的全部预测记录。

| PredictionSummary字段 | 类型 | 含义 |
| --- | --- | --- |
| `prediction_id` | string | 保存的预测编号 |
| `generated_at` | string | 原生成时间 |
| `profile` | string | 变量方案 |
| `model_version` | string | 使用的权重版本 |
| `data_version` | string | 使用的数据版本 |
| `latest_observed_month` | string | 生成时的输入末月 |
| `first_target_month` | string | 预测第1个月目标月 |
| `last_target_month` | string | 预测第3个月目标月 |
| `regions` | array[string] | 三地区编码 |
| `prediction_count` | integer | 正常为9 |
| `available_interval_count` | integer | 可用价格区间条数，0至9 |
| `detail_url` | string | 完整响应GET链接 |
| `csv_url` | string | CSV下载链接 |

### 8.3 GET /predictions/{prediction_id}

无请求体，成功200返回已保存的PredictionResponse。价格、区间、评价、模型与数据版本及`generated_at`均保留原生成时刻的值，不随当前默认设置变化。编号不存在时返回404。

### 8.4 GET /predictions/{prediction_id}/csv

无请求体，成功200返回带BOM的UTF-8 CSV，Content-Disposition文件名为`<prediction_id>.csv`。正常9条数据及1行表头；字段保持现有导出形式：

| CSV列 | 类型 | 含义 |
| --- | --- | --- |
| `region` | 文本 | 地区编码 |
| `region_name` | 文本 | 地区中文名 |
| `target_month` | YYYY-MM | 目标月 |
| `horizon` | 整数 | 1、2、3 |
| `prediction` | 数值 | 点预测价格，元/立方米 |
| `unit` | 文本 | 元/立方米 |
| `lower` | 数值或空 | 价格区间下限，不可用时空 |
| `upper` | 数值或空 | 价格区间上限，不可用时空 |
| `interval_available` | True/False文本 | 该条价格区间是否可用 |
| `model_version` | 文本 | 使用的权重版本 |
| `model_training_cutoff` | YYYY-MM | 权重拟合截止月 |
| `latest_observed_month` | YYYY-MM | 原输入末月 |

区间不可用时上下限为空。校准数量、不可用原因、历史评价及数据版本通过完整JSON查询。下载读取已保存结果；编号不存在时返回404。

## 9. 预测与评价的公共返回结构

本节定义预测与评价响应的完整嵌套结构。实际响应示例见[组合变量响应](../reference/service_v3_validation/selected_response.json)与[全变量响应](../reference/service_v3_validation/all_response.json)。示例包含对应验证环境的价格、指标及版本编号，集成时应使用当前实例返回的编号。

### 9.1 PredictionResponse与价格

| 顶层字段 | 类型 | 含义 |
| --- | --- | --- |
| `prediction_id` | string | 本次完整预测记录编号 |
| `generated_at` | string | 原始生成时间 |
| `predictions` | array[Price] | 三地区三步点预测，共9条 |
| `prediction_intervals` | Intervals | 价格区间设置及明细 |
| `evaluation` | Evaluation | 有证据的历史评价，不是本次未知未来准确率 |
| `model_status` | ModelStatus | 实际模型、输入、数据与版本状态 |

Price：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `region` | string | 地区编码 |
| `region_name` | string | 地区中文名 |
| `target_month` | string | 本条目标月 |
| `horizon` | integer | 相对最新观测月的第1、2、3个月 |
| `prediction` | number | 还原至原始单位的点预测价格 |
| `unit` | string | 元/立方米 |

### 9.2 价格区间

Intervals：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `method` | string | 当前采用matured_absolute_residual_order_statistic，即已成熟绝对误差的次序统计量 |
| `confidence` | number | 名义水平，默认0.95，不是实测历史覆盖率 |
| `calibration_basis` | string | 当前为same_parameters_features_rolling_refit_protocol，同结构、变量与训练计划的逐月重训参考 |
| `calibration_version` | string或null | 登记的校准证据版本编号，与模型版本分别记录 |
| `min_residuals` | integer | 至少需要多少条已成熟误差，默认24 |
| `residual_window` | integer | 最多使用最近多少条误差，默认24 |
| `interpretation` | string | 区间口径及适用范围的文字说明 |
| `items` | array[PriceInterval] | 三地区各预测步区间，共9条 |

PriceInterval：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `region` | string | 地区编码 |
| `target_month` | string | 目标月 |
| `horizon` | integer | 第1、2、3个月，不同步长分别校准 |
| `available` | boolean | 本条区间是否可用 |
| `lower` | number或null | 价格区间下限，元/立方米 |
| `upper` | number或null | 价格区间上限，元/立方米 |
| `calibration_count` | integer | 本次实际选取的已成熟误差数量，默认最多24 |
| `latest_calibration_target_month` | string或null | 这些误差中最新一条实际价格对应月份 |
| `reason` | string或null | 不可用原因，可用时null |
| `historical_coverage_percent` | number或null | 过去可评价区间实际包含真实价格的比例，已乘100 |
| `historical_coverage_count` | integer | 上述覆盖率分母，即过去可评价区间数，不是本次24条残差数量 |

**计算口径：**每个地区、每个预测步分别选取真实目标价格已经取得、且目标月不晚于输入末月的绝对残差。取最近`residual_window`条，数量达到`min_residuals`后计算区间。将n条残差升序排列为e₍₁₎至e₍ₙ₎：

```text
k = min(ceil((n + 1) × confidence), n)
区间半宽 = e₍ₖ₎
lower = prediction - e₍ₖ₎
upper = prediction + e₍ₖ₎
```

默认n=24、confidence=0.95，故k=24，取最近24条绝对残差的最大值作为半宽。上下限按上述公式返回，展示时保留其原始含义。

历史覆盖率按各旧起点当时已成熟的残差构造区间，再与该起点后续实际价格比较。它反映参考训练协议的历史表现；95%为名义水平，不代表当前固定权重已取得95%的未来覆盖保证。该价格区间与MAPE的Bootstrap置信区间含义不同。

结构、变量、预处理、训练计划及参考历史须与登记证据匹配。匹配失败或残差不足时，`available=false`、`lower=null`、`upper=null`，具体原因见`reason`。

### 9.3 模型状态ModelStatus

预测响应和模型详情共用下列字段。已完成预测的loaded_at为当时的加载时间；模型列表中尚未加载的版本允许loaded_at=null。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `model_name` | string | 三地区融合EMAformer |
| `service_version` | string | 生成该状态的接口程序版本 |
| `model_version` | string | 使用的权重版本 |
| `profile` | string | all或selected |
| `device` | string | 本次服务选用的设备；仅登记未加载时表示计划推理设备 |
| `registered_at` | string | 该权重版本首次登记时间 |
| `loaded_at` | string或null | 该模型最近加载时间；预测历史保留当时值 |
| `trained_at` | string或null | 实际训练完成时间；旧权重未记录时null，不用文件修改时间代替 |
| `training_time_status` | string | `recorded`表示已记录真实训练完成时间；`not_recorded`表示缺少该记录，此时trained_at为null |
| `model_training_cutoff` | string | 权重拟合截止月 |
| `parameter_selection_cutoff` | string | 参数及变量方案选择所用开发期截止月 |
| `latest_observed_month` | string | 本次选择的数据末月 |
| `data_imported_at` | string | 这份数据内容版本首次导入时间 |
| `data_version` | string | 本次实际使用或查询选定的数据版本 |
| `historical_reference_matches` | boolean | 与登记证据参考历史是否一致；不等同于能否执行点预测 |
| `seq_len` | integer | 此权重历史窗口月数 |
| `pred_len` | integer | 固定3 |
| `input_features` | object | 三地区编码分别对应输入列名数组，顺序严格对应权重，均包含历史目标价格 |

模型训练截止月与最新输入末月可以不同。更新输入数据后，预测继续使用所选权重；需要更新权重时提交训练任务。`trained_at`、`registered_at`、`loaded_at`及`data_imported_at`分别记录训练、登记、加载与数据导入时间。

### 9.4 历史评价Evaluation

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `recent_backtest` | Evidence | 最近若干完整已成熟起点的滚动重训协议评价；默认最多20个起点 |
| `registered_model_test` | Evidence | 当前登记权重的固定权重留出测试；默认最多20个完整起点 |
| `interpretation` | string | 指标公式及两类评价适用范围 |

Evidence：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `available` | boolean | 此类历史指标是否可用 |
| `reason` | string或null | 不可用原因；正常为null |
| `basis` | string | 滚动重训为same_parameters_features_rolling_refit_protocol；固定权重测试为registered_checkpoint_fixed_weight_holdout |
| `version` | string或null | 登记的评价预测明细版本编号 |
| `first_forecast_month` | string或null | 最早起点的第1个月目标月 |
| `last_forecast_month` | string或null | 最后起点的第1个月目标月 |
| `last_target_month` | string或null | 全部预测步中最晚目标月 |
| `regions` | array[RegionMetric] | 三地区分步指标；不可用时为空数组 |

RegionMetric：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `region` | string | 地区编码 |
| `region_name` | string | 中文名 |
| `origins` | integer | 本地区纳入评价的完整起点数量；调参复核的计数口径另见7.6 |
| `by_horizon` | array[StepMetric] | 第1、2、3个月分别一项 |
| `average_mape_percent` | number | 三个预测步MAPE的等权平均 |
| `average_accuracy_score_percent` | number | 100减平均MAPE |

StepMetric：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `horizon` | integer | 1、2、3 |
| `observations` | integer | 本步实际计分配对数量，不把缺失真实值计为0误差 |
| `mape_percent` | number | mean(abs(预测-真实)/真实)乘100 |
| `accuracy_score_percent` | number | 100减本步MAPE，负分不截断 |

最近20个完整起点对应每地区每步20组、三步合计60组预测与真实值配对。只有3个月留出标签时，每地区每步1组、三步合计3组。展示指标时应同时标明评价范围、协议及样本数，两类评价分别报告。

历史评价不可用时保留Evidence结构，时间字段为null、`regions`为空数组，`reason`说明原因。训练接口登记本权重的留出测试；逐月重训回测使用项目命令行入口另行执行，并登记与模型训练协议匹配的证据。一次训练完成后，价格区间仍须满足对应的校准证据及残差数量要求。

### 9.5 指标展示口径

指标表统一采用以下列序。指标均为百分数，表头应另行标明评价范围、协议及每步样本数。

| 地区 | 变量方案 | 第1个月MAPE（%） | 第1个月精度得分（%） | 第2个月MAPE（%） | 第2个月精度得分（%） | 第3个月MAPE（%） | 第3个月精度得分（%） | 平均MAPE（%） | 平均精度得分（%） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| region_name | profile | by_horizon[0].mape_percent | by_horizon[0].accuracy_score_percent | by_horizon[1].mape_percent | by_horizon[1].accuracy_score_percent | by_horizon[2].mape_percent | by_horizon[2].accuracy_score_percent | average_mape_percent | average_accuracy_score_percent |

调用方可按业务要求将得分分为：≥80%“较好”、70%至不足80%“可用”、60%至不足70%“需关注”、低于60%“需复核”。该分档用于业务展示，接口返回原始指标数值，不单独返回分档字段。无可用历史评价时应展示其原因。

## 10. 统一错误响应与冲突处理

### 10.1 错误响应结构

业务校验、资源状态及服务异常使用以下错误结构。`request_id`与响应头`X-Request-ID`一致，可供维护人员查询日志。

| 顶层字段 | 类型 | 含义 |
| --- | --- | --- |
| `error` | string | 稳定机器错误编码 |
| `message` | string | 可直接显示的中文摘要 |
| `request_id` | string | 日志定位编号，与X-Request-ID一致 |
| `details` | array[Issue] | 字段、地区、月份及修复建议；没有详情时空数组 |

```json
{
  "error": "INVALID_DATA",
  "message": "浙江数据存在缺失目标价格，请修正后重新上传。",
  "request_id": "req_example",
  "details": [
    {
      "code": "TARGET_MISSING",
      "region": "zhejiang",
      "month": "2026-04",
      "field": "终端非居民天然气价格（分省）",
      "message": "2026-04的目标价格为空。",
      "suggestion": "补齐真实目标价格后重新上传三地区处理表；不要用0填充。"
    }
  ]
}
```

### 10.2 HTTP状态与错误编码

| HTTP状态 | error | 含义及处理方式 |
| --- | --- | --- |
| 401 | `UNAUTHORIZED` | 密钥缺失或不正确；检查服务是否启用密钥及请求头配置 |
| 404 | `NOT_FOUND` | 数据、模型、任务、预测编号或路径不存在；使用当前实例返回的真实地址和编号 |
| 405 | `METHOD_NOT_ALLOWED` | 路径存在但HTTP方法不支持；按接口定义调整GET或POST |
| 409 | `VERSION_CONFLICT` | 默认版本前置条件不一致，或同一幂等键对应不同请求；刷新状态或为新操作生成新键 |
| 409 | `INVALID_STATE` | 当前资源状态不允许操作，例如任务不能取消、测试文件未生成、默认组合不兼容或原幂等请求仍在处理；根据message查询或调整后重试 |
| 413 | `FILE_TOO_LARGE` | 文件、请求或XLSX解压内容超限；检查上传文件及服务端限制 |
| 415 | `UNSUPPORTED_FORMAT` | 扩展名或文件内容不符合XLSX要求；按模板重新导出 |
| 422 | `INVALID_REQUEST` | 请求字段、参数值、时间划分或搜索范围不合法；按details修正 |
| 422 | `INVALID_DATA` | 数据表、单位、月份或数值校验失败；按地区、月份及字段定位问题 |
| 422 | `INCOMPATIBLE_MODEL` | 所选数据无法满足模型推理要求；选择兼容版本或重新训练 |
| 422 | `INVALID_OUTPUT` | 模型输出非有限值或非正价格；检查输入、预处理和权重 |
| 429 | `QUEUE_FULL` | 任务队列已满；查询已有任务，待容量释放后重新提交 |
| 500 | `INTERNAL_ERROR` | 未预期服务异常；保留request_id并联系维护人员检查日志 |
| 503 | `MODEL_LOAD_FAILED` | 模型加载失败；检查登记信息、权重文件和设备资源后重试 |

GET任务成功返回200表示任务记录查询成功，任务本身的执行结果由`status`、`error`及`result`判断。HTTP202表示提交或取消请求已接受，不能作为训练、调参完成的依据。

### 10.3 异步任务错误编码

以下编码位于任务或候选的`error.code`中，与HTTP错误响应的顶层`error`区分使用。

| 编码 | 含义 | 处理方式 |
| --- | --- | --- |
| `PREPARATION_FAILED` | 加载冻结配置或准备数据失败 | 检查任务配置及数据，修正后重新提交 |
| `EXECUTION_FAILED` | 训练、调参、评价或结果保存阶段异常 | 结合任务日志排查资源及执行错误，修正后重新提交 |
| `WORKER_STOPPED` | 工作进程已退出，任务未完成 | 检查服务及资源状态，另行提交任务 |
| `TIME_LIMIT` | 达到任务执行时限 | 查询已完成候选；需要最终推荐时调整预算或时限后重新提交 |
| `FIT_FAILED` | 单个候选拟合失败 | 查询该候选错误及其余候选状态；是否影响整项任务以任务状态为准 |
| `SERVICE_UPGRADED` | 历史任务因服务程序更新而中断 | 确认当前配置后重新提交 |

`retryable`提示排除问题后是否适合新建任务，服务不会据此自动重试。重跑属于新业务操作，应使用新的幂等键。

## 11. 处理表与变量约定

### 11.1 上传的25列

三地区各一个单工作表XLSX。第1行字段、第2行单位、第3行开始为月度记录。列名及顺序固定如下，表格无需装饰格式。日期存为YYYY-MM文本。上传过程不重新运行原始合并、GDP可用日期处理或汇率换算；这些应在生成处理表时完成。

| 字段 | 单位或格式 | 含义 |
| --- | --- | --- |
| 年月 | YYYY-MM | 数据所属月份，按月连续、升序且不重复 |
| JKM现货价格 | 美元/百万英热单位 | JKM天然气现货价格指标 |
| LNG进口价格 | 元/吨 | LNG进口价格指标 |
| TTF期货价格 | 美元/百万英热单位 | TTF天然气期货价格指标 |
| Henry Hub期货价格 | 美元/百万英热单位 | Henry Hub天然气期货价格指标 |
| Brent原油价格 | 美元/桶 | Brent原油价格指标 |
| 管道气进口价格 | 元/吨 | 管道天然气进口价格指标 |
| 美元/人民币汇率 | 人民币元/美元 | 1美元对应的人民币金额 |
| LNG进口量（全国） | 吨 | 全国LNG进口量 |
| 管道气进口量（全国） | 吨 | 全国管道天然气进口量 |
| LNG进口量（区域） | 吨 | 当前地区的LNG进口量 |
| 天然气产量（全国总量） | 亿立方米 | 全国天然气产量 |
| 天然气产量（区域） | 亿立方米 | 当前地区的天然气产量 |
| LNG产量（全国） | 万吨 | 全国LNG产量 |
| 门站价基准价格（分省） | 元/立方米 | 当前地区门站基准价格指标 |
| 中国液化天然气出厂价格指数 | 元/吨 | 全国LNG出厂价格指标，单位按处理表保存 |
| 工业增加值（分省） | % | 地区工业增加值相关指标的处理值 |
| 制造业PMI增速（全国） | % | 全国制造业PMI增速字段的处理值 |
| GDP增速（分省） | % | 当前地区GDP增速字段的处理值 |
| 平均气温（分省） | ℃ | 当前地区的月度平均气温指标 |
| 采暖度日数HDD（分省） | ℃·日 | 当前地区的采暖度日数指标 |
| 非居民天然气消费量（分省） | 亿立方米 | 当前地区非居民天然气消费量 |
| 柴油价格（0#柴油） | 元/升 | 0号柴油价格指标 |
| 动力煤价格 | 元/吨 | 动力煤价格指标 |
| 终端非居民天然气价格（分省） | 元/立方米 | 历史目标价格，始终作为输入，须完整且为正 |

全国共同指标按[data.py](../joint_emaformer/data.py)的COMMON清单核对三地一致，区域字段允许不同。非目标列允许空值，实际选为输入后须检查拟合历史中是否有可用值；目标标签须完整。

### 11.2 两种变量方案

两种方案均使用历史目标价格作为输入。请求通过`profile`选择方案，实际列名及顺序由模型配置和权重元数据确定。当前组合变量如下：

| 地区 | selected输入 |
| --- | --- |
| 上海 | 历史目标价格、管道气进口价格、管道气进口量（全国）、LNG产量（全国） |
| 浙江 | 历史目标价格、TTF期货价格、制造业PMI增速（全国） |
| 江苏 | 历史目标价格、天然气产量（全国总量）、LNG产量（全国）、采暖度日数HDD（分省）、柴油价格（0#柴油） |

`all`根据训练历史确定有效列，并保留历史目标价格。浙江排除LNG进口量（区域）及天然气产量（区域），江苏排除工业增加值（分省）、GDP增速（分省）及非居民天然气消费量（分省）。处理表仍须保留全部模板列，实际模型输入以`model_status.input_features`为准。

重新训练只在相应训练历史拟合中位数、插补和标准化；每个历史预测窗口只读取当时可见输入。HDD沿用已处理表的原值，不额外裁剪、重算或作业务性改写；如模型选中的非目标输入缺失，按明示的missing策略处理。每份权重保存自己的变量顺序与预处理，推理不重新拟合。

## 12. 运行管理与集成验证

### 12.1 数据与结果保存位置

运行文件保存在`service/config.yaml`的`storage_dir`中，默认目录为项目根目录下的`outputs/service`。可通过环境变量`LNG_SERVICE_STORAGE`指定其他位置。业务接口返回资源编号及查询、下载URL，以下路径供维护人员定位文件。

```text
outputs/service/
├── data_versions/<data_version>/
│   ├── shanghai.xlsx
│   ├── zhejiang.xlsx
│   ├── jiangsu.xlsx
│   └── metadata.json
├── trainings/<training_id>/
│   ├── task.json
│   ├── effective_config.json
│   ├── best.pt
│   ├── preprocessing.json
│   ├── training_audit.json
│   ├── history.csv
│   ├── test_predictions.csv       # 有测试时
│   ├── evaluation_predictions.csv # 有测试时，供内部证据核对
│   └── training.log
├── tuning_jobs/<tuning_id>/
│   ├── task.json
│   ├── effective_config.json
│   ├── search_plan.json
│   ├── screening_trials.json
│   ├── confirmed_candidates.json
│   ├── screening_public.json
│   ├── confirmation_public.json
│   ├── confirm_trial_<trial>.json
│   ├── selected_parameters.json
│   └── tuning.log
├── predictions/<prediction_id>/response.json
├── registry.sqlite3
└── logs/events.jsonl
```

目录按执行阶段逐步生成。取消、失败或尚未执行到相应阶段时，部分结果文件可能不存在。训练任务的`task.json`为提交或结束时的快照，实时任务状态应通过接口查询。

`registry.sqlite3`保存数据及模型登记、默认关系、任务状态与幂等记录；各资源目录保存实际文件。`predictions/<prediction_id>/response.json`是本次预测完整响应，包含价格、区间、历史评价及模型状态。查询和下载使用该已保存结果。

上传创建独立快照，项目`data`目录及已有快照保持原样。任务固定数据版本及有效配置，后续新上传、新训练和默认切换不改写其历史结果。取消、失败和超时记录保留，供问题定位与复核。

### 12.2 配置及模型版本管理

| 配置或记录 | 职责 |
| --- | --- |
| [service/config.yaml](config.yaml) | 服务地址、设备、运行目录、上传限制、任务预算及区间设置 |
| [service/models.yaml](models.yaml) | 启动时导入权重、模型版本及配套评价证据 |
| [config.yaml](../config.yaml) | 三地区初始处理表路径及公共数据设置 |
| [全变量配置](../joint_emaformer/configs/all.yaml)、[组合变量配置](../joint_emaformer/configs/selected.yaml) | 各方案的模型参数、训练计划及变量设置 |
| [search.yaml](../joint_emaformer/configs/search.yaml) | 默认搜索范围及标准搜索预算 |
| 任务effective_config.json | 任务接受时固定的实际参数、日期、数据版本、设备及程序版本 |
| registry.sqlite3 | 运行中的版本登记、任务状态及默认关系 |

服务启动时导入配置数据及模型。已有持久化默认关系优先；更新初始数据文件可能登记新数据版本，但不会自动切换已有默认值。训练接口生成的模型直接进入登记库，可通过启用接口切换，无需重启服务。

维护人员导入其他权重时，应在`models.yaml`中登记与权重元数据一致的新`model_version`及配套证据，重启后导入。固定权重测试须对应实际权重版本；滚动重训及价格区间证据须匹配结构、变量、预处理、训练计划和参考历史。原有版本文件应保留，回退通过启用旧版本完成。

### 12.3 调用方接入要求

| 场景 | 接入规则 |
| --- | --- |
| 版本选择 | 将接口返回的data_version、model_version及任务编号保存为业务引用；显示名称仅用于展示 |
| 上传后使用 | 后续请求显式携带新data_version；需改变全局默认时再调用activate |
| 任务提交 | 保存202响应中的任务编号及status_url，通过查询获取执行结果 |
| 进度查询 | 建议每2至5秒轮询，进入终态后停止；使用阶段名称与阶段进度共同展示 |
| 请求重试 | 网络结果不明时复用同一Idempotency-Key；修正请求或重新执行时使用新键 |
| 默认切换 | 需要防止并发覆盖时携带expected_current_*字段；409后重新查询当前状态 |
| 模型可用性 | 按本次数据查询compatibility，区分点预测兼容性、历史评价可用性及价格区间可用性 |
| 结果展示 | 展示3个目标月的价格与区间；历史精度须附带评价范围、协议及样本数 |
| 状态展示 | 从model_status读取模型版本、训练时间、训练截止月及数据更新时间 |
| 历史追溯 | 使用prediction_id读取原响应；更新默认版本后仍可查询原结果 |
| 异常处理 | 展示message及details中的修复建议，并保留request_id供日志查询 |

`latest_observed_month`表示数据覆盖末月，`data_imported_at`表示该版本首次导入时间，两者分别反映业务数据范围和系统接收时间。`trained_at=null`时按`training_time_status`展示“未记录”，不以登记时间代替。

### 12.4 部署验证与试运行

依赖安装、Windows CPU及Linux CUDA启动步骤见[部署说明](../deployment/README.md)。接口字段可通过运行实例的`/openapi.json`核对。已有验证记录见[服务验证报告](../reference/service_v3_validation/report.md)，目标部署环境仍须执行以下检查。

| 场景 | 应核对的结果 |
| --- | --- |
| 上传三地正确处理表 | 新数据201或重复200，月份一致，原表不变 |
| 日期断月、目标缺失、单位错误或全国指标冲突 | 返回具体地区/月份/字段，旧默认不变 |
| 同一幂等键网络重试 | 不重复训练、预测或切换；不同请求复用键409 |
| 留出3个月 | 训练截至末月减3；每地区每步1组指标与逐条预测一致 |
| 多月留出 | 完整起点数=留出月数减2，分步MAPE与明细可复算 |
| 全历史训练 | 使用实际历史范围，本次没有留出测试时对应评价标为不可用 |
| 调参 | 拟合、选轮数及排名仅使用开发历史；预算、复核次数及推荐规则可查 |
| 自动日期划分 | 数据截至2026-05时保留测试为2026-03至2026-05；更新至2026-06后为2026-04至2026-06，原实验配置不变 |
| 独立任务配置 | 接受后绑定数据版本和实际参数；排队期间变更默认数据或服务配置不改变任务，详情与执行范围一致 |
| 调参后训练 | 参数、布局、冻结轮数与学习率轨迹与推荐记录一致 |
| 最简调参后训练请求 | 仅传tuning_id和mode时正确推断变量方案及绑定数据 |
| 指定历史模型版本 | 使用该权重及其保存预处理，不因默认版本变化而改用其他权重 |
| 热启用和回退 | 校验成功后原子切换；失败保留旧版；历史预测不改变 |
| 缺少评价或区间证据 | 点预测有效时仍返回价格，相关available=false并明确原因 |
| 重启、失败或取消 | 任务和历史可查，不把中断当成功，不覆盖已完成权重 |
| Windows CPU与Linux CUDA部署 | 各自在目标机验证上传、训练、调参、切换和预测，记录实际环境与结果 |

试运行应保留数据版本、模型版本、训练及预测记录，定期核对真实价格成熟后的误差。更新数据后检查月份覆盖和缺失情况；调整参数或变量后重新检验，并按需要生成匹配的回测和校准证据。上线采用的具体版本、评价范围及启用时间应形成记录。
