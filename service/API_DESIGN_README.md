# 三地区价格预测应用：数据上传、训练、调参与预测接口说明

文档版本：3.0.0，2026-10-19。

**本文说明服务3.0.0的22项业务操作，包括数据上传、自动调参、按推荐方案训练、模型版本选择和热切换。** Windows/Linux启动命令及快速调用见[启动与调用说明](README.md)，部署步骤见[部署说明](../deployment/README.md)。请求字段和当前接口结构也可通过运行服务的/docs和/openapi.json查看。第12.2节为前端接入建议。

`/datasets`一次接收三地区处理后的XLSX表，形成不可变数据版本。在线业务模型为三地区融合EMAformer，一次联合处理上海、浙江、江苏；全变量和组合变量各有独立模型版本。LSTM、单区域EMAformer、MoFo保留各自命令行复现入口。

## 1. 使用流程

页面建议提供“预测”“训练与检验”“调参”“历史记录”四个入口。用户通过下拉框选择数据和模型，不手填文件路径、训练种子或记录编号。

1. 下载样例，准备三地区**处理后的**XLSX表；一次上传三张表。
2. 系统校验并保存数据版本，页面自动选中新版本。上传本身不覆盖原表、不改变全局默认数据。
3. 日常可直接使用兼容的已登记模型预测，无需每次重新训练。
4. 需要检验模型时，选择“留出最近3个月”，系统自动划分训练与测试范围。
5. 需要重新选参时，提交调参任务；搜索只使用开发期，测试期保留。
6. 选定参数后提交训练，训练成功自动登记候选模型。页面可以选择该版本预测，也可以将它设为默认。
7. 需要使用全部历史时，选择“全历史训练”；未来没有真实标签时，本次测试指标显示不可用。

假设上传数据为2015-01至2026-05：

| 操作 | 使用范围 | 结果含义 |
| --- | --- | --- |
| 留出3个月训练 | 拟合截至2026-02，测试2026-03至2026-05 | 一个固定新权重的三步留出测试，各地区各步1组配对 |
| 调参默认时间划分 | 初筛训练截至2024-06，验证2024-07至2026-02，保留测试2026-03至2026-05 | 根据验证历史选参，不读取保留测试标签选参 |
| 全历史训练 | 拟合截至2026-05 | 生成候选生产权重，本次没有留出测试标签 |
| 使用截至2026-05的观测预测 | 输入各权重要求的最近历史窗口 | 预测2026-06、2026-07、2026-08；不根据服务器日期改成其他月份 |

以上时间隔离属于历史留出评价。若人员此前已根据这段测试成绩调整过配置，不能仅因接口重新划分，就声称该测试是从未查看的独立样本。新产生的未来业务月份成熟后另行评价。

## 2. 通用约定

### 2.1 类型、编码和默认值

| 项目 | 约定 |
| --- | --- |
| JSON字段 | 英文小写下划线；前端配中文标签。未定义字段返回422，避免拼写错误被忽略 |
| 类型 | `string`文本、`integer`整数、`number`有限数值、`boolean`布尔、`object`对象、`array`数组；`null`表示未记录或不可用，不表示0 |
| 地区 | `shanghai`上海、`zhejiang`浙江、`jiangsu`江苏；三地一次联合处理 |
| 变量方案 | `all`全变量、`selected`组合变量；未指定模型版本或调参任务时默认selected，指定后按对应记录推断方案，显式填写须一致；默认选择不表示它在任何数据上都优于全变量 |
| 模型 | 第一版固定为三地区融合EMAformer；预测长度固定3个月 |
| 月份 | 严格`YYYY-MM`，例如`2026-02`，并检查真实年月合法性 |
| 时间戳 | 带时区ISO 8601；服务内部使用UTC，页面转换北京时间 |
| 百分数 | 名称带`_percent`的数值已乘100；2.5表示2.5%。`confidence=0.95`表示95% |
| 价格 | 元/立方米；JSON保留计算精度，页面显示位数不改变计算和导出结果 |
| 请求 | 除文件上传外使用`application/json`；POST预测和训练至少提交`{}`对象 |
| 默认数据 | 由服务维护的当前默认数据版本；省略`data_version`时通常解析为它，引用tuning_id训练时优先沿用该调参任务绑定的数据版本；解析结果在接受任务时固定 |
| 默认模型 | 每个变量方案各有一个默认模型。显式选版本不改变全局默认模型 |
| 标识 | 新数据版本用`data_`加32位随机十六进制；预测、训练、调参分别用`pred_`、`train_`、`tune_`加32位随机十六进制；旧数据24位编号继续可用 |

模型版本使用独立编号：随包模型为`all-bundled-v1`与`selected-bundled-v1`；API新训练为`变量方案-训练任务32位编号`，每次成功训练登记一个新版本。手工配置导入时填写与权重元数据一致的model_version和权重路径，并登记相应的评价证据版本。版本编号不从文件内容推导。

数据、模型、训练任务、调参任务及预测记录列表一次返回符合查询条件的全部记录，默认按创建时间降序，同时间按编号降序。模型列表使用`models`数组，其余记录列表使用`items`数组；空列表正常返回200。

### 2.2 请求头

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `Content-Type` | 有请求体时 | JSON用`application/json`；上传用自动带boundary的`multipart/form-data` |
| `X-API-Key` | 启用密钥时 | 沿用服务器`LNG_API_KEY`配置。`/docs`、`/redoc`、`/openapi.json`按现有方式免密钥 |
| `Idempotency-Key` | 否，建议页面自动生成 | 用于上传、任务提交、预测及默认切换，避免网络重试创建重复业务操作；非用户手填 |

相同密钥和相同业务请求在设计保留期24小时内返回第一次操作结果，不重新训练或预测；同一密钥对应不同业务请求返回409。上传按三份文件内容及表单字段识别请求，不按multipart边界识别。第一次省略的默认数据、默认模型和配置一经解析，重试继续使用首次结果。保留期是服务策略，可由维护人员配置。

响应统一带`X-Request-ID`用于日志查询。

## 3. 接口总览与现状

下面22项业务操作均由当前服务提供。示例中的版本和任务编号须替换为实际返回值。

| 方法与路径 | 当前状态 | 目标职责 |
| --- | --- | --- |
| `GET /templates/{region}` | 已实现 | 下载地区处理表样例 |
| `POST /datasets` | 已实现 | 同时上传并校验三张处理表 |
| `GET /datasets` | 已实现 | 数据版本列表 |
| `GET /datasets/{data_version}` | 已实现 | 数据范围、缺失情况及模型兼容性 |
| `POST /datasets/{data_version}/activate` | 已实现 | 切换默认数据版本 |
| `GET /models` | 已实现 | 全部已登记模型版本及兼容性 |
| `GET /models/{model_version}` | 已实现 | 模型参数、训练范围及评价详情 |
| `POST /models/{model_version}/activate` | 已实现 | 热加载并设为该方案默认模型 |
| `POST /trainings` | 已实现 | 提交固定配置的训练任务 |
| `GET /trainings` | 已实现 | 训练历史列表 |
| `GET /trainings/{training_id}` | 已实现 | 训练状态、实际配置及结果 |
| `GET /trainings/{training_id}/csv` | 已实现 | 下载该新权重的留出测试明细 |
| `POST /trainings/{training_id}/cancel` | 已实现 | 请求取消训练 |
| `POST /tuning-jobs` | 已实现 | 提交参数搜索任务 |
| `GET /tuning-jobs` | 已实现 | 调参历史列表 |
| `GET /tuning-jobs/{tuning_id}` | 已实现 | 搜索进度及当前已完成候选摘要 |
| `GET /tuning-jobs/{tuning_id}/results` | 已实现 | 推荐配置、训练计划及候选排名 |
| `POST /tuning-jobs/{tuning_id}/cancel` | 已实现 | 请求取消调参 |
| `POST /predictions` | 已实现 | 选择数据和模型版本，生成统一预测结果 |
| `GET /predictions` | 已实现 | 预测历史列表 |
| `GET /predictions/{prediction_id}` | 已实现 | 查询已保存完整响应 |
| `GET /predictions/{prediction_id}/csv` | 已实现 | 下载已保存的价格及区间明细 |

路径中的参数均必填，直接使用对应提交或列表接口返回的真实编号；页面自动传递。示例中的`data_111111111111111111111111`等仅是结构示意，不是现有服务器上的真实记录。

## 4. 数据接口

### 4.1 GET /templates/{region}

`region`为`shanghai`、`zhejiang`或`jiangsu`。无请求体，成功200，返回对应XLSX二进制文件；`Content-Type`为`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`，`Content-Disposition`包含下载文件名，不返回JSON。

样例是当前随包完整处理表，可供用户核对列名、单位和历史数据。第一版不增加空表与原始合并表上传；三张上传文件均须是经过处理的表。字段及单位见[表格约定](#11-处理表与变量约定)。

### 4.2 POST /datasets

请求类型为`multipart/form-data`。三个地区一次提交，全部通过才登记版本；禁止只更新一张表后默默混用其他版本的另外两张表。

| 请求字段 | 类型 | 必填及默认 | 含义 |
| --- | --- | --- | --- |
| `shanghai_file` | file | 必填 | 上海处理后XLSX |
| `zhejiang_file` | file | 必填 | 浙江处理后XLSX |
| `jiangsu_file` | file | 必填 | 江苏处理后XLSX |
| `name` | string | 可选，自动生成 | 页面显示的数据名称，最多100字符 |
| `note` | string | 可选，空文本 | 更新说明，最多500字符，例如“补充至2026年6月” |

默认限制为每文件20 MiB、三文件合计60 MiB、每表最多2400个月、每个工作簿解压后总内容最多200 MiB；限制由维护人员配置，不由客户端覆盖。检查真实XLSX结构，不只检查扩展名；原始文件名仅用于展示，不用它拼接服务端存储路径。

新版本成功201，完全相同的三份文件复用已有版本时200：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `created` | boolean | 是否新建了数据内容版本 |
| `dataset` | DataVersion | 数据版本完整信息，见下表 |

**DataVersion对象：**

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `data_version` | string | 三份文件内容共同确定的版本号 |
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

上传检查包括单工作表、25列顺序及单位行、年月合法连续且不重复、三地月份严格一致、全国共同指标一致、数值有限、目标价格为正且无缺失。允许非目标字段存在空值；插补参数在后续各次训练历史内拟合。检查通过不代表适用于全部已有权重，也不代表取得有效精度或区间证据。

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

无请求体。支持`source`（`configured`或`upload`，可选）和`is_default`（boolean，可选）过滤。成功200，返回`items: array[DataVersion]`，包含符合条件的全部数据版本。兼容性按查询时的模型登记清单计算，不写回原文件。

### 4.4 GET /datasets/{data_version}

无请求体，成功200直接返回DataVersion。不存在返回404。用于页面查看三地月份、缺失情况和可用模型。

### 4.5 POST /datasets/{data_version}/activate

将该版本设为全局默认数据，不修改`config.yaml`中的原始数据文件。页面在数据列表点击“设为默认”时调用；仅当前页面选择某数据无需调用本接口。

| 请求字段 | 类型 | 必填及默认 | 含义 |
| --- | --- | --- | --- |
| `expected_current_data_version` | string或null | 可选 | 页面读到的旧默认版本；提供后若已被别人切换，返回409，避免覆盖别人的操作 |

无附加条件可提交`{}`。成功200：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `data_version` | string | 新默认数据版本 |
| `previous_data_version` | string或null | 原默认版本 |
| `changed` | boolean | 是否实际改变默认版本；重复设为同版本为false |
| `activated_at` | string | 最近一次实际切换时间；重复调用不伪造新切换 |
| `compatible_models` | array[ModelCompatibility] | 对各已登记权重的兼容性 |

切换前检查当前默认组合模型能够使用该数据；不兼容时返回409并保留旧默认。若需要同时换新模型和新数据，可用模型启用接口一次原子切换，避免先后切换造成无可用组合。已提交任务及已保存预测不改变其数据版本。

## 5. 模型查询与热启用

### 5.1 GET /models

返回所有登记版本的列表，保留顶层`models`数组。

| 查询字段 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `profile` | string | 不过滤 | `all`或`selected` |
| `is_default` | boolean | 不过滤 | 是否为该方案默认模型 |
| `data_version` | string | 当前默认数据 | 按这份数据计算推理兼容性及评价可用性 |

成功200，返回`models: array[ModelRecord]`，包含符合条件的全部模型版本。模型数据状态按本次查询的数据版本说明，不把某次查询改成全局状态。

### 5.2 GET /models/{model_version}

可选查询参数`data_version`，默认当前默认数据；无请求体。成功200返回ModelRecord，不存在404。

ModelRecord包含[ModelStatus全部字段](#93-模型状态modelstatus)，并增加：

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `display_name` | string | 自动生成的中文名称，如“组合变量｜训练截至2026-05｜2026-10-18 训练” |
| `source` | string | `bundled`随包、`configured`其他配置导入，或`training`接口训练生成 |
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

| 状态值 | 页面显示 | 含义 |
| --- | --- | --- |
| `not_loaded` | 未加载 | 模型已登记，但权重尚未加载到CPU内存或GPU显存；需要使用时先加载 |
| `loading` | 加载中 | 正在读取权重、构建模型，并加载到指定的CPU或GPU设备 |
| `ready` | 已加载 | 模型加载成功，可供推理调用；实际预测仍须通过所选数据的兼容性检查 |
| `error` | 加载失败 | 本次加载未成功，模型暂时无法用于预测；可能原因包括权重文件无法读取、校验失败或内存/显存不足，排查后可重新尝试加载 |

登记是保存模型权重位置、版本、参数等信息；加载是实际读取权重，将模型放入内存或显存。登记成功后，模型可以处于`not_loaded`状态。例如登记10个模型版本时，可以只加载当前需要的版本，减少设备内存占用。选择尚未加载的版本进行预测时，服务先尝试加载；该过程使用已有权重，不会重新训练模型。加载失败时返回错误，不生成该次预测结果。

模型版本身份及训练范围不随查询数据改变；`latest_observed_month`、`data_version`、`data_imported_at`和历史参考匹配状态描述本次查询选择的数据。模型列表不强制把全部权重都加载到GPU。

### 5.3 POST /models/{model_version}/activate

将指定版本热加载并设为其变量方案的默认模型。请求体：

| 请求字段 | 类型 | 必填及默认 | 含义 |
| --- | --- | --- | --- |
| `data_version` | string或null | 可选，当前默认 | 与新模型一起使用的数据；显式填写时同时将它设为默认数据 |
| `expected_current_model_version` | string或null | 可选 | 预期的该方案旧默认模型，已变化则409 |
| `expected_current_data_version` | string或null | 可选 | 同时切换数据时预期的旧默认数据，已变化则409 |

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
| `warnings` | array[Issue] | 例如缺少区间证据，或新默认数据与另一个方案不兼容 |

服务先检查权重、保存的预处理参数、输入布局和选定数据，再加载并做一次推理验证，全部成功才提交默认指针变更。显式同时更换模型和数据时须原子提交。加载失败返回422或503并保留旧默认关系；不把失败权重设为默认。

切换后必须保证默认预测方案selected对应的模型与默认数据仍可用；否则409并保留旧关系。需要更换到旧组合模型无法使用的数据时，可先一次启用兼容的新selected模型和新数据，再调整all方案，避免默认预测入口失效。

切换无需重启服务，但可短暂排队；不承诺无资源代价的零延迟。每次预测接受时固定自己的模型和数据引用，切换不会改变正在执行的预测。缺少历史测试或校准证据不阻止点预测启用，相关指标和区间返回不可用及原因。若另一变量方案与新数据不兼容，明确提示，不能暗中改用别的模型。

启用旧版本即可完成回退；无需覆盖新文件。普通用户通过模型下拉框选择某个版本进行单次预测时，不需要先把它设为全局默认。

## 6. 训练接口

### 6.1 POST /trainings

JSON请求，一次任务训练一个变量方案的三地区融合权重。全变量与组合变量分别提交。训练使用固定配置和固定轮数；早停选轮数及结构搜索在调参任务内进行。

| 请求字段 | 类型 | 必填及默认 | 含义 |
| --- | --- | --- | --- |
| `profile` | string | 可选 | `all`或`selected`；指定tuning_id时从该调参任务推断，未指定时默认selected；同时明确填写时须与调参任务一致 |
| `data_version` | string或null | 可选 | 指定tuning_id时默认沿用该调参任务绑定的数据版本，否则使用当前默认数据；接受任务后固定 |
| `name` | string或null | 可选，自动生成 | 页面任务名称，最多100字符 |
| `mode` | string | 可选，`holdout` | `holdout`自动留出、`full_history`全部历史、`custom`指定边界 |
| `holdout_months` | integer | 仅holdout可用，默认3 | 留出末尾多少个月，至少3；按目标月划分，不随机抽取 |
| `training_cutoff` | string | custom必填，其他模式不传 | 指定训练截止月，YYYY-MM |
| `test_end_month` | string | custom可选，默认数据末月 | 指定测试末月，不晚于数据末月；须在训练之后至少保留3个月 |
| `tuning_id` | string或null | 可选，null | 使用成功调参任务的推荐结构和冻结训练计划；省略则用对应方案的随包最终配置 |
| `epochs` | integer | 可选，随配置，当前24 | 手动覆盖训练轮数，建议服务上限500；使用tuning_id时不传 |
| `learning_rate` | number | 可选，随配置 | 手动覆盖固定学习率，须为有限正数，建议上限0.01；使用tuning_id时不传 |

请求不接收模型文件路径、任意YAML路径、设备编号、训练种子、Shell命令或自动启用开关。服务器选择设备；全变量/组合变量清单跟随配置或调参结果，不在此自动重新筛选特征。

模式规则：holdout的截止月=数据末月减holdout_months；full_history截止月=数据末月且不留测试；custom使用显式边界。模式专属字段交叉传入返回422，不默默忽略。至少满足`历史月数 >= seq_len + 3`，并满足变量有效性和参数选择截止要求。整段训练范围必须覆盖所选参数、变量方案要求的数据；测试起点不能早于最终选参截止月之后。

指定tuning_id时，任务须成功、方案一致、训练截止月不早于该次选参截止月，数据在全部选参历史段的数值和单位须一致；允许其后追加新月份。修订选参历史不能直接沿用旧推荐。训练使用该调参任务冻结的轮数和学习率轨迹，禁止同时用epochs/learning_rate悄悄改变它；需要调整时另发不引用该调参任务的训练，按新配置记录。

推荐的自动训练流程只需在页面选择成功的调参方案和训练用途，后台提交`tuning_id`和`mode`。变量方案、默认数据版本、模型结构、批大小、训练轮数及逐轮学习率均从选中的调参记录取得，不需要用户重复填写；`mode`默认holdout，页面仍展示“留出最近3个月检验”和“全历史训练”两个用途供选择。省略的数据和方案在接受任务时解析并绑定，不随之后的全局默认切换而变化。页面显示方案名称，编号由页面自动传递。

只传epochs时，沿用源配置学习率轨迹并按现有训练入口规则截取或延长最后学习率；只传learning_rate时生成覆盖实际轮数的固定轨迹；二者都不传时完整沿用源计划。最终接受的参数和轨迹必须回显到任务详情。

简单留出测试：

```json
{
  "profile": "selected",
  "data_version": "data_111111111111111111111111",
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

接受前完成数据及边界检查，成功返回202和TrainingTask。202只表示接受任务；接受时`status=queued`、`stage=queued`、`result=null`，其余字段按公共任务结构返回。页面立即得到实际训练/测试月份，不必等训练完成才发现划分不合预期。

### 6.2 公共任务字段

TrainingTask与TuningTask都含有下列字段，编号与业务结果字段另列。POST提交、GET列表元素和GET详情使用同一任务对象；前端可先渲染摘要，再进入详情。

| 返回字段 | 类型 | 含义 |
| --- | --- | --- |
| `kind` | string | `training`或`tuning` |
| `name` | string | 中文任务名称 |
| `profile` | string | 变量方案 |
| `model_name` | string | 三地区融合EMAformer |
| `data_version` | string | 接受时绑定的数据版本，后续切换默认数据不影响它 |
| `device` | string | 服务器给本任务安排的实际设备，如cpu、cuda |
| `status` | string | 见任务状态表 |
| `stage` | string | `queued`、`preparing`、`training`、`evaluating`、`registering`、`screening`、`confirming`或`finalizing`；结束后保留最后阶段 |
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

TaskError：`code:string`为稳定编码，`message:string`为中文说明，`retryable:boolean`表示修复环境或资源问题后是否适合重新提交；不把内部绝对路径和完整堆栈直接返回。

| status | 中文状态 | 说明 |
| --- | --- | --- |
| `queued` | 排队中 | 已接受但未开始 |
| `running` | 执行中 | 结合stage展示具体阶段 |
| `cancel_requested` | 正在停止 | 已收到停止请求，等待安全检查点 |
| `succeeded` | 已完成 | 全部必需阶段成功，结果已持久化 |
| `failed` | 失败 | 执行异常或有效候选不足 |
| `cancelled` | 已取消 | 未执行完的任务已停止 |
| `interrupted` | 已中断 | 服务/工作进程异常或达到时限等，不能当成功 |

任务先持久化再返回202，关闭页面不影响执行。单机默认同时执行1个训练/搜索拟合任务，其余排队。服务重启后应检查工作进程真实状态，无法继续的标为interrupted，不假装成功或从头重复训练；第一版API不承诺自动断点续训，CLI已有的搜索恢复另保留。

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
| `parameter_selection_cutoff` | string | 当前配置/变量方案的选参截止月份，不能用最终训练截止月冒充 |
| `epochs_completed` | integer | 完成的训练轮数 |
| `evaluation` | Evaluation | 留出测试及匹配的滚动参考；具体结构见第9节 |
| `test_csv_url` | string或null | 存在留出测试时的下载URL，否则null |
| `model_url` | string | 该版本GET模型详情URL |

训练成功自动保存完整权重、预处理参数、实际配置、训练历史和审计记录，并登记模型；不自动替换默认模型。无完整测试标签时，evaluation对应项available=false，并说明原因。测试按固定新权重、逐起点历史输入计算；不等同于每个起点从头重训的滚动回测。

有H个月完整留出标签时，完整三步测试起点数为H-3+1，每地区每步各有这么多组配对；三步合计乘3。标签和未来输入不能进入权重、插补或标准化的拟合。某个后续测试起点可以使用当时已经观测到的前一个测试月输入，这仍是固定权重的历史滚动输入评价。

### 6.4 GET /trainings

无请求体。可按`profile`、`status`、`data_version`、`tuning_id`过滤，均可省略。成功200返回`items: array[TrainingTask]`，包含符合条件的全部训练任务，空记录返回空数组。

### 6.5 GET /trainings/{training_id}

无请求体，成功200返回TrainingTask，不存在404。运行中result为null；终态失败通过error说明，GET本身仍为200，表示成功查询到了一个失败任务。

### 6.6 GET /trainings/{training_id}/csv

无请求体，成功200返回带BOM的UTF-8 CSV，Content-Disposition文件名为`<training_id>_test.csv`。任务未成功或没有测试明细时返回409和中文原因，不下载一张全0的表。

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

无业务请求字段，提交`{}`。排队或运行中任务返回202，字段为`training_id:string`、`status:string`、`requested_at:string`、`message:string`。运行中先为cancel_requested，在下一安全检查点停止；不承诺强行即时中断GPU算子。已经cancelled时重复调用200并返回原记录；已成功、失败或中断任务返回409，不通过取消删除已完成模型。

## 7. 调参接口

### 7.1 POST /tuning-jobs

调参产生推荐配置和冻结训练计划。工作进程复用[tune.py](../joint_emaformer/tune.py)使用的候选生成和训练算法，以及[搜索范围配置](../joint_emaformer/configs/search.yaml)，按本节规则校验并记录任务。生产权重通过后续训练任务生成，测试成绩不参与本次搜索排名。

| 请求字段 | 类型 | 必填及默认 | 含义 |
| --- | --- | --- | --- |
| `profile` | string | 可选，selected | all或selected，一次只搜索一个变量方案 |
| `data_version` | string或null | 可选，当前默认 | 接受时绑定的数据版本 |
| `name` | string或null | 可选，自动生成 | 页面任务名称，最多100字符 |
| `preset` | string | 可选，standard | quick快速检查、standard标准、thorough较大预算；见下表 |
| `split_mode` | string | 可选，auto | auto自动边界，custom手动边界 |
| `holdout_months` | integer | 仅auto，默认3 | 从末尾保留的测试月份数，至少3 |
| `validation_months` | integer | 仅auto，默认20 | 初筛验证标签区间长度，至少3个月 |
| `train_end_month` | string | custom必填 | 初筛训练截止月 |
| `validation_end_month` | string | custom必填 | 开发期末月，即本次参数选择截止月 |
| `test_end_month` | string | custom可选，数据末月 | 保留测试末月；须在开发期后至少3个月 |
| `search_trials` | integer | 可选，随preset | 包括初始候选在内的不同候选总数；建议上限256 |
| `finalists` | integer | 可选，随preset | 前几名进入复核，至少1且不大于候选数；建议上限16 |
| `validation_folds` | integer | 可选，随preset | 开发期内复核折数，建议1至5 |
| `fold_months` | integer | 可选，随preset | 每折完整验证月份数，至少3，建议上限60 |
| `include_initial_candidates` | boolean | 可选，true | 是否优先评估现有8组初始配置 |
| `search_space` | object或null | 可选，随包范围 | 覆盖搜索范围中的指定字段；未给字段沿用默认范围 |
| `max_duration_minutes` | integer或null | 可选，服务器策略 | 从实际开始执行计算的总时限；不是排队时限。达到安全检查点后中断，保留部分记录 |

preset预算如下；standard读取服务端search_config指定的搜索配置，quick和thorough为内置预算，搜索范围仍读取同一配置：

| preset | 候选数 | 前列复核数 | 折数 | 每折月份 | 内部重复次数 |
| --- | --- | --- | --- | --- | --- |
| quick | 16 | 2 | 2 | 12 | 3 |
| standard | 64 | 4 | 3 | 20 | 3 |
| thorough | 128 | 8 | 3 | 20 | 3 |

内部复核使用现有固定随机性设置，种子不作为公开请求或响应字段。标准预算为64次初筛拟合，加4×3×3=36次复核拟合，共100次拟合/变量方案；失败拟合也消耗预算，不能把候选数理解成总训练次数。两个方案需要分别提交。

auto模式：`validation_end=数据末月-holdout_months`，`train_end=validation_end-validation_months`，测试从开发期次月开始。custom模式须满足训练末月早于验证末月、验证末月早于测试首月，且各评分段至少能构造完整3个月标签。每个复核折都位于开发期内，折的训练范围仅截至其验证首月之前；检查最长窗口及最早复核训练历史是否足够。模式专属字段混传返回422。

**自动日期划分与独立任务配置：**

日常自动调参只需选择数据版本和变量方案，其余请求字段可以省略。搜索范围、候选数、复核折数及训练设置读取服务端搜索配置，日期根据本次数据版本的实际末月计算，不要求用户手动填写参数或计算日期。留出测试默认3个月、初筛验证默认20个月，由服务端策略统一管理；下表示例采用这组默认规则。

| 调参阶段 | 数据截至2026-05时的范围 | 数据更新至2026-06时的范围 | 用途 |
| --- | --- | --- | --- |
| 初筛训练 | 2015-01至2024-06 | 2015-01至2024-07 | 拟合各候选模型 |
| 初筛验证 | 2024-07至2026-02 | 2024-08至2026-03 | 选择参数和训练轮数 |
| 保留测试 | 2026-03至2026-05 | 2026-04至2026-06 | 不参与选参，供正式训练后检验效果 |

表中假设历史首月为2015-01，实际首月以本次采用的数据范围和服务端策略为准。复核仍按搜索配置，在开发期内自动构造各折，不只重复同一初筛验证段。数据不足以覆盖最长候选窗口和全部复核折时，提交前返回具体原因，不读取测试月份补足历史。

“独立任务配置”指在接受任务时绑定本次数据版本、变量清单、搜索配置、实际月份边界及代码来源，并将解析结果随任务记录保存到第12.1节已有的`effective_config.json`中。任务执行时始终使用这份配置；排队期间上传新数据、切换默认数据或修改服务端配置，不回写已经接受的任务。用户无需创建或编辑配置文件，原实验的公共config.yaml和search.yaml保持原样，继续用于原实验复现。

命令行tune.py的初筛训练末月和开发期末月继续读取公共配置中的period.train_end与period.validation_end，以便复现原实验。自动接口的执行适配层使用本任务计算出的月份边界和数据快照调用同一算法，独立记录配置。参数搜索、插补、标准化、早停和候选排名只使用开发期；原表中已经存在的保留测试真实值也不参与选参。

数据版本内的测试月份可以已存在，但搜索代码只能切出截至validation_end的开发历史来拟合、插补、标准化、挑选轮数和排名。全变量有效列也仅用开发期的训练历史确定，并冻结为所有候选共用布局；不能根据测试段缺失情况筛变量。固定组合不自动重新跑XGBoost；较早复核折如果早于既有特征组合的选择时间，要标明是固定事后组合下的回顾验证。

组合变量方案的开发期末月不得早于公共配置记录的既定变量选择截止月；较早月份可作为开发期内的回顾复核折，但不能将尚未知晓该组合的历史月份冒充选参后的独立测试。

最简请求：

```json
{
  "profile": "selected",
  "data_version": "data_111111111111111111111111"
}
```

成功后，页面取得tuning_id并展示推荐方案。选择“按此方案训练”时，使用第6.1节的最简训练请求即可；技术参数无需再次提交。请求省略技术字段不会减少审计信息，实际时间范围、参数、搜索预算、冻结训练计划及评价结果仍通过任务详情返回并持久化。

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

成功202返回TuningTask。校验失败422，不启动GPU任务。时限到达后状态为interrupted，原因TIME_LIMIT；已完成记录可查询，但第一版不把部分搜索冒充最终推荐，也不允许训练接口引用未成功任务。

### 7.2 搜索范围与ModelParameters

search_space的每个键除lr_log10外均接收非空候选数组；数组元素类型见下表，未知键返回422。lr_log10是两个有限数构成的递增区间，实际学习率按10的该指数范围对数均匀抽取。ModelParameters是每项均已取定单值的对象，其中使用`lr:number`替代`lr_log10`，其余28项设置名称与配置一致。

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

还需检查隐藏维度能被头数整除、Dropout在[0,1)、学习率为正、窗口和网络规模不超过服务器预算等。level模式residual_scale规范为1，单层预测头head_multiplier规范为1；这类无实际影响的差异不重复计为新候选。

默认情况下保留原8组初始候选，包括4轮配置。用户覆盖某个搜索字段后，剔除不符合该字段有效约束的初始候选；未覆盖字段仍允许原预设。若不同有效候选不足search_trials，提交前返回422，不能无限循环采样。请求中的候选值也须遵守服务器给出的规模上限。

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

无请求体，可选`profile`、`status`、`data_version`过滤。200返回`items: array[TuningTask]`，包含符合条件的全部调参任务。

### 7.5 GET /tuning-jobs/{tuning_id}

无请求体，200返回TuningTask，不存在404。页面自动轮询此接口；搜索过程失败的个别候选不立即使整项任务失败，最终成功候选或完整复核候选不足时才标failed并解释。

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

复核RegionMetric的origins和各步observations为各次验证预测配对的总计数，重叠历史和不同重复保留，不解释为独立样本数。MAPE按各次完整拟合的分步指标等权平均，故不能将所有行简单拼接后平均来代替；界面必须标注“折与重复等权验证”。未完成全部复核的候选不能参与最终推荐。

当前选择规则沿用脚本：先按三地区等权平均验证MAPE初筛，复核再对地区、折和内部重复等权平均；并列时比较最差地区MAPE，再比较trial编号。最终轮数取最后一折三个最佳轮数的中位数，冻结对应学习率轨迹。结果称“本次候选中的推荐配置”，不宣称全局最优。真正产生权重须随后POST /trainings并传tuning_id。

### 7.7 POST /tuning-jobs/{tuning_id}/cancel

提交`{}`。响应与训练取消相同，但编号字段为`tuning_id`。有效取消请求202返回`tuning_id`、`status`、`requested_at`、`message`；已cancelled重复200，其他终态409。停止后保留已完成候选和错误记录，不能把没有完成全部复核的阶段最好结果自动发布成模型。

## 8. 预测接口

### 8.1 POST /predictions

JSON请求，至少提供`{}`。请求字段如下：

| 请求字段 | 类型 | 必填及默认 | 含义 |
| --- | --- | --- | --- |
| `data_version` | string或null | 可选，当前默认数据 | 使用已登记的数据快照，不接收本地/服务器路径 |
| `model_version` | string或null | 可选 | 显式选一个已登记模型；未加载时按受控缓存策略加载 |
| `profile` | string | 可选，未选模型时selected | 未传model_version时选择该方案默认模型；传了model_version则从版本元数据推断方案 |

若同时明确传profile和model_version，二者必须匹配，否则422。不能先把缺省profile强行填成selected，再拒绝单独传入的all版本；需要识别字段是否实际提供。显式版本号不改变全局默认。

最简调用：

```json
{}
```

从页面下拉框选择特定数据与权重：

```json
{
  "data_version": "data_111111111111111111111111",
  "model_version": "selected-a03df2b66e12a63a"
}
```

成功200返回PredictionResponse，完整字段见第9节。一次生成新prediction_id并保存完整响应，三地共9条价格。相同Idempotency-Key的重试复用第一次结果；没有该请求头的每次POST仍产生新记录。

接受后固定模型版本、数据版本和区间配置，即使默认关系随后变更，本次结果仍使用开始时的组合。检查数据覆盖权重历史首月、最新观测月不早于训练/选参截止月、列布局与保存预处理一致。非有限输出或非正预测价格返回明确计算错误，不填0或静默裁剪。价格区间缺失不阻止有效点预测，返回available=false及原因。

### 8.2 GET /predictions

用于页面历史记录，无请求体。

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

无请求体，成功200返回已保存的PredictionResponse，不重新预测。价格、区间、评价、模型与数据版本、generated_at均属于原生成时刻，不跟随当前默认设置变化；不存在404。仅旧字段名迁移可由兼容层处理，不重算或覆盖历史文件。

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

不含seed列。区间缺失不填0；校准数量、原因、历史评价和数据版本通过完整JSON查询。下载不重新计算；不存在的记录404。

## 9. 预测与评价的公共返回结构

本节列出全部业务字段。数组中的单条元素按对应对象解释，不把嵌套字段当作请求参数。样例真实价格和既有指标见[当前组合变量响应](../reference/service_validation/selected_response.json)及[当前全变量响应](../reference/service_validation/all_response.json)；它们属于现有服务的历史验证样例，编号不能直接用于另一个存储目录，也不证明本文新接口已经实现。

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

每个地区每个预测步独立选取目标月不晚于输入末月的已成熟绝对误差。n条误差排序，半宽取第`min(ceil((n+1)*confidence),n)`个值；默认n=24、confidence=0.95时取24条中的最大值，区间为点预测加减该半宽。

历史覆盖率按每个旧起点当时已经成熟的误差计算，不能用未来误差回填。区间是训练协议的历史误差参考，不保证固定当前权重的未来覆盖率；也不是MAPE的Bootstrap置信区间。更换结构、变量、训练计划或修订参考历史后，只有通过匹配检查的证据才能继续使用。没有匹配证据或残差不足时返回null上下限，不用MAPE范围替代价格区间。

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
| `training_time_status` | string | recorded或not_recorded |
| `model_training_cutoff` | string | 权重拟合截止月 |
| `parameter_selection_cutoff` | string | 参数及变量方案选择所用开发期截止月 |
| `latest_observed_month` | string | 本次选择的数据末月 |
| `data_imported_at` | string | 这份数据内容版本首次导入时间 |
| `data_version` | string | 本次实际使用的数据版本，未来可在请求中选择 |
| `historical_reference_matches` | boolean | 与登记证据参考历史是否一致；不等同于能否执行点预测 |
| `seq_len` | integer | 此权重历史窗口月数 |
| `pred_len` | integer | 固定3 |
| `input_features` | object | 三地区编码分别对应输入列名数组，顺序严格对应权重，均包含历史目标价格 |

原权重训练月份与新预测输入末月可以不同：更新输入不会自动重训。首次登记时间、加载时间、数据导入时间都不能冒充训练执行时间。

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

最近20个完整起点意味着各地区各步20组、三步合计60组，不是三个地区总计60组。只有3个月留出标签时，各地区各步1组、三步3组，页面明确显示样本数。固定权重测试和逐月重训回测不能合成一个平均分。

历史评价不可用时保留Evidence结构，时间字段为null、regions为空、reason说明原因。完整滚动重训回测暂沿用现有CLI，本文不新增回测任务接口；训练接口自动登记的是本权重的留出测试。新结构没有对应回测证据时，不能假装训练一次就已经获得24条可校准残差。

### 9.5 页面指标表

统一显示：地区、变量方案、第1个月MAPE、第1个月精度得分、第2个月MAPE、第2个月精度得分、第3个月MAPE、第3个月精度得分、平均MAPE、平均精度得分；均为百分数。表头同时显示评价范围、协议和每步样本数。

业务分档可沿用：得分不低于80为“较好”；70至不足80为“可用”；60至不足70为“需关注”；低于60为“需复核”。这是可配置业务提示，不是统计显著性结论，不能据此保证未来误差。没有评价时显示“暂无可用历史评价”，不显示0分或100分。

## 10. 统一错误响应与冲突处理

错误响应统一如下；request_id与响应头X-Request-ID一致，可用于日志查询。

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

| HTTP状态 | 典型error | 说明 |
| --- | --- | --- |
| 401 | UNAUTHORIZED | 已启用密钥但没有提供正确值 |
| 404 | NOT_FOUND | 数据、模型、任务或预测编号不存在 |
| 405 | METHOD_NOT_ALLOWED | 路径存在，但HTTP方法不支持 |
| 409 | VERSION_CONFLICT、INVALID_STATE | 默认关系已变化、幂等键冲突、无法取消终态任务、没有可下载测试等 |
| 413 | FILE_TOO_LARGE | 单文件、请求或XLSX解压规模超限 |
| 415 | UNSUPPORTED_FORMAT | 上传文件不是要求的XLSX |
| 422 | INVALID_REQUEST、INVALID_DATA、INCOMPATIBLE_MODEL | 参数、表结构、时间边界或所选权重与数据不匹配 |
| 429 | QUEUE_FULL | 排队容量已满，响应可带Retry-After，不伪装为训练成功 |
| 500 | INTERNAL_ERROR | 未预期服务异常，返回摘要并在日志保存详情 |
| 503 | WORKER_UNAVAILABLE、MODEL_LOAD_FAILED | 工作进程不可用、资源不足或模型无法加载 |

资源不存在优先404；结构校验失败422；资源存在但状态不允许操作409。任务在接受后发生训练错误，通过GET任务的status/error表达，不把每次状态查询都变为500。取消、默认切换和任务提交均需持久化及并发保护。

## 11. 处理表与变量约定

### 11.1 上传的25列

三地区各一个单工作表XLSX。第1行字段、第2行单位、第3行开始为月度记录。列名及顺序固定如下，表格无需装饰格式。日期存为YYYY-MM文本。上传过程不重新运行原始合并、GDP可用日期处理或汇率换算；这些应在生成处理表时完成。

| 字段 | 单位或格式 |
| --- | --- |
| 年月 | YYYY-MM |
| JKM现货价格 | 美元/百万英热单位 |
| LNG进口价格 | 元/吨 |
| TTF期货价格 | 美元/百万英热单位 |
| Henry Hub期货价格 | 美元/百万英热单位 |
| Brent原油价格 | 美元/桶 |
| 管道气进口价格 | 元/吨 |
| 美元/人民币汇率 | 人民币元/美元 |
| LNG进口量（全国） | 吨 |
| 管道气进口量（全国） | 吨 |
| LNG进口量（区域） | 吨 |
| 天然气产量（全国总量） | 亿立方米 |
| 天然气产量（区域） | 亿立方米 |
| LNG产量（全国） | 万吨 |
| 门站价基准价格（分省） | 元/立方米 |
| 中国液化天然气出厂价格指数 | 元/吨 |
| 工业增加值（分省） | % |
| 制造业PMI增速（全国） | % |
| GDP增速（分省） | % |
| 平均气温（分省） | ℃ |
| 采暖度日数HDD（分省） | ℃·日 |
| 非居民天然气消费量（分省） | 亿立方米 |
| 柴油价格（0#柴油） | 元/升 |
| 动力煤价格 | 元/吨 |
| 终端非居民天然气价格（分省） | 元/立方米 |

字段业务说明见[现有模板说明](README.md#get-templatesregion)。全国共同指标按当前data.py的COMMON清单核对三地一致；区域字段允许不同。非目标列可以缺失，实际选为输入后能否训练要检查对应拟合历史是否有可用值；目标标签永不插补。

### 11.2 两种变量方案

历史目标价格始终输入，不增加include_target开关。当前组合如下，接口展示中文变量清单；第一版用户选择方案，不通过请求任意修改列名。

| 地区 | selected输入 |
| --- | --- |
| 上海 | 历史目标价格、管道气进口价格、管道气进口量（全国）、LNG产量（全国） |
| 浙江 | 历史目标价格、TTF期货价格、制造业PMI增速（全国） |
| 江苏 | 历史目标价格、天然气产量（全国总量）、LNG产量（全国）、采暖度日数HDD（分省）、柴油价格（0#柴油） |

all按训练历史确定有效列，保留历史目标；浙江排除区域LNG进口量和区域天然气产量，江苏排除工业增加值、GDP和非居民消费量。表中仍保留这些模板列；排除输入不代表要求用户删除列。

重新训练只在相应训练历史拟合中位数、插补和标准化；每个历史预测窗口只读取当时可见输入。HDD沿用已处理表的原值，不额外裁剪、重算或作业务性改写；如模型选中的非目标输入缺失，按明示的missing策略处理。每份权重保存自己的变量顺序与预处理，推理不重新拟合。

## 12. 保存位置、页面与实施边界

### 12.1 建议运行目录

运行文件位于可配置storage_dir，默认outputs/service。业务响应返回编号与受控URL，文件路径只保存在内部任务和登记记录中。

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
│   ├── selected_parameters.json
│   └── tuning.log
├── predictions/<prediction_id>/response.json
├── registry.sqlite3
└── logs/events.jsonl
```

使用Python标准库SQLite维护数据/模型登记、默认指针、任务队列和幂等关系，默认切换以事务提交；文件保存实际数据和结果。模型登记指向任务产生的持久权重。训练原始审计内部保留复现所需信息，公开请求和CSV不展示seed。

上传产生新数据快照，不覆盖项目data目录、随包reference或旧快照。系统自动生成数据和训练模型的独立编号。models.yaml作为启动模型引导配置，填写独立model_version、权重路径及配套证据版本，运行中模型和默认关系由登记库管理。已登记文件按版本保留，替换权重应登记新版本。

已经提交的任务、已登记模型及预测都绑定其具体数据/配置/代码来源。新上传、新训练或新默认版本不回写历史结果。取消、失败和超时保留任务及完成记录供排查，不删除用户原始数据或已成功模型。

### 12.2 页面字段与按钮

| 页面 | 默认显示 | 主要操作 |
| --- | --- | --- |
| 数据 | 三个文件选择框、名称、说明、校验结果、月份范围 | 下载样例、上传；成功后选中新数据；另有“设为默认” |
| 预测 | 当前数据截至月、模型中文名和训练截至月、三个目标月份 | 开始预测、查看三地结果、下载明细 |
| 训练与检验 | 调参方案或随包配置、数据、留出3个月/全历史、实际日期范围 | 开始训练、查看进度、查看测试、选择新模型；选择调参方案时自动取得参数及绑定数据 |
| 调参 | 变量方案、数据、quick/standard/thorough、训练/验证/测试范围 | 开始搜索、查看排名、用推荐配置训练 |
| 模型 | 各版本名称、训练范围、历史指标及样本数、区间是否可用 | 选中单次使用、设为默认、回退旧版 |
| 历史记录 | 中文名称、生成时间、数据和模型版本、状态 | 点开详情、下载；无需手输编号 |

普通自动流程只让用户选择数据、变量方案、调参方案和训练用途，技术参数及日期由后台解析。训练高级设置默认折叠，调参默认使用preset；28项参数数组仅在高级区展开。表单提交前直接展示实际时间范围、计划拟合次数和目标月份，减少误操作，不额外要求用户计算日期。页面可每2秒查询运行任务，任务终态停止轮询；暂时关闭网页后从历史列表恢复查看。

“最新模型”和“当前默认模型”是不同概念，页面分别标识。模型下拉框按当前选定数据计算兼容性，不兼容项说明原因。缺少区间或评价证据时仍可显示有效价格，同时明确写“暂无匹配证据”，不捏造精度或上下限。

### 12.3 完整调用顺序

1. GET /templates/{region}下载三地样例，准备处理表。
2. POST /datasets上传三表，取得data_version；页面自动将它作为本页后续操作的数据。
3. GET /models?data_version=该版本查看能用的模型；直接POST /predictions也可，不必先训练。
4. 要检验现有结构时POST /trainings，mode=holdout；轮询任务，完成后查看evaluation并下载测试CSV。
5. 要搜索结构时POST /tuning-jobs，轮询后GET结果；成功后POST /trainings并引用tuning_id。训练和测试仍保持留出边界。
6. 完成历史评价并确定配置后，如需全部历史权重，再提交mode=full_history并使用同一成功tuning_id或固定最终配置。
7. 可以直接POST /predictions指定新model_version及data_version；需要更新全局默认时，POST /models/{model_version}/activate，并在必要时同时指定新data_version。
8. GET /predictions展示历史，点击记录查询原响应或下载CSV。真实价格成熟后的误差归集、重新回测和证据更新仍需单独的运维流程。

### 12.4 实现与运行说明

- **数据**：支持三表上传、不可变版本、显式数据选择和默认切换，不修改原配置表。启动时导入配置数据；已持久化的默认数据优先，更新配置表只登记新版本，使用新版本须显式选择或设为默认。
- **权重**：支持多版本登记、按版本选择、受控加载缓存和原子热切换。运行中由登记库管理默认模型，models.yaml用于启动导入。
- **任务**：HTTP接受任务后由独立工作进程执行。按本次数据末月生成独立配置，不覆盖原YAML；按tuning_id取得方案、绑定数据及冻结训练计划。实时状态以SQLite记录为准，task.json为提交或结束时的快照。
- **指标**：新调参使用最佳轮验证预测计算各地区第1、2、3个月的MAPE和得分；复核对折与内部重复等权汇总。既有实验记录保持原样。
- **变量布局**：每个新版本保存自己的布局与参考数据版本；历史窗口、预处理、变量顺序均随权重保存。新变量或结构须有对应证据才能返回校准区间。
- **证据**：固定权重测试必须绑定实际权重；滚动重训及价格区间按结构、变量、预处理、训练计划和参考历史匹配。完整回测继续使用现有CLI，暂不因一次训练就自动启动全历史回测。
- **持久化**：启动时导入既有数据快照、随包模型和已有预测，保留原文件及首次登记时间。活动工作进程继续运行，已退出的运行任务标为interrupted；停止服务时只终止本服务创建并持有句柄的工作进程。
- **上传依赖**：固定使用python-multipart==0.0.32，已写入部署依赖。[FastAPI文件上传说明](https://fastapi.tiangolo.com/tutorial/request-files/)
- **后台计算**：训练/搜索采用单机持久任务队列加独立进程，初期不要求Redis或Celery；并发数及资源预算由维护人员设置。较重计算与轻量应用内后台任务的区别见[FastAPI后台任务说明](https://fastapi.tiangolo.com/tutorial/background-tasks/#caveat)。
- **状态码**：202表示已接受、尚未完成；任务结果通过状态查询获取，语义见[HTTP 202规范](https://www.rfc-editor.org/rfc/rfc9110.html#name-202-accepted)。
- **兼容性**：服务版本为3.0.0。预测成功响应保留六个顶层字段，CSV保留原12列；model_status不再返回文件摘要。正常停止旧服务后启动新服务，旧数据、模型、预测及任务编号与默认关系保留；历史响应移除已停用字段并更新证据编号，价格和指标不重算。旧未完成任务标记interrupted并需重新提交，旧幂等缓存清空；升级后首次操作使用新的Idempotency-Key。旧数据24位编号继续可用。

### 12.5 实施后的验收项目

| 场景 | 应核对的结果 |
| --- | --- |
| 上传三地正确处理表 | 新数据201或重复200，月份一致，原表不变 |
| 日期断月、目标缺失、单位错误或全国指标冲突 | 返回具体地区/月份/字段，旧默认不变 |
| 同一幂等键网络重试 | 不重复训练、预测或切换；不同请求复用键409 |
| 留出3个月 | 训练截至末月减3；每地区每步1组指标与逐条预测一致 |
| 多月留出 | 完整起点数=留出月数减2，分步MAPE与明细可复算 |
| 全历史训练 | 使用全部历史；无测试标签时不伪造测试MAPE |
| 调参 | 所有拟合/选轮数/排名不读取保留测试；预算、复核次数及推荐规则可查 |
| 自动日期划分 | 数据截至2026-05时保留测试为2026-03至2026-05；更新至2026-06后为2026-04至2026-06，原实验配置不变 |
| 独立任务配置 | 接受后绑定数据版本和实际参数；排队期间变更默认数据或服务配置不改变任务，详情与执行范围一致 |
| 调参后训练 | 参数、布局、冻结轮数与学习率轨迹与推荐记录一致 |
| 最简调参后训练请求 | 仅传tuning_id和mode时正确推断方案及绑定数据，all方案不会被缺省selected误覆盖 |
| 指定历史模型版本 | 使用该权重及其保存预处理，不因默认版本变化而改用其他权重 |
| 热启用和回退 | 校验成功后原子切换；失败保留旧版；历史预测不改变 |
| 缺少评价或区间证据 | 点预测有效时仍返回价格，相关available=false并明确原因 |
| 重启、失败或取消 | 任务和历史可查，不把中断当成功，不覆盖已完成权重 |
| Windows CPU与Linux CUDA部署 | 各自在目标机验证上传、训练、调参、切换和预测；既有Windows CPU报告不能代替Linux GPU验收 |

本文示例编号仅用于说明请求结构。实际数据、模型、训练和调参编号均从对应接口返回值取得；运行服务的/openapi.json提供当前接口结构。
