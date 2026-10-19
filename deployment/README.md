# 部署、验证与试运行

本目录提供三地区融合模型统一接口的安装和启动方式。原实验入口及默认权重保持各自复现用途。服务默认CPU、127.0.0.1:8000、单进程；预测复用常驻模型，通过锁串行执行模型计算。训练、调参和回测继续使用既有命令行入口，更新模型时先评估再登记新版本。

## Windows环境与启动

已验证环境为Windows、Python 3.11，接口可在CPU执行。requirements-common.txt固定当前服务和实验依赖闭包，两份执行方案均沿用已验证的torch 2.14.0+cu130构建；CPU方案显式指定cpu，不调用GPU，但下载包较大。CPU专用wheel或Linux部署需要在目标环境重新核验，不能把本机执行验证当作所有平台的安装验证。

从experiment_reproduction目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r deployment/requirements-cpu.txt
.\.venv\Scripts\python.exe -B -m service.app --device cpu
```

CUDA机器使用requirements-cuda.txt并指定--device cuda；对应驱动和CUDA可用性由目标机检查。已有相同依赖环境时可直接使用该环境的Python启动。当前验收使用既有环境，不代表已完成隔离新环境或内网离线安装。

也可调用Windows原生启动脚本，并明确指定Python（可使用绝对路径）：

```powershell
$env:LNG_PYTHON = 'D:/your_env/python.exe'
.\deployment\start.cmd --device cpu --port 8000
```

start.cmd从脚本所在位置定位包根目录，可从其他工作目录启动，不依赖PowerShell脚本执行策略。也保留PowerShell入口，供允许执行.ps1的环境使用：

```powershell
powershell -NoProfile -File deployment/start.ps1 -Python ".venv/Scripts/python.exe" -Device cpu
```

PowerShell入口的-Python和-Config可用绝对路径，-BindAddress和-Port设置监听地址及端口。当前验证机器禁止执行.ps1，未改变其策略，使用直接Python命令及start.cmd验证。运行后访问http://127.0.0.1:8000/docs或/openapi.json，调用/models查询模型及数据状态。/docs默认依赖公共CDN脚本，完全离线的内网使用接口命令或系统；接口计算、/openapi.json、数据和权重不依赖外网。

向内网其他机器提供访问时，设置API密钥并绑定内网地址，例如：

```powershell
$env:LNG_API_KEY = '替换为配置的密钥'
python -B -m service.app --host 0.0.0.0 --port 8000 --device cpu
```

业务接口请求需携带X-API-Key请求头，文档入口无需密钥；不设置密钥时默认不启用该校验。按内网策略放行指定端口及访问范围。启动参数单进程、不启用reload；服务进程常驻，按Ctrl+C停止。可由既有进程管理工具启动，并调用/models检查模型及数据是否可读取；启用密钥时检查请求也须携带密钥。重启时使用同一配置和存储目录，已有数据、首次登记时间及预测记录可恢复。接口说明见[service/README.md](../service/README.md)。

## Linux部署（CUDA）

以下为Linux x86_64、Python 3.11、NVIDIA GPU的部署步骤，命令使用Bash。复制整个experiment_reproduction目录，包含data、模型权重、reference和配置文件；在目标机重新创建虚拟环境，不复制Windows的.venv。部署账户须能读取数据和权重，并能写入配置的outputs运行目录。默认相对数据路径无需修改；若曾改为D:/等Windows绝对路径，请在config.yaml的data中改为Linux实际路径。文件名大小写须与配置一致。

### 环境检查

目标机须已安装Python 3.11及venv模块、curl，以及适配GPU的NVIDIA驱动。当前requirements-cuda.txt固定torch==2.14.0+cu130；[PyTorch官方安装索引](https://download.pytorch.org/whl/cu130/torch/)提供Python 3.11、Linux x86_64的对应wheel，标记为manylinux_2_28，需glibc至少2.28。[NVIDIA兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)列出的CUDA 13.x驱动最低主版本为580；GPU架构和驱动仍须通过下面的实际运算检查。

```bash
uname -m
ldd --version | head -n 1
python3.11 --version
nvidia-smi
```

uname -m应为x86_64；nvidia-smi须能识别GPU和驱动。安装依赖需要访问Python包索引与PyTorch下载源；离线内网须先在相同Linux架构和Python版本的环境准备包含全部传递依赖的wheel包。

### 安装依赖并检查GPU

将下面目录替换为实际部署位置，进入包根目录执行：

```bash
cd /path/to/experiment_reproduction
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r deployment/requirements-cuda.txt
python -m pip check
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
import torch

print('PyTorch:', torch.__version__)
print('PyTorch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA unavailable: check GPU, driver and PyTorch installation'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.ones((16, 16), device='cuda')
print('CUDA matrix result:', (x @ x)[0, 0].item())
torch.cuda.synchronize()
PY
```

期望PyTorch为2.14.0+cu130、PyTorch CUDA为13.0、CUDA available为True，矩阵结果为16.0。该检查同时实际执行GPU运算；不要仅凭nvidia-smi中的CUDA Version判断项目已在GPU上运行。Linux下CUDA相关传递依赖可能与Windows不同，安装完成后保存目标机依赖清单：

```bash
mkdir -p outputs/deployment
python -m pip freeze > outputs/deployment/linux_cuda_requirements.txt
```

### 启动与调用

本机试运行：

```bash
python -B -m service.app --config service/config.yaml --device cuda --host 127.0.0.1 --port 8000
```

前台运行时按Ctrl+C停止。--device cuda显式要求使用GPU；CUDA不可用会启动失败。多个GPU时可在启动前用CUDA_VISIBLE_DEVICES选择设备，例如0表示第1张GPU。使用现有Python模块入口即可启动，不需要Windows的start.cmd或start.ps1。

供内网其他机器访问时，在启动服务的终端执行以下命令；密钥替换为实际配置值：

```bash
export LNG_API_KEY='replace-with-your-api-key'
export CUDA_VISIBLE_DEVICES=0
python -B -m service.app --config service/config.yaml --device cuda --host 0.0.0.0 --port 8000
```

以上两条启动命令分别用于本机或内网，选择一种运行，避免同一端口重复启动。下面在另一个终端调用内网模式的服务；新终端须设置同一密钥。服务器本机可用127.0.0.1，其他机器将它替换为服务器实际IP，0.0.0.0只用于服务监听。

```bash
export LNG_API_KEY='replace-with-your-api-key'
curl -fsS http://127.0.0.1:8000/models \
  -H "X-API-Key: $LNG_API_KEY"
curl -fsS -X POST http://127.0.0.1:8000/predictions \
  -H "X-API-Key: $LNG_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"profile":"selected"}'
```

/models返回的models数组中各模型的device和预测结果的model_status.device应为cuda。将profile改为all可验证全变量方案。两方案每次都应返回三地区未来3个月的9条预测，以及价格区间、历史评价和模型状态。接口仍读取配置中的处理表，数据更新后重启生效；模型无需在启动时重新训练。

### 后台运行与停止

需要退出终端后继续运行时，可使用已有进程管理工具；短期试运行也可使用下面的nohup方式。先停止前台服务，在包根目录、已激活虚拟环境的终端执行：

```bash
export LNG_API_KEY='replace-with-your-api-key'
export CUDA_VISIBLE_DEVICES=0
mkdir -p outputs/service
nohup .venv/bin/python -B -m service.app --config service/config.yaml \
  --device cuda --host 0.0.0.0 --port 8000 \
  > outputs/service/server.log 2>&1 &
echo $! > outputs/service/server.pid
tail -n 50 outputs/service/server.log
```

后台启动后，等待日志显示Application startup complete，再查询模型及数据状态；终端须已设置与服务端相同的密钥：

```bash
curl -fsS http://127.0.0.1:8000/models \
  -H "X-API-Key: $LNG_API_KEY"
```

server.pid记录本次后台进程编号；确认对应进程仍为这次启动的服务后，可在包根目录用下面命令停止。重启继续使用同一配置与存储目录，即可保留已有预测记录和数据版本。

```bash
kill -TERM "$(cat outputs/service/server.pid)"
```

### Linux验证与试运行

依赖安装后，在已激活的虚拟环境中执行：

```bash
python -B deployment/validate.py --output-dir outputs/linux_cpu_validation
```

validate.py明确使用CPU，自动执行单元测试、真实HTTP调用、与CPU命令行预测核对、上传、自动调参到训练、模型热切换、并发与重启恢复。报告scope为local_cpu_technical_trial，environment.os记录实际系统；即使在CUDA服务器运行，该报告也只表示CPU接口流程验证结果。

CUDA验证须另行执行前面的GPU运算检查、CUDA服务启动，以及all和selected两次实际HTTP预测，并检查返回设备为cuda。可用同一虚拟环境、同一配置和数据运行以下CUDA命令行预测，核对其future_predictions.csv与对应接口的地区、目标月份、预测步和价格：

```bash
python -B joint_emaformer/predict.py --mode future --device cuda \
  --output-dir outputs/linux_cuda_cli
```

该命令默认输出两方案，结果位于outputs/linux_cuda_cli/future_时间目录/all和selected。使用相同数据、权重及设备时，每个方案逐条核对9个价格，以0.0001元/立方米绝对差作为本次接口与CLI的一致性核对容差，同时保留模型状态查询结果、预测响应及GPU信息。

Linux GPU目标机仍需按本节命令实测。随包reference/service_v3_validation为本机CPU技术试运行证据；Linux服务器完成安装、GPU调用、重启和内网联调后，另存实际环境与验收记录，再按后文开展逐月业务试运行。

## 验证命令

```powershell
python -B -m unittest discover -s tests -p "test_*.py" -v
python -B deployment/validate.py
```

validate.py在本机空闲端口启动并关闭自己创建的服务，核对两方案HTTP与CLI预测，并实际执行上传→自动调参→按方案训练→新模型预测→热切换和回退。流程验证使用独立配置中的1个候选、1轮、1折和3次复核，不改变生产搜索YAML，也不作为正式最优参数结论。结果默认位于outputs/service_validation/时间目录；临时数据、权重和CLI输出在验证结束后自动清理，保留报告、响应样例和必要日志。--output-dir指定报告目录，--requests和--concurrency设置调用次数与并发数；

本次随包结果位于[reference/service_v3_validation](../reference/service_v3_validation/report.md)。报告包含环境、测试数量、HTTP调用耗时、数值差、样例响应和待完成事项，workflow.json保存小预算调参和训练流程的实际响应。CPU价格一致性使用0.0001元/立方米绝对容差。
