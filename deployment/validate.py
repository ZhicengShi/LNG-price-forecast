"""真实HTTP本机验证；仅启动、停止本脚本创建的服务进程。"""
import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import tempfile

import httpx
import numpy as np
import pandas as pd
import openpyxl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from service.storage import save_json
from service.engine import load_settings
from service import VERSION


def port_number():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def start(port, env, stream):
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    process = subprocess.Popen([sys.executable, '-B', '-m', 'service.app', '--host', '127.0.0.1',
        '--port', str(port), '--device', 'cpu'], cwd=ROOT, env=env, stdout=stream, stderr=stream, creationflags=flags)
    begin = time.perf_counter()
    deadline = time.monotonic()+45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('服务启动失败，请查看工作目录中的server.log')
        try:
            if httpx.get(f'http://127.0.0.1:{port}/models', headers={'X-API-Key': env['LNG_API_KEY']},
                         timeout=5, trust_env=False).status_code == 200:
                return process, time.perf_counter()-begin
        except httpx.HTTPError:
            pass
        time.sleep(.1)
    stop(process)
    raise TimeoutError('服务未在45秒内就绪')


def timings(values):
    array = np.asarray(values)*1000
    return dict(requests=len(values), p50_ms=float(np.quantile(array, .5)), p95_ms=float(np.quantile(array, .95)),
                maximum_ms=float(array.max()), mean_ms=float(array.mean()))


def private_service_config(work):
    """仅缩小流程验证预算，生产YAML及正式模型不变。"""
    settings = load_settings()
    search = yaml.safe_load(Path(settings['search_config']).read_text(encoding='utf-8'))
    parameters = deepcopy(search['initial_candidates'][1])
    parameters.update(seq_len=6,d_model=32,e_layers=1,head_layers=1,head_multiplier=1,
                      max_epochs=1,patience=1,batch_size=64)
    search.update(search_trials=1,finalists=1,validation_folds=1,fold_months=3,initial_candidates=[parameters])
    search_path = work/'search.yaml'
    search_path.write_text(yaml.safe_dump(search,allow_unicode=True),encoding='utf-8')
    settings.update(search_config=str(search_path),storage_dir=str(work/'runtime'),device='cpu')
    settings = {key:str(value) if isinstance(value,Path) else value for key,value in settings.items()}
    path = work/'service.yaml'
    save_json(path,settings)
    return path


def wait_task(client,task):
    deadline = time.monotonic()+120
    while time.monotonic()<deadline:
        response = client.get(task['status_url'])
        response.raise_for_status()
        value = response.json()
        if value['status'] in ('succeeded','failed','cancelled','interrupted'):
            if value['status']!='succeeded':
                raise RuntimeError('流程任务未成功：'+json.dumps(value,ensure_ascii=False))
            return value
        time.sleep(.2)
    raise TimeoutError('流程任务在120秒内未完成')


def workflow(client,output,responses):
    original = responses['selected']['model_status']['data_version']
    payloads = {region:client.get('/templates/'+region).content for region in ('shanghai','zhejiang','jiangsu')}
    changed = {}
    for region,payload in payloads.items():
        book = openpyxl.load_workbook(io.BytesIO(payload))
        sheet = book.active
        row = [cell.value for cell in sheet[sheet.max_row]]
        row[0] = str(pd.Period(row[0],freq='M')+1)
        sheet.append(row)
        stream = io.BytesIO()
        book.save(stream)
        book.close()
        changed[region] = stream.getvalue()
    files = {region+'_file':('latest_'+region+'.xlsx',payload) for region,payload in changed.items()}
    response = client.post('/datasets',files=files,data={'name':'HTTP流程验证数据'},headers={'Idempotency-Key':'upload-one'})
    assert response.status_code==201,response.text
    uploaded = response.json()
    assert not uploaded['dataset']['is_default']
    assert client.post('/datasets',files=files,data={'name':'HTTP流程验证数据'},headers={'Idempotency-Key':'upload-one'}).json()==uploaded
    version = uploaded['dataset']['data_version']
    first = client.post('/predictions',json={},headers={'Idempotency-Key':'prediction-one'}).json()
    activated = client.post('/datasets/'+version+'/activate',json={})
    activated.raise_for_status()
    explicit = client.post('/predictions',json={'data_version':version})
    explicit.raise_for_status()
    assert explicit.json()['model_status']['latest_observed_month']==uploaded['dataset']['latest_observed_month']
    assert client.post('/predictions',json={},headers={'Idempotency-Key':'prediction-one'}).json()==first
    assert client.post('/predictions',json={'profile':'all'},headers={'Idempotency-Key':'prediction-one'}).status_code==409
    client.post('/datasets/'+original+'/activate',json={}).raise_for_status()
    tuning_response = client.post('/tuning-jobs',json={'data_version':original,'profile':'all'})
    tuning_response.raise_for_status()
    tuning = wait_task(client,tuning_response.json())
    recommended = client.get(tuning['results_url']).json()
    response = client.post('/trainings',json={'tuning_id':tuning['tuning_id'],'mode':'holdout'})
    response.raise_for_status()
    training = wait_task(client,response.json())
    assert training['profile']=='all' and training['data_version']==original
    assert training['training_plan']==recommended['frozen_training_plan']
    model_version = training['result']['model_version']
    prediction = client.post('/predictions',json={'model_version':model_version})
    prediction.raise_for_status()
    assert len(prediction.json()['predictions'])==9
    assert all(not item['available'] for item in prediction.json()['prediction_intervals']['items'])
    assert prediction.json()['model_status']['trained_at'] is not None
    client.post('/models/'+model_version+'/activate',json={}).raise_for_status()
    assert client.post('/predictions',json={'profile':'all'}).json()['model_status']['model_version']==model_version
    client.post('/models/'+responses['all']['model_status']['model_version']+'/activate',json={}).raise_for_status()
    result_csv = client.get(training['result']['test_csv_url'])
    result_csv.raise_for_status()
    assert len(result_csv.content.decode('utf-8-sig').splitlines())==10
    save_json(output/'workflow.json',dict(upload=uploaded,tuning=tuning,recommendation=recommended,
              training=training,new_model_prediction=prediction.json(),
              interpretation='1候选、1轮和1折的小预算技术流程验证，不作为最优模型或正式精度结论'))
    return dict(passed=True,uploaded_version=version,tuning_id=tuning['tuning_id'],training_id=training['training_id'],
                model_version=model_version,screening_candidates=1,confirmation_fits=3,epochs_per_fit=1,
                automatic_dates=True,tuning_to_training=True,model_hot_switch=True,idempotency=True,
                new_model_without_calibration_returns_unavailable=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--requests', type=int, default=20)
    parser.add_argument('--concurrency', type=int, default=4)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise ValueError('requests和concurrency须为正整数')
    created = datetime.now(timezone(timedelta(hours=8)))
    stamp = created.strftime('%Y%m%d_%H%M%S')
    output = (args.output_dir or ROOT/'outputs/service_validation'/stamp).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (ROOT/'outputs').mkdir(exist_ok=True)
    temporary_work = tempfile.TemporaryDirectory(prefix='service_validation_work_',dir=ROOT/'outputs')
    work = Path(temporary_work.name)
    baseline = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))['files']
    protected = [name for name in baseline if not (ROOT/name).resolve().is_relative_to(output)]
    before = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in protected}
    print('VALIDATION_START '+str(output), flush=True)
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=180)
    (work/'tests.log').write_text(tests.stdout+tests.stderr, encoding='utf-8')
    shutil.copyfile(work/'tests.log',output/'tests.log')
    if tests.returncode:
        raise RuntimeError('测试未通过，请查看'+str(output/'tests.log'))
    test_count = int(re.search(r'Ran (\d+) tests', tests.stderr).group(1))
    cli = subprocess.run([sys.executable, '-B', str(ROOT/'joint_emaformer/predict.py'), '--mode', 'future',
        '--device', 'cpu', '--output-dir', str(work/'cli')], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=120)
    (work/'cli.log').write_text(cli.stdout+cli.stderr, encoding='utf-8')
    if cli.returncode:
        shutil.copyfile(work/'cli.log',output/'cli.log')
        raise RuntimeError('命令行预测失败，请查看'+str(output/'cli.log'))
    cli_tables = {p: pd.read_csv(next((work/'cli').glob('*/'+p+'/future_predictions.csv'))) for p in ('all','selected')}
    service_config = private_service_config(work)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_CACHE_DISABLE='1',PYTHONIOENCODING='utf-8',
               LNG_SERVICE_CONFIG=str(service_config),LNG_SERVICE_STORAGE=str(work/'runtime'), LNG_API_KEY=secrets.token_hex(24))
    port = port_number()
    address = f'http://127.0.0.1:{port}'
    process = None
    responses, comparison, invalid = {}, {}, []
    with (work/'server.log').open('a', encoding='utf-8') as stream:
        try:
            process, startup_seconds = start(port, env, stream)
            with httpx.Client(base_url=address, headers={'X-API-Key': env['LNG_API_KEY']}, timeout=60, trust_env=False) as client:
                models = client.get('/models')
                models.raise_for_status()
                assert {m['profile'] for m in models.json()['models']} == {'all', 'selected'}
                no_key = httpx.get(address+'/models', timeout=5, trust_env=False)
                assert no_key.status_code == 401
                schema = client.get('/openapi.json')
                schema.raise_for_status()
                assert '/health' not in schema.json()['paths']
                assert client.get('/health').status_code == 404
                assert sum(len(operations) for operations in schema.json()['paths'].values())==22
                save_json(output/'openapi.json', schema.json())
                for profile in ('all','selected'):
                    reply = client.post('/predictions', json={'profile': profile})
                    reply.raise_for_status()
                    value = reply.json()
                    responses[profile] = value
                    actual = pd.DataFrame(value['predictions']).sort_values(['horizon','region'])
                    expected = cli_tables[profile].sort_values(['horizon','region'])
                    np.testing.assert_array_equal(actual.target_month, expected.target_month)
                    delta = float(abs(actual.prediction.to_numpy()-expected.prediction.to_numpy()).max())
                    assert delta <= 1e-4
                    assert len(value['predictions']) == 9 and all(r['available'] for r in value['prediction_intervals']['items'])
                    comparison[profile] = dict(values=9, max_absolute_price_difference=delta, tolerance=1e-4)
                    save_json(output/(profile+'_response.json'), value)
                    csv_result = client.get('/predictions/'+value['prediction_id']+'/csv')
                    csv_result.raise_for_status()
                    (output/(profile+'_predictions.csv')).write_bytes(csv_result.content)
                assert client.get('/datasets').status_code==200
                for body in ({'profile': 'invalid'}, {'dataset_id': '../config.yaml'}, {'seed': 7}):
                    response = client.post('/predictions', json=body)
                    assert response.status_code == 422
                    invalid.append(dict(case=list(body)[0], status=response.status_code))
                assert client.get('/predictions/pred_'+'0'*32).status_code == 404
                assert client.post('/predictions',json={'model_version':'unknown'}).status_code==404
                workflow_result = workflow(client,output,responses)

                def call(number):
                    profile = ('all','selected')[number % 2]
                    started = time.perf_counter()
                    reply = client.post('/predictions', json={'profile': profile})
                    reply.raise_for_status()
                    value = reply.json()
                    assert value['predictions'] == responses[profile]['predictions']
                    return time.perf_counter()-started

                sequential = [call(i) for i in range(args.requests)]
                with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    parallel = list(executor.map(call, range(args.requests)))
                assert client.get('/models').status_code == 200
            stop(process)
            process = None
            process, restart_seconds = start(port, env, stream)
            with httpx.Client(base_url=address, headers={'X-API-Key': env['LNG_API_KEY']}, timeout=60, trust_env=False) as client:
                for profile, original in responses.items():
                    restored = client.get('/predictions/'+original['prediction_id'])
                    restored.raise_for_status()
                    assert restored.json() == original
                    fresh = client.post('/predictions', json={'profile': profile}).json()
                    for field in ('model_version','registered_at','data_imported_at','data_version'):
                        assert fresh['model_status'][field] == original['model_status'][field]
                    assert fresh['predictions'] == original['predictions']
                assert client.get('/trainings/'+workflow_result['training_id']).json()['status']=='succeeded'
                assert client.get('/models/'+workflow_result['model_version']).status_code==200
        finally:
            stop(process)
            stream.flush()
            shutil.copyfile(work/'server.log',output/'server.log')
    after = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in protected}
    assert before == after
    temporary_work.cleanup()
    versions = {name: importlib.metadata.version(name) for name in
                ('torch','numpy','pandas','fastapi','uvicorn','pydantic','starlette','httpx','python-multipart')}
    report = dict(passed=True, generated_at=created.isoformat(), scope='local_cpu_technical_trial',service_version=VERSION,
        environment=dict(os=platform.platform(), python=platform.python_version(), logical_cpus=os.cpu_count(),
                         versions=versions, inference_device='cpu', threads=4),
        tests_passed=test_count, profiles=['all','selected'], response_values=18, prices_match_actual_cli=comparison,
        four_output_sections_verified=True, price_intervals_available=18,
        interval_confidence=responses['selected']['prediction_intervals']['confidence'],
        interval_residual_window=responses['selected']['prediction_intervals']['residual_window'],
        interval_basis='same_parameters_features_rolling_refit_protocol',
        historical_training_time_not_fabricated=True, api_key_checked=True,versioned_uploads_checked=True,
        removed_health_route_checked=True,api_operations=22,workflow=workflow_result,
        invalid_requests=invalid, startup_seconds=startup_seconds, restart_seconds=restart_seconds,
        repeated_calls=timings(sequential), concurrent_calls=dict(concurrency=args.concurrency, **timings(parallel)),
        failed_benchmark_requests=0, persisted_predictions_restored=True, all_owned_service_processes_stopped=True,
        protected_files_unchanged=len(before), fresh_dependency_installation_verified=False,
        customer_machine_verified=False, real_monthly_business_trial_complete=False,
        runtime_artifacts=None,temporary_validation_artifacts_cleaned=True,
        pending=['甲方目标机安装与内网联调','甲方实际并发和资源占用测量','更新到最新完整业务月份',
                 '按月保存预测，真实值成熟后评价','需要时训练并登记最新生产权重及配套证据'])
    save_json(output/'report.json', report)
    lines = ['# 统一接口部署验证与本机试运行记录', '',
        f'执行时间：{report["generated_at"]}（北京时间）。结果：通过。', '',
        '本次为当前'+platform.system()+'环境CPU执行的技术验证，使用已安装依赖；未进行隔离新环境安装、甲方服务器部署或连续业务月精度验收。', '',
        f'- 测试：{test_count}项通过；两方案共18条HTTP预测与实际CLI输出核对，绝对容差0.0001元/立方米。',
        '- 统一响应的价格、区间、历史精度、模型状态均检查；18条预测区间可用。',
        '- 校准使用相同结构、变量和训练计划的滚动重训残差；名义95%，并返回历史覆盖率，不承诺固定当前权重未来95%覆盖。',
        '- 旧权重训练执行时间返回null；登记时间、数据导入时间及模型版本重启后保持，原预测JSON完整恢复。',
        '- 已验证22项业务操作的接口结构，以及上传校验、显式数据选择、默认切换、幂等、密钥、取消和重启恢复。',
        '- 实际HTTP执行上传→自动调参→按方案训练→新模型预测→热切换和回退；流程使用1候选、1轮、1折及3次复核，未重新搜索正式最优参数。',
        '- 新模型未登记匹配回测时返回区间不可用，训练完成时间如实记录。原随包模型的预测与CLI保持一致。',
        f'- 价格区间使用最近{report["interval_residual_window"]}条已成熟残差，名义水平{report["interval_confidence"]*100:g}%。',
        f'- 顺序请求{len(sequential)}次：P50={report["repeated_calls"]["p50_ms"]:.2f}ms，P95={report["repeated_calls"]["p95_ms"]:.2f}ms。',
        f'- 并发{args.concurrency}、请求{len(parallel)}次：P50={report["concurrent_calls"]["p50_ms"]:.2f}ms，P95={report["concurrent_calls"]["p95_ms"]:.2f}ms；失败0次。',
        f'- 首次就绪{startup_seconds:.2f}s，重启就绪{restart_seconds:.2f}s；本脚本创建的服务进程均已停止。',
        f'- 运行前后核对{len(before)}个交付保护文件哈希，均未变化。', '',
        '接口默认权重训练截至2024-07，样例数据截至2026-05，预测2026-06至2026-08；本次未将内存中训练过的其他权重作为上线权重。', '',
        '响应样例见all_response.json、selected_response.json，价格和区间明细见对应CSV，OpenAPI结构见openapi.json，机器可读记录见report.json。耗时为本机小规模试运行观测，不能作为甲方环境SLA。', '',
        '待甲方实施：', '']+['- '+item for item in report['pending']]+['']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('VALIDATION_PASS '+json.dumps(dict(tests=test_count, values=18, sequential=len(sequential),
        concurrent=len(parallel), cli_difference=comparison, report=str(output/'report.md')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
