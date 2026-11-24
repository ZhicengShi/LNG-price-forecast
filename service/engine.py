"""复用正式预测计算，组合价格、区间、历史评价及版本信息。"""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from datasets import load_global, read_frame, resolve_path
from CoRegionLng._runtime import configure, load_checkpoint
from CoRegionLng.data import REGIONS, NAMES, build_frame
from CoRegionLng.predict import forecast_future
from . import VERSION
from .intervals import future_intervals
from .registry import load_registry
from .schemas import PredictionResponse
from .storage import Store, now
from .catalog import Catalog
from .errors import APIError, issue
from .uploads import template_units, validate_tables
from provenance import without_file_checks

HERE = Path(__file__).resolve().parent


class DefaultModels:
    """按需取得默认版本，避免在缓存之外重复持有模型张量。"""
    def __init__(self,engine):
        self.engine = engine

    def __getitem__(self,profile):
        return self.engine.ensure_loaded(self.engine.catalog.default('model_'+profile)[0])

    def __setitem__(self,profile,model):
        # 默认指针在事务登记库中更新；此处不额外保留权重引用。
        return None

    def items(self):
        return [(p,self[p]) for p in ('all','selected')]

    def values(self):
        return [m for p,m in self.items()]


def load_settings(path=None):
    path = Path(path or os.environ.get('LNG_SERVICE_CONFIG', HERE/'config.yaml')).resolve()
    settings = yaml.safe_load(path.read_text(encoding='utf-8'))
    settings['search_config'] = resolve_path(settings.get('search_config') or HERE.parent/'CoRegionLng/configs/search.yaml', path)
    settings.setdefault('max_loaded_models', 2)
    settings.setdefault('idempotency_hours', 24)
    settings.setdefault('upload',{})
    for key,value in dict(file_bytes=20*1024**2,total_bytes=60*1024**2,expanded_bytes=200*1024**2).items():
        settings['upload'].setdefault(key,value)
    settings.setdefault('jobs', dict(holdout_months=3,validation_months=20,max_duration_minutes=None,max_queued=20))
    for key,value in dict(holdout_months=3,validation_months=20,max_duration_minutes=None,max_queued=20).items():
        settings['jobs'].setdefault(key,value)
    for key in ('data_config', 'model_registry', 'storage_dir'):
        settings[key] = resolve_path(settings[key], path)
    if os.environ.get('LNG_SERVICE_STORAGE'):
        settings['storage_dir'] = Path(os.environ['LNG_SERVICE_STORAGE']).resolve()
    if os.environ.get('LNG_SERVICE_DEVICE'):
        settings['device'] = os.environ['LNG_SERVICE_DEVICE']
    for key in ('threads', 'port', 'max_months', 'recent_origins'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key}须为正整数')
    if settings['port'] > 65535:
        raise ValueError('port超出有效范围')
    for value in (settings['max_loaded_models'],settings['idempotency_hours'],settings['jobs']['max_queued'],*settings['upload'].values()):
        if type(value) is not int or value<1:
            raise ValueError('缓存、幂等、队列及上传限制须为正整数')
    if any(type(settings['jobs'][key]) is not int or settings['jobs'][key]<3 for key in ('holdout_months','validation_months')):
        raise ValueError('自动划分的测试和验证月份数不能少于3')
    duration = settings['jobs']['max_duration_minutes']
    if duration is not None and (type(duration) is not int or duration<1):
        raise ValueError('执行时限须为正整数分钟或null')
    interval = settings['interval']
    if (not 0 < interval['confidence'] < 1 or type(interval['min_residuals']) is not int
            or type(interval['residual_window']) is not int
            or not 1 <= interval['min_residuals'] <= interval['residual_window']):
        raise ValueError('区间参数无效')
    return settings


def evidence(table, latest, count, matches, basis, version):
    empty = dict(available=False, reason=None, basis=basis, version=version,
                 first_forecast_month=None, last_forecast_month=None, last_target_month=None, regions=[])
    if not matches:
        return dict(empty, reason='历史数据与评价参考不一致，原指标不作为本数据版本的评价')
    if table is None:
        return dict(empty, reason='当前模型未登记此类评价证据')
    mature = table[table.target_month <= latest]
    complete = mature.groupby('origin_index').size()
    origins = sorted(complete[complete == 9].index)[-count:]
    part = mature[mature.origin_index.isin(origins)]
    if part.empty:
        return dict(empty, reason='截至最新观测月没有已成熟的完整三步评价起点')
    regions = []
    for region in REGIONS:
        group = part[part.region == region]
        steps = []
        for h in (1, 2, 3):
            g = group[group.horizon == h]
            mape = float(abs((g.prediction-g.actual)/g.actual).mean()*100)
            steps.append(dict(horizon=h, observations=len(g), mape_percent=mape, accuracy_score_percent=100-mape))
        mean = float(np.mean([s['mape_percent'] for s in steps]))
        regions.append(dict(region=region, region_name=NAMES[region], origins=len(origins), by_horizon=steps,
                            average_mape_percent=mean, average_accuracy_score_percent=100-mean))
    return dict(empty, available=True, first_forecast_month=part.first_forecast_month.min(),
                last_forecast_month=part.first_forecast_month.max(), last_target_month=part.target_month.max(), regions=regions)


class Engine:
    def __init__(self, settings):
        self.settings = settings
        self.store = Store(settings['storage_dir'])
        self.catalog = Catalog(self.store.root)
        from .migration import migrate_history
        migrate_history(self.store,self.catalog)
        self.shared = load_global(settings['data_config'])
        runtime = deepcopy(self.shared)
        runtime['runtime'].update(device=settings['device'], threads=settings['threads'])
        self.device = configure(runtime)
        self.lock = threading.RLock()
        self.models, self.references, self.reference_paths = load_registry(settings['model_registry'], self.device, self.store)
        self.units = template_units(self.reference_paths)
        self.cache = {model.version:model for model in self.models.values()}
        self.load_errors = {}
        self.loading = set()
        self.bootstrap_evidence = {m.version:(m.fixed_test,m.backtest,m.fixed_test_version,m.backtest_version) for m in self.models.values()}
        registry = yaml.safe_load(Path(settings['model_registry']).read_text(encoding='utf-8'))
        for profile,model in self.models.items():
            spec = registry['models'][profile]
            saved = {key:value for key,value in model.saved.items() if key != 'model_state_dict'}
            checkpoint_path = resolve_path(spec['checkpoint'],settings['model_registry'])
            bundled = checkpoint_path == (HERE.parent/'CoRegionLng/weights'/f'{profile}.pt').resolve()
            record = dict(model_version=model.version,profile=profile,saved=saved,
                checkpoint=str(checkpoint_path),registered_at=model.registered_at,
                source='bundled' if bundled else 'configured',training_id=None,tuning_id=None,
                training_data_version=None,reference_data_version=None,
                evidence_specs=deepcopy(spec),registry_path=str(settings['model_registry']),
                display_name=('全变量' if profile=='all' else '组合变量')+'｜训练截至'+saved['training_cutoff_month']+'｜随包' if bundled else profile+'｜配置导入',
                evidence='bootstrap')
            self.catalog.put('model',model.version,record,only_new=True)
            self.catalog.set_defaults({'model_'+profile:model.version},only_missing=True)
        payloads = {r: self.shared['data'][r].read_bytes() for r in REGIONS}
        metadata = self.store.capture_data(payloads, self.validate_paths)
        self.data_version = metadata['data_version']
        self.configured_data_version = self.data_version
        self.register_dataset(metadata,'configured',filenames={r:self.shared['data'][r].name for r in REGIONS})
        self.catalog.set_defaults({'data':self.data_version},only_missing=True)
        self.data_version = self.catalog.default('data')[0]
        for directory in (self.store.root/'data_versions').iterdir():
            if directory.name.startswith('data_') and (directory/'metadata.json').is_file():
                old_metadata,_ = self.store.data(directory.name)
                self.register_dataset(old_metadata,old_metadata.get('source','configured'))
        for path in (self.store.root/'predictions').glob('pred_*/response.json'):
            value = self.store.prediction(path.parent.name)
            self.catalog.put('prediction',value['prediction_id'],self.prediction_summary(value),only_new=True)
        for profile in ('all','selected'):
            self.models[profile] = self.ensure_loaded(self.catalog.default('model_'+profile)[0])
        self.models = DefaultModels(self)
        self.store.event('service_started', service_version=VERSION, device=str(self.device),
                         model_versions={r: m.version for r, m in self.models.items()}, data_version=self.data_version)

    def close(self):
        if hasattr(self,'jobs'):
            self.jobs.close()
        self.store.event('service_stopped')

    def start_jobs(self):
        from .tasks import JobManager
        self.jobs = JobManager(self)
        self.jobs.start()

    def sources(self, paths):
        result = {}
        for r in REGIONS:
            try:
                result[r] = read_frame(paths[r])
            except Exception as exc:
                raise ValueError(f'{NAMES[r]}数据表校验失败：{exc}') from exc
            if len(result[r]) > self.settings['max_months']:
                raise ValueError('数据月份数超过服务配置限制')
        return result

    def model_frame(self, sources, registered):
        saved = registered.saved
        start = pd.Period(saved['layout']['first_month'], freq='M')
        first = sources[REGIONS[0]]
        months = pd.period_range(start, first.index[-1], freq='M')
        if not len(months):
            raise ValueError('数据未覆盖模型要求的历史首月')
        sliced = {}
        for r in REGIONS:
            if not first.index.equals(sources[r].index):
                raise ValueError('三地区月份范围须严格一致')
            if not months.isin(sources[r].index).all():
                raise ValueError(f'数据须保留从{start}开始的完整观测历史')
            sliced[r] = sources[r].loc[months]
        if str(months[-1]) < max(saved['training_cutoff_month'], saved['selection_cutoff_month']):
            raise ValueError('最新观测月份早于模型训练或参数选择截止月')
        try:
            frame, _ = build_frame(sliced, saved['profile'], {}, saved_layout=saved['layout'])
        except AssertionError as exc:
            raise ValueError('三地区的全国共同指标不一致') from exc
        if len(frame) < saved['parameters']['seq_len']:
            raise ValueError('完整观测历史不足以构造预测窗口')
        return frame

    def validate_paths(self, paths):
        return validate_tables(paths,self.settings['max_months'],self.units)

    def reference_matches(self, sources, record=None):
        references = self.references
        if record and record.get('reference_data_version'):
            _,paths = self.store.data(record['reference_data_version'])
            references = self.sources(paths)
        for r in REGIONS:
            ref = references[r]
            months = ref.index[ref.index <= sources[r].index[-1]]
            if not months.isin(sources[r].index).all():
                return False
            if not np.allclose(ref.loc[months].to_numpy(dtype=float), sources[r].loc[months].to_numpy(dtype=float),
                               rtol=1e-12, atol=1e-12, equal_nan=True):
                return False
        return True

    def status(self, profile, metadata, matches, record=None):
        record = record or self.catalog.get('model',self.catalog.default('model_'+profile)[0])
        model = self.cache.get(record['model_version'])
        saved = record['saved']
        audit = saved.get('training_audit', {})
        trained_at = audit.get('finished_at') or audit.get('started_at')
        schema = saved['layout']['input_schema']
        return dict(model_name='CoRegionLng', service_version=VERSION, model_version=record['model_version'],
            profile=profile, device=str(self.device),
            registered_at=record['registered_at'], loaded_at=model.loaded_at if model else None, trained_at=trained_at,
            training_time_status='recorded' if trained_at else 'not_recorded',
            model_training_cutoff=saved['training_cutoff_month'], parameter_selection_cutoff=saved['selection_cutoff_month'],
            latest_observed_month=metadata['latest_observed_month'], data_imported_at=metadata['imported_at'],
            data_version=metadata['data_version'], historical_reference_matches=matches,
            seq_len=saved['parameters']['seq_len'], pred_len=3,
            input_features={r: schema['common']+schema[r] for r in REGIONS})

    def predict(self, request):
        with self.lock:
            record = self.resolve_model(request.profile,request.model_version)
            registered = self.ensure_loaded(record['model_version'])
            metadata, paths = self.store.data(request.data_version or self.catalog.default('data')[0])
            sources = self.sources(paths)
            frame = self.model_frame(sources, registered)
            forecasts = forecast_future(registered.model, frame, registered.saved)
            if any(not np.isfinite(row['prediction']) or row['prediction']<=0 for row in forecasts):
                raise APIError(422,'INVALID_OUTPUT','模型输出包含非正或非有限价格，请复核输入和权重')
            predictions = [dict(region=r['region'], region_name=NAMES[r['region']], target_month=r['target_month'],
                                horizon=r['horizon'], prediction=r['prediction'], unit='元/立方米') for r in forecasts]
            matches = self.reference_matches(sources,record)
            latest = metadata['latest_observed_month']
            evaluation = dict(
                recent_backtest=evidence(registered.backtest, latest, self.settings['recent_origins'], matches,
                    'same_parameters_features_rolling_refit_protocol', registered.backtest_version),
                registered_model_test=evidence(registered.fixed_test, latest, self.settings['recent_origins'], matches,
                    'registered_checkpoint_fixed_weight_holdout', registered.fixed_test_version),
                interpretation='精度得分=100-MAPE（百分数）。回测按每起点重新训练的协议评价；固定权重测试属于当前登记权重。'
                               '各自按最后若干完整且已成熟起点评分，注明月份及样本数；未来实际价格未知。')
            response = dict(prediction_id='pred_'+uuid4().hex, generated_at=now(), predictions=predictions,
                prediction_intervals=future_intervals(predictions, registered.backtest, len(frame), matches,
                    self.settings['interval'], registered.backtest_version), evaluation=evaluation,
                model_status=self.status(record['profile'], metadata, matches,record))
            # 校验公开输出后再落盘，读取重启前的记录时保持完全相同。
            response = PredictionResponse.model_validate(response).model_dump(mode='json')
            self.store.data(metadata['data_version'])
            self.store.save_prediction(response)
            self.catalog.put('prediction',response['prediction_id'],self.prediction_summary(response))
            return response

    def resolve_model(self, profile=None, version=None):
        record = self.catalog.get('model',version or self.catalog.default('model_'+(profile or 'selected'))[0])
        if profile is not None and record['profile'] != profile:
            raise APIError(422,'INVALID_REQUEST','变量方案与模型版本不一致')
        return record

    def ensure_loaded(self,version):
        with self.lock:
            if version in self.cache:
                result = self.cache.pop(version)
                self.cache[version] = result
                return result
            record = self.catalog.get('model',version)
            self.loading.add(version)
            try:
                model,saved = load_checkpoint(record['checkpoint'],self.device)
                if saved['layout'] != record['saved']['layout'] or saved['profile'] != record['profile']:
                    raise ValueError('权重布局与登记不一致')
                for key in ('parameters','preprocessing','epochs','lr_schedule','training_cutoff_month','selection_cutoff_month'):
                    if saved[key] != record['saved'][key]:
                        raise ValueError('权重配置、预处理或训练范围与登记不一致，请登记新版本')
                # 配套证据只在登记已通过验证的范围内读取。
                fixed = backtest = None
                fixed_version = backtest_version = None
                if record.get('evidence') == 'bootstrap':
                    fixed,backtest,fixed_version,backtest_version = self.model_evidence(record)
                elif record.get('fixed_predictions'):
                    fixed_path = Path(record['fixed_predictions'])
                    fixed,fixed_version = pd.read_csv(fixed_path),record['fixed_version']
                from .registry import RegisteredModel
                result = RegisteredModel(model,saved,version,record['registered_at'],now(),
                                         fixed,backtest,fixed_version,backtest_version)
                self.cache[version] = result
                while len(self.cache) > self.settings['max_loaded_models']:
                    self.cache.pop(next(iter(self.cache)))
                self.load_errors.pop(version,None)
                return result
            except Exception as exc:
                self.load_errors[version] = True
                self.store.event('model_load_failed',model_version=version,error_type=type(exc).__name__)
                raise APIError(503,'MODEL_LOAD_FAILED','模型加载失败，请检查权重、登记信息和设备资源') from exc
            finally:
                self.loading.discard(version)

    def model_evidence(self,record):
        if record['model_version'] in self.bootstrap_evidence:
            return self.bootstrap_evidence[record['model_version']]
        if record.get('evidence')=='bootstrap':
            from .registry import checked_table
            spec = record.get('evidence_specs',{})
            tables = []
            versions = []
            for key in ('fixed_test','backtest'):
                table,version = checked_table(spec[key],record['registry_path'],record['profile'],self.references) if spec.get(key) else (None,None)
                tables.append(table)
                versions.append(version)
            result = (*tables,*versions)
            self.bootstrap_evidence[record['model_version']] = result
            return result
        return None,None,None,None

    def register_dataset(self,metadata,source='upload',name=None,note='',filenames=None):
        try:
            return self.catalog.get('dataset',metadata['data_version']),False
        except APIError as exc:
            if exc.status != 404:
                raise
        _,paths = self.store.data(metadata['data_version'])
        summary = metadata if 'region_summaries' in metadata else dict(metadata,**self.validate_paths(paths))
        regions = deepcopy(summary['region_summaries'])
        for region in regions:
            region['filename'] = (filenames or {}).get(region['region'],region['filename'])
        body = dict(data_version=metadata['data_version'],name=name or '三地区截至'+metadata['latest_observed_month'],
            note=note,created_at=metadata['imported_at'],source=source,status='valid',
            first_month=metadata['first_month'],latest_observed_month=metadata['latest_observed_month'],months=metadata['months'],
            regions=regions,warnings=summary.get('warnings',[]))
        self.catalog.put('dataset',metadata['data_version'],body,only_new=True)
        return body,True

    def compatibility(self,record,sources):
        from types import SimpleNamespace
        reasons = []
        try:
            self.model_frame(sources,SimpleNamespace(saved=record['saved']))
        except (ValueError,AssertionError) as exc:
            reasons = [issue('MODEL_INCOMPATIBLE',str(exc))]
        return dict(model_version=record['model_version'],profile=record['profile'],can_predict=not reasons,reasons=reasons)

    def dataset(self,version):
        body = self.catalog.get('dataset',version)
        _,paths = self.store.data(version)
        sources = self.sources(paths)
        body['is_default'] = version == self.catalog.default('data')[0]
        body['compatible_models'] = [self.compatibility(record,sources) for record in self.catalog.list('model')]
        return body

    def dataset_list(self,source=None,is_default=None):
        values = [self.dataset(row['data_version']) for row in self.catalog.list('dataset')]
        return sorted([row for row in values if (source is None or row['source']==source)
                      and (is_default is None or row['is_default']==is_default)],key=lambda r:(r['created_at'],r['data_version']),reverse=True)

    def evaluation(self,record,metadata,sources):
        matches = self.reference_matches(sources,record)
        loaded = self.cache.get(record['model_version'])
        fixed = backtest = None
        fixed_version = backtest_version = None
        if loaded:
            fixed,backtest = loaded.fixed_test,loaded.backtest
            fixed_version,backtest_version = loaded.fixed_test_version,loaded.backtest_version
        elif record.get('evidence')=='bootstrap':
            fixed,backtest,fixed_version,backtest_version = self.model_evidence(record)
        elif record.get('fixed_predictions'):
            fixed,fixed_version = pd.read_csv(record['fixed_predictions']),record['fixed_version']
        latest = metadata['latest_observed_month']
        return dict(recent_backtest=evidence(backtest,latest,self.settings['recent_origins'],matches,
                'same_parameters_features_rolling_refit_protocol',backtest_version),
            registered_model_test=evidence(fixed,latest,self.settings['recent_origins'],matches,
                'registered_checkpoint_fixed_weight_holdout',fixed_version),
            interpretation='精度得分=100-MAPE（百分数）。固定权重测试与逐月重训回测分别报告；未来真实价格未知。')

    def model_record(self,version,data_version=None):
        record = self.catalog.get('model',version)
        metadata,paths = self.store.data(data_version or self.catalog.default('data')[0])
        sources = self.sources(paths)
        body = self.status(record['profile'],metadata,self.reference_matches(sources,record),record)
        saved = record['saved']
        body.update(display_name=record['display_name'],source=record['source'],training_id=record['training_id'],
            tuning_id=record['tuning_id'],training_data_version=record['training_data_version'],
            is_default=version==self.catalog.default('model_'+record['profile'])[0],
            load_state='loading' if version in self.loading else 'ready' if version in self.cache else 'error' if version in self.load_errors else 'not_loaded',
            parameters=saved['parameters'],training_plan=dict(epochs=saved['epochs'],lr_schedule=saved['lr_schedule']),
            compatibility=self.compatibility(record,sources),evaluation=self.evaluation(record,metadata,sources))
        return body

    def model_list(self,profile=None,is_default=None,data_version=None):
        rows = [self.model_record(record['model_version'],data_version) for record in self.catalog.list('model')
                if profile is None or record['profile']==profile]
        return sorted([r for r in rows if is_default is None or r['is_default']==is_default],
                      key=lambda r:(r['registered_at'],r['model_version']),reverse=True)

    def activate_data(self,version,request):
        with self.lock:
            data = self.dataset(version)
            selected = self.catalog.default('model_selected')[0]
            match = next(c for c in data['compatible_models'] if c['model_version']==selected)
            if not match['can_predict']:
                raise APIError(409,'INVALID_STATE','新数据与当前默认组合模型不兼容',match['reasons'])
            old,changed = self.catalog.set_defaults({'data':version},{'data':request.expected_current_data_version})
            self.data_version = version
            return dict(data_version=version,previous_data_version=old['data'],changed=changed,
                        activated_at=self.catalog.default('data')[1],compatible_models=data['compatible_models'])

    def activate_model(self,version,request):
        with self.lock:
            record = self.catalog.get('model',version)
            data_version = request.data_version or self.catalog.default('data')[0]
            metadata,paths = self.store.data(data_version)
            sources = self.sources(paths)
            model = self.ensure_loaded(version)
            try:
                values = forecast_future(model.model,self.model_frame(sources,model),model.saved)
                if any(not np.isfinite(v['prediction']) or v['prediction']<=0 for v in values):
                    raise ValueError('预测值无效')
                if record['profile'] != 'selected':
                    selected = self.catalog.get('model',self.catalog.default('model_selected')[0])
                    if not self.compatibility(selected,sources)['can_predict']:
                        raise APIError(409,'INVALID_STATE','切换后默认组合方案将不可用')
            except ValueError as exc:
                raise APIError(422,'INCOMPATIBLE_MODEL','模型与数据不能完成有效推理',[issue('INCOMPATIBLE_MODEL',str(exc))]) from exc
            key = 'model_'+record['profile']
            values = {key:version}
            if request.data_version is not None:
                values['data'] = data_version
            expected = {key:request.expected_current_model_version,'data':request.expected_current_data_version}
            old,changed = self.catalog.set_defaults(values,expected)
            self.models[record['profile']] = model
            self.data_version = self.catalog.default('data')[0]
            warnings = []
            if model.backtest is None:
                warnings.append(issue('NO_CALIBRATION','此版本没有匹配回测，价格区间暂不可用'))
            activated_at = self.catalog.default(key)[1]
            if request.data_version is not None:
                activated_at = max(activated_at,self.catalog.default('data')[1])
            return dict(model_version=version,profile=record['profile'],previous_model_version=old[key],
                data_version=self.data_version,previous_data_version=old['data'],changed=changed,
                activated_at=activated_at,warnings=warnings)

    @staticmethod
    def prediction_summary(value):
        status = value['model_status']
        months = sorted({r['target_month'] for r in value['predictions']})
        identifier = value['prediction_id']
        return dict(prediction_id=identifier,generated_at=value['generated_at'],profile=status['profile'],
            model_version=status['model_version'],data_version=status['data_version'],
            latest_observed_month=status['latest_observed_month'],first_target_month=months[0],last_target_month=months[-1],
            regions=list(REGIONS),prediction_count=len(value['predictions']),
            available_interval_count=sum(bool(v['available']) for v in value['prediction_intervals']['items']),
            detail_url='/predictions/'+identifier,csv_url='/predictions/'+identifier+'/csv')

    def prediction_list(self,**filters):
        rows = self.catalog.list('prediction')
        for key,value in filters.items():
            if value is None:
                continue
            if key in ('generated_from','generated_to'):
                try:
                    stamp = pd.Timestamp(value)
                    if stamp.tz is None:
                        raise ValueError('时间须带时区')
                except Exception as exc:
                    raise APIError(422,'INVALID_REQUEST','生成时间须为带时区ISO 8601') from exc
                rows = [r for r in rows if (pd.Timestamp(r['generated_at'])>=stamp if key=='generated_from' else pd.Timestamp(r['generated_at'])<stamp)]
            else:
                rows = [r for r in rows if r[key]==value]
        return sorted(rows,key=lambda r:(r['generated_at'],r['prediction_id']),reverse=True)

    def idempotent(self,key,fingerprint,scope,operation):
        if key is None:
            return operation()
        if not key.strip() or len(key)>200:
            raise APIError(422,'INVALID_REQUEST','Idempotency-Key须为1至200字符')
        with self.lock:
            cached = self.catalog.reserve(scope,key,fingerprint,time.time(),self.settings['idempotency_hours']*3600)
            if cached is not None:
                return cached['body'],cached['status']
            try:
                body,status = operation()
            except APIError as exc:
                self.catalog.finish(scope,key,dict(body=dict(_cached_error=dict(status=exc.status,code=exc.code,message=exc.message,details=exc.details)),status=exc.status))
                raise
            self.catalog.finish(scope,key,dict(body=body,status=status))
            return body,status

    def task(self,identifier,kind=None,public=True):
        from .tasks import public_task
        value = self.catalog.get('job',identifier)
        if kind and value['kind'] != kind:
            raise APIError(404,'NOT_FOUND','任务不存在')
        return public_task(self,value) if public else value

    def task_config(self,value):
        from .tasks import task_directory
        path = task_directory(self.store.root,value)/'effective_config.json'
        config = without_file_checks(json.loads(path.read_text(encoding='utf-8')))
        if config != value['_internal']['config_snapshot']:
            raise APIError(422,'INVALID_DATA','已冻结的任务配置被修改，请重新提交任务')
        return config
