"""年度预算版模型目录、预测记录与持久训练提交。"""
from copy import deepcopy
from contextlib import contextmanager
import csv
import importlib
import io
import json
import math
from pathlib import Path
import re
from uuid import uuid4

import pandas as pd
import torch

from . import VERSION as SERVICE_VERSION
from .annual_schemas import AnnualPredictionResponse, AnnualTrainingTask
from .errors import APIError, issue
from .storage import now, save_json

VERSION = '年度预算版'
PACKAGE = Path(__file__).resolve().parents[1] / VERSION
MODEL_NAMES = {'lstm': 'LSTM', 'emaformer': 'EMAformer', 'mofo': 'MoFo',
               'emmoformer': 'EMMoFormer', 'CoRegionLng': 'CoRegionLng'}


def runtime():
    return importlib.import_module('年度预算版.annual_runtime')


@contextmanager
def annual_precision(device):
    previous = torch.get_float32_matmul_precision()
    try:
        if str(device).startswith('cuda'):
            torch.set_float32_matmul_precision('high')
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def annual_period(budget_year):
    if type(budget_year) is not int or not 2000 <= budget_year <= 9998:
        raise APIError(422, 'INVALID_REQUEST', '预算年度无效')
    return dict(budget_year=budget_year, cutoff_month=f'{budget_year-1}-08',
                target_start=f'{budget_year}-01', target_end=f'{budget_year}-12')


def select_configs(configs, model=None, profile=None):
    if model not in (None, 'all', *MODEL_NAMES) or profile not in (None, 'all', 'selected', 'both'):
        raise APIError(422, 'INVALID_REQUEST', '年度模型或变量方案无效')
    return [deepcopy(c) for c in configs
            if (model in (None, 'all') or c['spec']['model'] == model)
            and (profile in (None, 'both') or c['spec']['profile'] == profile)]


def spec_key(config):
    spec = config['spec']
    return spec['model'], spec['profile'], tuple(spec['regions'])


def checked_identifier(value, prefix):
    if not re.fullmatch(prefix + r'_[a-f0-9]{32}', value):
        raise APIError(404, 'NOT_FOUND', '年度记录编号无效')
    return value


def weight_metadata(checkpoint, source):
    spec = checkpoint['spec']
    return dict(model=spec['model'], profile=spec['profile'], regions=spec['regions'],
                budget_year=checkpoint['budget_year'], cutoff_month=checkpoint['cutoff_month'], source=source)


def price_rows(checkpoint, prices):
    if any(not math.isfinite(float(price)) or float(price) <= 0 for price in prices):
        raise APIError(422, 'INVALID_OUTPUT', '年度预算模型输出须为有限正数')
    return [{key: row[key] for key in ('model', 'profile', 'region', 'region_name', 'unit', 'n_train')}
            | {'prediction': row['predicted']} for row in runtime().forecast_rows(checkpoint, prices)]


def prediction_response(budget_year, data_version, metadata, device, source, training_id,
                        training_data_version, weights, predictions):
    identifier = 'apred_' + uuid4().hex
    url = '/annual-budget/predictions/' + identifier
    value = dict(version=VERSION, prediction_id=identifier, generated_at=now(), **annual_period(budget_year),
                 data_version=data_version, data_latest_observed_month=metadata['latest_observed_month'],
                 device=str(device), weight_source=source, training_id=training_id,
                 training_data_version=training_data_version, weights=weights, predictions=predictions,
                 detail_url=url, csv_url=url+'/csv')
    return AnnualPredictionResponse.model_validate(value).model_dump(mode='json')


def save_prediction(store, catalog, response):
    identifier = checked_identifier(response['prediction_id'], 'apred')
    save_json(store.root/'annual_predictions'/identifier/'response.json', response)
    catalog.put('annual_prediction', identifier, response)
    store.event('annual_prediction_complete', prediction_id=identifier,
                budget_year=response['budget_year'], data_version=response['data_version'])


class AnnualService:
    def __init__(self, engine):
        self.engine = engine
        self.configs = json.loads((PACKAGE/'configs/models.json').read_text(encoding='utf-8-sig'))
        for file in (engine.store.root/'annual_predictions').glob('apred_*/response.json'):
            record = AnnualPredictionResponse.model_validate(json.loads(file.read_text(encoding='utf-8'))).model_dump(mode='json')
            if record['training_id'] is not None:
                try:
                    task = engine.catalog.get('job', record['training_id'])
                except APIError:
                    continue
                if task['status'] != 'succeeded':
                    continue
            engine.catalog.put('annual_prediction', record['prediction_id'], record, only_new=True)

    def models(self, budget_year=2026, model=None, profile=None):
        period = annual_period(budget_year)
        rows = []
        for config in select_configs(self.configs, model, profile):
            spec = config['spec']
            path = runtime().checkpoint_path(PACKAGE/'weights', budget_year, spec)
            rows.append(dict(model=spec['model'], model_name=MODEL_NAMES[spec['model']], profile=spec['profile'],
                             regions=spec['regions'], input_months=spec['seq_len'], output='次年1–12月算术平均价格',
                             input_features=dict(zip(spec['regions'], spec['columns'])),
                             packed_channels=spec.get('layout', {}).get('input_size'),
                             training_epochs=config['training']['epochs'], available=path.is_file(),
                             budget_year=budget_year, cutoff_month=period['cutoff_month'], weight_source='bundled'))
        return dict(version=VERSION, **period, models=rows)

    def dataset(self, requested, cutoff_month):
        version = requested or self.engine.catalog.default('data')[0]
        self.engine.catalog.get('dataset', version)
        try:
            metadata, paths = self.engine.store.data(version)
        except (KeyError, ValueError) as exc:
            raise APIError(422, 'INVALID_DATA', '年度预测数据版本不可用') from exc
        if not metadata['first_month'] <= cutoff_month <= metadata['latest_observed_month']:
            raise APIError(422, 'INVALID_DATA', f'数据须覆盖预算截止月 {cutoff_month}')
        return version, metadata, paths

    def training(self, identifier):
        checked_identifier(identifier, 'atrain')
        return self.engine.task(identifier, 'annual_training', public=False)

    def predict(self, body):
        with self.engine.lock:
            period = annual_period(body.budget_year)
            source, training_data_version = 'bundled', None
            weights_dir = PACKAGE/'weights'
            configs = self.configs
            model, profile, requested_data = body.model, body.profile, body.data_version
            if body.training_id is not None:
                task = self.training(body.training_id)
                if task['status'] != 'succeeded':
                    raise APIError(409, 'INVALID_STATE', '年度训练任务尚未成功完成')
                if task['budget_year'] != body.budget_year:
                    raise APIError(422, 'INVALID_REQUEST', '预算年度与年度训练任务不一致')
                source, training_data_version = 'training', task['data_version']
                requested_data = requested_data or task['data_version']
                configs = deepcopy(task['_internal']['config_snapshot']['configs'])
                model = model if model is not None else task['model']
                profile = profile if profile is not None else task['profile']
                requested = select_configs(self.configs, model, profile)
                if not {spec_key(c) for c in requested} <= {spec_key(c) for c in configs}:
                    raise APIError(422, 'INVALID_REQUEST', '所选范围超出该年度训练任务的权重范围')
                weights_dir = self.engine.store.root/'annual_trainings'/body.training_id/'weights'
            configs = select_configs(configs, model, profile)
            if not configs:
                raise APIError(422, 'INVALID_REQUEST', '没有符合条件的年度模型')
            version, metadata, paths = self.dataset(requested_data, period['cutoff_month'])
            checkpoints = [runtime().checkpoint_path(weights_dir, body.budget_year, c['spec']) for c in configs]
            if any(not p.is_file() for p in checkpoints):
                raise APIError(409, 'ANNUAL_WEIGHTS_UNAVAILABLE', '该预算年度的权重尚未准备，请先提交年度训练',
                               [issue('ANNUAL_WEIGHTS_UNAVAILABLE', '缺少已选择的年度模型权重',
                                      suggestion='POST /annual-budget/trainings')])
            predictions, weights = [], []
            for config, path in zip(configs, checkpoints):
                try:
                    checkpoint = runtime().load_checkpoint(path, body.budget_year)
                    if checkpoint['spec'] != config['spec']:
                        raise ValueError('年度模型配置与权重不一致')
                    with annual_precision(self.engine.device):
                        prices = runtime().predict_checkpoint(checkpoint, paths, str(self.engine.device))
                except (ValueError, KeyError) as exc:
                    raise APIError(422, 'INVALID_DATA', str(exc)) from exc
                predictions.extend(price_rows(checkpoint, prices))
                weights.append(weight_metadata(checkpoint, source))
            response = prediction_response(period['budget_year'], version, metadata, self.engine.device, source,
                                           body.training_id, training_data_version, weights, predictions)
            save_prediction(self.engine.store, self.engine.catalog, response)
            return response

    def prediction(self, identifier):
        checked_identifier(identifier, 'apred')
        response = self.engine.catalog.get('annual_prediction', identifier)
        if response['training_id'] is not None:
            if self.training(response['training_id'])['status'] != 'succeeded':
                raise APIError(404, 'NOT_FOUND', '年度预测尚未发布')
        return response

    def predictions(self, budget_year=None, model=None, profile=None, data_version=None):
        if budget_year is not None:
            annual_period(budget_year)
        select_configs(self.configs, model, profile)
        rows = []
        for value in self.engine.catalog.list('annual_prediction'):
            if budget_year is not None and value['budget_year'] != budget_year:
                continue
            if data_version is not None and value['data_version'] != data_version:
                continue
            if not any((model in (None, 'all') or row['model']==model)
                       and (profile in (None, 'both') or row['profile']==profile) for row in value['predictions']):
                continue
            try:
                rows.append(self.prediction(value['prediction_id']))
            except APIError as exc:
                if exc.status != 404:
                    raise
        return dict(version=VERSION, items=sorted(rows, key=lambda r:(r['generated_at'], r['prediction_id']), reverse=True))

    def prediction_csv(self, identifier):
        response = self.prediction(identifier)
        stream = io.StringIO(newline='')
        shared = {k:response[k] for k in ('version', 'prediction_id', 'budget_year', 'cutoff_month',
                                         'target_start', 'target_end', 'data_version', 'weight_source', 'training_id')}
        fields = [*shared, 'model', 'profile', 'region', 'region_name', 'prediction', 'unit', 'n_train']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(shared, **row) for row in response['predictions'])
        return stream.getvalue().encode('utf-8-sig')

    def submit_training(self, body):
        with self.engine.lock:
            queued = [task for task in self.engine.catalog.list('job') if task['status']=='queued']
            if len(queued) >= self.engine.settings['jobs']['max_queued']:
                raise APIError(429, 'QUEUE_FULL', '任务队列已满，请稍后重试')
            period = annual_period(body.budget_year)
            configs = select_configs(self.configs, body.model, body.profile)
            data_version, metadata, _ = self.dataset(body.data_version, period['cutoff_month'])
            first_month = pd.Period(metadata['first_month'], freq='M')
            cutoff = pd.Period(period['cutoff_month'], freq='M')
            history_months = cutoff.ordinal - first_month.ordinal + 1
            if history_months < max(c['spec']['seq_len'] for c in configs) + 16:
                raise APIError(422, 'INVALID_DATA', '历史长度不足以形成完整年度训练标签')
            identifier = 'atrain_' + uuid4().hex
            folder = self.engine.store.root/'annual_trainings'/identifier
            config = dict(version=VERSION, service_version=SERVICE_VERSION, kind='annual_training',
                          budget_year=body.budget_year, cutoff_month=period['cutoff_month'],
                          configs=configs, data_version=data_version, device=str(self.engine.device),
                          threads=self.engine.settings['threads'], max_duration_minutes=self.engine.settings['jobs']['max_duration_minutes'])
            total = sum(c['training']['epochs'] for c in configs)
            task = dict(version=VERSION, kind='annual_training', training_id=identifier,
                        name=body.name or f'{body.budget_year}年度预算训练', **period,
                        model=body.model, profile=body.profile, data_version=data_version,
                        device=str(self.engine.device), status='queued', stage='queued', submitted_at=now(),
                        started_at=None, finished_at=None, elapsed_seconds=0.0, queue_position=len(queued)+1,
                        progress=dict(unit='epoch', completed_units=0, total_units=total, phase_percent=0.0,
                                      current_epoch=None, max_epochs=None, completed_configurations=0,
                                      total_configurations=len(configs)), warnings=[], error=None,
                        request=body.model_dump(exclude_unset=True), result=None,
                        status_url='/annual-budget/trainings/'+identifier,
                        _internal=dict(pid=None, nonce=uuid4().hex, config_snapshot=deepcopy(config)))
            public = AnnualTrainingTask.model_validate({k:v for k,v in task.items() if not k.startswith('_')}).model_dump(mode='json')
            folder.mkdir(parents=True, exist_ok=False)
            save_json(folder/'effective_config.json', config)
            save_json(folder/'task.json', public)
            self.engine.catalog.put('job', identifier, task)
            self.engine.store.event('job_submitted', job_id=identifier, job_kind='annual_training')
            from .tasks import public_task
            return public_task(self.engine, task)
