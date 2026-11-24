"""在统一任务队列的独立进程内执行年度预算训练。"""
from datetime import datetime
import json
from pathlib import Path
import time

import torch

from . import VERSION as SERVICE_VERSION
from .annual import (VERSION, checked_identifier, runtime, annual_period, price_rows,
                     weight_metadata, prediction_response, save_prediction)
from .catalog import Catalog
from .storage import Store, now, save_json


class Cancelled(Exception):
    pass


class TimeLimit(Exception):
    pass


class AnnualWorker:
    def __init__(self, storage, task_id):
        self.task_id = checked_identifier(task_id, 'atrain')
        self.store, self.catalog = Store(storage), Catalog(storage)
        self.task = self.catalog.get('job', task_id)
        if self.task['kind'] != 'annual_training':
            raise ValueError('年度训练任务类别不匹配')
        self.folder = self.store.root/'annual_trainings'/task_id
        self.config = json.loads((self.folder/'effective_config.json').read_text(encoding='utf-8'))
        if self.config != self.task['_internal']['config_snapshot']:
            raise ValueError('年度训练配置与冻结快照不一致')
        if self.config['version'] != VERSION or self.config['service_version'] != SERVICE_VERSION:
            raise ValueError('年度训练版本与提交时不一致')
        self.metadata, self.paths = self.store.data(self.config['data_version'])
        self.device = runtime().resolve_device(self.config['device'])
        torch.set_num_threads(self.config['threads'])
        if self.device == 'cuda':
            torch.set_float32_matmul_precision('high')
        elapsed = max(0.0, time.time()-datetime.fromisoformat(self.task['started_at']).timestamp())
        self.begin = time.monotonic()-elapsed

    def check(self):
        task = self.catalog.get('job', self.task_id)
        if task['status'] != 'running':
            raise Cancelled()
        limit = self.config['max_duration_minutes']
        if limit is not None and time.monotonic()-self.begin >= limit*60:
            raise TimeLimit()

    def progress(self, completed, total, epoch, epochs, configurations, count):
        self.check()
        def update(task):
            if task['status'] != 'running':
                raise Cancelled()
            return dict(stage='training', elapsed_seconds=time.monotonic()-self.begin,
                        progress=dict(unit='epoch', completed_units=completed, total_units=total,
                                      phase_percent=100*completed/total, current_epoch=epoch, max_epochs=epochs,
                                      completed_configurations=configurations, total_configurations=count))
        self.catalog.update_job(self.task_id, update)

    def training(self):
        cfg = self.config
        total = sum(c['training']['epochs'] for c in cfg['configs'])
        completed, predictions, weights = 0, [], []
        count = len(cfg['configs'])
        for index, config in enumerate(cfg['configs']):
            self.check()
            def callback(epoch, epochs):
                self.progress(completed+epoch, total, epoch, epochs, index, count)
            checkpoint = runtime().train_checkpoint(config, cfg['budget_year'], self.paths, self.device, progress=callback)
            self.check()
            path = runtime().checkpoint_path(self.folder/'weights', cfg['budget_year'], config['spec'])
            runtime().save_checkpoint(path, checkpoint)
            prices = runtime().predict_checkpoint(checkpoint, self.paths, self.device)
            predictions.extend(price_rows(checkpoint, prices))
            weights.append(weight_metadata(checkpoint, 'training'))
            completed += config['training']['epochs']
            self.progress(completed, total, None, None, index+1, count)
        self.check()
        response = prediction_response(cfg['budget_year'], cfg['data_version'], self.metadata, self.device,
                                       'training', self.task_id, cfg['data_version'], weights, predictions)
        save_prediction(self.store, self.catalog, response)
        finished = now()
        result = dict(configurations=count, weights_count=len(weights), trained_at=finished,
                      prediction_id=response['prediction_id'], predictions_url=response['detail_url'], csv_url=response['csv_url'])
        def publish(task):
            if task['status'] != 'running':
                raise Cancelled()
            return dict(status='succeeded', stage='complete', finished_at=finished,
                        elapsed_seconds=time.monotonic()-self.begin, result=result,
                        progress=dict(unit='epoch', completed_units=total, total_units=total, phase_percent=100.0,
                                      current_epoch=None, max_epochs=None, completed_configurations=count,
                                      total_configurations=count))
        self.catalog.update_job(self.task_id, publish)
        try:
            self.store.event('annual_training_complete', training_id=self.task_id, budget_year=cfg['budget_year'])
        except OSError:
            import traceback
            traceback.print_exc()

    def execute(self):
        try:
            self.training()
        except (Cancelled, TimeLimit) as exc:
            cancelled = isinstance(exc, Cancelled)
            self.catalog.update_job(self.task_id, dict(status='cancelled' if cancelled else 'interrupted',
                finished_at=now(), elapsed_seconds=time.monotonic()-self.begin, result=None,
                error=None if cancelled else dict(code='TIME_LIMIT', message='年度训练达到执行时限', retryable=True)))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.catalog.update_job(self.task_id, dict(status='failed', finished_at=now(),
                elapsed_seconds=time.monotonic()-self.begin, result=None,
                error=dict(code='EXECUTION_FAILED', message='年度训练失败：'+type(exc).__name__+'；请查看任务日志', retryable=True)))
        finally:
            task = self.catalog.get('job', self.task_id)
            save_json(self.folder/'task.json', {key:value for key,value in task.items() if not key.startswith('_')})
