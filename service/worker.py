"""独立进程执行冻结任务；仅由服务协调器启动。"""
import argparse
from copy import deepcopy
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch

from datasets import read_frame
from joint_emaformer.data import REGIONS, NAMES, PreparedData, build_frame, windows
from joint_emaformer._training import train, checkpoint, predict
from joint_emaformer._runtime import package_checkpoint, save_checkpoint, prediction_rows, csv
from .catalog import Catalog, encode
from .storage import Store, now, save_json
from . import VERSION


class Cancelled(Exception):
    pass


class TimeLimit(Exception):
    pass


def region_metrics(actual,forecast):
    errors = abs((forecast-actual)/actual).mean(axis=0)*100
    return [dict(region=region,region_name=NAMES[region],origins=len(actual),
        by_horizon=[dict(horizon=h+1,observations=len(actual),mape_percent=float(errors[h,i]),
                         accuracy_score_percent=100-float(errors[h,i])) for h in range(3)],
        average_mape_percent=float(errors[:,i].mean()),average_accuracy_score_percent=100-float(errors[:,i].mean()))
        for i,region in enumerate(REGIONS)]


def aggregate(records):
    result = []
    for index,region in enumerate(REGIONS):
        counts = sum(record['regions'][index]['origins'] for record in records)
        steps = [float(np.mean([record['regions'][index]['by_horizon'][h]['mape_percent'] for record in records])) for h in range(3)]
        result.append(dict(region=region,region_name=NAMES[region],origins=counts,
            by_horizon=[dict(horizon=h+1,observations=counts,mape_percent=steps[h],accuracy_score_percent=100-steps[h]) for h in range(3)],
            average_mape_percent=float(np.mean(steps)),average_accuracy_score_percent=100-float(np.mean(steps))))
    return result


class Worker:
    def __init__(self,root,task_id):
        self.store = Store(root)
        self.catalog = Catalog(root)
        self.task_id = task_id
        self.task = self.catalog.get('job',task_id)
        self.folder = self.store.root/('trainings' if self.task['kind']=='training' else 'tuning_jobs')/task_id
        path = self.folder/'effective_config.json'
        self.config = json.loads(path.read_text(encoding='utf-8'))
        if self.config != self.task['_internal']['config_snapshot']:
            raise ValueError('任务配置与提交时的配置快照不一致')
        if self.config['service_version'] != VERSION:
            raise ValueError('执行服务版本与提交任务时不一致，请重新提交')
        self.begin = time.monotonic()-max(0.,time.time()-datetime.fromisoformat(self.task['started_at']).timestamp())
        self.last_progress = 0.
        torch.set_num_threads(self.config['threads'])
        self.device = torch.device(self.config['device'])
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA设备不可用')
        _,paths = self.store.data(self.config['data_version'])
        sources = {r:read_frame(paths[r]).loc[self.config['split']['first_month']:] for r in REGIONS}
        if self.config['kind']=='tuning':
            sources = {r:frame.loc[:self.config['selection_cutoff']] for r,frame in sources.items()}
        self.frame,_ = build_frame(sources,self.config['profile'],{},saved_layout=self.config['layout'])
        self.prepared = PreparedData(self.frame,self.config['layout'],self.device)

    def check(self):
        task = self.catalog.get('job',self.task_id)
        if task['status']=='cancel_requested':
            raise Cancelled()
        if task['status'] not in ('running',):
            raise Cancelled()
        minutes = self.config['max_duration_minutes']
        if minutes is not None and time.monotonic()-self.begin >= minutes*60:
            raise TimeLimit()

    def update(self,**patch):
        self.check()
        patch['elapsed_seconds'] = time.monotonic()-self.begin
        self.catalog.update_job(self.task_id,patch)

    def callback(self,event):
        self.check()
        if event['event'] in ('epoch_start','epoch_end'):
            current = self.catalog.get('job',self.task_id)['progress']
            current.update(current_epoch=event['epoch'],max_epochs=event['epochs'])
            if self.config['kind']=='training':
                done = event['epoch'] if event['event']=='epoch_end' else event['epoch']-1
                current.update(unit='epoch',completed_units=done,total_units=event['epochs'],phase_percent=100*done/event['epochs'])
            self.update(progress=current)

    def cutoff(self,month):
        return int(self.frame.index.get_loc(pd.Period(month,freq='M')))+1

    def training(self):
        cfg = self.config
        self.update(stage='training')
        fit_end = self.cutoff(cfg['split']['training_cutoff'])
        model,prep,history = train(self.prepared,cfg['parameters'],fit_end,cfg['seed'],
                                  replay_lrs=cfg['schedule'],verbose=True,callback=self.callback)
        self.check()
        audit = dict(started_at=self.task['started_at'],finished_at=now(),elapsed_seconds=time.monotonic()-self.begin,
            seed=cfg['seed'],device=str(self.device),data_version=cfg['data_version'],
            training_cutoff_month=cfg['split']['training_cutoff'],selection_cutoff_month=cfg['selection_cutoff'],
            epochs=len(history),input_layout=cfg['layout'],parameters=cfg['parameters'],configuration=cfg,
            service_version=VERSION)
        saved = package_checkpoint(checkpoint(model,cfg['layout'],cfg['parameters'],prep,cfg['seed'],cfg['schedule'],fit_end),
                                   cfg['profile'],self.frame,{'data_version':cfg['data_version']},cfg['selection_cutoff'],audit)
        checkpoint_path = self.folder/'best.pt'
        version = cfg['profile']+'-'+self.task_id.removeprefix('train_')
        saved['model_version'] = version
        save_checkpoint(checkpoint_path,saved)
        save_json(self.folder/'preprocessing.json',prep)
        save_json(self.folder/'training_audit.json',audit)
        csv(self.folder/'history.csv',history)
        self.update(stage='evaluating')
        rows = []
        if cfg['split']['test_start']:
            end = self.cutoff(cfg['split']['test_end'])
            data = windows(self.frame,cfg['layout'],prep,fit_end,end,cfg['parameters']['seq_len'])
            forecast = predict(model,data,prep)
            if not np.isfinite(forecast).all():
                raise FloatingPointError('测试输出不是有限数')
            rows = prediction_rows(self.frame,data.starts,data.actual,forecast,cfg['profile'],cfg['split']['training_cutoff'])
            csv(self.folder/'evaluation_predictions.csv',rows)
            public_rows = [dict(training_id=self.task_id,model_version=version,data_version=cfg['data_version'],
                profile=cfg['profile'],region=row['region'],region_name=NAMES[row['region']],history_cutoff=row['history_cutoff'],
                first_forecast_month=row['first_forecast_month'],target_month=row['target_month'],horizon=row['horizon'],
                actual=row['actual'],prediction=row['prediction'],
                absolute_percentage_error_percent=abs(row['prediction']-row['actual'])/row['actual']*100) for row in rows]
            csv(self.folder/'test_predictions.csv',public_rows)
        from .engine import evidence
        fixed_path = self.folder/'evaluation_predictions.csv' if rows else None
        fixed_version = version+'-test' if fixed_path else None
        evaluation = dict(recent_backtest=evidence(None,cfg['split']['latest_observed_month'],20,True,
                'same_parameters_features_rolling_refit_protocol',None),
            registered_model_test=evidence(pd.DataFrame(rows) if rows else None,cfg['split']['latest_observed_month'],20,True,
                'registered_checkpoint_fixed_weight_holdout',fixed_version),
            interpretation='精度得分=100-MAPE（百分数）。本次为固定新权重留出测试；无保留标签时不生成测试指标。')
        record = dict(model_version=version,profile=cfg['profile'],checkpoint=str(checkpoint_path),
            saved={k:v for k,v in saved.items() if k!='model_state_dict'},registered_at=now(),source='training',
            training_id=self.task_id,tuning_id=cfg['tuning_id'],training_data_version=cfg['data_version'],
            reference_data_version=cfg['data_version'],evidence=None,fixed_predictions=str(fixed_path) if fixed_path else None,
            fixed_version=fixed_version,display_name=('全变量' if cfg['profile']=='all' else '组合变量')+'｜训练截至'+cfg['split']['training_cutoff']+'｜'+audit['finished_at'][:10]+' 训练')
        result = dict(model_version=version,display_name=record['display_name'],registered_at=record['registered_at'],
            trained_at=audit['finished_at'],model_training_cutoff=cfg['split']['training_cutoff'],
            parameter_selection_cutoff=cfg['selection_cutoff'],epochs_completed=len(history),evaluation=evaluation,
            test_csv_url='/trainings/'+self.task_id+'/csv' if rows else None,model_url='/models/'+version)
        self.update(stage='registering')
        self.store.data(cfg['data_version'])
        self.check()
        # 模型登记与任务成功在同一事务中提交，取消时不登记半成品。
        with self.catalog.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            task = json.loads(db.execute("SELECT body FROM records WHERE kind='job' AND id=?",(self.task_id,)).fetchone()[0])
            if task['status']!='running':
                raise Cancelled()
            db.execute("INSERT OR IGNORE INTO records VALUES ('model',?,?)",(version,encode(record)))
            task.update(status='succeeded',finished_at=now(),elapsed_seconds=time.monotonic()-self.begin,result=result)
            db.execute("UPDATE records SET body=? WHERE kind='job' AND id=?",(encode(task),self.task_id))
        self.store.event('training_complete',training_id=self.task_id,model_version=version)

    def fit(self,parameters,fit_end,eval_end,seed):
        start = time.monotonic()
        result,forecast = train(self.prepared,parameters,fit_end,seed,eval_end,callback=self.callback)
        actual = self.prepared.partition(parameters,fit_end,fit_end,eval_end)[0].actual
        result['regions'] = region_metrics(actual,forecast)
        result['elapsed_seconds'] = time.monotonic()-start
        return result

    def publish_candidates(self,rows,phase):
        valid = sorted([r for r in rows if r['status']=='complete'],key=lambda row:(row['score_mape_percent'],
                       max(v['average_mape_percent'] for v in row['regions']),row['trial']))
        ranks = {row['trial']:i+1 for i,row in enumerate(valid)}
        for row in rows:
            row['rank'] = ranks.get(row['trial'])
        save_json(self.folder/('screening_public.json' if phase=='screening' else 'confirmation_public.json'),
                  sorted(rows,key=lambda r:(r['rank'] is None,r['rank'] or r['trial'])))
        self.update(best_so_far=valid[0] if valid else None)

    def tuning(self):
        cfg = self.config
        settings = cfg['settings']
        screen,raw = [],[]
        fit_end = self.cutoff(cfg['split']['training_cutoff'])
        end = len(self.frame)
        for number,parameters in enumerate(cfg['candidates'],1):
            self.update(stage='screening',progress=dict(unit='fit',completed_units=number-1,total_units=len(cfg['candidates']),
                phase_percent=100*(number-1)/len(cfg['candidates']),current_epoch=None,max_epochs=parameters['max_epochs']))
            row = dict(trial=number,rank=None,phase='screening',status='running',parameters=parameters,
                score_mape_percent=None,regions=[],fits_completed=0,fits_planned=1,best_epoch=None,elapsed_seconds=0.,error=None)
            screen.append(row)
            self.publish_candidates(screen,'screening')
            started = time.monotonic()
            try:
                result = self.fit(parameters,fit_end,end,settings['primary_seed'])
                raw.append(dict(trial=number,parameters=parameters,**result))
                row.update(status='complete',score_mape_percent=result['score'],regions=result['regions'],best_epoch=result['best_epoch'])
            except (FloatingPointError,RuntimeError,ValueError) as exc:
                row.update(status='failed',error=dict(code='FIT_FAILED',message='此候选拟合失败：'+type(exc).__name__,retryable=False))
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row.update(fits_completed=1,elapsed_seconds=time.monotonic()-started)
            save_json(self.folder/'screening_trials.json',raw)
            self.publish_candidates(screen,'screening')
        ranked = sorted([r for r in screen if r['status']=='complete'],key=lambda row:(row['score_mape_percent'],
                        max(v['average_mape_percent'] for v in row['regions']),row['trial']))
        if len(ranked)<settings['finalists']:
            raise ValueError('成功候选数量不足')
        confirmed,public = [],[]
        total = settings['finalists']*len(cfg['folds'])*len(settings['training_seeds'])
        finished = 0
        for item in ranked[:settings['finalists']]:
            parameters = item['parameters']
            records = []
            row = dict(trial=item['trial'],rank=None,phase='confirmation',status='running',parameters=parameters,
                score_mape_percent=None,regions=[],fits_completed=0,fits_planned=len(cfg['folds'])*len(settings['training_seeds']),
                best_epoch=None,elapsed_seconds=0.,error=None)
            public.append(row)
            started = time.monotonic()
            for fold in cfg['folds']:
                fit_end,eval_end = self.cutoff(fold['training_cutoff']),self.cutoff(fold['validation_end'])
                for seed in settings['training_seeds']:
                    self.update(stage='confirming',progress=dict(unit='fit',completed_units=finished,total_units=total,
                        phase_percent=100*finished/total,current_epoch=None,max_epochs=parameters['max_epochs']))
                    self.publish_candidates(public,'confirmation')
                    try:
                        result = self.fit(parameters,fit_end,eval_end,seed)
                        records.append(dict(fold=fold['fold'],**result))
                    except (FloatingPointError,RuntimeError,ValueError) as exc:
                        row.update(status='failed',error=dict(code='FIT_FAILED',message='此候选复核失败：'+type(exc).__name__,retryable=False))
                    finished += 1
                    row['fits_completed'] += 1
                    row['elapsed_seconds'] = time.monotonic()-started
                    save_json(self.folder/f'confirm_trial_{item["trial"]}.json',records)
            if len(records)==row['fits_planned']:
                regions = aggregate(records)
                score = float(np.mean([r['average_mape_percent'] for r in regions]))
                row.update(status='complete',regions=regions,score_mape_percent=score)
                confirmed.append(dict(trial=item['trial'],parameters=parameters,score=score,
                                      mean_region_MAPE=[r['average_mape_percent'] for r in regions],records=records))
            self.publish_candidates(public,'confirmation')
            save_json(self.folder/'confirmed_candidates.json',confirmed)
        if len(confirmed)<settings['finalists']:
            raise ValueError('完整复核候选数量不足')
        self.update(stage='finalizing')
        chosen = min(confirmed,key=lambda row:(row['score'],max(row['mean_region_MAPE']),row['trial']))
        last = sorted([r for r in chosen['records'] if r['fold']==len(cfg['folds'])],key=lambda r:(r['best_epoch'],r['seed']))
        representative = last[len(last)//2]
        schedule = [r['lr'] for r in representative['history'][:representative['best_epoch']]]
        recommendation = dict(trial=chosen['trial'],parameters=chosen['parameters'],frozen_epochs=len(schedule),
                              frozen_lr_schedule=schedule,selection_end_month=cfg['selection_cutoff'])
        save_json(self.folder/'selected_parameters.json',dict(chosen,**recommendation))
        self.store.data(cfg['data_version'])
        self.check()
        def finish(task):
            if task['status']!='running':
                raise Cancelled()
            return dict(status='succeeded',finished_at=now(),elapsed_seconds=time.monotonic()-self.begin,result_available=True,
                        _internal=dict(task['_internal'],recommendation=recommendation),
                        progress=dict(unit='fit',completed_units=total,total_units=total,phase_percent=100.,current_epoch=None,max_epochs=None))
        self.catalog.update_job(self.task_id,finish)
        self.store.event('tuning_complete',tuning_id=self.task_id)

    def execute(self):
        try:
            self.check()
            if self.config['kind']=='training':
                self.training()
            else:
                self.tuning()
        except (Cancelled,TimeLimit) as exc:
            cancelled = isinstance(exc,Cancelled)
            self.catalog.update_job(self.task_id,dict(status='cancelled' if cancelled else 'interrupted',finished_at=now(),
                elapsed_seconds=time.monotonic()-self.begin,error=None if cancelled else dict(code='TIME_LIMIT',message='达到执行时限，已保留完成记录',retryable=True)))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.catalog.update_job(self.task_id,dict(status='failed',finished_at=now(),elapsed_seconds=time.monotonic()-self.begin,
                error=dict(code='EXECUTION_FAILED',message='任务执行失败：'+type(exc).__name__+'；请由维护人员查看任务日志',retryable=True)))
        finally:
            task = self.catalog.get('job',self.task_id)
            if task['status']!='succeeded':
                for filename in ('screening_public.json','confirmation_public.json'):
                    path = self.folder/filename
                    if path.exists():
                        rows = json.loads(path.read_text(encoding='utf-8'))
                        for row in rows:
                            if row['status']=='running':
                                row['status']='partial'
                        save_json(path,rows)
            save_json(self.folder/'task.json',{k:v for k,v in task.items() if not k.startswith('_')})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--storage',type=Path,required=True)
    parser.add_argument('--job',required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'(train|tune|atrain)_[a-f0-9]{32}',args.job):
        raise ValueError('任务编号无效')
    try:
        task = Catalog(args.storage).get('job',args.job)
        if task['kind']=='annual_training':
            from .annual_worker import AnnualWorker
            worker = AnnualWorker(args.storage,args.job)
        else:
            worker = Worker(args.storage,args.job)
    except Exception:
        catalog = Catalog(args.storage)
        catalog.update_job(args.job,dict(status='failed',finished_at=now(),error=dict(code='PREPARATION_FAILED',message='任务准备失败，请检查冻结配置和数据',retryable=True)))
        raise
    worker.execute()


if __name__=='__main__':
    main()
