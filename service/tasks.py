"""持久任务队列和独立工作进程；HTTP进程不执行模型拟合。"""
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4

from .errors import APIError, issue
from .planning import training_config, tuning_config
from .storage import now, save_json

TERMINAL = {'succeeded','failed','cancelled','interrupted'}
ROOT = Path(__file__).resolve().parents[1]


def identifier(task):
    return task['tuning_id'] if task['kind']=='tuning' else task['training_id']


def task_directory(storage, task):
    folders = {'training':'trainings', 'tuning':'tuning_jobs', 'annual_training':'annual_trainings'}
    return Path(storage)/folders[task['kind']]/identifier(task)


def public_task(engine,task):
    value = {key:deepcopy(body) for key,body in task.items() if not key.startswith('_')}
    queued = sorted([t for t in engine.catalog.list('job') if t['status']=='queued'],key=lambda t:(t['submitted_at'],identifier(t)))
    value['queue_position'] = next((i+1 for i,t in enumerate(queued) if identifier(t)==identifier(task)),None) if task['status']=='queued' else None
    if task['started_at'] and task['status'] not in TERMINAL:
        value['elapsed_seconds'] = max(0.,time.time()-datetime.fromisoformat(task['started_at']).timestamp())
    return value


def submit(engine,request,kind):
    with engine.lock:
        pending = [task for task in engine.catalog.list('job') if task['status']=='queued']
        if len(pending)>=engine.settings['jobs']['max_queued']:
            raise APIError(429,'QUEUE_FULL','任务队列已满，请稍后重试')
        config = training_config(engine,request) if kind=='training' else tuning_config(engine,request)
        task_id = ('train_' if kind=='training' else 'tune_')+uuid4().hex
        folder = engine.store.root/('trainings' if kind=='training' else 'tuning_jobs')/task_id
        folder.mkdir(parents=True,exist_ok=False)
        save_json(folder/'effective_config.json',config)
        path = '/trainings/' if kind=='training' else '/tuning-jobs/'
        warnings = []
        if kind=='tuning' and config['profile']=='selected':
            cutoff = engine.shared['period']['validation_end']
            if any(fold['training_cutoff'] < cutoff for fold in config['folds']):
                warnings.append(issue('RETROSPECTIVE_FIXED_FEATURES','较早复核折属于固定事后组合下的回顾验证'))
        task = dict(kind=kind,name=request.name or ('自动训练' if kind=='training' else '自动调参')+'｜'+config['profile']+'｜截至'+config['split']['latest_observed_month'],
            profile=config['profile'],model_name='CoRegionLng',data_version=config['data_version'],
            device=config['device'],status='queued',stage='queued',submitted_at=now(),started_at=None,finished_at=None,
            elapsed_seconds=0.,queue_position=len(pending)+1,
            progress=dict(unit='epoch' if kind=='training' else 'fit',completed_units=0,total_units=len(config['schedule']) if kind=='training' else None,
                          phase_percent=0.,current_epoch=None,max_epochs=None),warnings=warnings,error=None,status_url=path+task_id,
            request=request.model_dump(exclude_unset=True),resolved_split=config['split'],
            _internal=dict(pid=None,nonce=uuid4().hex,config_snapshot=deepcopy(config)))
        task[kind+'_id'] = task_id
        if kind=='training':
            task.update(parameters=config['parameters'],training_plan=dict(epochs=len(config['schedule']),lr_schedule=config['schedule']),
                        tuning_id=config['tuning_id'],result=None)
        else:
            settings = config['settings']
            search = dict(search_trials=settings['search_trials'],finalists=settings['finalists'],
                validation_folds=settings['validation_folds'],fold_months=settings['fold_months'],
                replications_per_fold=len(settings['training_seeds']),include_initial_candidates=request.include_initial_candidates,
                initial_candidates_count=min(len(settings['initial_candidates']),len(config['candidates'])),search_space=settings['search_space'],
                max_duration_minutes=config['max_duration_minutes'],
                planned_fits=settings['search_trials']+settings['finalists']*settings['validation_folds']*len(settings['training_seeds']),folds=config['folds'])
            task.update(resolved_search=search,best_so_far=None,result_available=False,results_url=path+task_id+'/results')
            save_json(folder/'search_plan.json',config['candidates'])
        save_json(folder/'task.json',{k:v for k,v in task.items() if not k.startswith('_')})
        engine.catalog.put('job',task_id,task)
        engine.store.event('job_submitted',job_id=task_id,job_kind=kind)
        return public_task(engine,task)


def list_tasks(engine,kind,**filters):
    rows = [t for t in engine.catalog.list('job') if t['kind']==kind]
    for key,value in filters.items():
        if value is not None:
            rows = [t for t in rows if t.get(key)==value]
    return [public_task(engine,t) for t in sorted(rows,key=lambda t:(t['submitted_at'],identifier(t)),reverse=True)]


def cancel(engine,task_id,kind):
    engine.task(task_id,kind,public=False)
    repeated = False
    def change(task):
        nonlocal repeated
        if task['status']=='cancelled':
            repeated = True
            return {}
        if task['status'] in TERMINAL:
            raise APIError(409,'INVALID_STATE','已结束任务不能取消')
        status = 'cancelled' if task['status']=='queued' else 'cancel_requested'
        internal = dict(task['_internal'])
        internal.setdefault('cancel_requested_at',now())
        return dict(status=status,finished_at=now() if status=='cancelled' else None,_internal=internal)
    task = engine.catalog.update_job(task_id,change)
    id_field = 'tuning_id' if kind=='tuning' else 'training_id'
    return {id_field:task_id,'status':task['status'],
            'requested_at':task['_internal'].get('cancel_requested_at',task['finished_at'] or now()),
            'message':'任务已取消' if task['status']=='cancelled' else '已请求停止，等待安全检查点'},200 if repeated else 202


def process_alive(pid):
    if not pid:
        return False
    if os.name=='nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code))) and code.value==259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def task_active(task):
    pid = task['_internal'].get('pid')
    if process_alive(pid):
        return True
    # 领取任务与写入子进程编号之间留出启动宽限，避免重叠重启误判。
    return not pid and task.get('started_at') and time.time()-datetime.fromisoformat(task['started_at']).timestamp()<10


class JobManager:
    def __init__(self,engine):
        self.engine = engine
        self.stop_event = threading.Event()
        self.process = None
        self.current_id = None
        self.stream = None
        self.thread = threading.Thread(target=self.loop,name='lng-job-coordinator',daemon=True)

    def start(self):
        for task in self.engine.catalog.list('job'):
            if task['status'] in ('running','cancel_requested') and not task_active(task):
                self.interrupt(identifier(task),'工作进程已停止，任务未完成')
        self.thread.start()

    def interrupt(self,task_id,message):
        def update(task):
            if task['status'] in TERMINAL:
                return {}
            return dict(status='interrupted',finished_at=now(),error=dict(code='WORKER_STOPPED',message=message,retryable=True))
        self.engine.catalog.update_job(task_id,update)

    def loop(self):
        while not self.stop_event.wait(.2):
            try:
                self.tick()
            except Exception as exc:
                self.engine.store.event('job_coordinator_error',error_type=type(exc).__name__)

    def tick(self):
        if self.process is not None:
            if self.process.poll() is None:
                return
            task = self.engine.catalog.get('job',self.current_id)
            if task['status'] not in TERMINAL:
                self.interrupt(self.current_id,'工作进程意外退出，请查看服务日志')
            self.stream.close()
            self.process = self.stream = self.current_id = None
        tasks = self.engine.catalog.list('job')
        running = [t for t in tasks if t['status'] in ('running','cancel_requested')]
        for task in running:
            if not task_active(task):
                self.interrupt(identifier(task),'先前工作进程已退出')
        if any(task_active(t) for t in running):
            return
        queued = sorted([t for t in tasks if t['status']=='queued'],key=lambda t:(t['submitted_at'],identifier(t)))
        if not queued:
            return
        task = queued[0]
        task_id = identifier(task)
        claimed = False
        def claim(value):
            nonlocal claimed
            if value['status']!='queued':
                return {}
            claimed = True
            return dict(status='running',stage='preparing',started_at=now())
        self.engine.catalog.update_job(task_id,claim)
        if not claimed:
            return
        folder = task_directory(self.engine.store.root,task)
        self.stream = (folder/('tuning.log' if task['kind']=='tuning' else 'training.log')).open('a',encoding='utf-8')
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
        try:
            self.process = subprocess.Popen([sys.executable,'-X','utf8','-B','-m','service.worker',
                '--storage',str(self.engine.store.root),'--job',task_id],cwd=ROOT,env=env,
                stdout=self.stream,stderr=self.stream,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except Exception:
            self.stream.close()
            self.stream = None
            self.interrupt(task_id,'无法启动训练工作进程')
            raise
        self.current_id = task_id
        self.engine.catalog.update_job(task_id,lambda value:dict(_internal=dict(value['_internal'],pid=self.process.pid)))

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.process is not None and self.process.poll() is None:
            # 只停止本协调器创建并持有句柄的进程。
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.interrupt(self.current_id,'服务停止，未完成任务已中断')
        if self.stream is not None:
            self.stream.close()


def tuning_results(engine,task_id,phase):
    task = engine.task(task_id,'tuning',public=False)
    folder = engine.store.root/'tuning_jobs'/task_id
    path = folder/('screening_public.json' if phase=='screening' else 'confirmation_public.json')
    items = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    result = task['_internal'].get('recommendation') if task['status']=='succeeded' else None
    return dict(tuning_id=task_id,status=task['status'],result_available=result is not None,profile=task['profile'],
        data_version=task['data_version'],selection_cutoff=task['resolved_split']['validation_end'],
        objective='mean_validation_mape_percent',recommended_trial=result['trial'] if result else None,
        recommended_parameters=result['parameters'] if result else None,
        frozen_training_plan=dict(epochs=result['frozen_epochs'],lr_schedule=result['frozen_lr_schedule']) if result else None,
        selection_rule='按三地区、折与内部重复等权验证MAPE选择；并列比较最差地区，再比较候选编号',phase=phase,items=items)
