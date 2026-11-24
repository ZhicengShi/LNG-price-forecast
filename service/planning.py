"""把业务请求解析为任务私有配置；提交时冻结数据、参数和日期。"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from datasets import read_frame
from CoRegionLng.data import REGIONS, build_frame
from CoRegionLng._runtime import validate_parameters
from CoRegionLng._search import candidates
from .errors import APIError, issue
from . import VERSION

HERE = Path(__file__).resolve().parents[1]/'CoRegionLng'


def invalid(message):
    raise APIError(422,'INVALID_REQUEST',message)


def date_split(first, latest, cutoff, validation=None, test_end=None):
    cutoff = pd.Period(str(cutoff),freq='M')
    end = pd.Period(str(test_end or latest),freq='M')
    if cutoff < first or end > latest or cutoff > end:
        invalid('训练或评价边界超出可用月份')
    return dict(first_month=str(first),training_cutoff=str(cutoff),
        validation_start=str(cutoff+1) if validation else None,
        validation_end=str(validation) if validation else None,
        test_start=str((pd.Period(validation,freq='M') if validation else cutoff)+1) if end > (pd.Period(validation,freq='M') if validation else cutoff) else None,
        test_end=str(end) if end > (pd.Period(validation,freq='M') if validation else cutoff) else None,
        latest_observed_month=str(latest),pred_len=3)


def frames_for(engine, data_version):
    metadata, paths = engine.store.data(data_version)
    sources = {r:read_frame(paths[r]) for r in REGIONS}
    first = max(pd.Period(engine.shared['period']['start'],freq='M'),sources[REGIONS[0]].index[0])
    latest = sources[REGIONS[0]].index[-1]
    if first > latest:
        invalid('数据不包含可用训练历史')
    return metadata, {r: f.loc[first:latest].copy() for r,f in sources.items()}, first, latest


def training_config(engine, request):
    chosen = None
    if request.tuning_id:
        chosen = engine.task(request.tuning_id,'tuning',public=False)
        if chosen['status'] != 'succeeded':
            invalid('只能引用已成功完成的调参任务')
        if request.profile is not None and request.profile != chosen['profile']:
            invalid('变量方案与调参任务不一致')
        if request.epochs is not None or request.learning_rate is not None:
            invalid('使用tuning_id时不能另行覆盖epochs或learning_rate')
    profile = chosen['profile'] if chosen else request.profile or 'selected'
    version = request.data_version or (chosen['data_version'] if chosen else engine.data_version)
    metadata,sources,first,latest = frames_for(engine,version)
    cfg = yaml.safe_load((HERE/'configs'/f'{profile}.yaml').read_text(encoding='utf-8'))
    supplied = request.model_fields_set
    if request.mode == 'holdout':
        if supplied & {'training_cutoff','test_end_month'}:
            invalid('holdout模式不能指定training_cutoff或test_end_month')
        cutoff,test_end = latest-(request.holdout_months or 3),latest
    elif request.mode == 'full_history':
        if supplied & {'holdout_months','training_cutoff','test_end_month'}:
            invalid('full_history模式不接收留出月份或手动边界')
        cutoff = test_end = latest
    else:
        if 'holdout_months' in supplied or not request.training_cutoff:
            invalid('custom模式须指定training_cutoff且不能指定holdout_months')
        cutoff = pd.Period(request.training_cutoff,freq='M')
        test_end = pd.Period(request.test_end_month or str(latest),freq='M')
    if request.mode != 'full_history' and test_end.ordinal-cutoff.ordinal < 3:
        invalid('训练截止月之后须至少保留3个月完整测试标签')
    split = date_split(first,latest,cutoff,test_end=test_end)
    if chosen:
        effective = engine.task_config(chosen)
        result = chosen['_internal']['recommendation']
        parameters = deepcopy(result['parameters'])
        schedule = list(result['frozen_lr_schedule'])
        selection_end = effective['selection_cutoff']
        if str(cutoff) < selection_end:
            invalid('训练截止月不能早于参数选择截止月')
        _,original,_,_ = frames_for(engine,chosen['data_version'])
        month_range = pd.period_range(effective['split']['first_month'],selection_end,freq='M')
        for region in REGIONS:
            if not month_range.isin(sources[region].index).all() or not np.allclose(
                original[region].loc[month_range].to_numpy(dtype=float),sources[region].loc[month_range].to_numpy(dtype=float),
                rtol=1e-12,atol=1e-12,equal_nan=True):
                invalid('调参所用历史被修改，须重新调参或选择原数据版本')
        layout = deepcopy(effective['layout'])
        if str(first) != layout['first_month']:
            invalid('训练历史首月与调参布局不一致')
        frame,_ = build_frame(sources,profile,{},saved_layout=layout)
        features = effective['features']
    else:
        parameters = deepcopy(cfg['parameters'])
        epochs = request.epochs or cfg['training']['epochs']
        schedule = cfg['training']['lr_schedule'] or [parameters['lr']]*cfg['training']['epochs']
        schedule = (schedule+[schedule[-1]]*epochs)[:epochs]
        if request.learning_rate is not None:
            parameters['lr'] = request.learning_rate
            schedule = [request.learning_rate]*epochs
        selection_end = engine.shared['period']['validation_end']
        if str(cutoff) < selection_end:
            invalid('训练历史早于随包配置的参数选择截止月')
        features = deepcopy(engine.shared['features'])
        fit_end = cutoff.ordinal-first.ordinal+1
        frame,layout = build_frame(sources,profile,features['selected'],fit_end,features['all'])
    validate_parameters(parameters)
    fit_end = cutoff.ordinal-first.ordinal+1
    if fit_end < parameters['seq_len']+3:
        invalid('训练历史不足以构造完整输入和3个月标签')
    if frame.iloc[:fit_end].median().isna().any():
        invalid('训练范围存在全空输入列，请检查变量方案或增加历史')
    return dict(kind='training',profile=profile,data_version=version,split=split,layout=layout,
        parameters=parameters,schedule=schedule,selection_cutoff=selection_end,features=features,
        seed=cfg['training']['seed'],service_version=VERSION,
        device=str(engine.device),threads=engine.settings['threads'],max_duration_minutes=None,
        tuning_id=request.tuning_id,mode=request.mode)


def tuning_config(engine, request):
    version = request.data_version or engine.data_version
    metadata,sources,first,latest = frames_for(engine,version)
    settings = yaml.safe_load(Path(engine.settings['search_config']).read_text(encoding='utf-8'))
    supplied = request.model_fields_set
    defaults = {'quick':(16,2,2,12),'standard':(settings['search_trials'],settings['finalists'],settings['validation_folds'],settings['fold_months']),
                'thorough':(128,8,3,20)}[request.preset]
    for key,value in zip(('search_trials','finalists','validation_folds','fold_months'),defaults):
        settings[key] = getattr(request,key) if getattr(request,key) is not None else value
    if settings['finalists'] > settings['search_trials']:
        invalid('复核名额不能超过候选数')
    if request.split_mode == 'auto':
        if supplied & {'train_end_month','validation_end_month','test_end_month'}:
            invalid('auto模式不能指定手动月份边界')
        end = latest-(request.holdout_months or engine.settings['jobs']['holdout_months'])
        train_end = end-(request.validation_months or engine.settings['jobs']['validation_months'])
        test_end = latest
    else:
        if supplied & {'holdout_months','validation_months'} or not request.train_end_month or not request.validation_end_month:
            invalid('custom模式须指定训练和验证末月，且不能指定自动划分长度')
        train_end,end = [pd.Period(v,freq='M') for v in (request.train_end_month,request.validation_end_month)]
        test_end = pd.Period(request.test_end_month or str(latest),freq='M')
    if end.ordinal-train_end.ordinal < 3 or test_end.ordinal-end.ordinal < 3:
        invalid('验证和保留测试均须包含至少3个月完整标签')
    if request.profile=='selected' and str(end)<engine.shared['period']['validation_end']:
        invalid('开发期末月不能早于既定组合变量的选择截止月；较早历史仅用于开发期内的回顾复核')
    split = date_split(first,latest,train_end,validation=str(end),test_end=test_end)
    overrides = request.search_space or {}
    if set(overrides)-set(settings['search_space']):
        invalid('search_space含未知字段')
    for key,values in overrides.items():
        if not isinstance(values,list) or not values:
            invalid('搜索范围每项须为非空数组')
        original = settings['search_space'][key]
        if key == 'lr_log10':
            if len(values)!=2 or any(type(v) not in (int,float) or not np.isfinite(v) for v in values) or not -8 <= values[0] < values[1] <= -2:
                invalid('lr_log10须为[-8,-2]内递增的两个有限数')
        elif any(type(v) is not type(original[0]) and not (type(original[0]) is float and type(v) is int) for v in values):
            invalid('搜索字段类型不正确：'+key)
        settings['search_space'][key] = values
    anchors = settings['initial_candidates'] if request.include_initial_candidates else []
    def allowed(p):
        return all((10**values[0] <= p['lr'] <= 10**values[1]) if key=='lr_log10' else p[key] in values
                   for key,values in overrides.items())
    settings['initial_candidates'] = [p for p in anchors if allowed(p)]
    settings['profiles'] = [request.profile]
    # 验证整个候选空间，而不仅验证本次随机抽到的候选。
    base = deepcopy(yaml.safe_load((HERE/'configs/selected.yaml').read_text(encoding='utf-8'))['parameters'])
    maxima = dict(seq_len=120,d_model=512,n_heads=8,e_layers=4,ff_multiplier=8,head_layers=3,
                  head_multiplier=2,batch_size=256,max_epochs=500,patience=500,cycle=120)
    for key,values in settings['search_space'].items():
        if key == 'lr_log10':
            continue
        for value in values:
            if type(value) in (int,float) and not np.isfinite(value):
                invalid('搜索范围含非有限数值：'+key)
            p = dict(base,**{key:value})
            if key == 'n_heads':
                p['d_model'] = max(8,value)
            if key == 'd_model':
                p['n_heads'] = 1
            if key in maxima and (type(value) is not int or not 0 <= value <= maxima[key]):
                invalid('搜索字段超过服务器规模限制：'+key)
            try:
                validate_parameters(p)
            except (ValueError,TypeError) as exc:
                invalid('搜索范围无效：'+key)
    plan = candidates(settings)
    for parameters in plan:
        validate_parameters(parameters)
    fit_end,development_end = train_end.ordinal-first.ordinal+1,end.ordinal-first.ordinal+1
    minimum = development_end-settings['fold_months']*settings['validation_folds']
    if min(minimum,fit_end) < max(p['seq_len'] for p in plan)+3:
        invalid('历史不足以覆盖最长候选窗口及全部复核折')
    features = deepcopy(engine.shared['features'])
    # 布局只在初筛训练历史决定，保留测试完全不进入搜索数据。
    dev_sources = {r:f.loc[:end].copy() for r,f in sources.items()}
    frame,layout = build_frame(dev_sources,request.profile,features['selected'],fit_end,features['all'])
    folds = []
    for i in range(settings['validation_folds']):
        last = development_end-settings['fold_months']*(settings['validation_folds']-i-1)
        cutoff = last-settings['fold_months']
        if frame.iloc[:cutoff].median().isna().any():
            invalid('较早复核训练段存在全空输入列，请调整历史或方案')
        folds.append(dict(fold=i+1,training_cutoff=str(frame.index[cutoff-1]),
                          validation_start=str(frame.index[cutoff]),validation_end=str(frame.index[last-1])))
    return dict(kind='tuning',profile=request.profile,data_version=version,split=split,layout=layout,
        settings=settings,candidates=plan,features=features,selection_cutoff=str(end),folds=folds,
        service_version=VERSION,device=str(engine.device),threads=engine.settings['threads'],
        max_duration_minutes=request.max_duration_minutes or engine.settings['jobs']['max_duration_minutes'])
