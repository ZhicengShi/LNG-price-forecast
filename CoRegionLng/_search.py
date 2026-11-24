"""可配置且确定性的参数候选生成。"""
import json
import numpy as np


def canonical(parameters):
    parameters = dict(parameters)
    if parameters['output_mode']=='level':
        parameters['residual_scale'] = 1.0
    if parameters['head_layers']==1:
        parameters['head_multiplier'] = 1
    return parameters


def candidates(settings):
    space = settings['search_space']
    anchored = [dict(p) for p in settings['initial_candidates']]
    rng = np.random.default_rng(settings['search_seed'])
    result,seen = [],set()
    attempts = 0
    while len(result)<settings['search_trials']:
        attempts += 1
        if attempts>settings['search_trials']*100:
            raise ValueError('当前搜索空间不足以生成所需数量的不同候选，请减少search_trials或扩展搜索范围')
        if anchored:
            parameters = anchored.pop(0)
        else:
            parameters = {key:values[int(rng.integers(len(values)))] for key,values in space.items()
                          if key!='lr_log10'}
            parameters['lr'] = float(10**rng.uniform(*space['lr_log10']))
        parameters = canonical(parameters)
        if settings.get('epoch_cap'):
            parameters['max_epochs'] = min(parameters['max_epochs'],settings['epoch_cap'])
            parameters['patience'] = min(parameters['patience'],parameters['max_epochs'])
        if parameters['d_model']%parameters['n_heads']:
            continue
        signature = json.dumps(parameters,sort_keys=True)
        if signature not in seen:
            result.append(parameters)
            seen.add(signature)
    return result

