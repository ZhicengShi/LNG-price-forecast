"""三地区参数候选的训练、历史验证和固定学习率轨迹重训。"""
import gc
import random

import numpy as np
import torch

from .model import ARCHITECTURE, CoRegionLng
def loss_function(predicted,actual,parameters,y_scale,y_mean):
    if parameters['loss']=='mse':
        return torch.nn.functional.mse_loss(predicted,actual)
    if parameters['loss']=='smooth_l1':
        return torch.nn.functional.smooth_l1_loss(predicted,actual)
    if parameters['loss']=='mae':
        return torch.nn.functional.l1_loss(predicted,actual)
    if parameters['loss']=='mape':
        original = (actual*y_scale+y_mean).abs().clamp_min(1e-6)
        return ((predicted-actual).abs()*y_scale/original).mean()
    raise ValueError('未知训练损失')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mape(actual, pred):
    if actual.shape != pred.shape or not np.isfinite(pred).all() or not np.isfinite(actual).all() or (actual <= 0).any():
        raise ValueError('指标输入无效')
    return abs((pred-actual)/actual).mean(axis=(0, 1))*100


def predict(model, data, prep):
    model.eval()
    with torch.inference_mode():
        out = model(data.x.to(next(model.parameters()).device), data.t.to(next(model.parameters()).device)).cpu().numpy()
    return out*np.asarray(prep['y_scale'])+np.asarray(prep['y_mean'])


def train(prepared, p, fit_end, seed, eval_end=None, replay_lrs=None, verbose=False, callback=None):
    if callback:
        callback(dict(event='preparing'))
    seed_all(seed)
    data, prep = prepared.partition(p, fit_end, 0, fit_end)
    validation = prepared.partition(p, fit_end, fit_end, eval_end)[0] if eval_end else None
    model = CoRegionLng(prepared.layout, p).to(prepared.device)
    optimizer_class = torch.optim.Adam if p['optimizer'] == 'Adam' else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    scheduler = None
    if replay_lrs is None:
        if p['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, p['max_epochs'], eta_min=p['lr']*.05)
        elif p['scheduler'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=max(2, p['patience']//3))
    x, y, t = [v.to(prepared.device) for v in (data.x, data.y, data.t)]
    y_scale = torch.tensor(prep['y_scale'], dtype=torch.float32, device=prepared.device)
    y_mean = torch.tensor(prep['y_mean'], dtype=torch.float32, device=prepared.device)
    generator = torch.Generator().manual_seed(seed)
    history = []
    best, best_epoch, stale, best_forecast = float('inf'), 0, 0, None
    epochs = len(replay_lrs) if replay_lrs is not None else p['max_epochs']
    for epoch in range(1, epochs+1):
        if callback:
            callback(dict(event='epoch_start', epoch=epoch, epochs=epochs))
        if replay_lrs is not None:
            for group in optimizer.param_groups:
                group['lr'] = replay_lrs[epoch-1]
        lr = optimizer.param_groups[0]['lr']
        order = torch.randperm(len(x), generator=generator).to(prepared.device)
        model.train()
        total = 0.
        for offset in range(0, len(x), p['batch_size']):
            if callback:
                callback(dict(event='batch', epoch=epoch, epochs=epochs))
            indexes = order[offset:offset+p['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            predicted = model(x[indexes], t[indexes])
            loss = loss_function(predicted, y[indexes], p, y_scale, y_mean)
            if not torch.isfinite(loss):
                raise FloatingPointError('非有限训练损失')
            loss.backward()
            if p['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), p['grad_clip'])
            optimizer.step()
            total += float(loss.detach())*len(indexes)
        row = dict(epoch=epoch, lr=lr, training_loss=total/len(x))
        if validation is not None:
            forecast = predict(model, validation, prep)
            errors = mape(validation.actual, forecast)
            score = float(errors.mean())
            row.update(score=score, MAPE=errors.tolist())
            if score < best-1e-10:
                best, best_epoch, stale, best_forecast = score, epoch, 0, forecast
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step(score) if p['scheduler'] == 'plateau' else scheduler.step()
        history.append(row)
        if callback:
            callback(dict(event='epoch_end', epoch=epoch, epochs=epochs))
        if verbose:
            print(f'epoch={epoch}/{epochs} train_loss={row["training_loss"]:.8f}', flush=True)
        if validation is not None and stale >= p['patience']:
            break
    if validation is None:
        return model, prep, history
    result = dict(score=best, best_epoch=best_epoch, epochs_run=len(history),
                  MAPE=mape(validation.actual, best_forecast).tolist(), history=history,
                  parameter_count=sum(v.numel() for v in model.parameters()), seed=seed,
                  fit_end=fit_end, eval_end=eval_end)
    del model, optimizer, x, y, t
    gc.collect()
    torch.cuda.empty_cache()
    return result, best_forecast


def checkpoint(model, layout, p, prep, seed, schedule, fit_end):
    return dict(architecture=ARCHITECTURE, layout=layout, parameters=p, preprocessing=prep,
                seed=seed, epochs=len(schedule), lr_schedule=schedule, fit_end=fit_end,
                model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()})
