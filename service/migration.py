"""升级服务记录，保留既有编号、预测数值、默认关系和查询地址。"""
import json

from provenance import without_file_checks, normalise_evidence_versions
from .catalog import encode
from .storage import save_json, now
from . import VERSION


def migrate_history(store,catalog):
    marker = 'independent_versions_v3'
    with catalog.connect() as db:
        if db.execute("SELECT 1 FROM records WHERE kind='migration' AND id=?",(marker,)).fetchone():
            return
        for kind,identifier,raw in db.execute('SELECT kind,id,body FROM records').fetchall():
            value = normalise_evidence_versions(without_file_checks(json.loads(raw)))
            if kind=='model':
                for key in ('fixed_test','backtest'):
                    spec = value.get('evidence_specs',{}).get(key)
                    if spec:
                        spec.setdefault('version',identifier+'-'+key)
                        if key=='fixed_test':
                            spec.setdefault('model_version',identifier)
                if value.get('fixed_predictions'):
                    value['fixed_version'] = identifier+'-test'
            if kind=='job':
                folder = 'trainings' if value['kind']=='training' else 'tuning_jobs'
                path = store.root/folder/identifier/'effective_config.json'
                if path.is_file():
                    config = without_file_checks(json.loads(path.read_text(encoding='utf-8')))
                    save_json(path,config)
                    value['_internal']['config_snapshot'] = config
                if value['status'] in ('queued','running','cancel_requested'):
                    # 算法/接口升级后不自动续跑旧任务；结果和已完成任务仍保留。
                    value.update(status='interrupted',stage='interrupted',finished_at=now(),
                                 error=dict(code='SERVICE_UPGRADED',message='服务版本已升级，请重新提交此任务',retryable=True))
            db.execute('UPDATE records SET body=? WHERE kind=? AND id=?',(encode(value),kind,identifier))
        # 旧缓存无法按新请求比对规则复用；业务记录不受影响。
        db.execute('DELETE FROM idempotency')
        db.execute("INSERT INTO records VALUES ('migration',?,?)",(marker,encode(dict(version=VERSION))))
    path = store.root/'models.json'
    if path.is_file():
        save_json(path,without_file_checks(json.loads(path.read_text(encoding='utf-8'))))
