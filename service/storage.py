"""服务独立存储：独立数据版本、原子结果及追加事件日志。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import threading
from uuid import uuid4
from provenance import without_file_checks, normalise_evidence_versions

REGIONS = ('shanghai', 'zhejiang', 'jiangsu')


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        for name in ('data_versions', 'predictions', 'logs'):
            (self.root/name).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def event(self, kind, **fields):
        with self.lock:
            with (self.root/'logs/events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(time=now(), event=kind, **fields), ensure_ascii=False, allow_nan=False)+'\n')

    def register_model(self, version, metadata):
        with self.lock:
            path = self.root/'models.json'
            records = without_file_checks(json.loads(path.read_text(encoding='utf-8'))) if path.exists() else {}
            if version not in records:
                records[version] = dict(registered_at=now(), **metadata)
                save_json(path, records)
            return records[version]['registered_at']

    def capture_data(self, payloads, validator, **extra):
        with self.lock:
            # 重复上传直接比较三份文件内容，不依赖任何摘要算法。
            for directory in sorted((self.root/'data_versions').glob('data_*')):
                if not (directory/'metadata.json').is_file():
                    continue
                if all((directory/(r+'.xlsx')).is_file() and (directory/(r+'.xlsx')).stat().st_size == len(payloads[r]) and
                       (directory/(r+'.xlsx')).read_bytes() == payloads[r] for r in REGIONS):
                    metadata, _ = self.data(directory.name)
                    return metadata
            version = 'data_'+uuid4().hex
            final = self.root/'data_versions'/version
            staging = self.root/'data_versions'/('_pending_'+uuid4().hex)
            staging.mkdir()
            try:
                paths = {}
                for region in REGIONS:
                    paths[region] = staging/(region+'.xlsx')
                    paths[region].write_bytes(payloads[region])
                summary = validator(paths)
                metadata = dict(data_version=version, imported_at=now(), **summary, **extra)
                save_json(staging/'metadata.json', metadata)
                staging.rename(final)
                self.event('data_registered', data_version=version, latest_observed_month=metadata['latest_observed_month'])
                return metadata
            finally:
                if staging.exists():
                    # 只清理本次创建、且位于服务数据目录中的暂存目录。
                    assert staging.resolve().parent == (self.root/'data_versions').resolve()
                    assert staging.name.startswith('_pending_')
                    shutil.rmtree(staging)

    def data(self, version):
        if not re.fullmatch(r'data_(?:[a-f0-9]{24}|[a-f0-9]{32})', version):
            raise KeyError('数据版本编号无效')
        directory = self.root/'data_versions'/version
        if not (directory/'metadata.json').is_file():
            raise KeyError('数据版本不存在')
        original = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        metadata = without_file_checks(original)
        if metadata != original:
            save_json(directory/'metadata.json',metadata)
        paths = {r: directory/(r+'.xlsx') for r in REGIONS}
        if any(not path.is_file() for path in paths.values()):
            raise ValueError('已登记的数据表不存在，请检查服务运行目录')
        return metadata, paths

    def save_prediction(self, response):
        path = self.root/'predictions'/response['prediction_id']/'response.json'
        save_json(path, response)
        self.event('prediction_complete', prediction_id=response['prediction_id'],
                   model_version=response['model_status']['model_version'],
                   data_version=response['model_status']['data_version'])

    def prediction(self, version):
        if not re.fullmatch(r'pred_[a-f0-9]{32}', version):
            raise KeyError('预测编号无效')
        path = self.root/'predictions'/version/'response.json'
        if not path.is_file():
            raise KeyError('预测记录不存在')
        original = json.loads(path.read_text(encoding='utf-8'))
        result = normalise_evidence_versions(without_file_checks(original))
        # 保留既有预测文件，查询时只将旧的只读字段名转换为当前名称。
        status = result['model_status']
        if 'dataset_id' in status:
            status['data_version'] = status.pop('dataset_id')
        if result != original:
            save_json(path,result)
        return result
