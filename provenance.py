"""记录来源、时间与版本；读取旧记录时移除已停用的文件摘要字段。"""
from datetime import datetime, timezone
from pathlib import Path
import re

def source_info(path):
    path = Path(path)
    stat = path.stat()
    return dict(filename=path.name, bytes=stat.st_size,
                updated_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())


def without_file_checks(value):
    """兼容已有权重和历史记录，不计算或验证旧摘要。"""
    if isinstance(value, dict):
        return {key: without_file_checks(item) for key, item in value.items()
                if 'sha256' not in str(key).lower() and key != 'hash_algorithm'}
    if isinstance(value, list):
        return [without_file_checks(item) for item in value]
    if isinstance(value, tuple):
        return tuple(without_file_checks(item) for item in value)
    return value


def normalise_evidence_versions(value, profile='legacy'):
    """将旧证据摘要编号转换为可读的历史证据编号，保留预测和指标。"""
    if isinstance(value, dict):
        profile = value.get('profile',value.get('model_status',{}).get('profile',profile))
        result = {key:normalise_evidence_versions(item,profile) for key,item in value.items()}
        for key in ('version','calibration_version'):
            if re.fullmatch(r'[a-fA-F0-9]{64}',str(result.get(key,''))):
                name = 'backtest' if key=='calibration_version' else result.get('basis','evaluation')
                result[key] = 'legacy-'+profile+'-'+name
        return result
    if isinstance(value,list):
        return [normalise_evidence_versions(item,profile) for item in value]
    return value
