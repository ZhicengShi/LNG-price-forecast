"""处理表验证、上传限制及数据统计；不依赖任何已加载模型。"""
import io
from itertools import islice
from pathlib import Path
import zipfile
import numpy as np
import openpyxl
from datasets import DATE, FIELDS, TARGET, read_frame
from joint_emaformer.data import REGIONS, NAMES, COMMON
from .errors import APIError, issue


def inspect_xlsx(payload, filename, region, limits):
    if Path(filename or '').suffix.lower() != '.xlsx':
        raise APIError(415, 'UNSUPPORTED_FORMAT', '请上传XLSX文件', [issue('UNSUPPORTED_FORMAT', '文件扩展名须为.xlsx', region)])
    if len(payload) > limits['file_bytes']:
        raise APIError(413, 'FILE_TOO_LARGE', '单文件大小超过限制')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if sum(e.file_size for e in entries) > limits['expanded_bytes']:
                raise APIError(413, 'FILE_TOO_LARGE', '工作簿解压后内容超过限制')
            if not {'[Content_Types].xml','xl/workbook.xml'} <= set(archive.namelist()):
                raise ValueError('工作簿结构无效')
            if any(e.flag_bits & 1 for e in entries):
                raise ValueError('不支持加密工作簿')
    except (zipfile.BadZipFile, ValueError) as exc:
        raise APIError(415, 'UNSUPPORTED_FORMAT', '文件不是可读取的XLSX工作簿',
                       [issue('UNSUPPORTED_FORMAT', '请使用处理后XLSX表重新上传', region)]) from exc


def validate_tables(paths, max_months, template_units):
    frames, summaries, warnings = {}, [], []
    for region in REGIONS:
        try:
            with Path(paths[region]).open('rb') as stream:
                book = openpyxl.load_workbook(stream, read_only=True, data_only=True)
                try:
                    if len(book.worksheets) != 1:
                        raise ValueError('仅支持一个工作表')
                    worksheet = book.active
                    header, units = list(worksheet.iter_rows(min_row=1, max_row=2, values_only=True))
                    if list(header) != [DATE,*FIELDS]:
                        raise ValueError('字段名称或顺序与模板不一致')
                    if list(units) != template_units[region]:
                        wrong = next((i for i,(a,b) in enumerate(zip(units, template_units[region])) if a != b), 0)
                        raise APIError(422, 'INVALID_DATA', NAMES[region]+'单位行与模板不一致',
                            [issue('INVALID_UNIT','请使用样例中的单位',region,field=header[wrong])])
                    row_count = worksheet.max_row
                    if row_count is None:
                        row_count = sum(1 for _ in islice(worksheet.iter_rows(values_only=True),max_months+3))
                    if row_count > max_months+2:
                        raise ValueError('数据月份数超过服务配置限制')
                finally:
                    book.close()
            frame = read_frame(paths[region])
        except APIError:
            raise
        except Exception as exc:
            raise APIError(422, 'INVALID_DATA', NAMES[region]+'数据表校验失败',
                           [issue('INVALID_TABLE', str(exc) if isinstance(exc,ValueError) else '工作簿无法读取',region,
                                  suggestion='请核对单工作表、字段、单位及连续YYYY-MM月份')]) from exc
        labels = frame[TARGET]
        invalid = ~np.isfinite(labels.to_numpy(dtype=float)) | (labels.to_numpy(dtype=float) <= 0)
        if invalid.any():
            index = int(np.flatnonzero(invalid)[0])
            raise APIError(422, 'INVALID_DATA', NAMES[region]+'存在缺失或非正目标价格',
                           [issue('TARGET_MISSING','目标价格须为正且不可缺失',region,str(frame.index[index]),TARGET,
                                  '补齐真实价格，不要填0或插补标签')])
        frames[region] = frame
        missing = {field:int(frame[field].isna().sum()) for field in FIELDS}
        empty = [field for field,count in missing.items() if count == len(frame)]
        summaries.append(dict(region=region,region_name=NAMES[region],filename=Path(paths[region]).name,
            months=len(frame),first_month=str(frame.index[0]),latest_observed_month=str(frame.index[-1]),
            missing_counts=missing,all_empty_columns=empty))
        if empty:
            warnings.append(issue('EMPTY_COLUMNS','存在全空的非目标字段，训练时按方案处理',region))
    first = frames[REGIONS[0]]
    for region in REGIONS[1:]:
        if not frames[region].index.equals(first.index):
            raise APIError(422,'INVALID_DATA','三地区年月范围须严格一致',
                           [issue('MONTH_RANGE_MISMATCH','三地区须包含相同的连续月份',region)])
        a,b = first[COMMON].to_numpy(dtype=float),frames[region][COMMON].to_numpy(dtype=float)
        mismatch = ~np.isclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True)
        if mismatch.any():
            row,col = np.argwhere(mismatch)[0]
            raise APIError(422,'INVALID_DATA','三地区的全国共同指标不一致',
                           [issue('COMMON_VALUE_CONFLICT','请核对全国指标',region,str(first.index[row]),COMMON[col])])
    return dict(first_month=str(first.index[0]),latest_observed_month=str(first.index[-1]),months=len(first),
                regions=list(REGIONS),region_summaries=summaries,warnings=warnings,input_format='xlsx_fields_units_monthly')


def template_units(paths):
    result = {}
    for region,path in paths.items():
        book = openpyxl.load_workbook(path,read_only=True,data_only=True)
        try:
            result[region] = list(next(book.active.iter_rows(min_row=2,max_row=2,values_only=True)))
        finally:
            book.close()
    return result
