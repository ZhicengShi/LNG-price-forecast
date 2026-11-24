"""三地区统一HTTP接口；python -B -m service.app启动。"""
import argparse
import base64
from contextlib import asynccontextmanager
import csv
import hmac
import io
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import APIKeyHeader
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import VERSION
from .engine import Engine, load_settings
from .schemas import (PredictRequest, PredictionResponse, Region, Profile, ActivateData, ActivateModel,
                      TrainingRequest, TuningRequest, EmptyRequest)
from .errors import APIError, issue
from .uploads import inspect_xlsx
from .tasks import submit, list_tasks, cancel, tuning_results
from . import schemas as api
from .annual import AnnualService
from . import annual_schemas as annual_api


def create_app(config_path=None):
    settings = load_settings(config_path)
    api_key = os.environ.get(settings['api_key_env'], '')

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = await run_in_threadpool(Engine,settings)
        try:
            app.state.engine.annual = AnnualService(app.state.engine)
            app.state.engine.start_jobs()
            yield
        finally:
            await run_in_threadpool(app.state.engine.close)

    app = FastAPI(title='三地区天然气价格预测接口',version=VERSION,lifespan=lifespan,
        responses={code:{'model':api.ErrorResponse} for code in (401,404,405,409,413,415,422,429,500,503)})
    key_header = APIKeyHeader(name='X-API-Key',auto_error=False)

    def authorize(key=Depends(key_header)):
        if api_key and (key is None or not hmac.compare_digest(key.encode('utf-8'),api_key.encode('utf-8'))):
            raise HTTPException(status_code=401,detail='API密钥无效')

    def query_guard(request: Request):
        route = request.scope.get('route')
        allowed = {
            '/datasets': {'source','is_default'},
            '/models': {'profile','is_default','data_version'},
            '/models/{model_version}': {'data_version'},
            '/trainings': {'profile','status','data_version','tuning_id'},
            '/tuning-jobs': {'profile','status','data_version'},
            '/tuning-jobs/{tuning_id}/results': {'phase'},
            '/predictions': {'profile','model_version','data_version','generated_from','generated_to'},
            '/annual-budget/models': {'budget_year','model','profile'},
            '/annual-budget/predictions': {'budget_year','model','profile','data_version'},
            '/annual-budget/trainings': {'budget_year','model','profile','status','data_version'},
        }.get(getattr(route,'path',''),set()) if request.method=='GET' else set()
        unknown = set(request.query_params)-allowed
        if unknown:
            raise APIError(422,'INVALID_REQUEST','请求含未知查询字段',
                           [issue('UNKNOWN_FIELD','未定义的查询字段',field=key) for key in sorted(unknown)])

    router = APIRouter(dependencies=[Depends(authorize),Depends(query_guard)])

    def error_response(request,exc):
        return JSONResponse(status_code=exc.status,content=dict(error=exc.code,message=exc.message,
            request_id=getattr(request.state,'request_id','req_'+uuid4().hex),details=exc.details))

    @app.middleware('http')
    async def request_log(request,call_next):
        started = perf_counter()
        request.state.request_id = 'req_'+uuid4().hex
        maximum = settings['upload']['total_bytes']+1024**2
        if request.headers.get('content-length','').isdigit() and int(request.headers['content-length'])>maximum:
            response = error_response(request,APIError(413,'FILE_TOO_LARGE','请求内容超过服务限制'))
        else:
            received = 0
            original_receive = request._receive
            async def limited_receive():
                nonlocal received
                message = await original_receive()
                received += len(message.get('body',b''))
                if received>maximum:
                    raise StarletteHTTPException(413,'请求内容超过服务限制')
                return message
            request._receive = limited_receive
            try:
                response = await call_next(request)
            except APIError as exc:
                response = error_response(request,exc)
        response.headers['X-Request-ID'] = request.state.request_id
        engine = getattr(request.app.state,'engine',None)
        if engine is not None:
            engine.store.event('http_request',method=request.method,path=request.url.path,status=response.status_code,
                               request_id=request.state.request_id,elapsed_ms=(perf_counter()-started)*1000)
        return response

    @app.exception_handler(APIError)
    async def business_error(request,exc):
        return error_response(request,exc)

    @app.exception_handler(ValueError)
    async def invalid_data(request,exc):
        return error_response(request,APIError(422,'INVALID_DATA','数据或参数校验失败',[issue('INVALID_VALUE',str(exc))]))

    @app.exception_handler(KeyError)
    async def missing_record(request,exc):
        return error_response(request,APIError(404,'NOT_FOUND','指定的记录不存在'))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request,exc):
        code = {401:'UNAUTHORIZED',404:'NOT_FOUND',405:'METHOD_NOT_ALLOWED',413:'FILE_TOO_LARGE'}.get(exc.status_code,'INVALID_REQUEST')
        return error_response(request,APIError(exc.status_code,code,str(exc.detail)))

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request,exc):
        fields = [issue('INVALID_FIELD',str(e['msg']),field='.'.join(map(str,e['loc']))) for e in exc.errors()]
        return error_response(request,APIError(422,'INVALID_REQUEST','请求字段校验失败',fields))

    @app.exception_handler(Exception)
    async def unexpected(request,exc):
        request.app.state.engine.store.event('internal_error',request_id=request.state.request_id,error_type=type(exc).__name__)
        response = error_response(request,APIError(500,'INTERNAL_ERROR','服务内部异常，请凭请求编号联系维护人员'))
        response.headers['X-Request-ID'] = request.state.request_id
        return response

    def mutation(request,body,operation):
        fingerprint = json.dumps(dict(method=request.method,path=request.url.path,body=body),
                                 sort_keys=True,ensure_ascii=False,separators=(',',':'))
        response_model = getattr(request.scope.get('route'),'response_model',None)
        def checked():
            value,status = operation()
            if response_model is not None:
                value = response_model.model_validate(value).model_dump(mode='json')
            return value,status
        value,status = request.app.state.engine.idempotent(request.headers.get('Idempotency-Key'),fingerprint,'service-v3',checked)
        if isinstance(value,dict) and '_cached_error' in value:
            raise APIError(**value['_cached_error'])
        return JSONResponse(status_code=status,content=value)

    @router.get('/models',summary='模型状态、变量及版本',response_model=api.ModelList)
    def models(request: Request,profile: Profile | None=None,is_default: bool | None=None,data_version: str | None=None):
        return dict(models=request.app.state.engine.model_list(profile,is_default,data_version))

    @router.get('/models/{model_version}',summary='查看指定模型版本',response_model=api.ModelRecord)
    def model(model_version: str,request: Request,data_version: str | None=None):
        return request.app.state.engine.model_record(model_version,data_version)

    @router.post('/models/{model_version}/activate',summary='热加载并切换默认模型',response_model=api.ModelActivation)
    def activate_model(model_version: str,body: ActivateModel,request: Request):
        return mutation(request,body.model_dump(exclude_unset=True),lambda:(request.app.state.engine.activate_model(model_version,body),200))

    @router.get('/templates/{region}',summary='下载地区数据样例',response_class=FileResponse,
                responses={200:{'content':{'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':{'schema':{'type':'string','format':'binary'}}}}})
    def template(region: Region,request: Request):
        path = request.app.state.engine.reference_paths[region]
        return FileResponse(path,filename=path.name,media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    @router.post('/predictions',response_model=PredictionResponse,summary='返回价格、区间、精度和模型状态')
    def predict(body: PredictRequest,request: Request):
        return mutation(request,body.model_dump(exclude_unset=True),lambda:(request.app.state.engine.predict(body),200))

    @router.get('/predictions',summary='查询预测历史',response_model=api.PredictionList)
    def predictions(request: Request,profile: Profile | None=None,model_version: str | None=None,data_version: str | None=None,
                    generated_from: str | None=None,generated_to: str | None=None):
        return dict(items=request.app.state.engine.prediction_list(profile=profile,model_version=model_version,data_version=data_version,
                    generated_from=generated_from,generated_to=generated_to))

    @router.get('/predictions/{prediction_id}',response_model=PredictionResponse,summary='查询已保存预测')
    def prediction(prediction_id: str,request: Request):
        return request.app.state.engine.store.prediction(prediction_id)

    @router.get('/predictions/{prediction_id}/csv',summary='下载价格和区间明细',response_class=Response,
                responses={200:{'content':{'text/csv':{'schema':{'type':'string','format':'binary'}}}}})
    def download(prediction_id: str,request: Request):
        result = request.app.state.engine.store.prediction(prediction_id)
        items = {(r['region'],r['horizon']):r for r in result['prediction_intervals']['items']}
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream,fieldnames=['region','region_name','target_month','horizon','prediction','unit',
            'lower','upper','interval_available','model_version','model_training_cutoff','latest_observed_month'])
        writer.writeheader()
        status = result['model_status']
        for row in result['predictions']:
            interval = items[(row['region'],row['horizon'])]
            writer.writerow(dict(row,lower=interval['lower'],upper=interval['upper'],interval_available=interval['available'],
                **{key:status[key] for key in ('model_version','model_training_cutoff','latest_observed_month')}))
        return Response(stream.getvalue().encode('utf-8-sig'),media_type='text/csv',
                        headers={'Content-Disposition':f'attachment; filename="{prediction_id}.csv"'})

    @router.post('/datasets',summary='上传三地区处理后XLSX',response_model=api.UploadResponse,responses={201:{'model':api.UploadResponse}})
    async def upload(request: Request,shanghai_file: UploadFile=File(...),zhejiang_file: UploadFile=File(...),jiangsu_file: UploadFile=File(...),
                     name: str | None=Form(default=None,max_length=100),note: str=Form(default='',max_length=500)):
        engine = request.app.state.engine
        form = await request.form()
        unknown = set(form)-{'shanghai_file','zhejiang_file','jiangsu_file','name','note'}
        if unknown or any(len(form.getlist(key))>1 for key in form):
            raise APIError(422,'INVALID_REQUEST','上传含未知或重复的字段')
        payloads,filenames = {},{}
        total = 0
        for region,file in zip(('shanghai','zhejiang','jiangsu'),(shanghai_file,zhejiang_file,jiangsu_file)):
            payload = await file.read(settings['upload']['file_bytes']+1)
            filename = (file.filename or '').replace('\\','/').split('/')[-1]
            await run_in_threadpool(inspect_xlsx,payload,filename,region,settings['upload'])
            total += len(payload)
            if total>settings['upload']['total_bytes']:
                raise APIError(413,'FILE_TOO_LARGE','三份文件总大小超过限制')
            payloads[region],filenames[region] = payload,filename[:255]
        fingerprint = dict(files={region:base64.b64encode(payload).decode('ascii') for region,payload in payloads.items()},name=name,note=note)
        def operation():
            with engine.lock:
                metadata = engine.store.capture_data(payloads,engine.validate_paths,source='upload')
                _,created = engine.register_dataset(metadata,'upload',name,note,filenames)
                return dict(created=created,dataset=engine.dataset(metadata['data_version'])),201 if created else 200
        return await run_in_threadpool(mutation,request,fingerprint,operation)

    @router.get('/datasets',summary='数据版本列表',response_model=api.DataList)
    def datasets(request: Request,source: Literal['configured','upload'] | None=None,is_default: bool | None=None):
        return dict(items=request.app.state.engine.dataset_list(source,is_default))

    @router.get('/datasets/{data_version}',summary='数据版本详情',response_model=api.DataVersion)
    def dataset(data_version: str,request: Request):
        return request.app.state.engine.dataset(data_version)

    @router.post('/datasets/{data_version}/activate',summary='切换默认数据',response_model=api.DataActivation)
    def activate_data(data_version: str,body: ActivateData,request: Request):
        return mutation(request,body.model_dump(exclude_unset=True),lambda:(request.app.state.engine.activate_data(data_version,body),200))

    @router.post('/trainings',status_code=202,summary='按固定方案提交训练',response_model=api.TrainingTask)
    def training(body: TrainingRequest,request: Request):
        return mutation(request,body.model_dump(exclude_unset=True),lambda:(submit(request.app.state.engine,body,'training'),202))

    @router.get('/trainings',summary='训练任务列表',response_model=api.TrainingList)
    def trainings(request: Request,profile: Profile | None=None,status: api.TaskStatus | None=None,data_version: str | None=None,tuning_id: str | None=None):
        return dict(items=list_tasks(request.app.state.engine,'training',profile=profile,status=status,data_version=data_version,tuning_id=tuning_id))

    @router.get('/trainings/{training_id}',summary='训练进度及结果',response_model=api.TrainingTask)
    def training_detail(training_id: str,request: Request):
        return request.app.state.engine.task(training_id,'training')

    @router.get('/trainings/{training_id}/csv',summary='下载新权重留出测试明细',response_class=Response,
                responses={200:{'content':{'text/csv':{'schema':{'type':'string','format':'binary'}}}}})
    def training_csv(training_id: str,request: Request):
        engine = request.app.state.engine
        task = engine.task(training_id,'training')
        if task['status']!='succeeded' or not task['result']['test_csv_url']:
            raise APIError(409,'INVALID_STATE','此任务尚无可下载的测试明细')
        return FileResponse(engine.store.root/'trainings'/training_id/'test_predictions.csv',media_type='text/csv',filename=training_id+'.csv')

    @router.post('/trainings/{training_id}/cancel',summary='取消训练任务',response_model=api.CancelTraining,responses={202:{'model':api.CancelTraining}})
    def cancel_training(training_id: str,body: EmptyRequest,request: Request):
        return mutation(request,{},lambda:cancel(request.app.state.engine,training_id,'training'))

    @router.post('/tuning-jobs',status_code=202,summary='提交自动调参任务',response_model=api.TuningTask)
    def tuning(body: TuningRequest,request: Request):
        return mutation(request,body.model_dump(exclude_unset=True),lambda:(submit(request.app.state.engine,body,'tuning'),202))

    @router.get('/tuning-jobs',summary='调参任务列表',response_model=api.TuningList)
    def tunings(request: Request,profile: Profile | None=None,status: api.TaskStatus | None=None,data_version: str | None=None):
        return dict(items=list_tasks(request.app.state.engine,'tuning',profile=profile,status=status,data_version=data_version))

    @router.get('/tuning-jobs/{tuning_id}',summary='调参进度',response_model=api.TuningTask)
    def tuning_detail(tuning_id: str,request: Request):
        return request.app.state.engine.task(tuning_id,'tuning')

    @router.get('/tuning-jobs/{tuning_id}/results',summary='调参推荐方案及候选排名',response_model=api.TuningResults)
    def tuning_result(tuning_id: str,request: Request,phase: Literal['screening','confirmation']='confirmation'):
        return tuning_results(request.app.state.engine,tuning_id,phase)

    @router.post('/tuning-jobs/{tuning_id}/cancel',summary='取消调参任务',response_model=api.CancelTuning,responses={202:{'model':api.CancelTuning}})
    def cancel_tuning(tuning_id: str,body: EmptyRequest,request: Request):
        return mutation(request,{},lambda:cancel(request.app.state.engine,tuning_id,'tuning'))

    @router.get('/annual-budget/models', tags=['年度预算版'], summary='年度预算模型、变量与可用年份',
                response_model=annual_api.AnnualModelList)
    def annual_models(request: Request, budget_year: int=Query(default=2026,ge=2000,le=9998),
                      model: Literal['all','lstm','emaformer','mofo','emmoformer','joint_emaformer'] | None=None,
                      profile: Literal['all','selected','both'] | None=None):
        return request.app.state.engine.annual.models(budget_year,model,profile)

    @router.post('/annual-budget/predictions', tags=['年度预算版'], summary='预测下一年度平均价格',
                 response_model=annual_api.AnnualPredictionResponse)
    def annual_predict(body: annual_api.AnnualPredictRequest,request: Request):
        return mutation(request,body.model_dump(exclude_unset=True),
                        lambda:(request.app.state.engine.annual.predict(body),200))

    @router.get('/annual-budget/predictions', tags=['年度预算版'], summary='年度预算预测历史',
                response_model=annual_api.AnnualPredictionList)
    def annual_predictions(request: Request,budget_year: int | None=Query(default=None,ge=2000,le=9998),
                           model: Literal['all','lstm','emaformer','mofo','emmoformer','joint_emaformer'] | None=None,
                           profile: Literal['all','selected','both'] | None=None,data_version: str | None=None):
        return request.app.state.engine.annual.predictions(budget_year,model,profile,data_version)

    @router.get('/annual-budget/predictions/{prediction_id}', tags=['年度预算版'], summary='查询年度预算预测结果',
                response_model=annual_api.AnnualPredictionResponse)
    def annual_prediction(prediction_id: str,request: Request):
        return request.app.state.engine.annual.prediction(prediction_id)

    @router.get('/annual-budget/predictions/{prediction_id}/csv', tags=['年度预算版'], summary='下载年度预算预测明细',
                response_class=Response,
                responses={200:{'content':{'text/csv':{'schema':{'type':'string','format':'binary'}}}}})
    def annual_csv(prediction_id: str,request: Request):
        payload = request.app.state.engine.annual.prediction_csv(prediction_id)
        return Response(payload,media_type='text/csv',
                        headers={'Content-Disposition':f'attachment; filename="{prediction_id}.csv"'})

    @router.post('/annual-budget/trainings', tags=['年度预算版'], summary='提交年度预算训练任务',status_code=202,
                 response_model=annual_api.AnnualTrainingTask)
    def annual_training(body: annual_api.AnnualTrainingRequest,request: Request):
        return mutation(request,body.model_dump(exclude_unset=True),
                        lambda:(request.app.state.engine.annual.submit_training(body),202))

    @router.get('/annual-budget/trainings', tags=['年度预算版'], summary='年度预算训练任务列表',
                response_model=annual_api.AnnualTrainingList)
    def annual_trainings(request: Request,budget_year: int | None=Query(default=None,ge=2000,le=9998),
                         model: Literal['all','lstm','emaformer','mofo','emmoformer','joint_emaformer'] | None=None,
                         profile: Literal['all','selected','both'] | None=None,status: api.TaskStatus | None=None,
                         data_version: str | None=None):
        return dict(version='年度预算版',items=list_tasks(request.app.state.engine,'annual_training',
                    budget_year=budget_year,model=model,profile=profile,status=status,data_version=data_version))

    @router.get('/annual-budget/trainings/{training_id}', tags=['年度预算版'], summary='年度预算训练进度与结果',
                response_model=annual_api.AnnualTrainingTask)
    def annual_training_detail(training_id: str,request: Request):
        return request.app.state.engine.task(training_id,'annual_training')

    @router.post('/annual-budget/trainings/{training_id}/cancel', tags=['年度预算版'], summary='取消年度预算训练',
                 response_model=annual_api.AnnualCancelTraining,responses={202:{'model':annual_api.AnnualCancelTraining}})
    def annual_cancel(training_id: str,body: EmptyRequest,request: Request):
        def operation():
            value,status = cancel(request.app.state.engine,training_id,'annual_training')
            return dict(version='年度预算版',**value),status
        return mutation(request,{},operation)

    app.include_router(router)
    return app


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--host')
    parser.add_argument('--port',type=int)
    parser.add_argument('--device',choices=['cpu','cuda','auto'])
    args = parser.parse_args()
    if args.config:
        os.environ['LNG_SERVICE_CONFIG'] = str(args.config.resolve())
    if args.device:
        os.environ['LNG_SERVICE_DEVICE'] = args.device
    settings = load_settings()
    import uvicorn
    uvicorn.run('service.app:create_app',factory=True,host=args.host or settings['host'],port=args.port or settings['port'],workers=1,reload=False)


if __name__=='__main__':
    main()
