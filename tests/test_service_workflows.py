"""版本、上传、幂等、自动调参到真实独立进程训练的业务验收。"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import openpyxl
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from service.app import create_app
from joint_emaformer.data import REGIONS


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='lng_workflow_')
        self.workspace = Path(self.temporary.name)
        shared = yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
        self.payloads = {r:(ROOT/shared['data'][r]).read_bytes() for r in REGIONS}
        shared['data'] = {r:str(ROOT/shared['data'][r]) for r in REGIONS}
        shared_path = self.workspace/'data.yaml'
        shared_path.write_text(yaml.safe_dump(shared,allow_unicode=True),encoding='utf-8')
        settings = yaml.safe_load((ROOT/'service/config.yaml').read_text(encoding='utf-8'))
        search = yaml.safe_load((ROOT/'joint_emaformer/configs/search.yaml').read_text(encoding='utf-8'))
        small = deepcopy(search['initial_candidates'][1])
        small.update(seq_len=6,d_model=32,e_layers=1,head_layers=1,head_multiplier=1,max_epochs=1,patience=1,batch_size=64)
        search.update(search_trials=1,finalists=1,validation_folds=1,fold_months=3,initial_candidates=[small])
        search_path = self.workspace/'search.yaml'
        search_path.write_text(yaml.safe_dump(search,allow_unicode=True),encoding='utf-8')
        settings.update(data_config=str(shared_path),model_registry=str(ROOT/'service/models.yaml'),search_config=str(search_path))
        path = self.workspace/'service.yaml'
        path.write_text(yaml.safe_dump(settings,allow_unicode=True),encoding='utf-8')
        self.env = patch.dict(os.environ,LNG_SERVICE_CONFIG=str(path),LNG_SERVICE_STORAGE=str(self.workspace/'runtime'),LNG_API_KEY='',LNG_SERVICE_DEVICE='cpu')
        self.env.start()
        self.client = TestClient(create_app())
        self.client.__enter__()
        self.original = self.client.get('/datasets').json()['items'][0]['data_version']

    def tearDown(self):
        self.client.__exit__(None,None,None)
        self.env.stop()
        self.temporary.cleanup()

    def edited(self,append=False,missing=False):
        payloads = {}
        for region,payload in self.payloads.items():
            book = openpyxl.load_workbook(io.BytesIO(payload))
            worksheet = book.active
            if append:
                row = [cell.value for cell in worksheet[worksheet.max_row]]
                row[0] = '2026-06'
                worksheet.append(row)
            elif region=='shanghai':
                worksheet.cell(worksheet.max_row,25).value = None if missing else 4.25
            output = io.BytesIO()
            book.save(output)
            book.close()
            payloads[region] = output.getvalue()
        return payloads

    def upload(self,payloads=None,**kwargs):
        return self.client.post('/datasets',files={r+'_file':('custom_'+r+'.xlsx',p,'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            for r,p in (payloads or self.payloads).items()},**kwargs)

    def wait_task(self,task,timeout=65):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            reply = self.client.get(task['status_url'])
            self.assertEqual(reply.status_code,200,reply.text)
            value = reply.json()
            if value['status'] in ('succeeded','failed','cancelled','interrupted'):
                return value
            time.sleep(.15)
        self.fail('Task timed out: '+json.dumps(value,ensure_ascii=False))

    def test_upload_versions_defaults_and_immutable_history(self):
        duplicate = self.upload(data={'name':'ignored duplicate'})
        self.assertEqual(duplicate.status_code,200,duplicate.text)
        self.assertFalse(duplicate.json()['created'])
        old = self.client.post('/predictions',json={}).json()
        reply = self.upload(self.edited(append=True),data={'name':'new month','note':'append'})
        self.assertEqual(reply.status_code,201,reply.text)
        data = reply.json()['dataset']
        version = data['data_version']
        self.assertFalse(data['is_default'])
        self.assertEqual(data['latest_observed_month'],'2026-06')
        self.assertEqual(data['regions'][0]['filename'],'custom_shanghai.xlsx')
        self.assertEqual(self.client.post('/predictions',json={}).json()['model_status']['data_version'],self.original)
        selected = self.client.post('/predictions',json={'data_version':version}).json()
        self.assertEqual(selected['predictions'][0]['target_month'],'2026-07')
        activated = self.client.post('/datasets/'+version+'/activate',json={'expected_current_data_version':self.original})
        self.assertEqual(activated.status_code,200,activated.text)
        self.assertTrue(activated.json()['changed'])
        self.assertFalse(self.client.post('/datasets/'+version+'/activate',json={}).json()['changed'])
        self.assertEqual(self.client.post('/datasets/'+self.original+'/activate',json={'expected_current_data_version':self.original}).status_code,409)
        self.assertEqual(self.client.get('/predictions/'+old['prediction_id']).json(),old)
        with TestClient(create_app()) as restarted:
            self.assertEqual(restarted.post('/predictions',json={}).json()['model_status']['data_version'],version)

    def test_upload_validation_and_detailed_errors(self):
        before = len(self.client.get('/datasets').json()['items'])
        bad = self.upload(self.edited(missing=True))
        self.assertEqual(bad.status_code,422,bad.text)
        self.assertEqual(bad.json()['details'][0]['month'],'2026-05')
        self.assertEqual(bad.json()['details'][0]['region'],'shanghai')
        self.assertEqual(bad.json()['request_id'],bad.headers['X-Request-ID'])
        self.assertEqual(len(self.client.get('/datasets').json()['items']),before)
        self.assertEqual(self.client.post('/datasets',files={'shanghai_file':('x.csv',b'bad')}).status_code,422)
        files = {r+'_file':('x.xlsx',b'bad' if r=='shanghai' else p) for r,p in self.payloads.items()}
        self.assertEqual(self.client.post('/datasets',files=files).status_code,415)
        self.assertEqual(self.client.get('/models?page=1').status_code,422)
        self.assertEqual(self.client.get('/health').status_code,404)

    def test_idempotency_retries_and_conflict(self):
        headers = {'Idempotency-Key':'predict-one'}
        first = self.client.post('/predictions',json={},headers=headers)
        second = self.client.post('/predictions',json={},headers=headers)
        self.assertEqual(first.status_code,200,first.text)
        self.assertEqual(second.json(),first.json())
        self.assertEqual(self.client.post('/predictions',json={'profile':'all'},headers=headers).status_code,409)
        other = self.client.post('/predictions',json={}).json()
        self.assertNotEqual(first.json()['prediction_id'],other['prediction_id'])
        with TestClient(create_app()) as restarted:
            self.assertEqual(restarted.post('/predictions',json={},headers=headers).json(),first.json())
        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(lambda _:self.client.post('/predictions',json={},headers={'Idempotency-Key':'concurrent-one'}).json(),range(3)))
        self.assertTrue(all(value==responses[0] for value in responses))

    def test_upload_limits(self):
        settings = self.client.app.state.engine.settings['upload']
        original = deepcopy(settings)
        try:
            settings['file_bytes']=100
            self.assertEqual(self.upload().status_code,413)
            settings.update(original)
            settings['expanded_bytes']=100
            self.assertEqual(self.upload().status_code,413)
            settings.update(original)
            settings['total_bytes']=sum(map(len,self.payloads.values()))-1
            self.assertEqual(self.upload().status_code,413)
            settings.update(original)
            reply = self.client.post('/predictions',json={},headers={'Content-Length':str(original['total_bytes']+2*1024**2)})
            self.assertEqual(reply.status_code,413)
            self.assertEqual(reply.json()['request_id'],reply.headers['X-Request-ID'])
        finally:
            settings.update(original)

    def test_failed_hot_load_retains_default_and_reports_error(self):
        engine = self.client.app.state.engine
        original = engine.catalog.default('model_selected')[0]
        record = deepcopy(engine.catalog.get('model',original))
        version = 'selected-'+'0'*16
        record.update(model_version=version,checkpoint=str(self.workspace/'missing.pt'),source='configured')
        engine.catalog.put('model',version,record)
        reply = self.client.post('/models/'+version+'/activate',json={})
        self.assertEqual(reply.status_code,503,reply.text)
        self.assertEqual(engine.catalog.default('model_selected')[0],original)
        self.assertEqual(self.client.get('/models/'+version).json()['load_state'],'error')
        self.assertEqual(self.client.post('/predictions',json={}).status_code,200)
        data = self.upload(self.edited(append=True)).json()['dataset']['data_version']
        switched = self.client.post('/models/'+original+'/activate',json={'data_version':data})
        self.assertEqual(switched.status_code,200,switched.text)
        self.assertTrue(switched.json()['changed'])
        self.assertEqual(switched.json()['activated_at'],engine.catalog.default('data')[1])

    def test_running_cancel_and_shutdown_interrupt(self):
        for terminate in (False,True):
            reply = self.client.post('/trainings',json={'epochs':500})
            self.assertEqual(reply.status_code,202,reply.text)
            task = reply.json()
            deadline = time.monotonic()+10
            while time.monotonic()<deadline:
                current = self.client.app.state.engine.task(task['training_id'],'training',public=False)
                if current['status']=='running' and current['_internal'].get('pid'):
                    break
                time.sleep(.05)
            self.assertEqual(current['status'],'running')
            if terminate:
                self.client.__exit__(None,None,None)
                self.client = TestClient(create_app())
                self.client.__enter__()
                self.assertEqual(self.client.get(task['status_url']).json()['status'],'interrupted')
            else:
                cancelled = self.client.post(task['status_url']+'/cancel',json={})
                self.assertEqual(cancelled.status_code,202,cancelled.text)
                self.assertEqual(self.wait_task(task)['status'],'cancelled')

    def test_auto_dates_follow_uploaded_month_and_reject_invalid_plans(self):
        self.client.app.state.engine.jobs.close()
        version = self.upload(self.edited(append=True)).json()['dataset']['data_version']
        reply = self.client.post('/tuning-jobs',json={'data_version':version})
        self.assertEqual(reply.status_code,202,reply.text)
        split = reply.json()['resolved_split']
        self.assertEqual(split['training_cutoff'],'2024-07')
        self.assertEqual(split['validation_end'],'2026-03')
        self.assertEqual(split['test_start'],'2026-04')
        for body in ({'split_mode':'auto','train_end_month':'2024-06'},
                     {'search_space':{'unknown':[1]}},{'search_space':{'dropout':[2.]}},
                     {'search_space':{'seq_len':[True]}},{'search_trials':1,'finalists':2},
                     {'split_mode':'custom','train_end_month':'2019-12','validation_end_month':'2020-12','test_end_month':'2026-05'},
                     {'split_mode':'custom','train_end_month':'2026-13','validation_end_month':'2026-02'}):
            response = self.client.post('/tuning-jobs',json=body)
            self.assertEqual(response.status_code,422,response.text)

    def test_queue_cancellation_and_frozen_config(self):
        self.client.app.state.engine.jobs.close()
        task = self.client.post('/trainings',json={'epochs':1}).json()
        self.assertEqual(task['status'],'queued')
        changed = self.upload(self.edited(append=True)).json()['dataset']['data_version']
        self.client.post('/datasets/'+changed+'/activate',json={})
        saved = self.client.get(task['status_url']).json()
        self.assertEqual(saved['data_version'],self.original)
        self.assertEqual(saved['resolved_split']['training_cutoff'],'2026-02')
        cancelled = self.client.post(task['status_url']+'/cancel',json={})
        self.assertEqual(cancelled.status_code,202,cancelled.text)
        self.assertEqual(cancelled.json()['status'],'cancelled')
        repeated = self.client.post(task['status_url']+'/cancel',json={})
        self.assertEqual(repeated.status_code,200,repeated.text)
        self.assertEqual(repeated.json(),cancelled.json())
        self.assertEqual(self.client.get(task['status_url']+'/csv').status_code,409)

    def test_auto_tuning_training_full_history_and_hot_switch(self):
        response = self.client.post('/tuning-jobs',json={'profile':'all'})
        self.assertEqual(response.status_code,202,response.text)
        tuning = response.json()
        self.assertEqual(tuning['resolved_split']['training_cutoff'],'2024-06')
        self.assertEqual(tuning['resolved_split']['validation_end'],'2026-02')
        final = self.wait_task(tuning)
        self.assertEqual(final['status'],'succeeded',json.dumps(final,ensure_ascii=False))
        result = self.client.get(final['results_url']).json()
        self.assertTrue(result['result_available'])
        self.assertEqual(len(result['items'][0]['regions'][0]['by_horizon']),3)
        self.assertNotIn('"seed"',json.dumps(final))
        request = {'tuning_id':final['tuning_id'],'mode':'holdout'}
        reply = self.client.post('/trainings',json=request,headers={'Idempotency-Key':'training-one'})
        self.assertEqual(reply.status_code,202,reply.text)
        self.assertEqual(self.client.post('/trainings',json=request,headers={'Idempotency-Key':'training-one'}).json(),reply.json())
        training = self.wait_task(reply.json())
        self.assertEqual(training['status'],'succeeded',json.dumps(training,ensure_ascii=False))
        self.assertEqual(training['profile'],'all')
        self.assertEqual(training['data_version'],self.original)
        self.assertEqual(training['training_plan'],result['frozen_training_plan'])
        self.assertEqual(training['resolved_split']['training_cutoff'],'2026-02')
        self.assertEqual(len(self.client.get(training['result']['test_csv_url']).content.decode('utf-8-sig').splitlines()),10)
        for region in training['result']['evaluation']['registered_model_test']['regions']:
            self.assertEqual(region['origins'],1)
            self.assertEqual(len(region['by_horizon']),3)
        version = training['result']['model_version']
        record = self.client.get('/models/'+version).json()
        self.assertIsNotNone(record['trained_at'])
        self.assertEqual(record['load_state'],'not_loaded')
        predicted = self.client.post('/predictions',json={'model_version':version}).json()
        self.assertEqual(predicted['model_status']['profile'],'all')
        self.assertEqual(len(predicted['predictions']),9)
        self.assertTrue(all(not row['available'] for row in predicted['prediction_intervals']['items']))
        self.assertEqual(self.client.post('/predictions',json={'model_version':version,'profile':'selected'}).status_code,422)
        self.assertEqual(self.client.post('/models/'+version+'/activate',json={}).status_code,200)
        self.assertEqual(self.client.post('/predictions',json={'profile':'all'}).json()['model_status']['model_version'],version)
        full_response = self.client.post('/trainings',json={'tuning_id':final['tuning_id'],'mode':'full_history'})
        self.assertEqual(full_response.status_code,202,full_response.text)
        full = self.wait_task(full_response.json())
        self.assertEqual(full['status'],'succeeded',json.dumps(full,ensure_ascii=False))
        self.assertEqual(full['result']['model_training_cutoff'],'2026-05')
        self.assertFalse(full['result']['evaluation']['registered_model_test']['available'])
        self.assertEqual(self.client.get(full['status_url']+'/csv').status_code,409)
        self.assertEqual(self.client.post('/trainings',json=dict(request,epochs=1)).status_code,422)
        self.assertEqual(self.client.post('/trainings',json=dict(request,profile='selected')).status_code,422)
        with TestClient(create_app()) as restarted:
            self.assertEqual(restarted.get(training['status_url']).json()['status'],'succeeded')
            self.assertEqual(restarted.post('/predictions',json={'profile':'all'}).json()['model_status']['model_version'],version)


if __name__=='__main__':
    unittest.main()
