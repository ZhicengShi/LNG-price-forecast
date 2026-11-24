"""年度预算版接口的真实权重预测、持久化和异步训练验收。"""
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import openpyxl
import pandas as pd
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from service.app import create_app


BASE = '/annual-budget'
VERSION = '年度预算版'
TERMINAL = {'succeeded', 'failed', 'cancelled', 'interrupted'}


class AnnualServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='lng_annual_api_')
        cls.workspace = Path(cls.temporary.name)
        shared = yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
        cls.payloads = {region:(ROOT/name).read_bytes() for region,name in shared['data'].items()}
        shared['data'] = {region:str((ROOT/name).resolve()) for region,name in shared['data'].items()}
        shared_path = cls.workspace/'data.yaml'
        shared_path.write_text(yaml.safe_dump(shared, allow_unicode=True), encoding='utf-8')
        settings = yaml.safe_load((ROOT/'service/config.yaml').read_text(encoding='utf-8'))
        settings.update(data_config=str(shared_path), model_registry=str(ROOT/'service/models.yaml'),
                        search_config=str(ROOT/'CoRegionLng/configs/search.yaml'))
        cls.config_path = cls.workspace/'service.yaml'
        cls.config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding='utf-8')
        cls.environment = patch.dict(os.environ, LNG_SERVICE_CONFIG=str(cls.config_path),
            LNG_SERVICE_STORAGE=str(cls.workspace/'runtime'), LNG_SERVICE_DEVICE='cpu', LNG_API_KEY='')
        cls.environment.start()
        cls.client = TestClient(create_app())
        cls.client.__enter__()
        cls.engine = cls.client.app.state.engine
        cls.reference = pd.read_csv(ROOT/'年度预算版/reference/年度预算逐年预测.csv')

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.environment.stop()
        cls.temporary.cleanup()

    def predict(self, body=None, **kwargs):
        response = self.client.post(BASE+'/predictions', json=body or {}, **kwargs)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def upload(self, payloads):
        response = self.client.post('/datasets', files={
            region+'_file':(region+'.xlsx', content, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            for region,content in payloads.items()})
        self.assertIn(response.status_code, (200, 201), response.text)
        return response.json()['dataset']['data_version']

    def wait_task(self, task, timeout=120):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            response = self.client.get(task['status_url'])
            self.assertEqual(response.status_code, 200, response.text)
            value = response.json()
            progress = value['progress']
            self.assertGreaterEqual(progress['completed_units'], 0, value)
            self.assertLessEqual(progress['completed_units'], progress['total_units'], value)
            self.assertGreaterEqual(progress['phase_percent'], 0, value)
            self.assertLessEqual(progress['phase_percent'], 100, value)
            if value['status'] in TERMINAL:
                return value
            time.sleep(.2)
        self.fail('年度训练任务超时 '+json.dumps(value, ensure_ascii=False))

    def test_default_thirty_results_match_reference_and_calendar(self):
        value = self.predict()
        self.assertEqual(value['version'], VERSION)
        self.assertEqual(value['budget_year'], 2026)
        self.assertEqual(value['cutoff_month'], '2025-08')
        self.assertEqual((value['target_start'], value['target_end']), ('2026-01','2026-12'))
        self.assertEqual(value['weight_source'], 'bundled')
        self.assertIsNone(value['training_id'])
        self.assertEqual(len(value['predictions']), 30)
        keys = ['model','profile','region']
        expected = self.reference[self.reference.budget_year.eq(2026)].set_index(keys).predicted.sort_index()
        actual = pd.DataFrame(value['predictions']).set_index(keys).prediction.sort_index()
        self.assertEqual(actual.index.tolist(), expected.index.tolist())
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
        self.assertTrue(all(row['unit']=='元/立方米' and row['n_train']==107 for row in value['predictions']))
        self.assertEqual(len(value['weights']), 26)

    def test_filters_and_saved_results_csv_list_survive_restart(self):
        value = self.predict({'budget_year':2025, 'model':'CoRegionLng', 'profile':'selected'})
        self.assertEqual(len(value['predictions']), 3)
        self.assertEqual(value['cutoff_month'], '2024-08')
        self.assertEqual({r['model'] for r in value['predictions']}, {'CoRegionLng'})
        self.assertEqual({r['profile'] for r in value['predictions']}, {'selected'})
        saved = self.client.get(value['detail_url'])
        self.assertEqual(saved.json(), value)
        download = self.client.get(value['csv_url'])
        self.assertEqual(download.status_code, 200, download.text)
        rows = list(csv.DictReader(io.StringIO(download.content.decode('utf-8-sig'))))
        self.assertEqual(len(rows), 3)
        self.assertEqual(sorted(float(r['prediction']) for r in rows), sorted(r['prediction'] for r in value['predictions']))
        listing = self.client.get(BASE+'/predictions', params={'budget_year':2025})
        self.assertEqual(listing.status_code, 200, listing.text)
        self.assertIn(value['prediction_id'], {r['prediction_id'] for r in listing.json()['items']})
        with TestClient(create_app()) as restarted:
            self.assertEqual(restarted.get(value['detail_url']).json(), value)

    def test_models_schema_and_strict_request_validation(self):
        response = self.client.get(BASE+'/models', params={'budget_year':2026,'model':'CoRegionLng','profile':'selected'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['version'], VERSION)
        self.assertEqual([(r['model'], r['model_name']) for r in response.json()['models']],
                         [('CoRegionLng', 'CoRegionLng')])
        schema = self.client.get('/openapi.json').json()
        for path in (BASE+'/models', BASE+'/predictions', BASE+'/trainings', BASE+'/trainings/{training_id}/cancel'):
            self.assertIn(path, schema['paths'])
        for body in ({'model':'unknown'}, {'profile':'unknown'}, {'budget_year':True}, {'budget_year':'2026'},
                     {'data_version':'../../config.yaml'}, {'seed':1}):
            reply = self.client.post(BASE+'/predictions', json=body)
            self.assertEqual(reply.status_code, 422, reply.text)
        self.assertEqual(self.client.get(BASE+'/models?unknown=1').status_code, 422)
        unknown = self.client.get(BASE+'/predictions/apred_'+'0'*32)
        self.assertEqual(unknown.status_code, 404, unknown.text)
        missing = self.client.post(BASE+'/predictions', json={'budget_year':2023})
        self.assertEqual(missing.status_code, 409, missing.text)
        self.assertEqual(missing.json()['error'], 'ANNUAL_WEIGHTS_UNAVAILABLE')

    def test_short_history_can_reach_inference_but_cannot_train(self):
        # The 2016 budget cutoff has eight observed months: enough for inference, but no mature label.
        body = {'budget_year': 2016, 'model': 'CoRegionLng', 'profile': 'selected'}
        prediction = self.client.post(BASE+'/predictions', json=body)
        self.assertEqual(prediction.status_code, 409, prediction.text)
        self.assertEqual(prediction.json()['error'], 'ANNUAL_WEIGHTS_UNAVAILABLE')
        training = self.client.post(BASE+'/trainings', json=body)
        self.assertEqual(training.status_code, 422, training.text)
        self.assertEqual(training.json()['error'], 'INVALID_DATA')

    def test_future_budget_requires_august_data(self):
        for endpoint in ('/predictions','/trainings'):
            response = self.client.post(BASE+endpoint, json={'budget_year':2027,'model':'lstm','profile':'selected'})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(response.json()['error'], 'INVALID_DATA')
        tasks = self.client.get(BASE+'/trainings', params={'budget_year':2027}).json()['items']
        self.assertEqual(tasks, [])

    def test_changes_after_august_do_not_enter_annual_inputs(self):
        before = self.predict({'model':'CoRegionLng', 'profile':'selected'})
        payloads = dict(self.payloads)
        book = openpyxl.load_workbook(io.BytesIO(payloads['shanghai']))
        sheet = book.active
        self.assertEqual(sheet.cell(sheet.max_row,1).value, '2026-05')
        sheet.cell(sheet.max_row,25).value = 99.0
        stream = io.BytesIO()
        book.save(stream)
        book.close()
        payloads['shanghai'] = stream.getvalue()
        version = self.upload(payloads)
        after = self.predict({'model':'CoRegionLng', 'profile':'selected','data_version':version})
        self.assertNotEqual(after['data_version'],before['data_version'])
        self.assertEqual(after['predictions'],before['predictions'])

    def test_idempotency_and_existing_short_term_forecast(self):
        body = {'model':'lstm','profile':'selected'}
        headers = {'Idempotency-Key':'annual-predict-repeat'}
        first = self.predict(body, headers=headers)
        second = self.predict(body, headers=headers)
        self.assertEqual(first, second)
        conflict = self.client.post(BASE+'/predictions',json={'model':'mofo','profile':'selected'},headers=headers)
        self.assertEqual(conflict.status_code, 409, conflict.text)
        short = self.client.post('/predictions', json={'profile':'selected'})
        self.assertEqual(short.status_code, 200, short.text)
        self.assertEqual(len(short.json()['predictions']), 9)
        self.assertEqual({r['target_month'] for r in short.json()['predictions']}, {'2026-06','2026-07','2026-08'})

    def test_protected_routes_reuse_api_key(self):
        with patch.dict(os.environ, LNG_API_KEY='annual-test-key'):
            with TestClient(create_app()) as protected:
                self.assertEqual(protected.get(BASE+'/models').status_code, 401)
                self.assertEqual(protected.post(BASE+'/predictions', json={}).status_code, 401)
                response = protected.get(BASE+'/models', headers={'X-API-Key':'annual-test-key'})
                self.assertEqual(response.status_code, 200, response.text)

    def test_queued_training_can_cancel_without_publishing_weights(self):
        self.engine.jobs.close()
        try:
            response = self.client.post(BASE+'/trainings', json={'model':'lstm','profile':'selected','name':'排队取消检查'})
            self.assertEqual(response.status_code, 202, response.text)
            task = response.json()
            self.assertEqual(task['status'], 'queued')
            pending = self.client.post(BASE+'/predictions', json={'training_id':task['training_id'],'model':'lstm','profile':'selected'})
            self.assertEqual(pending.status_code, 409, pending.text)
            cancel = self.client.post(task['status_url']+'/cancel', json={})
            self.assertEqual(cancel.status_code, 202, cancel.text)
            self.assertEqual(cancel.json()['status'], 'cancelled')
            again = self.client.post(task['status_url']+'/cancel', json={})
            self.assertEqual(again.status_code, 200, again.text)
            self.assertEqual(self.client.get(task['status_url']).json()['status'], 'cancelled')
        finally:
            self.engine.start_jobs()

    def test_real_training_publishes_prediction_and_reusable_weights(self):
        headers = {'Idempotency-Key':'annual-training-repeat'}
        request = {'budget_year':2026,'model':'lstm','profile':'selected','name':'年度训练验收'}
        response = self.client.post(BASE+'/trainings', json=request, headers=headers)
        self.assertEqual(response.status_code, 202, response.text)
        task = response.json()
        repeat = self.client.post(BASE+'/trainings', json=request, headers=headers)
        self.assertEqual(repeat.status_code, 202, repeat.text)
        self.assertEqual(repeat.json()['training_id'], task['training_id'])
        final = self.wait_task(task)
        self.assertEqual(final['status'], 'succeeded', json.dumps(final,ensure_ascii=False))
        self.assertEqual(final['version'], VERSION)
        self.assertEqual(final['kind'], 'annual_training')
        self.assertEqual(final['result']['weights_count'], 3)
        self.assertEqual(final['progress']['completed_units'], final['progress']['total_units'])
        self.assertEqual(final['progress']['phase_percent'], 100)
        self.assertEqual(final['progress']['completed_configurations'], 3)
        saved = self.client.get(final['result']['predictions_url'])
        self.assertEqual(saved.status_code,200,saved.text)
        self.assertEqual(saved.json()['weight_source'],'training')
        self.assertEqual(saved.json()['training_id'],task['training_id'])
        self.assertEqual(len(saved.json()['predictions']),3)
        loaded = self.predict({'budget_year':2026,'model':'lstm','profile':'selected','training_id':task['training_id']})
        self.assertEqual(loaded['predictions'],saved.json()['predictions'])
        expanded = self.client.post(BASE+'/predictions',json={'training_id':task['training_id'],'model':'mofo','profile':'selected'})
        self.assertEqual(expanded.status_code,422,expanded.text)
        with TestClient(create_app()) as restarted:
            restored = restarted.get(task['status_url'])
            self.assertEqual(restored.status_code,200,restored.text)
            self.assertEqual(restored.json()['status'],'succeeded')
            self.assertEqual(restarted.get(final['result']['predictions_url']).json(),saved.json())


if __name__=='__main__':
    unittest.main()
