"""接口集成：数值、校准边界、配置数据校验、版本与恢复。"""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
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
from service.intervals import future_intervals
from service.errors import APIError
from joint_emaformer.predict import forecast_future
from joint_emaformer.data import REGIONS


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='lng_service_tests_')
        cls.workspace = Path(cls.temporary.name)
        cls.input_paths = {r: cls.workspace/(r+'.xlsx') for r in REGIONS}
        shared = yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
        for r in REGIONS:
            cls.input_paths[r].write_bytes((ROOT/shared['data'][r]).read_bytes())
        shared['data'] = {r: str(p) for r, p in cls.input_paths.items()}
        shared_path = cls.workspace/'data_config.yaml'
        shared_path.write_text(yaml.safe_dump(shared, allow_unicode=True), encoding='utf-8')
        settings = yaml.safe_load((ROOT/'service/config.yaml').read_text(encoding='utf-8'))
        settings.update(data_config=str(shared_path), model_registry=str(ROOT/'service/models.yaml'),
                        search_config=str(ROOT/'joint_emaformer/configs/search.yaml'))
        config_path = cls.workspace/'service.yaml'
        config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding='utf-8')
        cls.env = patch.dict(os.environ, {'LNG_SERVICE_STORAGE': str(cls.workspace/'runtime'),
            'LNG_SERVICE_CONFIG': str(config_path), 'LNG_API_KEY': '', 'LNG_SERVICE_DEVICE':'cpu'})
        cls.env.start()
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.client.__enter__()
        cls.engine = cls.app.state.engine
        cls.payloads = {r: cls.engine.reference_paths[r].read_bytes() for r in REGIONS}

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.env.stop()
        cls.temporary.cleanup()

    def tearDown(self):
        for r, payload in self.payloads.items():
            self.input_paths[r].write_bytes(payload)

    def write_inputs(self, payloads):
        for r, payload in payloads.items():
            self.input_paths[r].write_bytes(payload)

    def edited(self, region, edit):
        payloads = dict(self.payloads)
        book = openpyxl.load_workbook(io.BytesIO(payloads[region]))
        edit(book.active)
        stream = io.BytesIO()
        book.save(stream)
        book.close()
        payloads[region] = stream.getvalue()
        return payloads

    def test_both_profiles_match_direct_cpu_forecast_and_four_part_contract(self):
        metadata, paths = self.engine.store.data(self.engine.data_version)
        for profile in ('all', 'selected'):
            reply = self.client.post('/predictions', json={'profile': profile})
            self.assertEqual(reply.status_code, 200, reply.text)
            value = reply.json()
            self.assertEqual(set(value)-{'prediction_id', 'generated_at'},
                {'predictions', 'prediction_intervals', 'evaluation', 'model_status'})
            expected = forecast_future(self.engine.models[profile].model,
                self.engine.model_frame(self.engine.sources(paths), self.engine.models[profile]), self.engine.models[profile].saved)
            np.testing.assert_array_equal([r['prediction'] for r in value['predictions']], [r['prediction'] for r in expected])
            self.assertEqual(len(value['predictions']), 9)
            self.assertTrue(all(r['available'] for r in value['prediction_intervals']['items']))
            self.assertEqual(sorted({r['target_month'] for r in value['predictions']}), ['2026-06', '2026-07', '2026-08'])
            self.assertIsNone(value['model_status']['trained_at'])
            self.assertEqual(value['model_status']['training_time_status'], 'not_recorded')
            self.assertEqual(value['model_status']['model_training_cutoff'], '2024-07')
            self.assertNotIn('"seed"', json.dumps(value))
            saved = self.client.get('/predictions/'+value['prediction_id'])
            self.assertEqual(saved.json(), value)
            csv_result = self.client.get('/predictions/'+value['prediction_id']+'/csv')
            self.assertEqual(csv_result.status_code, 200)
            self.assertEqual(len(csv_result.content.decode('utf-8-sig').splitlines()), 10)

    def test_recent_backtest_metrics_match_reference_analysis(self):
        table = pd.read_csv(ROOT/'joint_emaformer/reference/analysis/metrics.csv')
        for profile in ('all', 'selected'):
            reply = self.client.post('/predictions', json={'profile': profile}).json()
            metric = reply['evaluation']['recent_backtest']
            self.assertTrue(metric['available'])
            self.assertEqual(metric['first_forecast_month'], '2024-08')
            self.assertEqual(metric['last_forecast_month'], '2026-03')
            for region in metric['regions']:
                expected = table[(table.iloc[:, 0] == 'after_selection') & (table.iloc[:, 1] == region['region_name'])
                                 & (table.iloc[:, 2] == profile)].iloc[0]
                self.assertAlmostEqual(region['average_mape_percent'], expected.iloc[-2], places=12)
                for step in region['by_horizon']:
                    self.assertEqual(step['accuracy_score_percent'], 100-step['mape_percent'])
                    self.assertEqual(step['observations'], 20)

    def test_versioned_data_endpoints_and_request_selection(self):
        schema = self.client.get('/openapi.json').json()
        self.assertIn('/datasets',schema['paths'])
        self.assertEqual(set(schema['components']['schemas']['PredictRequest']['properties']), {'profile', 'model_version','data_version'})
        self.assertEqual(self.client.get('/datasets').status_code,200)
        self.assertEqual(self.client.post('/datasets',json={}).status_code,422)
        self.assertEqual(self.client.get('/datasets/default').status_code,404)
        self.assertEqual(self.client.get('/datasets/ds_'+'0'*24).status_code,404)
        self.assertEqual(self.client.post('/predictions', json={'dataset_id': 'ds_'+'0'*24}).status_code, 422)
        response = self.client.post('/predictions', json={}).json()
        self.assertIn('data_version', response['model_status'])
        self.assertNotIn('dataset_id', response['model_status'])

    def test_bad_configured_units_duplicate_months_and_missing_labels_leave_no_version(self):
        before = {p.name for p in (self.engine.store.root/'data_versions').iterdir()}
        edits = [lambda w: setattr(w.cell(2, 25), 'value', '元/吨'),
                 lambda w: setattr(w.cell(4, 1), 'value', w.cell(3, 1).value),
                 lambda w: setattr(w.cell(w.max_row, 25), 'value', None)]
        for edit in edits:
            self.write_inputs(self.edited('shanghai', edit))
            with self.assertRaises((ValueError,APIError)):
                with TestClient(create_app()):
                    pass
        after = {p.name for p in (self.engine.store.root/'data_versions').iterdir()}
        self.assertEqual(before, after)

    def test_common_indicator_conflict_and_broken_file_are_rejected(self):
        bad = self.edited('zhejiang', lambda w: setattr(w.cell(3, 2), 'value', 99999))
        for payloads in (bad, dict(self.payloads, shanghai=b'not a workbook')):
            self.write_inputs(payloads)
            with self.assertRaises((ValueError,APIError)):
                with TestClient(create_app()):
                    pass

    def test_historical_revision_disables_old_intervals_and_metrics(self):
        changed = self.edited('shanghai', lambda w: setattr(w.cell(w.max_row-1, 25), 'value', 4.5))
        self.write_inputs(changed)
        with TestClient(create_app()) as restarted:
            reply = restarted.post('/predictions', json={'profile': 'selected',
                'data_version':restarted.app.state.engine.configured_data_version})
        self.assertEqual(reply.status_code, 200, reply.text)
        value = reply.json()
        self.assertFalse(value['model_status']['historical_reference_matches'])
        self.assertTrue(all(not i['available'] and i['lower'] is None and i['upper'] is None
                            for i in value['prediction_intervals']['items']))
        self.assertFalse(value['evaluation']['recent_backtest']['available'])
        self.assertFalse(value['evaluation']['registered_model_test']['available'])

    def test_append_month_preserves_history_and_advances_forecast_months(self):
        payloads = {}
        for r in REGIONS:
            def edit(ws):
                values = [c.value for c in ws[ws.max_row]]
                values[0] = '2026-06'
                ws.append(values)
            payloads[r] = self.edited(r, edit)[r]
        self.write_inputs(payloads)
        with TestClient(create_app()) as restarted:
            value = restarted.post('/predictions', json={'data_version':restarted.app.state.engine.configured_data_version}).json()
        self.assertNotEqual(value['model_status']['data_version'], self.engine.data_version)
        self.assertEqual(sorted({i['target_month'] for i in value['predictions']}), ['2026-07', '2026-08', '2026-09'])
        self.assertTrue(value['model_status']['historical_reference_matches'])
        self.assertTrue(all(i['available'] for i in value['prediction_intervals']['items']))

    def test_no_future_residuals_and_insufficient_calibration(self):
        entry = self.engine.models['selected']
        table = entry.backtest.copy()
        forecasts = [dict(region=r, horizon=h, target_month=str(pd.Period('2024-08',freq='M')+h-1), prediction=4.)
                     for r in REGIONS for h in (1,2,3)]
        origin = 115
        settings = self.engine.settings['interval']
        before = future_intervals(forecasts, table, origin, True, settings, entry.backtest_version)
        table.loc[table.origin_index+table.horizon > origin, 'actual'] = 1e9
        after = future_intervals(forecasts, table, origin, True, settings, entry.backtest_version)
        self.assertEqual(before, after)
        self.assertTrue(all(i['latest_calibration_target_month'] <= '2024-07' for i in before['items']))
        short = future_intervals(forecasts, entry.backtest.iloc[:9], origin, True, settings, entry.backtest_version)
        self.assertTrue(all(not i['available'] and i['lower'] is None for i in short['items']))

    def test_invalid_price_outputs_are_not_saved(self):
        sample = self.client.post('/predictions',json={}).json()['predictions']
        before = len(self.client.get('/predictions').json()['items'])
        for invalid in (-1.,float('nan')):
            forecast = [dict(row,prediction=invalid) for row in sample]
            with patch('service.engine.forecast_future',return_value=forecast):
                reply = self.client.post('/predictions',json={})
            self.assertEqual(reply.status_code,422,reply.text)
            self.assertEqual(reply.json()['error'],'INVALID_OUTPUT')
        self.assertEqual(len(self.client.get('/predictions').json()['items']),before)

    def test_bad_ids_profiles_versions_and_unknown_records(self):
        for body in ({'profile': 'bad'}, {'dataset_id': '../config.yaml'}, {'seed': 7}):
            self.assertEqual(self.client.post('/predictions', json=body).status_code, 422)
        self.assertEqual(self.client.post('/predictions',json={'model_version':'wrong'}).status_code,404)
        self.assertEqual(self.client.get('/predictions/pred_'+'0'*32).status_code, 404)
        self.assertEqual(self.client.get('/templates/shanghai').status_code, 200)

    def test_restart_keeps_registered_times_dataset_and_prediction(self):
        before = self.client.post('/predictions', json={}).json()
        with TestClient(create_app()) as restarted:
            self.assertEqual(restarted.get('/predictions/'+before['prediction_id']).json(), before)
            after = restarted.post('/predictions', json={}).json()
        for field in ('model_version', 'registered_at', 'data_imported_at', 'data_version'):
            self.assertEqual(before['model_status'][field], after['model_status'][field])
        self.assertEqual(before['predictions'], after['predictions'])

    def test_previously_saved_predictions_remain_readable(self):
        value = self.client.post('/predictions', json={}).json()
        legacy = deepcopy(value)
        legacy['model_status']['dataset_id'] = legacy['model_status'].pop('data_version')
        path = self.engine.store.root/'predictions'/value['prediction_id']/'response.json'
        path.write_text(json.dumps(legacy, ensure_ascii=False), encoding='utf-8')
        reply = self.client.get('/predictions/'+value['prediction_id'])
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(reply.json(), value)
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), legacy)

    def test_configured_api_key_is_required(self):
        with patch.dict(os.environ, {'LNG_API_KEY': 'local-test-key'}):
            with TestClient(create_app()) as protected:
                self.assertNotIn('/health', protected.get('/openapi.json').json()['paths'])
                self.assertEqual(protected.get('/health').status_code, 404)
                self.assertEqual(protected.get('/health', headers={'X-API-Key': 'local-test-key'}).status_code, 404)
                for path in ('/docs', '/redoc', '/openapi.json'):
                    self.assertEqual(protected.get(path).status_code, 200)
                self.assertEqual(protected.get('/models').status_code, 401)
                self.assertEqual(protected.get('/models', headers={'X-API-Key': 'local-test-key'}).status_code, 200)


if __name__ == '__main__':
    unittest.main()
