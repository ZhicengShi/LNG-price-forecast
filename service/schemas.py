"""公开请求与四部分预测响应；公开字段不包含训练种子。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
import pandas as pd

Profile = Literal['all', 'selected']
Region = Literal['shanghai', 'zhejiang', 'jiangsu']


class PublicModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class PredictRequest(PublicModel):
    model_config = ConfigDict(json_schema_extra={'examples':[{'profile':'selected'}]})
    profile: Profile | None = None
    model_version: str | None = Field(default=None, max_length=80)
    data_version: str | None = Field(default=None, pattern=r'^data_[a-f0-9]{24}$')


class ActivateData(PublicModel):
    model_config = ConfigDict(json_schema_extra={'examples':[{}]})
    expected_current_data_version: str | None = None


class ActivateModel(PublicModel):
    model_config = ConfigDict(json_schema_extra={'examples':[{}]})
    data_version: str | None = None
    expected_current_model_version: str | None = None
    expected_current_data_version: str | None = None


class TrainingRequest(PublicModel):
    model_config = ConfigDict(json_schema_extra={'examples':[{'mode':'holdout'}]})
    profile: Profile | None = None
    data_version: str | None = Field(default=None, pattern=r'^data_[a-f0-9]{24}$')
    name: str | None = Field(default=None, max_length=100)
    mode: Literal['holdout','full_history','custom'] = 'holdout'
    holdout_months: int | None = Field(default=None, ge=3, le=2400, strict=True)
    training_cutoff: str | None = None
    test_end_month: str | None = None
    tuning_id: str | None = Field(default=None, pattern=r'^tune_[a-f0-9]{32}$')
    epochs: int | None = Field(default=None, ge=1, le=500, strict=True)
    learning_rate: float | None = Field(default=None, gt=0, le=.01)

    @field_validator('training_cutoff','test_end_month')
    @classmethod
    def month(cls,value):
        if value is not None:
            import re
            if not re.fullmatch(r'\d{4}-\d{2}',value):
                raise ValueError('月份须使用YYYY-MM')
            pd.Period(value,freq='M')
        return value


class TuningRequest(PublicModel):
    model_config = ConfigDict(json_schema_extra={'examples':[{'profile':'selected'}]})
    profile: Profile = 'selected'
    data_version: str | None = Field(default=None, pattern=r'^data_[a-f0-9]{24}$')
    name: str | None = Field(default=None, max_length=100)
    preset: Literal['quick','standard','thorough'] = 'standard'
    split_mode: Literal['auto','custom'] = 'auto'
    holdout_months: int | None = Field(default=None, ge=3, le=2400, strict=True)
    validation_months: int | None = Field(default=None, ge=3, le=2400, strict=True)
    train_end_month: str | None = None
    validation_end_month: str | None = None
    test_end_month: str | None = None
    search_trials: int | None = Field(default=None, ge=1, le=256, strict=True)
    finalists: int | None = Field(default=None, ge=1, le=16, strict=True)
    validation_folds: int | None = Field(default=None, ge=1, le=5, strict=True)
    fold_months: int | None = Field(default=None, ge=3, le=60, strict=True)
    include_initial_candidates: bool = True
    search_space: dict | None = None
    max_duration_minutes: int | None = Field(default=None, ge=1, le=10080, strict=True)

    _months = field_validator('train_end_month','validation_end_month','test_end_month')(TrainingRequest.month.__func__)


class EmptyRequest(PublicModel):
    pass


class Price(PublicModel):
    region: Region
    region_name: str
    target_month: str
    horizon: int = Field(ge=1, le=3)
    prediction: float
    unit: str


class PriceInterval(PublicModel):
    region: Region
    target_month: str
    horizon: int
    available: bool
    lower: float | None
    upper: float | None
    calibration_count: int
    latest_calibration_target_month: str | None
    reason: str | None
    historical_coverage_percent: float | None
    historical_coverage_count: int


class Intervals(PublicModel):
    method: str
    confidence: float
    calibration_basis: str
    calibration_version: str | None
    min_residuals: int
    residual_window: int
    interpretation: str
    items: list[PriceInterval]


class StepMetric(PublicModel):
    horizon: int
    observations: int
    mape_percent: float
    accuracy_score_percent: float


class RegionMetric(PublicModel):
    region: Region
    region_name: str
    origins: int
    by_horizon: list[StepMetric]
    average_mape_percent: float
    average_accuracy_score_percent: float


class Evidence(PublicModel):
    available: bool
    reason: str | None
    basis: str
    version: str | None
    first_forecast_month: str | None
    last_forecast_month: str | None
    last_target_month: str | None
    regions: list[RegionMetric]


class Evaluation(PublicModel):
    recent_backtest: Evidence
    registered_model_test: Evidence
    interpretation: str


class ModelStatus(PublicModel):
    model_name: str
    service_version: str
    model_version: str
    checkpoint_sha256: str
    profile: Profile
    device: str
    registered_at: str
    loaded_at: str | None
    trained_at: str | None
    training_time_status: str
    model_training_cutoff: str
    parameter_selection_cutoff: str
    latest_observed_month: str
    data_imported_at: str
    data_version: str
    data_sha256: dict[str, str]
    historical_reference_matches: bool
    seq_len: int
    pred_len: int
    input_features: dict[str, list[str]]


class PredictionResponse(PublicModel):
    prediction_id: str
    generated_at: str
    predictions: list[Price]
    prediction_intervals: Intervals
    evaluation: Evaluation
    model_status: ModelStatus


class Issue(PublicModel):
    code: str
    region: Region | None
    month: str | None
    field: str | None
    message: str
    suggestion: str | None


class ErrorResponse(PublicModel):
    error: str
    message: str
    request_id: str
    details: list[Issue]


class ModelCompatibility(PublicModel):
    model_version: str
    profile: Profile
    can_predict: bool
    reasons: list[Issue]


class DataRegion(PublicModel):
    region: Region
    region_name: str
    filename: str
    months: int
    first_month: str
    latest_observed_month: str
    missing_counts: dict[str,int]
    all_empty_columns: list[str]


class DataVersion(PublicModel):
    data_version: str
    name: str
    note: str
    created_at: str
    source: Literal['configured','upload']
    is_default: bool
    status: Literal['valid']
    first_month: str
    latest_observed_month: str
    months: int
    regions: list[DataRegion]
    warnings: list[Issue]
    compatible_models: list[ModelCompatibility]


class DataList(PublicModel):
    items: list[DataVersion]


class UploadResponse(PublicModel):
    created: bool
    dataset: DataVersion


class TrainingPlan(PublicModel):
    epochs: int
    lr_schedule: list[float]


class ModelRecord(ModelStatus):
    display_name: str
    source: Literal['bundled','configured','training']
    training_id: str | None
    tuning_id: str | None
    training_data_version: str | None
    is_default: bool
    load_state: Literal['not_loaded','loading','ready','error']
    parameters: dict
    training_plan: TrainingPlan
    compatibility: ModelCompatibility
    evaluation: Evaluation


class ModelList(PublicModel):
    models: list[ModelRecord]


class DataActivation(PublicModel):
    data_version: str
    previous_data_version: str | None
    changed: bool
    activated_at: str
    compatible_models: list[ModelCompatibility]


class ModelActivation(PublicModel):
    model_version: str
    profile: Profile
    previous_model_version: str | None
    data_version: str
    previous_data_version: str | None
    changed: bool
    activated_at: str
    warnings: list[Issue]


class DateSplit(PublicModel):
    first_month: str
    training_cutoff: str
    validation_start: str | None
    validation_end: str | None
    test_start: str | None
    test_end: str | None
    latest_observed_month: str
    pred_len: int


class TaskProgress(PublicModel):
    unit: Literal['epoch','fit','origin','step']
    completed_units: int
    total_units: int | None
    phase_percent: float | None
    current_epoch: int | None
    max_epochs: int | None


class TaskError(PublicModel):
    code: str
    message: str
    retryable: bool


TaskStatus = Literal['queued','running','cancel_requested','succeeded','failed','cancelled','interrupted']


class Task(PublicModel):
    kind: Literal['training','tuning']
    name: str
    profile: Profile
    model_name: str
    data_version: str
    device: str
    status: TaskStatus
    stage: str
    submitted_at: str
    started_at: str | None
    finished_at: str | None
    elapsed_seconds: float
    queue_position: int | None
    progress: TaskProgress
    warnings: list[Issue]
    error: TaskError | None
    status_url: str
    request: dict
    resolved_split: DateSplit


class TrainingResult(PublicModel):
    model_version: str
    display_name: str
    registered_at: str
    trained_at: str
    model_training_cutoff: str
    parameter_selection_cutoff: str
    epochs_completed: int
    evaluation: Evaluation
    test_csv_url: str | None
    model_url: str


class TrainingTask(Task):
    training_id: str
    parameters: dict
    training_plan: TrainingPlan
    tuning_id: str | None
    result: TrainingResult | None


class TrainingList(PublicModel):
    items: list[TrainingTask]


class ValidationFold(PublicModel):
    fold: int
    training_cutoff: str
    validation_start: str
    validation_end: str


class TuningPlan(PublicModel):
    search_trials: int
    finalists: int
    validation_folds: int
    fold_months: int
    replications_per_fold: int
    include_initial_candidates: bool
    initial_candidates_count: int
    search_space: dict
    max_duration_minutes: int | None
    planned_fits: int
    folds: list[ValidationFold]


class Candidate(PublicModel):
    trial: int
    rank: int | None
    phase: Literal['screening','confirmation']
    status: Literal['complete','running','failed','partial']
    parameters: dict
    score_mape_percent: float | None
    regions: list[RegionMetric]
    fits_completed: int
    fits_planned: int
    best_epoch: int | None
    elapsed_seconds: float
    error: TaskError | None


class TuningTask(Task):
    tuning_id: str
    resolved_search: TuningPlan
    best_so_far: Candidate | None
    result_available: bool
    results_url: str


class TuningList(PublicModel):
    items: list[TuningTask]


class TuningResults(PublicModel):
    tuning_id: str
    status: TaskStatus
    result_available: bool
    profile: Profile
    data_version: str
    selection_cutoff: str
    objective: str
    recommended_trial: int | None
    recommended_parameters: dict | None
    frozen_training_plan: TrainingPlan | None
    selection_rule: str
    phase: Literal['screening','confirmation']
    items: list[Candidate]


class CancelTraining(PublicModel):
    training_id: str
    status: TaskStatus
    requested_at: str
    message: str


class CancelTuning(PublicModel):
    tuning_id: str
    status: TaskStatus
    requested_at: str
    message: str


class PredictionSummary(PublicModel):
    prediction_id: str
    generated_at: str
    profile: Profile
    model_version: str
    data_version: str
    latest_observed_month: str
    first_target_month: str
    last_target_month: str
    regions: list[Region]
    prediction_count: int
    available_interval_count: int
    detail_url: str
    csv_url: str


class PredictionList(PublicModel):
    items: list[PredictionSummary]
