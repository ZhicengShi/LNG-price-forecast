"""年度预算版 HTTP 请求与响应。"""
from typing import Literal

from pydantic import Field

from .schemas import PublicModel, Region, Profile, Issue

AnnualModel = Literal['all', 'lstm', 'emaformer', 'mofo', 'emmoformer', 'joint_emaformer']
AnnualProfile = Literal['all', 'selected', 'both']
AnnualVersion = Literal['年度预算版']
AnnualStatus = Literal['queued', 'running', 'cancel_requested', 'succeeded', 'failed', 'cancelled', 'interrupted']


class AnnualPredictRequest(PublicModel):
    budget_year: int = Field(default=2026, ge=2000, le=9998, strict=True)
    model: AnnualModel | None = None
    profile: AnnualProfile | None = None
    data_version: str | None = Field(default=None, pattern=r'^data_(?:[a-f0-9]{24}|[a-f0-9]{32})$')
    training_id: str | None = Field(default=None, pattern=r'^atrain_[a-f0-9]{32}$')


class AnnualTrainingRequest(PublicModel):
    budget_year: int = Field(default=2026, ge=2000, le=9998, strict=True)
    model: AnnualModel = 'all'
    profile: AnnualProfile = 'both'
    data_version: str | None = Field(default=None, pattern=r'^data_(?:[a-f0-9]{24}|[a-f0-9]{32})$')
    name: str | None = Field(default=None, max_length=100)


class AnnualPrice(PublicModel):
    model: str
    profile: Profile
    region: Region
    region_name: str
    prediction: float
    unit: str
    n_train: int


class AnnualWeight(PublicModel):
    model: str
    profile: Profile
    regions: list[Region]
    budget_year: int
    cutoff_month: str
    source: Literal['bundled', 'training']


class AnnualPredictionResponse(PublicModel):
    version: AnnualVersion
    prediction_id: str
    generated_at: str
    budget_year: int
    cutoff_month: str
    target_start: str
    target_end: str
    data_version: str
    data_latest_observed_month: str
    device: str
    weight_source: Literal['bundled', 'training']
    training_id: str | None
    training_data_version: str | None
    weights: list[AnnualWeight]
    predictions: list[AnnualPrice]
    detail_url: str
    csv_url: str


class AnnualPredictionList(PublicModel):
    version: AnnualVersion
    items: list[AnnualPredictionResponse]


class AnnualModelRecord(PublicModel):
    model: str
    model_name: str
    profile: Profile
    regions: list[Region]
    input_months: int
    output: str
    input_features: dict[str, list[str]]
    packed_channels: int | None
    training_epochs: int
    available: bool
    budget_year: int
    cutoff_month: str
    weight_source: Literal['bundled']


class AnnualModelList(PublicModel):
    version: AnnualVersion
    budget_year: int
    cutoff_month: str
    target_start: str
    target_end: str
    models: list[AnnualModelRecord]


class AnnualProgress(PublicModel):
    unit: Literal['epoch']
    completed_units: int
    total_units: int
    phase_percent: float
    current_epoch: int | None
    max_epochs: int | None
    completed_configurations: int
    total_configurations: int


class AnnualTaskError(PublicModel):
    code: str
    message: str
    retryable: bool


class AnnualTrainingResult(PublicModel):
    configurations: int
    weights_count: int
    trained_at: str
    prediction_id: str
    predictions_url: str
    csv_url: str


class AnnualTrainingTask(PublicModel):
    version: AnnualVersion
    kind: Literal['annual_training']
    training_id: str
    name: str
    budget_year: int
    cutoff_month: str
    target_start: str
    target_end: str
    model: AnnualModel
    profile: AnnualProfile
    data_version: str
    device: str
    status: AnnualStatus
    stage: str
    submitted_at: str
    started_at: str | None
    finished_at: str | None
    elapsed_seconds: float
    queue_position: int | None
    progress: AnnualProgress
    warnings: list[Issue]
    error: AnnualTaskError | None
    request: dict
    result: AnnualTrainingResult | None
    status_url: str


class AnnualTrainingList(PublicModel):
    version: AnnualVersion
    items: list[AnnualTrainingTask]


class AnnualCancelTraining(PublicModel):
    version: AnnualVersion
    training_id: str
    status: AnnualStatus
    requested_at: str
    message: str
