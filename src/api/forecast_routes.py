"""`/v1/forecast`: the endpoint AIAIC calls. `/v1/predict` is retired and answers 410 (src/api/routes.py; see
src/forecast/__init__.py for why it could not forecast)."""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.forecast.service import ForecastUnavailable, forecast, list_meta

router = APIRouter(prefix="/v1")


class HistoryPoint(BaseModel):
    date: date
    modal_price: float = Field(gt=0, le=1_000_000)


class ForecastRequest(BaseModel):
    commodity: str = Field(min_length=1, max_length=64, examples=["Onion"])
    state: str = Field(min_length=1, max_length=64, examples=["Maharashtra"])
    market: str = Field(min_length=1, max_length=96, examples=["Lasalgaon"])
    as_of: date = Field(examples=["2025-10-01"])
    horizon_days: int = Field(ge=1, le=90, examples=[7])
    replay: bool = Field(default=False, description="Allow historical examples whose target date is in the past")
    history: Optional[List[HistoryPoint]] = Field(default=None, max_length=400,
                                                  description="Optional: this mandi's recent prices, oldest first")


@router.post("/forecast")
def post_forecast(req: ForecastRequest) -> dict:
    try:
        out = forecast(req.model_dump(mode="json"), str(settings.models_dir))
    except ForecastUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    out["replay"] = req.replay
    return out


@router.get("/forecast/meta")
def get_forecast_meta() -> dict:
    try:
        return list_meta(str(settings.models_dir))
    except ForecastUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
