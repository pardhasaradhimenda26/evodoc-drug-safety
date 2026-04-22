"""
EvoDoc Clinical Drug Safety Engine — Pydantic Models
All request/response schemas with full validation.
"""

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# ─────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────

class PatientHistory(BaseModel):
    current_medications: List[str] = Field(default_factory=list, description="Medicines already being taken")
    known_allergies: List[str] = Field(default_factory=list, description="Known drug/substance allergies")
    conditions: List[str] = Field(default_factory=list, description="Active medical conditions")
    age: Optional[int] = Field(None, ge=0, le=130, description="Patient age in years")
    weight_kg: Optional[float] = Field(None, gt=0, le=500, description="Patient weight in kilograms")
    renal_function: Optional[Literal["normal", "mild_impairment", "moderate_impairment", "severe_impairment", "esrd"]] = Field(
        None, description="Kidney function status"
    )
    hepatic_function: Optional[Literal["normal", "mild_impairment", "moderate_impairment", "severe_impairment"]] = Field(
        None, description="Liver function status"
    )
    pregnancy_status: Optional[Literal["not_pregnant", "first_trimester", "second_trimester", "third_trimester", "breastfeeding"]] = Field(
        None, description="Pregnancy/lactation status"
    )

    @field_validator("current_medications", "known_allergies", "conditions", mode="before")
    @classmethod
    def strip_and_deduplicate(cls, v):
        if isinstance(v, list):
            cleaned = list({item.strip() for item in v if isinstance(item, str) and item.strip()})
            return cleaned
        return v

    @field_validator("age", mode="before")
    @classmethod
    def validate_age(cls, v):
        if v is not None and v < 0:
            raise ValueError("Age cannot be negative")
        return v


class DrugSafetyRequest(BaseModel):
    proposed_medicines: List[str] = Field(..., min_length=1, description="New drugs to be prescribed")
    patient_history: PatientHistory = Field(default_factory=PatientHistory)
    request_id: Optional[str] = Field(None, description="Optional trace ID for audit logging")

    @field_validator("proposed_medicines", mode="before")
    @classmethod
    def validate_medicines(cls, v):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("At least one medicine must be provided")
        cleaned = []
        seen = set()
        for med in v:
            if not isinstance(med, str) or not med.strip():
                continue
            name = med.strip()
            # Normalize: capitalize first letter of each word
            normalized = " ".join(word.capitalize() for word in name.split())
            if normalized.lower() not in seen:
                seen.add(normalized.lower())
                cleaned.append(normalized)
        if not cleaned:
            raise ValueError("No valid medicine names provided")
        return cleaned


# ─────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────

class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: Literal["high", "medium", "low"]
    mechanism: str
    clinical_recommendation: str
    source_confidence: Literal["high", "medium", "low", "fallback"]
    interaction_type: Optional[str] = Field(None, description="e.g., pharmacokinetic, pharmacodynamic")
    evidence_level: Optional[Literal["A", "B", "C", "D"]] = Field(
        None, description="Evidence level: A=RCT, B=cohort, C=case report, D=theoretical"
    )


class AllergyAlert(BaseModel):
    medicine: str
    allergen: str
    reason: str
    severity: Literal["critical", "high", "medium", "low"]
    cross_reactivity: bool = Field(False, description="True if this is a cross-reactivity flag, not exact match")
    cross_reactivity_rate: Optional[str] = Field(None, description="e.g., '10-15%' estimated cross-reactivity")


class ContraindicationAlert(BaseModel):
    medicine: str
    condition: str
    severity: Literal["contraindicated", "use_with_caution", "monitor_closely"]
    reason: str
    alternative_suggestion: Optional[str] = None


class RiskScoreBreakdown(BaseModel):
    base_score: float = Field(description="Score from drug-drug interactions")
    allergy_score: float = Field(description="Score from allergy alerts")
    condition_score: float = Field(description="Score from contraindications")
    polypharmacy_penalty: float = Field(description="Penalty for high number of concurrent drugs")
    age_modifier: float = Field(description="Age-based risk modifier")
    renal_modifier: float = Field(description="Renal impairment modifier")
    final_score: float = Field(ge=0, le=100)
    interpretation: str


class DrugSafetyResponse(BaseModel):
    interactions: List[DrugInteraction] = Field(default_factory=list)
    allergy_alerts: List[AllergyAlert] = Field(default_factory=list)
    contraindication_alerts: List[ContraindicationAlert] = Field(default_factory=list)
    safe_to_prescribe: bool
    overall_risk_level: Literal["low", "medium", "high", "critical"]
    requires_doctor_review: bool
    patient_risk_score: Optional[float] = Field(None, ge=0, le=100, description="Composite risk score 0-100")
    risk_score_breakdown: Optional[RiskScoreBreakdown] = None
    source: Literal["llm", "fallback", "hybrid"]
    cache_hit: bool
    processing_time_ms: int
    validated_medicines: List[str] = Field(default_factory=list, description="Medicines after normalization")
    warnings: List[str] = Field(default_factory=list, description="Non-critical notices e.g. unrecognized drug name")
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    llm_available: bool
    cache_backend: str
    cache_size: int
    model_name: str
    version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
