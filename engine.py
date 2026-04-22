"""
EvoDoc Clinical Drug Safety Engine — Core Engine
Handles LLM inference, fallback logic, allergy detection,
contraindication checking, and risk scoring.
"""

import json
import logging
import re
import time
from pathlib import Path
from itertools import combinations
from typing import Optional

import httpx
from models import (
    DrugSafetyRequest, DrugSafetyResponse,
    DrugInteraction, AllergyAlert, ContraindicationAlert,
    RiskScoreBreakdown,
)
from cache import get_cache

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
PROMPTS_DIR = Path(__file__).parent / "prompts"

with open(DATA_DIR / "fallback_interactions.json") as f:
    FALLBACK_DATA = json.load(f)

with open(DATA_DIR / "contraindication_rules.json") as f:
    CONTRAINDICATION_DATA = json.load(f)

with open(PROMPTS_DIR / "system_prompt.txt") as f:
    SYSTEM_PROMPT = f.read()

DRUG_CLASS_MAP: dict[str, list[str]] = FALLBACK_DATA["drug_class_map"]
CROSS_REACTIVITY = FALLBACK_DATA.get("cross_reactivity_rates", {})


# ─────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────

class OllamaClient:
    """Thin async wrapper around Ollama local API."""

    def __init__(self, base_url: str, model: str, timeout: float = 25.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    async def generate(self, prompt: str) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,   # Low temperature = more deterministic clinical output
                "top_p": 0.9,
                "num_predict": 2048,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(f"{self.base_url}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
                return data.get("message", {}).get("content", "")
        except httpx.TimeoutException:
            logger.warning("Ollama request timed out")
        except Exception as e:
            logger.error(f"Ollama error: {e}")
        return None


# ─────────────────────────────────────────────
# DRUG NAME NORMALIZATION
# ─────────────────────────────────────────────

def normalize_drug_name(name: str) -> str:
    """Lowercase, strip, remove dose info (e.g. 'Metformin 500mg' → 'metformin')."""
    name = name.lower().strip()
    # Remove dosage patterns
    name = re.sub(r"\d+\s*(mg|mcg|g|ml|iu|units?|%)", "", name)
    # Remove route info
    name = re.sub(r"\b(oral|iv|im|sc|topical|inhaled|sublingual)\b", "", name)
    return name.strip()


def get_drug_classes(drug_name: str) -> list[str]:
    """Return all drug classes this drug belongs to."""
    normalized = normalize_drug_name(drug_name)
    classes = []
    for cls, members in DRUG_CLASS_MAP.items():
        if normalized in [normalize_drug_name(m) for m in members]:
            classes.append(cls)
    return classes


def drugs_match(drug_a: str, drug_b: str) -> bool:
    """Check if two drug references are the same drug."""
    na, nb = normalize_drug_name(drug_a), normalize_drug_name(drug_b)
    if na == nb:
        return True
    # Check if they belong to the same class
    classes_a = get_drug_classes(drug_a)
    classes_b = get_drug_classes(drug_b)
    return bool(set(classes_a) & set(classes_b))


# ─────────────────────────────────────────────
# ALLERGY DETECTION
# ─────────────────────────────────────────────

CROSS_REACTIVITY_PAIRS = {
    # allergen_class → {proposed_class → rate}
    "penicillin": {"cephalosporin": "1-2%", "carbapenem": "<1%"},
    "cephalosporin": {"penicillin": "1-2%"},
    "sulfonamide": {"thiazide_diuretic": "<1% (different sulfonamide moiety)"},
    "nsaid": {"nsaid": "high — all NSAIDs cross-react via COX inhibition"},
    "fluoroquinolone": {"fluoroquinolone": "possible — class effect"},
}


def get_all_allergen_classes(allergen: str) -> list[str]:
    """
    Get drug classes for an allergen.
    Handles the case where the allergen name IS a class name
    (e.g., patient lists 'Penicillin' as allergen = allergic to the whole class).
    """
    allergen_normalized = normalize_drug_name(allergen)
    # Check if the allergen IS a class name directly
    if allergen_normalized in DRUG_CLASS_MAP:
        return [allergen_normalized]
    # Otherwise look it up as a drug
    return get_drug_classes(allergen)


def detect_allergies(proposed_medicines: list[str], known_allergies: list[str]) -> list[AllergyAlert]:
    alerts = []

    for med in proposed_medicines:
        med_normalized = normalize_drug_name(med)
        med_classes = get_drug_classes(med)

        for allergen in known_allergies:
            allergen_normalized = normalize_drug_name(allergen)
            allergen_classes = get_all_allergen_classes(allergen)

            # Exact name match
            if med_normalized == allergen_normalized:
                alerts.append(AllergyAlert(
                    medicine=med,
                    allergen=allergen,
                    reason=f"Exact match — {med} is listed as a known allergen",
                    severity="critical",
                    cross_reactivity=False,
                ))
                continue

            # If proposed drug belongs to a class that the allergen IS or belongs to → same class
            for med_cls in med_classes:
                if med_cls in allergen_classes:
                    alerts.append(AllergyAlert(
                        medicine=med,
                        allergen=allergen,
                        reason=f"Same drug class: {med} belongs to {med_cls} class — allergy reported to {allergen}",
                        severity="critical" if allergen_normalized == med_cls else "high",
                        cross_reactivity=allergen_normalized != med_cls,
                        cross_reactivity_rate="high — same class",
                    ))
                    break

            # Cross-reactivity between different classes
            for med_cls in med_classes:
                for allergen_cls in allergen_classes:
                    if med_cls in allergen_classes:
                        continue  # Already handled above
                    if allergen_cls in CROSS_REACTIVITY_PAIRS:
                        cross_map = CROSS_REACTIVITY_PAIRS[allergen_cls]
                        if med_cls in cross_map:
                            rate = cross_map[med_cls]
                            alerts.append(AllergyAlert(
                                medicine=med,
                                allergen=allergen,
                                reason=f"Cross-reactivity: {med} ({med_cls}) with reported allergy to {allergen} ({allergen_cls})",
                                severity="high",
                                cross_reactivity=True,
                                cross_reactivity_rate=rate,
                            ))
                            break

    # Deduplicate
    seen = set()
    deduped = []
    for a in alerts:
        key = (a.medicine.lower(), a.allergen.lower())
        if key not in seen:
            seen.add(key)
            deduped.append(a)
    return deduped


# ─────────────────────────────────────────────
# CONTRAINDICATION CHECKER (BONUS C)
# ─────────────────────────────────────────────

def check_contraindications(
    proposed_medicines: list[str],
    conditions: list[str],
    renal_function: Optional[str] = None,
    hepatic_function: Optional[str] = None,
    pregnancy_status: Optional[str] = None,
) -> list[ContraindicationAlert]:
    alerts = []

    # Build effective condition list
    effective_conditions = [c.lower().replace(" ", "_") for c in conditions]

    # Map renal/hepatic function to condition labels
    if renal_function in ("moderate_impairment", "severe_impairment", "esrd"):
        effective_conditions.extend(["renal_failure", "chronic_kidney_disease"])
    elif renal_function == "mild_impairment":
        effective_conditions.append("chronic_kidney_disease")

    if hepatic_function in ("moderate_impairment", "severe_impairment"):
        effective_conditions.append("hepatic_failure")
    elif hepatic_function == "mild_impairment":
        effective_conditions.append("active_liver_disease")

    if pregnancy_status in ("first_trimester", "second_trimester", "third_trimester", "breastfeeding"):
        effective_conditions.append("pregnancy")

    for rule in CONTRAINDICATION_DATA["contraindications"]:
        rule_drugs_normalized = [normalize_drug_name(d) for d in rule["drugs"]]
        rule_conditions = [c.lower().replace(" ", "_") for c in rule["conditions"]]

        for med in proposed_medicines:
            med_normalized = normalize_drug_name(med)
            med_classes = get_drug_classes(med)

            # Check if drug matches rule (by name or class)
            drug_matches = (
                med_normalized in rule_drugs_normalized or
                rule["drug_class"] in med_classes
            )

            if not drug_matches:
                continue

            # Check if any condition is present
            for cond in rule_conditions:
                if cond in effective_conditions:
                    alerts.append(ContraindicationAlert(
                        medicine=med,
                        condition=cond.replace("_", " ").title(),
                        severity=rule["severity"],
                        reason=rule["reason"],
                        alternative_suggestion=rule.get("alternative"),
                    ))
                    break  # One condition match per rule per drug is enough

    return alerts


# ─────────────────────────────────────────────
# FALLBACK INTERACTION ENGINE
# ─────────────────────────────────────────────

def _match_fallback(drug_a: str, drug_b: str, rule: dict) -> bool:
    """Check if a fallback rule applies to this drug pair."""
    rule_a = normalize_drug_name(rule["drug_a"])
    rule_b = normalize_drug_name(rule["drug_b"])
    na = normalize_drug_name(drug_a)
    nb = normalize_drug_name(drug_b)

    def matches_any(drug_name: str, rule_token: str) -> bool:
        if normalize_drug_name(drug_name) == rule_token:
            return True
        # Check if rule_token is a class
        if rule_token in DRUG_CLASS_MAP:
            drug_classes = get_drug_classes(drug_name)
            return rule_token in drug_classes
        # Check if drug belongs to a class that contains rule_token
        for cls, members in DRUG_CLASS_MAP.items():
            if rule_token in [normalize_drug_name(m) for m in members]:
                if cls in get_drug_classes(drug_name):
                    return True
        return False

    forward = matches_any(drug_a, rule_a) and matches_any(drug_b, rule_b)
    reverse = matches_any(drug_b, rule_a) and matches_any(drug_a, rule_b)
    return forward or reverse


def run_fallback_engine(
    proposed_medicines: list[str],
    current_medications: list[str],
) -> list[DrugInteraction]:
    """
    Rule-based fallback when LLM unavailable.
    Checks all pairs: proposed×proposed and proposed×current.
    """
    all_interactions = []
    all_drugs = proposed_medicines + current_medications

    # Generate all unique pairs
    pairs = list(combinations(proposed_medicines, 2))
    # Also check proposed vs current meds
    for pm in proposed_medicines:
        for cm in current_medications:
            pairs.append((pm, cm))

    for drug_a, drug_b in pairs:
        for rule in FALLBACK_DATA["interactions"]:
            if _match_fallback(drug_a, drug_b, rule):
                all_interactions.append(DrugInteraction(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    severity=rule["severity"],
                    mechanism=rule["mechanism"],
                    clinical_recommendation=rule["clinical_recommendation"],
                    source_confidence="fallback",
                    interaction_type=rule.get("interaction_type"),
                    evidence_level=rule.get("evidence_level"),
                ))
                break  # One rule match per pair

    return all_interactions


# ─────────────────────────────────────────────
# LLM RESPONSE PARSER & VALIDATOR
# ─────────────────────────────────────────────

VALID_SEVERITY = {"high", "medium", "low"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_EVIDENCE = {"A", "B", "C", "D"}
VALID_INTERACTION_TYPE = {"pharmacokinetic", "pharmacodynamic", "mixed"}


def _safe_str(val, default="Not specified") -> str:
    if isinstance(val, str) and val.strip():
        return val.strip()
    return default


def parse_llm_response(raw_text: str) -> Optional[dict]:
    """Parse and strictly validate LLM JSON output."""
    if not raw_text:
        return None

    # Extract JSON from response (handle any markdown fences)
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not json_match:
        logger.warning("No JSON found in LLM response")
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"LLM JSON parse error: {e}")
        return None

    # Validate and sanitize interactions
    validated_interactions = []
    for item in data.get("interactions", []):
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "")).lower()
        if severity not in VALID_SEVERITY:
            severity = "medium"  # Safe default

        confidence = str(item.get("source_confidence", "")).lower()
        if confidence not in VALID_CONFIDENCE:
            confidence = "low"

        itype = str(item.get("interaction_type", "")).lower()
        if itype not in VALID_INTERACTION_TYPE:
            itype = None

        evidence = str(item.get("evidence_level", "")).upper()
        if evidence not in VALID_EVIDENCE:
            evidence = None

        drug_a = _safe_str(item.get("drug_a"))
        drug_b = _safe_str(item.get("drug_b"))

        if drug_a == "Not specified" or drug_b == "Not specified":
            continue

        validated_interactions.append({
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": severity,
            "mechanism": _safe_str(item.get("mechanism")),
            "clinical_recommendation": _safe_str(item.get("clinical_recommendation")),
            "source_confidence": confidence,
            "interaction_type": itype,
            "evidence_level": evidence,
        })

    # Validate allergy alerts
    validated_alerts = []
    for alert in data.get("allergy_alerts", []):
        if not isinstance(alert, dict):
            continue
        severity = str(alert.get("severity", "")).lower()
        if severity not in {"critical", "high", "medium", "low"}:
            severity = "high"
        validated_alerts.append({
            "medicine": _safe_str(alert.get("medicine")),
            "allergen": _safe_str(alert.get("allergen")),
            "reason": _safe_str(alert.get("reason")),
            "severity": severity,
            "cross_reactivity": bool(alert.get("cross_reactivity", False)),
            "cross_reactivity_rate": alert.get("cross_reactivity_rate"),
        })

    assessment = data.get("overall_assessment", {})

    return {
        "interactions": validated_interactions,
        "allergy_alerts": validated_alerts,
        "safe_to_prescribe": bool(assessment.get("safe_to_prescribe", False)),
        "overall_risk_level": assessment.get("overall_risk_level", "high"),
        "requires_doctor_review": bool(assessment.get("requires_doctor_review", True)),
        "unrecognized_drugs": data.get("unrecognized_drugs", []),
    }


# ─────────────────────────────────────────────
# RISK SCORER (BONUS B)
# ─────────────────────────────────────────────

def calculate_risk_score(
    interactions: list[DrugInteraction],
    allergy_alerts: list[AllergyAlert],
    contraindications: list[ContraindicationAlert],
    age: Optional[int],
    renal_function: Optional[str],
    total_drug_count: int,
) -> RiskScoreBreakdown:
    """
    Patient Risk Score (0–100) calculation.
    Transparent, auditable formula.
    """
    # Interaction scoring
    severity_weights = {"high": 25, "medium": 12, "low": 4}
    base_score = min(sum(severity_weights.get(i.severity, 0) for i in interactions), 50)

    # Allergy scoring
    allergy_weights = {"critical": 30, "high": 20, "medium": 10, "low": 5}
    allergy_score = min(sum(allergy_weights.get(a.severity, 0) for a in allergy_alerts), 40)

    # Contraindication scoring
    contra_weights = {"contraindicated": 20, "use_with_caution": 8, "monitor_closely": 5}
    condition_score = min(sum(contra_weights.get(c.severity, 0) for c in contraindications), 30)

    # Polypharmacy penalty (>5 concurrent drugs)
    polypharmacy_penalty = max(0, (total_drug_count - 5) * 2)

    # Age modifier: infants and elderly carry highest risk
    # Must check <= 2 before <= 12 to avoid dead-code branch
    age_modifier = 0.0
    if age is not None:
        if age >= 80:
            age_modifier = 10.0
        elif age >= 65:
            age_modifier = 6.0
        elif age <= 2:      # Infant: immature CYP enzymes, narrow therapeutic windows
            age_modifier = 8.0
        elif age <= 12:     # Child: reduced renal/hepatic clearance vs adults
            age_modifier = 5.0

    # Renal modifier
    renal_modifiers = {
        "mild_impairment": 3.0,
        "moderate_impairment": 8.0,
        "severe_impairment": 15.0,
        "esrd": 20.0,
    }
    renal_modifier = renal_modifiers.get(renal_function or "normal", 0.0)

    # Final score (capped at 100)
    final_score = min(
        base_score + allergy_score + condition_score +
        polypharmacy_penalty + age_modifier + renal_modifier,
        100.0
    )

    # Interpretation
    if final_score >= 70:
        interpretation = "CRITICAL RISK — Prescription requires immediate specialist review. Multiple severe safety concerns identified."
    elif final_score >= 50:
        interpretation = "HIGH RISK — Significant safety concerns. Close monitoring and possible dose adjustment required."
    elif final_score >= 30:
        interpretation = "MODERATE RISK — Some safety concerns present. Physician awareness and monitoring recommended."
    else:
        interpretation = "LOW RISK — No significant safety concerns identified with current information."

    return RiskScoreBreakdown(
        base_score=round(base_score, 1),
        allergy_score=round(allergy_score, 1),
        condition_score=round(condition_score, 1),
        polypharmacy_penalty=round(polypharmacy_penalty, 1),
        age_modifier=round(age_modifier, 1),
        renal_modifier=round(renal_modifier, 1),
        final_score=round(final_score, 1),
        interpretation=interpretation,
    )


def determine_overall_risk(
    interactions: list[DrugInteraction],
    allergy_alerts: list[AllergyAlert],
    contraindications: list[ContraindicationAlert],
    risk_score: float,
) -> tuple[str, bool, bool]:
    """Returns (overall_risk_level, safe_to_prescribe, requires_doctor_review)"""

    has_critical_allergy = any(a.severity == "critical" for a in allergy_alerts)
    has_high_interaction = any(i.severity == "high" for i in interactions)
    has_contraindicated = any(c.severity == "contraindicated" for c in contraindications)

    if has_critical_allergy or has_contraindicated:
        return "critical", False, True
    elif has_high_interaction or risk_score >= 60:
        return "high", False, True
    elif risk_score >= 35:
        return "medium", True, True
    else:
        return "low", True, False


# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────

class DrugSafetyEngine:

    def __init__(self, ollama_client: OllamaClient):
        self.llm = ollama_client
        self.cache = get_cache()

    def _build_llm_prompt(self, request: DrugSafetyRequest) -> str:
        h = request.patient_history
        return f"""Analyze the following drug safety case for an Indian clinic patient.

PROPOSED MEDICINES (to be newly prescribed):
{json.dumps(request.proposed_medicines, indent=2)}

PATIENT HISTORY:
- Age: {h.age if h.age is not None else 'Not provided'}
- Weight: {f"{h.weight_kg} kg" if h.weight_kg else 'Not provided'}
- Current Medications (already taking): {json.dumps(h.current_medications) if h.current_medications else 'None'}
- Known Allergies: {json.dumps(h.known_allergies) if h.known_allergies else 'None'}
- Active Medical Conditions: {json.dumps(h.conditions) if h.conditions else 'None'}
- Renal Function: {h.renal_function or 'Not provided'}
- Hepatic Function: {h.hepatic_function or 'Not provided'}
- Pregnancy Status: {h.pregnancy_status or 'Not provided'}

INSTRUCTIONS:
1. Check ALL proposed medicines against each other (proposed × proposed)
2. Check ALL proposed medicines against current medications (proposed × current)
3. Check ALL proposed medicines against known allergies (including drug class cross-reactivity)
4. Return ONLY valid JSON matching the required schema. No extra text."""

    async def analyze(self, request: DrugSafetyRequest) -> DrugSafetyResponse:
        start_time = time.time()
        warnings = []

        # Check cache
        cache_key = self.cache.build_cache_key(
            request.proposed_medicines,
            request.patient_history.current_medications,
        )
        cache_hit, cached_result = self.cache.get(cache_key)

        if cache_hit and cached_result:
            logger.info(f"Cache hit for key {cache_key[:16]}...")
            # Update processing time for cached response
            cached_result["cache_hit"] = True
            cached_result["processing_time_ms"] = int((time.time() - start_time) * 1000)
            return DrugSafetyResponse(**cached_result)

        # ── ALLERGY DETECTION (always rule-based for reliability) ──
        allergy_alerts = detect_allergies(
            request.proposed_medicines,
            request.patient_history.known_allergies,
        )

        # ── CONTRAINDICATION CHECK (Bonus C) ──
        contraindications = check_contraindications(
            request.proposed_medicines,
            request.patient_history.conditions,
            request.patient_history.renal_function,
            request.patient_history.hepatic_function,
            request.patient_history.pregnancy_status,
        )
        # Also check current medications for contraindications
        if request.patient_history.current_medications:
            current_contra = check_contraindications(
                request.patient_history.current_medications,
                request.patient_history.conditions,
                request.patient_history.renal_function,
                request.patient_history.hepatic_function,
                request.patient_history.pregnancy_status,
            )
            contraindications.extend(current_contra)

        # ── LLM INFERENCE ──
        source = "llm"
        interactions = []
        requires_doctor_review = False
        unrecognized_drugs = []

        llm_available = await self.llm.is_available()

        if llm_available:
            logger.info(f"Running LLM inference with {self.llm.model}")
            prompt = self._build_llm_prompt(request)
            raw_response = await self.llm.generate(prompt)
            parsed = parse_llm_response(raw_response) if raw_response else None

            if parsed:
                interactions = [DrugInteraction(**i) for i in parsed["interactions"]]
                # Merge LLM allergy alerts with our rule-based ones (deduplicate)
                llm_allergy_keys = {(a["medicine"].lower(), a["allergen"].lower()) for a in parsed["allergy_alerts"]}
                existing_keys = {(a.medicine.lower(), a.allergen.lower()) for a in allergy_alerts}
                for llm_alert in parsed["allergy_alerts"]:
                    key = (llm_alert["medicine"].lower(), llm_alert["allergen"].lower())
                    if key not in existing_keys:
                        allergy_alerts.append(AllergyAlert(**llm_alert))
                requires_doctor_review = parsed.get("requires_doctor_review", False)
                unrecognized_drugs = parsed.get("unrecognized_drugs", [])
            else:
                logger.warning("LLM response failed validation, activating hybrid mode")
                interactions = run_fallback_engine(
                    request.proposed_medicines,
                    request.patient_history.current_medications,
                )
                source = "hybrid"
                requires_doctor_review = True
                warnings.append("LLM response could not be validated. Fallback engine used for interactions.")
        else:
            logger.warning("LLM unavailable — using fallback engine")
            interactions = run_fallback_engine(
                request.proposed_medicines,
                request.patient_history.current_medications,
            )
            source = "fallback"
            requires_doctor_review = True
            warnings.append("Medical LLM is offline. Results from rule-based fallback dataset. Physician review mandatory.")

        if unrecognized_drugs:
            warnings.append(f"Unrecognized drug names (verify spelling): {', '.join(unrecognized_drugs)}")

        # ── RISK SCORING (Bonus B) ──
        total_drugs = len(request.proposed_medicines) + len(request.patient_history.current_medications)
        risk_breakdown = calculate_risk_score(
            interactions=interactions,
            allergy_alerts=allergy_alerts,
            contraindications=contraindications,
            age=request.patient_history.age,
            renal_function=request.patient_history.renal_function,
            total_drug_count=total_drugs,
        )

        overall_risk, safe_to_prescribe, dr_review_flag = determine_overall_risk(
            interactions, allergy_alerts, contraindications, risk_breakdown.final_score
        )
        requires_doctor_review = requires_doctor_review or dr_review_flag

        processing_time_ms = int((time.time() - start_time) * 1000)

        response_data = dict(
            interactions=interactions,
            allergy_alerts=allergy_alerts,
            contraindication_alerts=contraindications,
            safe_to_prescribe=safe_to_prescribe,
            overall_risk_level=overall_risk,
            requires_doctor_review=requires_doctor_review,
            patient_risk_score=risk_breakdown.final_score,
            risk_score_breakdown=risk_breakdown,
            source=source,
            cache_hit=False,
            processing_time_ms=processing_time_ms,
            validated_medicines=request.proposed_medicines,
            warnings=warnings,
            request_id=request.request_id,
        )

        # Cache the result (serialize for storage)
        self.cache.set(cache_key, {
            k: v.model_dump() if hasattr(v, "model_dump") else
               [item.model_dump() for item in v] if isinstance(v, list) and v and hasattr(v[0], "model_dump") else v
            for k, v in response_data.items()
        })

        return DrugSafetyResponse(**response_data)
