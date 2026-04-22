"""
EvoDoc Clinical Drug Safety Engine — Test Suite
Tests cover: interactions, allergy detection, contraindications,
caching, fallback, risk scoring, and edge cases.
"""

import json
import hashlib
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from models import DrugSafetyRequest, PatientHistory
from cache import DrugSafetyCache
from engine import (
    normalize_drug_name,
    get_drug_classes,
    detect_allergies,
    check_contraindications,
    run_fallback_engine,
    calculate_risk_score,
    parse_llm_response,
    DrugSafetyEngine,
    OllamaClient,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def basic_request():
    return DrugSafetyRequest(
        proposed_medicines=["Warfarin", "Aspirin"],
        patient_history=PatientHistory(
            current_medications=["Metoprolol"],
            known_allergies=["Penicillin"],
            conditions=["hypertension"],
            age=65,
            weight_kg=72.0,
        )
    )


@pytest.fixture
def penicillin_allergy_request():
    return DrugSafetyRequest(
        proposed_medicines=["Amoxicillin", "Paracetamol"],
        patient_history=PatientHistory(
            known_allergies=["Penicillin"],
            age=35,
        )
    )


@pytest.fixture
def renal_failure_request():
    return DrugSafetyRequest(
        proposed_medicines=["Metformin", "Ibuprofen"],
        patient_history=PatientHistory(
            conditions=["chronic_kidney_disease"],
            renal_function="severe_impairment",
            age=58,
        )
    )


@pytest.fixture
def mock_engine():
    client = OllamaClient(base_url="http://localhost:11434", model="biomistral")
    client.is_available = AsyncMock(return_value=False)  # Force fallback
    return DrugSafetyEngine(ollama_client=client)


# ─────────────────────────────────────────────
# CACHE TESTS
# ─────────────────────────────────────────────

class TestCache:

    def test_cache_key_is_order_independent(self):
        """Same drugs in different order must produce same cache key."""
        key1 = DrugSafetyCache.build_cache_key(
            ["Warfarin", "Aspirin", "Metformin"],
            ["Metoprolol"]
        )
        key2 = DrugSafetyCache.build_cache_key(
            ["Metformin", "Warfarin", "Aspirin"],
            ["Metoprolol"]
        )
        assert key1 == key2, "Cache keys must be identical regardless of drug order"

    def test_cache_key_is_case_insensitive(self):
        key1 = DrugSafetyCache.build_cache_key(["warfarin", "aspirin"], [])
        key2 = DrugSafetyCache.build_cache_key(["WARFARIN", "ASPIRIN"], [])
        assert key1 == key2

    def test_different_drugs_produce_different_keys(self):
        key1 = DrugSafetyCache.build_cache_key(["Warfarin"], [])
        key2 = DrugSafetyCache.build_cache_key(["Aspirin"], [])
        assert key1 != key2

    def test_cache_set_and_get(self):
        cache = DrugSafetyCache(ttl_seconds=60)
        key = "test-key-123"
        cache.set(key, {"result": "test"})
        hit, value = cache.get(key)
        assert hit is True
        assert value == {"result": "test"}

    def test_cache_miss_returns_false(self):
        cache = DrugSafetyCache(ttl_seconds=60)
        hit, value = cache.get("nonexistent-key")
        assert hit is False
        assert value is None

    def test_cache_ttl_expiry(self):
        import time
        cache = DrugSafetyCache(ttl_seconds=1)  # 1 second TTL
        key = "expiry-test"
        cache.set(key, "data")
        time.sleep(1.1)
        hit, value = cache.get(key)
        assert hit is False

    def test_cache_hit_rate_tracking(self):
        cache = DrugSafetyCache(ttl_seconds=60)
        key = "rate-test"
        cache.set(key, "value")
        cache.get("nonexistent")  # miss
        cache.get(key)            # hit
        cache.get(key)            # hit
        assert cache.hit_rate == pytest.approx(66.67, rel=0.01)

    def test_cache_current_medications_in_key(self):
        """Current medications must be part of the cache key."""
        key1 = DrugSafetyCache.build_cache_key(["Warfarin"], ["Aspirin"])
        key2 = DrugSafetyCache.build_cache_key(["Warfarin"], ["Metformin"])
        assert key1 != key2


# ─────────────────────────────────────────────
# DRUG NORMALIZATION TESTS
# ─────────────────────────────────────────────

class TestDrugNormalization:

    def test_removes_dosage(self):
        assert normalize_drug_name("Metformin 500mg") == "metformin"
        assert normalize_drug_name("Aspirin 75 mg") == "aspirin"

    def test_lowercase(self):
        assert normalize_drug_name("WARFARIN") == "warfarin"

    def test_strips_whitespace(self):
        assert normalize_drug_name("  Aspirin  ") == "aspirin"

    def test_drug_class_detection(self):
        classes = get_drug_classes("Amoxicillin")
        assert "penicillin" in classes

    def test_nsaid_class_detection(self):
        classes = get_drug_classes("Ibuprofen")
        assert "nsaid" in classes

    def test_unknown_drug_returns_empty_classes(self):
        classes = get_drug_classes("XYZ_UNKNOWN_DRUG_9999")
        assert classes == []


# ─────────────────────────────────────────────
# ALLERGY DETECTION TESTS
# ─────────────────────────────────────────────

class TestAllergyDetection:

    def test_exact_allergy_match(self):
        """Exact drug name match must return critical severity."""
        alerts = detect_allergies(["Aspirin"], ["Aspirin"])
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"
        assert alerts[0].cross_reactivity is False

    def test_penicillin_amoxicillin_cross_reactivity(self):
        """Penicillin allergy must flag Amoxicillin (same class)."""
        alerts = detect_allergies(["Amoxicillin"], ["Penicillin"])
        assert len(alerts) >= 1
        assert any(a.medicine == "Amoxicillin" for a in alerts)
        flagged = [a for a in alerts if a.medicine == "Amoxicillin"][0]
        assert flagged.severity in ("critical", "high")

    def test_no_allergy_when_unrelated(self):
        """No alert when drugs are unrelated to known allergens."""
        alerts = detect_allergies(["Metformin", "Atorvastatin"], ["Penicillin"])
        assert len(alerts) == 0

    def test_case_insensitive_detection(self):
        """Allergy detection must be case-insensitive."""
        alerts = detect_allergies(["ASPIRIN"], ["aspirin"])
        assert len(alerts) >= 1

    def test_multiple_allergies(self):
        alerts = detect_allergies(
            ["Amoxicillin", "Ibuprofen"],
            ["Penicillin", "Aspirin"]
        )
        # Should flag amoxicillin for penicillin class
        medicines_flagged = [a.medicine for a in alerts]
        assert "Amoxicillin" in medicines_flagged

    def test_no_duplicate_alerts(self):
        """Same allergy pair must not appear twice."""
        alerts = detect_allergies(["Amoxicillin"], ["Penicillin"])
        pairs = [(a.medicine, a.allergen) for a in alerts]
        assert len(pairs) == len(set(pairs))


# ─────────────────────────────────────────────
# CONTRAINDICATION TESTS (BONUS C)
# ─────────────────────────────────────────────

class TestContraindications:

    def test_nsaid_flagged_for_renal_failure(self):
        alerts = check_contraindications(
            ["Ibuprofen"],
            conditions=[],
            renal_function="severe_impairment"
        )
        assert len(alerts) >= 1
        assert any(a.medicine == "Ibuprofen" for a in alerts)

    def test_metformin_flagged_for_renal_failure(self):
        alerts = check_contraindications(
            ["Metformin"],
            conditions=["renal_failure"],
        )
        assert len(alerts) >= 1

    def test_beta_blocker_flagged_for_asthma(self):
        alerts = check_contraindications(
            ["Metoprolol"],
            conditions=["asthma"],
        )
        assert len(alerts) >= 1
        assert alerts[0].severity in ("contraindicated", "use_with_caution")

    def test_tetracycline_flagged_in_pregnancy(self):
        alerts = check_contraindications(
            ["Doxycycline"],
            conditions=[],
            pregnancy_status="first_trimester"
        )
        assert len(alerts) >= 1

    def test_no_contraindication_when_none_applicable(self):
        alerts = check_contraindications(
            ["Paracetamol"],
            conditions=["hypertension"],
        )
        assert len(alerts) == 0


# ─────────────────────────────────────────────
# FALLBACK ENGINE TESTS
# ─────────────────────────────────────────────

class TestFallbackEngine:

    def test_warfarin_aspirin_interaction(self):
        interactions = run_fallback_engine(["Warfarin", "Aspirin"], [])
        assert len(interactions) >= 1
        pair_drugs = {(i.drug_a.lower(), i.drug_b.lower()) for i in interactions}
        found = any(
            ("warfarin" in d and "aspirin" in d)
            for d in [f"{a} {b}" for a, b in pair_drugs]
        )
        assert found

    def test_interaction_with_current_medications(self):
        """Proposed drugs must be checked against current medications."""
        interactions = run_fallback_engine(["Warfarin"], ["Aspirin"])
        assert len(interactions) >= 1

    def test_never_empty_result(self):
        """Even with unknown drugs, should return empty list not None."""
        result = run_fallback_engine(["Paracetamol"], [])
        assert result is not None
        assert isinstance(result, list)

    def test_duplicate_medicines_handled(self):
        """Duplicate entries in proposed_medicines must not cause double-counting."""
        interactions = run_fallback_engine(["Warfarin", "warfarin"], [])
        # Should not error out
        assert isinstance(interactions, list)

    def test_source_confidence_is_fallback(self):
        interactions = run_fallback_engine(["Warfarin", "Aspirin"], [])
        for i in interactions:
            assert i.source_confidence == "fallback"


# ─────────────────────────────────────────────
# RISK SCORING TESTS (BONUS B)
# ─────────────────────────────────────────────

class TestRiskScoring:

    def _make_interaction(self, severity):
        from models import DrugInteraction
        return DrugInteraction(
            drug_a="A", drug_b="B", severity=severity,
            mechanism="test", clinical_recommendation="test",
            source_confidence="high"
        )

    def _make_allergy(self, severity):
        from models import AllergyAlert
        return AllergyAlert(
            medicine="X", allergen="Y", reason="test",
            severity=severity, cross_reactivity=False
        )

    def test_high_interaction_increases_score(self):
        high_result = calculate_risk_score(
            [self._make_interaction("high")], [], [], None, None, 2
        )
        low_result = calculate_risk_score(
            [self._make_interaction("low")], [], [], None, None, 2
        )
        assert high_result.final_score > low_result.final_score

    def test_critical_allergy_maxes_allergy_score(self):
        result = calculate_risk_score(
            [], [self._make_allergy("critical")], [], None, None, 2
        )
        assert result.allergy_score == 30.0

    def test_elderly_age_increases_score(self):
        young = calculate_risk_score([], [], [], age=25, renal_function=None, total_drug_count=2)
        elderly = calculate_risk_score([], [], [], age=80, renal_function=None, total_drug_count=2)
        assert elderly.final_score > young.final_score

    def test_renal_failure_increases_score(self):
        normal = calculate_risk_score([], [], [], None, "normal", 2)
        esrd = calculate_risk_score([], [], [], None, "esrd", 2)
        assert esrd.final_score > normal.final_score

    def test_score_capped_at_100(self):
        interactions = [self._make_interaction("high")] * 10
        allergies = [self._make_allergy("critical")] * 5
        result = calculate_risk_score(interactions, allergies, [], 85, "esrd", 15)
        assert result.final_score <= 100.0

    def test_score_in_range_0_to_100(self):
        result = calculate_risk_score([], [], [], None, None, 0)
        assert 0.0 <= result.final_score <= 100.0


# ─────────────────────────────────────────────
# LLM RESPONSE PARSER TESTS
# ─────────────────────────────────────────────

class TestLLMResponseParser:

    def test_parses_valid_response(self):
        valid_json = json.dumps({
            "interactions": [{
                "drug_a": "Warfarin", "drug_b": "Aspirin",
                "severity": "high",
                "mechanism": "Antiplatelet + anticoagulant synergism",
                "clinical_recommendation": "Avoid or monitor INR",
                "source_confidence": "high",
                "interaction_type": "pharmacodynamic",
                "evidence_level": "A"
            }],
            "allergy_alerts": [],
            "overall_assessment": {
                "safe_to_prescribe": False,
                "overall_risk_level": "high",
                "requires_doctor_review": True,
                "clinical_summary": "High-risk combination"
            },
            "unrecognized_drugs": []
        })
        result = parse_llm_response(valid_json)
        assert result is not None
        assert len(result["interactions"]) == 1
        assert result["interactions"][0]["severity"] == "high"

    def test_handles_invalid_severity(self):
        """Invalid severity values must be corrected to 'medium'."""
        json_str = json.dumps({
            "interactions": [{
                "drug_a": "A", "drug_b": "B",
                "severity": "CRITICAL",  # Invalid
                "mechanism": "test", "clinical_recommendation": "test",
                "source_confidence": "high"
            }],
            "allergy_alerts": [],
            "overall_assessment": {"safe_to_prescribe": True, "overall_risk_level": "low", "requires_doctor_review": False, "clinical_summary": "ok"},
            "unrecognized_drugs": []
        })
        result = parse_llm_response(json_str)
        assert result["interactions"][0]["severity"] == "medium"

    def test_handles_empty_response(self):
        result = parse_llm_response("")
        assert result is None

    def test_handles_non_json_response(self):
        result = parse_llm_response("Sorry, I cannot analyze this.")
        assert result is None

    def test_handles_json_in_markdown_fence(self):
        response = '```json\n{"interactions": [], "allergy_alerts": [], "overall_assessment": {"safe_to_prescribe": true, "overall_risk_level": "low", "requires_doctor_review": false, "clinical_summary": "ok"}, "unrecognized_drugs": []}\n```'
        result = parse_llm_response(response)
        assert result is not None


# ─────────────────────────────────────────────
# INPUT VALIDATION TESTS
# ─────────────────────────────────────────────

class TestInputValidation:

    def test_negative_age_raises_error(self):
        with pytest.raises(Exception):
            PatientHistory(age=-5)

    def test_empty_medicines_raises_error(self):
        with pytest.raises(Exception):
            DrugSafetyRequest(proposed_medicines=[])

    def test_duplicate_medicines_deduplicated(self):
        req = DrugSafetyRequest(
            proposed_medicines=["Aspirin", "aspirin", "ASPIRIN"],
        )
        assert len(req.proposed_medicines) == 1

    def test_whitespace_only_medicine_ignored(self):
        req = DrugSafetyRequest(
            proposed_medicines=["Aspirin", "   ", "Warfarin"],
        )
        assert "   " not in req.proposed_medicines

    def test_medicine_names_normalized(self):
        req = DrugSafetyRequest(
            proposed_medicines=["aspirin"],
        )
        # Should be title-cased
        assert req.proposed_medicines[0] == "Aspirin"


# ─────────────────────────────────────────────
# INTEGRATION TESTS (Fallback mode)
# ─────────────────────────────────────────────

class TestIntegration:

    @pytest.mark.asyncio
    async def test_full_analysis_fallback_mode(self, mock_engine, basic_request):
        """Full pipeline test with fallback engine (no LLM needed)."""
        result = await mock_engine.analyze(basic_request)

        assert result is not None
        assert result.source == "fallback"
        assert isinstance(result.interactions, list)
        assert isinstance(result.allergy_alerts, list)
        assert result.cache_hit is False
        assert result.processing_time_ms > 0
        assert 0 <= result.patient_risk_score <= 100

    @pytest.mark.asyncio
    async def test_second_call_is_cached(self, mock_engine, basic_request):
        """Second identical request must return cache_hit=True."""
        await mock_engine.analyze(basic_request)
        result2 = await mock_engine.analyze(basic_request)
        assert result2.cache_hit is True

    @pytest.mark.asyncio
    async def test_penicillin_allergy_flagged(self, mock_engine, penicillin_allergy_request):
        result = await mock_engine.analyze(penicillin_allergy_request)
        assert len(result.allergy_alerts) >= 1
        flagged_meds = [a.medicine for a in result.allergy_alerts]
        assert "Amoxicillin" in flagged_meds

    @pytest.mark.asyncio
    async def test_renal_failure_contraindications(self, mock_engine, renal_failure_request):
        result = await mock_engine.analyze(renal_failure_request)
        assert len(result.contraindication_alerts) >= 1

    @pytest.mark.asyncio
    async def test_safe_to_prescribe_false_for_critical_allergy(self, mock_engine):
        req = DrugSafetyRequest(
            proposed_medicines=["Amoxicillin"],
            patient_history=PatientHistory(known_allergies=["Penicillin"])
        )
        result = await mock_engine.analyze(req)
        assert result.safe_to_prescribe is False
        assert result.requires_doctor_review is True

    @pytest.mark.asyncio
    async def test_response_never_empty(self, mock_engine):
        req = DrugSafetyRequest(
            proposed_medicines=["Paracetamol"],
            patient_history=PatientHistory()
        )
        result = await mock_engine.analyze(req)
        # Should always return a result, even if no interactions
        assert result is not None
        assert result.overall_risk_level is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
