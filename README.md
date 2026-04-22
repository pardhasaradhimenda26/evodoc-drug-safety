# EvoDoc Clinical Drug Safety Engine

> **A wrong output here does not fail a test. It can harm a patient.**

AI-powered drug interaction and safety checker built for Indian clinics. Accepts proposed medicines and patient history, returns a structured clinical safety assessment with drug interactions, allergy alerts, contraindications, and a risk score — in under 3 seconds.

---

## Architecture Overview

```
POST /analyze
    │
    ├── Input Validation (Pydantic)
    │       └── Normalize drug names, deduplicate, validate age/weight
    │
    ├── Cache Check (SHA-256 hash of sorted drugs)
    │       └── HIT → return cached result instantly
    │
    ├── Allergy Detection (rule-based, always runs)
    │       └── Exact match + drug class cross-reactivity
    │
    ├── Contraindication Check (rule-based, always runs)
    │       └── Drug × condition, renal/hepatic/pregnancy modifiers
    │
    ├── Interaction Analysis
    │       ├── LLM available → BioMistral-7B via Ollama
    │       └── LLM offline  → Fallback rule dataset (25 interactions)
    │
    ├── Risk Scoring (0–100)
    │       └── Interaction score + allergy score + condition score +
    │           polypharmacy penalty + age modifier + renal modifier
    │
    └── Response Assembly → DrugSafetyResponse
```

---

## LLM Choice: BioMistral-7B

### Why BioMistral over GPT-4 / Gemini / Claude?

| Factor | BioMistral-7B | Generic LLMs |
|--------|--------------|--------------|
| Medical training | ✅ Pre-trained on PubMed + MIMIC | ❌ General corpus |
| VRAM (4-bit quant) | ~4.5GB (full), ~2.5GB (Q4_K_M) | Cloud only |
| Zero cloud exposure | ✅ Fully local | ❌ Data leaves clinic |
| Drug interaction accuracy | High on clinical pharmacology | Moderate |
| Hallucination rate | Lower (domain-specific) | Higher (generic) |
| CDSCO India compliant | ✅ No data transfer | ❌ Cloud dependency |
| Cost at scale | ₹0 per query | ₹3-15 per query |

**Why not Meditron?** Meditron-7B is strong for clinical reasoning but BioMistral-7B shows better structured JSON output consistency in benchmarks, critical for our zero-hallucination requirement.

**Fallback:** If BioMistral is unavailable, the engine falls back to a curated rule-based dataset of 25 clinically validated drug interactions. The system **never returns an empty result**.

---

## Caching Strategy

**Algorithm:** SHA-256 hash of `sorted(proposed_medicines) + sorted(current_medications)`

```python
key_payload = {
    "proposed": sorted(m.lower() for m in proposed_medicines),
    "current":  sorted(m.lower() for m in current_medications),
}
cache_key = sha256(json.dumps(key_payload, sort_keys=True))
```

**Why this works:**
- Order-independent: `[Warfarin, Aspirin]` == `[Aspirin, Warfarin]` → same key
- Case-insensitive: `ASPIRIN` == `aspirin` → same key
- Current medications in key: same drugs + different history = different result

**TTL:** 1 hour (configurable via `CACHE_TTL_SECONDS`)

**Caching tradeoffs:**
| Factor | Decision | Reason |
|--------|----------|--------|
| Key includes patient history? | No | Allergy and contraindication checks always run fresh from rules — never cached. Only LLM interaction results (the expensive call) are cached. |
| Redis vs in-memory? | In-memory | Zero infrastructure overhead for clinic-grade hardware. `DrugSafetyCache` is drop-in replaceable with a Redis adapter — same interface, no other changes needed. |
| TTL? | 1 hour | Drug interaction data is clinically stable within hours. Balances freshness vs compute savings. |
| Max size? | 1000 entries | Prevents memory exhaustion on constrained clinic hardware. LRU eviction when full. |

---

## Fallback Dataset

Located at `data/fallback_interactions.json`. Contains **25 clinically validated drug interactions** sourced from:
- Stockley's Drug Interactions (9th edition)
- FDA drug interaction tables
- WHO Model Formulary 2023
- CDSCO India prescribing guidelines

**Notable interactions covered:**
| Drug Pair | Severity | Mechanism |
|-----------|----------|-----------|
| Warfarin + Aspirin | High | Protein binding displacement + platelet inhibition |
| Simvastatin + Amlodipine | Medium | CYP3A4 inhibition → rhabdomyolysis risk |
| SSRIs + MAOIs | High | Serotonin syndrome (potentially fatal) |
| Lithium + Ibuprofen | High | Reduced renal clearance → toxicity |
| Sildenafil + Nitrates | High | Severe hypotension (absolute contraindication) |
| Digoxin + Amiodarone | High | P-gp inhibition → digoxin toxicity |
| Clopidogrel + Omeprazole | Medium | CYP2C19 competitive inhibition |
| Rifampicin + OCP | High | CYP3A4 induction → contraceptive failure |
| Ciprofloxacin + Theophylline | High | CYP1A2 inhibition → theophylline toxicity |
| Tacrolimus + Clarithromycin | High | CYP3A4 inhibition → immunosuppressant overdose |

---

## Bonus Features Implemented

### ✅ Bonus A — Prompt Engineering
See `prompts/system_prompt.txt`. Key design decisions:
- Explicit JSON schema enforcement (model cannot return text)
- Hallucination prevention: confidence thresholding below 70%
- Indian clinical context (generic drug name aliases)
- Temperature set to 0.1 for deterministic clinical output

### ✅ Bonus B — Risk Scoring (0–100)
Transparent formula with audit trail:
```
final_score = min(
    interaction_score  (max 50: high=25pts, medium=12pts, low=4pts)
  + allergy_score      (max 40: critical=30pts, high=20pts)
  + condition_score    (max 30: contraindicated=20pts)
  + polypharmacy_penalty (2pts per drug above 5)
  + age_modifier       (elderly ≥80: +10pts, ≥65: +6pts)
  + renal_modifier     (ESRD: +20pts, severe: +15pts)
, 100)
```

### ✅ Bonus C — Condition Contraindications
15 drug-condition contraindication rules in `data/contraindication_rules.json`. Examples:
- NSAIDs → Renal failure, peptic ulcer, heart failure: **CONTRAINDICATED**
- Beta-blockers → Asthma, COPD, heart block: **CONTRAINDICATED**
- Metformin → Renal failure, hepatic failure: **CONTRAINDICATED**
- Tetracyclines → Pregnancy: **CONTRAINDICATED**
- Statins → Active liver disease: **CONTRAINDICATED**

### ✅ Bonus D — Performance

`processing_time_ms` is returned in every response (including cache hits).

**Measured on local machine (Intel i7, 16GB RAM, no GPU, BioMistral Q4_K_M via Ollama):**

| Scenario | `processing_time_ms` |
|----------|----------------------|
| Cache HIT | < 5 ms |
| Fallback engine, 5 drugs | 45 – 120 ms |
| BioMistral 4-bit, 5 drugs | 1,200 – 2,800 ms |
| BioMistral 4-bit, 10 drugs | 2,100 – 3,200 ms ✅ under 3s target |

**Why it stays fast:** Allergy detection and contraindication checks are pure Python dict lookups (~1ms). Only the LLM call is slow. Cache eliminates it entirely for repeat queries.

**At scale (if processing_time_ms exceeds 3 seconds):**
1. GPU inference (RTX 3060, 6GB VRAM): BioMistral Q4 runs at ~15 tokens/sec → consistently under 2 seconds
2. Ollama multi-instance behind nginx upstream for concurrent clinic users
3. Startup cache pre-warming for the 50 most common Indian drug combinations
4. Async Celery queue with Redis broker for high-concurrency peak loads (outpatient morning rush)

---

## Project Structure

```
evodoc-drug-safety/
├── main.py                          # FastAPI app, routes, middleware
├── engine.py                        # Core analysis engine
├── cache.py                         # Caching layer
├── models.py                        # Pydantic request/response schemas
├── prompts/
│   └── system_prompt.txt            # Medical LLM system prompt (Bonus A)
├── data/
│   ├── fallback_interactions.json   # 25 drug interactions (fallback)
│   └── contraindication_rules.json  # 15 drug-condition rules (Bonus C)
├── tests/
│   └── test_engine.py               # 52 test cases
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### 1. Clone and install dependencies
```bash
git clone https://github.com/pardhasaradhimenda26/evodoc-drug-safety
cd evodoc-drug-safety
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 3. Install Ollama and pull BioMistral
```bash
# Install Ollama (https://ollama.com)
curl -fsSL https://ollama.com/install.sh | sh

# Pull BioMistral (medical LLM, ~4.1GB)
ollama pull biomistral

# Verify it's running
ollama list
```

### 4. Run the server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test with curl
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "proposed_medicines": ["Warfarin", "Aspirin", "Ibuprofen"],
    "patient_history": {
      "current_medications": ["Metoprolol"],
      "known_allergies": ["Penicillin"],
      "conditions": ["hypertension", "chronic_kidney_disease"],
      "age": 68,
      "weight_kg": 72,
      "renal_function": "mild_impairment"
    }
  }'
```

### 6. Run tests
```bash
pytest tests/ -v --tb=short
```

### 7. View interactive API docs
Open `http://localhost:8000/docs` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Core drug safety analysis |
| POST | `/analyze/batch` | Batch analysis (up to 10 patients) |
| GET | `/health` | System health + LLM status |
| GET | `/cache/stats` | Cache hit rate and performance |
| POST | `/cache/clear` | Clear cache (admin) |
| GET | `/interactions/fallback` | View fallback dataset |
| GET | `/drug-classes` | View drug class mappings |

---

## Design Decisions

**Why in-memory cache over Redis?**
Clinic hardware often can't run Redis. In-memory is zero-config and sufficient for single-instance deployments. The `DrugSafetyCache` class is drop-in replaceable with a Redis adapter without changing any calling code.

**Why rule-based allergy detection even when LLM is available?**
LLM hallucination risk on allergy cross-reactivity is too high for patient safety. Rule-based detection is 100% predictable, auditable, and fast. LLM output supplements it; rule-based always takes precedence.

**Why validate all LLM fields?**
Blind trust in LLM output is dangerous in clinical settings. Every field is validated: invalid severity defaults to "medium", missing mechanisms get flagged, hallucinated drug names are moved to `unrecognized_drugs`.

---

## Author

**Pardhasaradhi Menda** | Founder & CTO, HEILC  
B.Tech CSE (AI/ML), SRM Institute of Science and Technology  
GitHub: [pardhasaradhimenda26](https://github.com/pardhasaradhimenda26)
