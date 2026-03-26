# HydroUA-GPT

HydroUA-GPT is a Gradio-based application that combines two capabilities in a single chat interface:

1. **Hydrology Q&A** powered by a shared language model (fine-tuned hydrology expert)
2. **Build-site feasibility screening** for a proposed building at a specific address using geocoding, FEMA flood-hazard lookup, and a block-level vulnerability index

The app routes each user message into the correct path, keeps follow-up questions within the same context when possible, and can display a location map for build-site screening results.

---

## What the app does

### 1) Hydrology assistant
The hydrology path handles hydrology and water-resources questions such as runoff, floods, rainfall, groundwater, SMAP, watersheds, and related topics. Responses are streamed token-by-token from the loaded model for a chat-like experience.

### 2) Build-site feasibility advisor
The advisor path supports queries like:

- `Can I build a warehouse at 123 Main St, Tuscaloosa AL?`
- `hospital at 302 Reed St, Tuscaloosa AL`
- `What is a warehouse?`

For advisor queries, the app:
- extracts the **building type** and **address**
- geocodes the address using the **U.S. Census Geocoder**
- looks up **FEMA NFHL** flood hazard at the point
- looks up a **block-level vulnerability index** from a Parquet file
- combines those signals into an overall tier and a simple feasibility score
- returns a screening summary and, when coordinates are available, shows the point on an embedded map

---

## Architecture overview

The app is intentionally split into focused modules so the logic is easier to maintain and update.

### Core files

- `app.py` — Gradio UI, message dispatch, follow-up handling, streaming handoff, and map embed generation
- `router.py` — rule-based first-pass intent routing
- `llm_router.py` — LLM fallback router that can upgrade `offtopic` to `hydrology` or `advisor`
- `hydrology_chat.py` — hydrology chat prompt construction and streaming generation
- `advisor_flow.py` — end-to-end advisor logic and response formatting
- `advisor_llm.py` — LLM slot filling for building type and address extraction
- `advisor_tools.py` — geocoding, FEMA lookup, vulnerability lookup, caching, tier combination, and scoring
- `model_runtime.py` — one-time model/tokenizer loading and reuse across modules

### High-level request flow

```text
User message
   |
   v
Rule router (router.py)
   |
   +--> advisor ------------------------------+
   |                                           |
   |                                   advisor_flow.py
   |                                           |
   |                      +--------------------+-------------------+
   |                      |                                        |
   |                advisor_llm.py                         advisor_tools.py
   |           (slot filling / follow-up)        (geocode + FEMA + VI + score)
   |                                           |
   |                                           v
   |                                   screening response + map
   |
   +--> hydrology ----------------------------+
   |                                           |
   |                                   hydrology_chat.py
   |                                           |
   |                                   streamed model reply
   |
   +--> offtopic -> refusal / guidance
```

### Routing behavior

The routing logic is hybrid:

- **Rules first** decide whether a message looks like hydrology, advisor, or off-topic.
- If the rule router returns `offtopic`, the app can call an **LLM router** as a fallback.
- Short follow-ups such as “examples” or “explain more” can inherit the previous route.
- Weather and flood-forecast questions are treated carefully so the app does not pretend to have guaranteed real-time forecast access.

---

## Build-site advisor logic

The advisor flow is designed as a screening tool, not a formal engineering or regulatory determination.

### Inputs it tries to extract
- **Building type** such as warehouse, hospital, school, residential, industrial, or commercial
- **Address** such as a street address or, in some cases, a city/state fallback

### Data sources and processing
1. **U.S. Census Geocoder**
   - Converts address to latitude/longitude
   - Attempts to recover a block GEOID

2. **FEMA National Flood Hazard Layer (NFHL)**
   - Queries the flood hazard polygon intersecting the point
   - Interprets zone, subtype, and SFHA flag into a FEMA risk tier

3. **Block-level vulnerability table**
   - Loaded from a Parquet file referenced by `VULN_PARQUET_PATH`
   - Looks up `Vulnerability_Index` by GEOID10
   - Converts the value into a vulnerability tier using fixed quantile thresholds

4. **Combined scoring**
   - Uses the max of FEMA tier and vulnerability tier
   - Applies a building-type penalty
   - Produces a simple score, label, and recommended next steps

### Important limitation
This is a **screening-level** workflow. It is not a replacement for:
- a surveyed site boundary
- a local floodplain determination
- a drainage study
- a site-specific engineering review
- code and ordinance review for critical facilities

---

## Hydrology chat behavior

Hydrology responses are generated from the same loaded model used elsewhere in the app. The hydrology module:

- builds a system prompt focused on hydrology and water resources
- formats the conversation using the tokenizer chat template when available
- streams responses using `TextIteratorStreamer`
- keeps non-hydrology rejection out of the generation layer because routing already handles that decision

The app also includes a safety guard for forecast-style questions. Instead of inventing live weather or flood predictions, it tells the user what forecast sources and signals to check.

---

## Model loading strategy

`model_runtime.py` loads the model and tokenizer once, caches them in module globals, and shares them across the rest of the application.

Key points:
- Default model path is:
  - `/icebox/data/shares/mh2/shassan6/gradio/models/hydrology_aqua_llm_strict_11B`
- It first tries to load via `AutoPeftModelForCausalLM`
- If that fails, it falls back to `AutoModelForCausalLM`
- The model is loaded with `torch.bfloat16` and `device_map="auto"`
- `flash_attention_2` is enabled when supported

Because the model is loaded up front in `app.py`, startup can take time, but repeated requests reuse the same in-memory model.

---

## Caching

The advisor utilities cache expensive lookups in SQLite databases.

### Geocode cache
- Stored in:
  - `${SLURM_TMPDIR}/hydroua_geocode_cache_${USER}.sqlite`
  - or `/tmp/...` if `SLURM_TMPDIR` is not set

### FEMA cache
- Stored in:
  - `${SLURM_TMPDIR}/hydroua_fema_cache_${USER}.sqlite`
  - or `/tmp/...`

These caches reduce repeated network calls for geocoding and FEMA queries.

---

## Environment variables

The application already has sensible defaults, but these environment variables control the important runtime behavior.

### App/UI
- `HOST` — Gradio host, default: `0.0.0.0`
- `PORT` — Gradio port, default: `7860`
- `MAP_BASE_URL` — base URL for the embedded map view
- `DEBUG_ROUTER` — set to `1` to print routing/debug info

### Advisor data/runtime
- `VULN_PARQUET_PATH` — path to the vulnerability Parquet file
- `SLURM_TMPDIR` — preferred temp/cache directory on HPC
- `FEMA_TIMEOUT` — FEMA request timeout in seconds
- `FEMA_RETRIES` — FEMA retry count
- `FEMA_CACHE_TTL_SECONDS` — cache lifetime for FEMA results
- `FEMA_CACHE_DECIMALS` — coordinate rounding precision for FEMA cache keys
- `SSL_CERT_FILE` — optional custom CA bundle

### Model
The model path is currently hardcoded inside `model_runtime.py` as `MODEL_PATH`. If you want this to be configurable, the simplest improvement is to replace that constant with an environment-variable lookup.

---

## Python dependencies

Based on the code, the app depends on the following Python packages:

- `gradio`
- `torch`
- `transformers`
- `duckdb`
- `peft` (optional but preferred if the model is PEFT-based)
- `certifi` (optional fallback for SSL certificates)

It also uses only standard-library modules for:
- threading
- regex
- JSON
- SQLite
- HTTP requests via `urllib`
- path and environment handling

A minimal install might look like:

```bash
pip install gradio torch transformers duckdb peft certifi
```

Depending on your cluster or GPU image, you may want a more controlled environment for PyTorch and CUDA.

---

## Expected data and model paths

Before launching the app, make sure the following resources exist:

### 1) Model directory
Default:

```bash
/icebox/data/shares/mh2/shassan6/gradio/models/hydrology_aqua_llm_strict_11B
```

### 2) Vulnerability Parquet
Default:

```bash
/icebox/data/shares/mh2/shassan6/gradio/data/vulnerability.parquet
```

### 3) Map service
Default embedded base URL:

```bash
https://homes-design-brave-anna.trycloudflare.com/
```

If that map endpoint changes, set `MAP_BASE_URL` before launch.

---

## Running the app

From the project directory:

```bash
python app.py
```

By default, Gradio launches on:

```text
http://0.0.0.0:7860
```

### Example HPC-style launch

```bash
export HOST=0.0.0.0
export PORT=7860
export VULN_PARQUET_PATH=/path/to/vulnerability.parquet
export MAP_BASE_URL=https://your-map-service.example.com/
python app.py
```

---

## Example prompts

### Hydrology
- `What is a hydrograph?`
- `Explain runoff generation in simple terms.`
- `How does SMAP soil moisture help hydrology?`

### Build-site advisor
- `Can I build a warehouse at 123 Main St, Tuscaloosa AL?`
- `Can I build a hospital at 302 Reed St, Tuscaloosa AL?`
- `commercial at 2500 6th St, Tuscaloosa AL`

### Definition shortcut inside advisor flow
- `What is a warehouse?`
- `Define hospital`

---

## Notes on current design choices

### 1) Hot reload during interaction
`app.py` reloads several logic modules at request time using `importlib.reload(...)`. This is convenient while iterating on routing or advisor logic, but it may not be ideal for a production deployment.

### 2) Shared model across tasks
The same runtime model is reused for:
- hydrology answer generation
- LLM-based intent routing fallback
- advisor slot filling

This keeps the architecture simple and avoids loading separate models.

### 3) Conservative handling of live forecasts
The app explicitly avoids pretending it has guaranteed real-time weather or flood forecast data.

---

## Suggested next improvements

Some practical next steps for this project would be:

1. Move `MODEL_PATH` to an environment variable
2. Add a `requirements.txt` or `environment.yml`
3. Add logging instead of print-based debugging
4. Add tests for routing, slot filling, and scoring
5. Add structured error handling around external APIs
6. Add a deployment script for Slurm or another HPC launcher
7. Add a configuration file for thresholds and scoring weights

---

## Repository layout

```text
.
├── app.py
├── advisor_flow.py
├── advisor_llm.py
├── advisor_tools.py
├── hydrology_chat.py
├── llm_router.py
├── model_runtime.py
└── router.py
```

---

## Summary

HydroUA-GPT is a single-chat Gradio app that combines a hydrology-focused LLM assistant with a lightweight build-site feasibility screening workflow. It uses hybrid routing, a shared model runtime, live geocoding and FEMA queries, local caching, and a simple map embed to keep both capabilities in one interface.
