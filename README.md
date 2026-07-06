# HydroSite-GPT

**HydroSite-GPT** is a multi-agent hydrology and build-site feasibility assistant that combines a domain-tuned hydrology language model with deterministic geospatial, flood, rainfall, terrain, and vulnerability tools. The system is designed for screening-level site assessment: a user can ask general hydrology questions or evaluate whether a proposed building site has flood, rainfall, terrain-drainage, and socio-economic-infrastructure vulnerability concerns.

The application is built with **Gradio** for the user interface and **LangGraph** for the multi-agent workflow orchestration.

---

## What the system does

HydroSite-GPT supports two main modes:

1. **Hydrology chat**
   - Answers hydrology and water-resources questions.
   - Streams responses from the local HydroUA-GPT model.
   - Handles concepts such as runoff, rainfall, soil moisture, floods, drought, watersheds, groundwater, and stormwater.

2. **Multi-agent build-site feasibility screening**
   - Accepts a building type and location/address.
   - Geocodes the site.
   - Retrieves flood hazard, rainfall frequency, terrain/drainage context, and vulnerability data.
   - Computes a composite screening-level site risk score.
   - Produces an explainable report with specialist-agent outputs, verifier checks, recommendations, and map location.

Example query:

```text
Can I build a hospital at 302 Reed St, Tuscaloosa AL?
```

---

## Project architecture

```text
HydroSite-GPT/
├── app.py                         # Gradio UI and chat/map state handling
├── advisor_graph.py               # Multi-agent LangGraph workflow
├── advisor_utils.py               # Shared parsing and formatting helpers
├── advisor_tools.py               # Geocoding, FEMA, NOAA, SEIV, terrain, scoring tools
├── advisor_llm.py                 # Slot extraction and LLM-based interpretations
├── router.py                      # Fast rule-based intent router
├── llm_router.py                  # LLM fallback router
├── hydrology_chat.py              # Streaming hydrology chat generation
├── model_runtime.py               # Shared model/tokenizer loader
├── requirements_langgraph.txt     # LangGraph dependencies
└── README.md
```

---

## Multi-agent workflow

The build-site feasibility path is implemented as an explicit multi-agent LangGraph workflow.

```text
User message
   ↓
Supervisor/Router Agent
   ↓
Planner Agent
   ↓
Geocoder Agent
   ↓
FEMA Flood Hazard Agent
   ↓
Rainfall/Hydrology Agent
   ↓
Terrain/Drainage Agent
   ↓
SEIV Vulnerability Agent
   ↓
Scoring/Synthesis Agent
   ↓
Critic/Verifier Agent
   ↓
Report Writer Agent
   ↓
Final screening report + map update
```

### Agent roles

| Agent | Purpose |
|---|---|
| **Supervisor/Router Agent** | Classifies the user request as hydrology, advisor, or off-topic. Uses rules first and LLM fallback only when needed. |
| **Planner Agent** | Extracts building type and address, resolves follow-up context, and decides which specialist agents should run. |
| **Geocoder Agent** | Converts the site address into latitude, longitude, and Census GEOID. |
| **FEMA Flood Hazard Agent** | Looks up mapped FEMA NFHL flood zone and Special Flood Hazard Area information. |
| **Rainfall/Hydrology Agent** | Retrieves NOAA Atlas 14 rainfall frequency values and interprets extreme rainfall implications. |
| **Terrain/Drainage Agent** | Samples local elevation context using USGS EPQS and screens for flat, convergent, or depressional terrain. |
| **SEIV Vulnerability Agent** | Looks up block-level Socio-Economic-Infrastructure Vulnerability using a local parquet dataset. |
| **Scoring/Synthesis Agent** | Combines FEMA, NOAA, terrain, SEIV, and building criticality into a composite site risk score. |
| **Critic/Verifier Agent** | Checks which evidence sources succeeded, flags missing components, and prevents overclaiming. |
| **Report Writer Agent** | Produces the final user-facing screening report with agent trace, limitations, and recommendations. |

---

## Data and external services

The advisor workflow uses the following data/services:

- **U.S. Census Geocoder** for address-to-point geocoding and Census GEOID extraction.
- **FEMA NFHL** for flood hazard zone lookup.
- **NOAA Atlas 14** for precipitation frequency estimates.
- **USGS EPQS / 3DEP** for local elevation samples.
- **Local SEIV parquet dataset** for block-level vulnerability lookup.

The SEIV dataset path is configured in `advisor_tools.py` through the `VULN_PARQUET_PATH` environment variable.

Default:

```bash
/icebox/data/shares/mh2/mkarimiziarani/agent/data/vulnerability.parquet
```

Override it with:

```bash
export VULN_PARQUET_PATH=/path/to/vulnerability.parquet
```

---

## Model configuration

The local hydrology model is loaded in `model_runtime.py`.

Default model path:

```python
MODEL_PATH = "/icebox/data/shares/mh2/shassan6/gradio/models/hydrology_aqua_llm_strict_11B"
```

Update this path in `model_runtime.py` if the model is stored elsewhere.

The model is shared by:

- `hydrology_chat.py`
- `llm_router.py`
- `advisor_llm.py`

This avoids loading multiple copies of the same model.

---

## Installation

Create and activate a Python environment first.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install LangGraph dependencies:

```bash
pip install -r requirements_langgraph.txt
```

You may also need the project runtime dependencies if they are not already installed in your environment:

```bash
pip install gradio torch transformers peft duckdb certifi
```

Depending on your model and GPU setup, install the correct PyTorch/CUDA build for your system.

---

## Running the app

From the project folder:

```bash
python app.py
```

By default, the app launches on:

```text
0.0.0.0:7860
```

You can override the host and port:

```bash
export HOST=0.0.0.0
export PORT=7860
python app.py
```

The Gradio interface includes:

- A chat panel for hydrology and site-screening questions.
- A map panel that updates after successful geocoding.

---

## Map configuration

The map iframe base URL is configured with `MAP_BASE_URL`.

Default:

```bash
https://homes-design-brave-anna.trycloudflare.com/
```

Override it with:

```bash
export MAP_BASE_URL=https://your-map-app-url.example.com/
```

The app appends latitude, longitude, and zoom parameters:

```text
?lat=<lat>&lng=<lon>&z=<zoom>
```

---

## Example prompts

Hydrology:

```text
What is runoff and how does soil moisture affect it?
```

```text
Explain the difference between infiltration, percolation, and groundwater recharge.
```

Build-site feasibility:

```text
Can I build a warehouse at 123 Main St, Tuscaloosa AL?
```

```text
Screen a hospital site at 302 Reed St, Tuscaloosa AL.
```

Follow-up flow:

```text
Can I build a hospital?
```

The system will ask for the missing location.

```text
302 Reed St, Tuscaloosa AL
```

The planner will reuse the building type from the previous turn.

---

## Output report

For site-screening requests, the final report includes:

- Geocoded coordinates and Census GEOID.
- Planner decision.
- FEMA flood hazard analysis.
- NOAA Atlas 14 rainfall intensity analysis.
- Terrain and local drainage screening.
- SEIV vulnerability lookup.
- Composite site risk score.
- Main risk drivers.
- Integrated interpretation.
- Critic/verifier checks and warnings.
- Recommended next steps.
- Agent trace.
- Screening-level limitations.

---

## Important limitations

HydroSite-GPT is a **screening-level decision-support tool**, not a final engineering, permitting, or legal determination.

It does **not** replace:

- Full hydrologic and hydraulic modeling.
- Detailed DEM-based flow accumulation.
- Storm sewer capacity modeling.
- Site grading and drainage design.
- FEMA map panel verification by a qualified professional.
- Local floodplain ordinance review.
- Engineering or permitting review.

The system should use conservative language and avoid making final build/no-build claims unless supported by direct prohibitive evidence.

---

## Development notes

### Intent routing

Routing is intentionally hybrid:

1. `router.py` handles fast deterministic rules.
2. `llm_router.py` can upgrade uncertain/off-topic classifications to hydrology or advisor.
3. The LangGraph supervisor uses the final route to select the correct path.

### Hydrology streaming

Hydrology chat still streams through `hydrology_chat.py`. The LangGraph workflow returns `stream_required=True`, and `app.py` delegates streaming to the hydrology chat module.


