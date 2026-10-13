# Aperture

Follow an agent investigation from the conversation log to the evidence behind its diagnosis.

Aperture is a Python harness built for the MongoDB Harness Engineering Hackathon. It uses AEGIS multi-agent execution traces to compare how an investigator works with different amounts of context. MongoDB Atlas keeps the traces, received evidence, model responses, and evaluations together so each investigation can be replayed in Streamlit.

![Aperture's Streamlit replay showing the three S4 context strategies](docs/images/s4-replay.jpg)

*The recorded S4 comparison in the Streamlit dashboard.*

## What the investigator reads

A trace is a conversation log: the original task, messages from each agent, and any recorded tool activity. Each event has an ID and an author. The investigator reads that log and identifies agents whose actions appear faulty, with citations for each claim.

The targeted strategy starts with a short preview of each event. Clipped previews show how much text was left out, such as `[1,180 characters omitted]`. When previews are clipped, the investigator must choose events to expand before giving a final diagnosis. The harness fetches those events and fits the evidence into the request budget.

There are two logs worth knowing:

- **Execution trace:** what happened during the original multi-agent task.
- **Received-evidence log:** the exact text delivered to the investigator for a particular call, after clipping and budget checks.

The second log makes the citations inspectable. The investigator returns a chunk ID and a span ID; application code attaches the exact text from that call's received-evidence log. The provenance gate checks those references before accepting the diagnosis. Its rules live separately from retrieval policies.

**Provenance verified** means the citation points to evidence the investigator received. Attribution is scored separately against AEGIS labels, which stay outside the investigator's input.

## Three ways to supply context

| Strategy | What the investigator receives |
| --- | --- |
| Capped-large | Conversation text up to the configured context cap. |
| Short context | A 180-character preview of each event. |
| Mandatory first expansion | The short overview, followed by investigator-selected event expansions. |

The recorded comparisons use the same `gpt-4o-mini` model, temperature of `0`, citation protocol, and scoring rules across all three arms.

Targeted retrieval allows two rounds, with up to five events per round and 4,000 source characters per expanded event. The harness enforces a 12,000-token request budget and a 10,000-token evidence budget. Packing includes instructions, IDs, tool framing, and retained history. Expanded text replaces the corresponding preview, and the run records whether each selected event was delivered fully, partially, or deferred.

## Walk through the replay

The dashboard opens as a **Recorded experiment replay** and reads saved results from the official MongoDB Atlas Hackathon Sandbox.

### S4: follow the evidence

Start with **S4 · Evidence recovered**. The three cards show predictions, attribution F1, exact match, total tokens, and provenance decisions.

Below the cards, follow the short overview into the investigator's selected expansions. In this recording, it requested events **2, 4, 6, 10, and 12**, and received all five. The expanded evidence included both tracked formulas from events 10 and 12. The targeted diagnosis identified **Assistant1 and Assistant3**, matching the reference agent set for this case.

Open **Diagnosis & clickable citations** to read the exact received text behind a claim. **Run, code & call ledger** shows the recorded versions and token counts for every selection and diagnosis call.

### L2: inspect the run history

Choose **L2 · Limits & cost** to inspect the original recording and the later packing revision side by side. Each version keeps its own predictions, budget details, gate decisions, and token accounting. The original baseline results remain available alongside the revised targeted run.

### Gate: try a reference check

Choose **Gate · Injected test**, then click **Run injected invalid-reference test**. The demo changes one span ID in a copy of a recorded citation and runs the actual provenance gate. The page shows the original reference, the injected reference, and the rejection reason.

This test runs locally. The replay uses stored model responses, and the recorded evidence stays intact.

## Run locally

Use Python 3.11 or later.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

Create a local `.env` using [`.env.example`](.env.example) as the template. Keep an existing `.env` if you already have one configured.

```dotenv
MONGODB_URI=mongodb+srv://<username>:<password>@<sandbox-host>/?appName=Aperture
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_MODEL=gpt-4o-mini
```

Use the connection string from the **official MongoDB Atlas Hackathon Sandbox**. Add your current IP under Atlas Network Access and use the Python connection string from the cluster's **Connect → Drivers** page.

The replay needs access to the saved experiment records in the `aperture` database. `OPENAI_API_KEY` is used when running investigations; opening the replay reads the existing recordings.

```sh
python -m streamlit run dashboard.py
```

Open [localhost:8501](http://localhost:8501).

Credentials stay in the local `.env`. The repository's `.gitignore` covers environment files, keys, virtual environments, and downloaded datasets.

## What Atlas stores

| Collections | Contents |
| --- | --- |
| `traces`, `labels` | Execution traces and separately stored evaluation labels. |
| `citation_studies`, `citation_policies` | Experiment settings, versions, code hashes, and retrieval policies. |
| `citation_sources`, `citation_deliveries` | Source events and the exact evidence delivered, including citation spans. |
| `citation_runs`, `citation_calls`, `citation_retrievals` | Run history, model responses, selected events, delivery details, and token usage. |
| `citation_results`, `citation_evaluations` | Accepted diagnoses and attribution scores. |
| `citation_tests`, `citation_reports` | Verification results and saved comparison reports. |

These records preserve the harness's experiment history across sessions.

## Project files

```text
aperture.py                Original harness, ingestion, and scoring
targeted_retrieval.py      Event selection protocol and context packing
provenance_gate.py         Citation checks against received evidence
run_s4_targeted.py         Versioned S4 experiment runner
dashboard.py              Streamlit recorded replay
download_data.py           AEGIS download helper
inspect_data.py            Dataset inspection
test_targeted_retrieval.py Citation and retrieval tests
test_context_packing.py    Context-budget regression tests
reports/                  Saved experiment reports and run artifacts
```

Saved reports: [S4](reports/s4-mandatory-expansion-spans-20261011-v3/report.html) · [L2 original](reports/l2-frozen-mandatory-expansion-20261011-v1/report.html) · [L2 packing revision](reports/l2-packing-repair-20261011-v4/report.html).

## Dataset and stack

The recordings use validation examples from [AEGIS on Hugging Face](https://huggingface.co/datasets/Fancylalala/AEGIS). To download the validation data for local inspection:

```sh
python download_data.py val
```

Built with Python, MongoDB Atlas, the OpenAI API, Streamlit, pandas, and tiktoken.
