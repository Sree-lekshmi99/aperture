# Aperture — a five-hour AEGIS harness project

**What it is:** a small investigator agent that labels faulty agents in AEGIS multi-agent execution traces, plus a **Python harness** that compares three context-selection strategies and records a better-performing strategy in MongoDB Atlas. The existing agent model is unchanged. This is a proof of concept, not a production system.

**What it is not:** a reproduction of Raindrop's proprietary production traces, a fix for live multi-agent executions, or a claim that adaptive retrieval did not exist before.

## 1. Get ready

- Install Python 3.11+ and VS Code.
- Obtain the **special Atlas Hackathon Sandbox invitation from MongoDB** and create your cluster through that link, as required by the guide.
- Obtain an OpenAI **API** key (separately billed from ChatGPT subscriptions), or adapt the single `ask_investigator` call to your available model API.

## 2. Create a virtual environment

Mac/Linux:
```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 3. Download AEGIS

```sh
python download_data.py test
python inspect_data.py --file data/test.jsonl --count 3
python download_data.py val
```

Source: https://huggingface.co/datasets/Fancylalala/AEGIS

- `test.jsonl`: official held-out benchmark, approximately 15.6 MB.
- `val.jsonl`: official validation data, approximately 43.6 MB.
- We intentionally skip the 173 MB training split.

`inspect_data.py` is the first sanity check: verify `input.conversation_history` is a list of agent events and `output.faulty_agents` contains evaluation labels. When checking examples by hand it is okay to inspect labels, **but never supply those labels to the model**.

## 4. Connect to the required MongoDB Atlas sandbox

1. Follow the emailed invitation and create the project and cluster **there**. Don't create an unrelated cluster.
2. Under **Database Access**, add a database user with read/write access to your hackathon database.
3. Under **Network Access**, allow your current IP address.
4. Open your cluster's **Connect** flow and copy the **Drivers → Python** URI.
5. Copy `.env.example` to `.env` and fill in the actual credentials. URL-encode reserved characters in URI passwords (or make a username/password without reserved characters). Never commit `.env`.

Mac/Linux:
```sh
cp .env.example .env
```
Windows PowerShell:
```powershell
Copy-Item .env.example .env
```

Then edit `.env` in VS Code:
```ini
MONGODB_URI=mongodb+srv://...
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## 5. Ingest a small sample into MongoDB

```sh
python aperture.py ingest --split val --limit 40
python aperture.py ingest --split test --limit 20
```

The MongoDB database `aperture` has four collections:

- `traces`: task, agent conversation, public metadata, and split.
- `labels`: true faulty agents, **never exposed to the investigator**.
- `results`: each investigator prediction, explanation, token usage, score.
- `policies`: current stored policy per task type.

## 6. Run one investigator

```sh
python aperture.py run --split val --limit 1 --strategy large
python aperture.py run --split val --limit 1 --strategy outline
```

Confirm that you see an agent prediction, the actual agent labels, F1, and input-token count. The first run makes calls to a model API and may incur charges; later identical calls are cached in MongoDB.

## 7. Learn a context policy

```sh
python aperture.py tune --limit 8
```

Three candidate context strategies are evaluated on *validation only*:

1. **large**: the full recorded conversation up to an explicit 25,000-character cap.
2. **outline**: short snippet from each event.
3. **focused**: a short overview of early events plus greater detail for the last four.

For each validation task, `output.faulty_agents` is read **only after prediction** to score the response. The controller chooses the best strategy using `mean(F1 - 0.02 * prompt_tokens/1000)` and stores it as a durable policy in `policies`. If a task type has at least three validation examples it gets a task-specific choice; otherwise the global policy applies.

## 8. Compare on held-out examples

```sh
python aperture.py compare --limit 8
```

This reports F1, exact-match rate, and prompt token usage for all three candidate strategies and the **adaptive selection**. Same task records, same model, different context policies. Held-out test results are for **reporting only**, never policy tuning.

Caution: AEGIS injects several errors into many trajectories; this prototype measures *faulty-agent attribution* rather than correctly classifying every failure-mode code. For stronger evaluation, add error-mode metrics and separately assess whether the quoted evidence actually supports each diagnosis.

## 9. Run the dashboard

```sh
streamlit run dashboard.py
```





### Attribution and ethical demo claims

Cite the AEGIS dataset: https://huggingface.co/datasets/Fancylalala/AEGIS.

