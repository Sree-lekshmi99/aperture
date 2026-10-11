"""Run with: streamlit run dashboard.py"""
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
st.set_page_config(page_title="Aperture", page_icon="🧭", layout="wide")
st.title("Aperture — adaptive agent context harness")
st.caption("AEGIS traces · MongoDB Atlas memory and evaluation · candidate context policies")
uri = os.getenv("MONGODB_URI", "")
if not uri or "YOUR_" in uri:
    st.error("Set MONGODB_URI in .env first.")
    st.stop()
try:
    db = MongoClient(uri, serverSelectionTimeoutMS=5000)["aperture"]
    db.command("ping")
except Exception as exc:
    st.error(f"Could not connect to MongoDB Atlas: {exc}")
    st.stop()

policies = list(db.policies.find({}, {"_id": 1, "strategy": 1, "n": 1}))
st.subheader("Policies learned from validation records")
if policies:
    st.dataframe(pd.DataFrame(policies), use_container_width=True, hide_index=True)
else:
    st.info("Run `python aperture.py tune --limit 8` first.")

records = list(db.results.find({"split": "test"}, {"_id": 0}).limit(1000))
st.subheader("Held-out AEGIS evaluation")
if records:
    df = pd.DataFrame(records)
    summary = df.groupby("strategy", as_index=False).agg(
        runs=("trace_id", "count"), mean_f1=("f1", "mean"),
        exact_rate=("exact_match", "mean"), mean_input_tokens=("prompt_tokens", "mean")
    )
    st.dataframe(summary.round(3), use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        st.caption("Faulty-agent identification F1 (higher is better)")
        st.bar_chart(summary.set_index("strategy")["mean_f1"])
    with right:
        st.caption("Input tokens per diagnosis (lower is better)")
        st.bar_chart(summary.set_index("strategy")["mean_input_tokens"])
    st.subheader("Inspect one diagnosis")
    options = [f"{r['trace_id']} [{r['strategy']}]" for r in records]
    idx = st.selectbox("Execution", range(len(records)), format_func=lambda x: options[x])
    selected = records[idx]
    a, b, c = st.columns(3)
    a.metric("Faulty-agent F1", round(selected.get("f1", 0), 2))
    b.metric("Input tokens", selected.get("prompt_tokens", 0))
    c.metric("Exact match", "Yes" if selected.get("exact_match") else "No")
    st.write("**Predicted agents:**", selected.get("predicted_agents"))
    st.write("**Reference agents:**", selected.get("true_agents"))
    st.write("**Agent's explanation:**", selected.get("explanation"))
else:
    st.info("Run `python aperture.py compare --limit 8` to generate test results.")
