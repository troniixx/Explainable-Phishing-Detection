#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

import os, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# --- Ensure project root is on sys.path ---
from pathlib import Path
import sys

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent  # go one level up from app/ to repo root

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fact_checker.checker import extract_urls

from lime.lime_text import LimeTextExplainer
from sklearn.utils.validation import check_is_fitted

from typing import Any, Optional

try:
    from src.fact_checker.checker import FactChecker, FactCheckResult, Evidence
    FACT_CHECKER_AVAILABLE = True
except Exception:
    FACT_CHECKER_AVAILABLE = False
    FactChecker = None  # type: ignore
    Evidence = Any      # type: ignore


# ---- Auto add project root to pythonpath ----
THIS_FILE = Path(__file__).resolve()
for parent in THIS_FILE.parents:
    if (parent / "src").is_dir():
        REPO_ROOT = parent
        break
else:
    REPO_ROOT = THIS_FILE.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- Custom LIME explainer import ----
try:
    from src.explain.lime_explain import explain_with_lime
except Exception:
    # Fallback to inline LIME
    print("[WARNING] Failed to import custom LIME explainer, falling back to inline LIME.")
    
# ---- Streamlit Configs ----
st.set_page_config(
    page_title="Explainable Phishing Detector (LIME DEMO)",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_DIR_DEFAULT = "models/runs/20251022-124353/transformer_distilroberta-base"
MODEL_FILE_NAME = "pipeline.joblib"
FEEDBACK_PATH = Path("feedback/feedback.csv")
CLASS_NAMES = ["Benign", "Phishing"]

# ---- Helper functions ----
def is_hf_model_dir(p: Path) -> bool:
    # A HF folder typically has config.json + tokenizer files
    return (p / "config.json").exists() and ((p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists())

def best_hf_checkpoint(root: Path) -> Path:
    """
    Choose a best checkpoint if available:
    1) trainer_state.json -> 'best_model_checkpoint'
    2) latest 'checkpoint-*' folder
    3) fall back to root itself (root contains model.safetensors)
    """
    ts = root / "trainer_state.json"
    if ts.exists():
        try:
            import json
            data = json.load(open(ts, "r"))
            best = data.get("best_model_checkpoint")
            if best:
                bp = Path(best)
                return bp if bp.is_dir() else root
        except Exception:
            pass

    ckpts = sorted([d for d in root.glob("checkpoint-*") if d.is_dir()],
                    key=lambda d: int(d.name.split("-")[-1]))
    return ckpts[-1] if ckpts else root

class HFWrapper:
    """
    Minimal wrapper so the rest of the app can call predict_proba(df[['text','sender']]).
    We ignore 'sender' for the transformer (unless you later add a 2-field model).
    """
    def __init__(self, pipeline: TextClassificationPipeline, id2label: dict[int,str]): # type: ignore
        self.pipe = pipeline # type: ignore
        self.id2label = id2label
        # normalize positive label index
        # assume binary with labels like {0: 'LABEL_0', 1: 'LABEL_1'} or {0:'ham',1:'spam'}
        self.pos_idx = 1 if 1 in id2label else 0

    def predict_proba(self, df_2col: pd.DataFrame) -> np.ndarray:
        texts = df_2col["text"].astype(str).tolist()
        # return dicts with 'score' for positive class; we’ll compute both probs
        # Use return_all_scores=True to ensure both classes returned
        outputs = self.pipe(texts, truncation=True, return_all_scores=True)
        probs = []
        for row in outputs:
            # row is a list of dicts: [{'label': 'LABEL_0', 'score': p0}, {'label':'LABEL_1','score': p1}]
            # make sure ordering is consistent: sort by label id if possible
            # build map label->score
            m = {d["label"]: d["score"] for d in row} # type: ignore
            # heuristics to fetch p1
            # try label names in id2label; else fall back to lexicographic
            labels = [self.id2label.get(i, f"LABEL_{i}") for i in range(len(row))]
            try:
                p1 = float(m[labels[self.pos_idx]])
            except Exception:
                # fallback: try LABEL_1 or the last element
                p1 = float(m.get("LABEL_1", row[-1]["score"])) # type: ignore
            p0 = 1.0 - p1
            probs.append([p0, p1])
        return np.asarray(probs, dtype=float)


@st.cache_data(show_spinner="Loading model...")
def load_pipeline(model_dir: str):
    """
    Load either:
    - sklearn pipeline from <model_dir>/{pipeline.joblib,model.joblib}
    - or a Hugging Face transformer from <model_dir> (root or checkpoint-*).
    Returns an object with either:
    - sklearn pipeline (has predict_proba/decision_function), or
    - _HFWrapper (has predict_proba(df))
    """
    model_dir_path = Path(model_dir)
    if not model_dir_path.is_absolute():
        model_dir_path = (REPO_ROOT / model_dir_path).resolve()

    # 1) Try sklearn first (keeps existing behaviour)
    p = model_dir_path / MODEL_FILE_NAME
    if p.exists():
        pipe = load(p)
        try:
            check_is_fitted(pipe)
        except Exception:
            pass
        return pipe

    alt = model_dir_path / "model.joblib"
    if alt.exists():
        pipe = load(alt)
        try:
            check_is_fitted(pipe)
        except Exception:
            pass
        return pipe

    # 2) Try Hugging Face
    if not HF_AVAILABLE:
        raise FileNotFoundError(
            f"No sklearn model file found in {model_dir_path} and transformers not available."
        )

    # If user points to the parent folder (e.g., models/.../transformer_distilroberta-base),
    # pick the best checkpoint inside it. If they point to a checkpoint-* folder, use it as-is.
    hf_root = model_dir_path
    if (model_dir_path / "config.json").exists():
        hf_ckpt = model_dir_path
    else:
        hf_ckpt = best_hf_checkpoint(hf_root)

    if not is_hf_model_dir(hf_ckpt):
        raise FileNotFoundError(
            f"Could not find a Hugging Face checkpoint in {model_dir_path} "
            f"(looked at {hf_ckpt}). Expected config.json/tokenizer files."
        )

    # Device hint: MPS on Apple, CUDA else CPU
    device = -1
    if torch.backends.mps.is_available(): # type: ignore
        device = 0  # transformers treats mps as device 0
        dtype = torch.float32 # type: ignore
    elif torch.cuda.is_available(): # type: ignore
        device = 0
        dtype = torch.float16 # type: ignore
    else:
        dtype = torch.float32 # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(str(hf_ckpt)) # type: ignore
    model = AutoModelForSequenceClassification.from_pretrained(str(hf_ckpt), dtype=dtype) # type: ignore

    hf_pipe = TextClassificationPipeline( # type: ignore
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,            # we’ll ask for all scores
        return_all_scores=True, # ensures both classes
        batch_size=16,
    )

    id2label = model.config.id2label if hasattr(model.config, "id2label") else {0: "LABEL_0", 1: "LABEL_1"}
    return HFWrapper(hf_pipe, id2label)



def predict_proba_safe(pipeline, df_2col: pd.DataFrame):
    # Hugging Face wrapper
    if isinstance(pipeline, HFWrapper):
        return pipeline.predict_proba(df_2col)

    # sklearn branches (your existing code) ...
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(df_2col)
        if proba.ndim == 1 or proba.shape[1] == 1:
            p1 = proba.ravel().astype(float)
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        return proba
    
    elif hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function(df_2col), dtype=float)
        if scores.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-scores))
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        
        e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    
    preds = np.asarray(pipeline.predict(df_2col), dtype=int)
    p1 = preds.astype(float)
    p0 = 1.0 - p1
    
    return np.vstack([p0, p1]).T


def lime_explain_for_single_text(pipeline, text: str, sender: str, num_features: int = 10):
    """
    Generate LIME explanation for one email
    """
    explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=42)
    
    def classifier_fn(text_list):
        df = pd.DataFrame({
            "text": text_list,
            "sender": [sender] * len(text_list)
        })
        return predict_proba_safe(pipeline, df)

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=classifier_fn,
        num_features=num_features,
        num_samples=1000,
    )

    return exp

def save_feedback_row(row: dict):
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    
    if FEEDBACK_PATH.exists():
        df.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(FEEDBACK_PATH, mode="w", header=True, index=False)

@st.cache_resource
def get_fact_checker():
    if not FACT_CHECKER_AVAILABLE:
        return None
    try:
        return FactChecker() # type: ignore
    except Exception:
        return None
    
def level_to_label(level):
    if isinstance(level, (int, float, np.floating)):
        if level >= 0.8: return "Critical"
        if level >= 0.5: return "High"
        if level >= 0.2: return "Medium"
        return "Low"
    s = str(level).lower()
    if s in {"critical", "high", "medium", "low"}:
        return s.capitalize()
    return s

def severity_badge(label: str, level) -> str:
    lvl = level_to_label(level)
    color = { 
            "Critical": "#B71C1C",
            "High": "#E65100",
            "Medium": "#FBC02D",
            "Low": "#2E7D32"
    }.get(lvl, "#616161")
    
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:9999px;font-size:0.80rem;margin-right:6px">{label}: {lvl}</span>'

def render_factcheck_panel(res):
    badges = [
        severity_badge("Overall Risk", res.fact_risk),
        severity_badge("Brand Mismatch", res.components.get("brand_mismatch", 0)),
        severity_badge("TLD Risk", res.components.get("tld_risk", 0)),
        severity_badge("URL Obfuscation", res.components.get("url_obfuscation", 0)),
        severity_badge("Claim Risk", res.components.get("claim_risk", 0)),
        severity_badge("Unverified Entity", res.components.get("unverified_entity", 0)),
    ]
    st.markdown(" ".join(badges), unsafe_allow_html=True)

    if getattr(res, "message", None):
        st.markdown("**Summary**")
        for m in res.message:
            st.markdown(f"- {m}")

    if getattr(res, "evidence", None):
        rows = []
        for ev in res.evidence:
            flat = {"type": getattr(ev, "label", "")}
            det = getattr(ev, "details", {})  # <- fixed name
            if isinstance(det, dict):
                for k, v in det.items():
                    flat[k] = v
            rows.append(flat)
        if rows:
            st.markdown("**Evidence Details**")
            st.dataframe(pd.DataFrame(rows), width="stretch")


# ---- Controls ----

st.sidebar.title("Settings")
model_dir = st.sidebar.text_input("Model Directory", value=MODEL_DIR_DEFAULT)
num_features = st.sidebar.slider("LIME: number of features", min_value=5, max_value=20, value=10, step=1)
threshold = st.sidebar.slider("Decision threshold (phishing)", min_value=0.50, max_value=0.90, value=0.65, step=0.01)
st.sidebar.caption(f"Emails are flagged as phishing if P(Phishing) >= {threshold:.2f}")
st.sidebar.markdown("---")
st.sidebar.caption("Leave sender empty if unknown.")

run_factcheck = st.sidebar.checkbox("Run fact-checker panel", value=True)
if not FACT_CHECKER_AVAILABLE:
    st.sidebar.caption("Fact-checker: not available (module import failed)")
else:
    st.sidebar.caption("Fact-checker: available")

pipe, load_error = None, None
try:
    pipe = load_pipeline(model_dir)
except Exception as e:
    load_error = str(e)
    
# ---- Main App ----

st.title("Explainable Phishing Detector - LIME Prototype")

if load_error:
    st.error(f"Failed to load model pipeline: {load_error}")
    st.stop()
    
with st.form("email_form"):
    sender = st.text_input("Sender (optional)", placeholder="e.g. support@example.com").strip().lower()
    email_text = st.text_area("Email Text", height=220, placeholder="Paste the email body here (subject + content)...")
    submitted = st.form_submit_button("Analyze Email")
    
if submitted:
    if not email_text.strip():
        st.warning("Please paste an email text to analyze.")
        st.stop()

    df_one = pd.DataFrame([{"text" : email_text, "sender": sender or ""}])
    proba = predict_proba_safe(pipe, df_one)
    p_legit, p_phish = float(proba[0,0]), float(proba[0,1])
    
    fc_res: Optional[Any] = None
    
    # --- Fact-Checker panel ---
    if run_factcheck:
        fc = get_fact_checker()
        st.subheader("Fact-Check Findings")
        if fc is None:
            st.info("Fact-checker not available or failed to initialize.")
        else:
            try:
                fc_res = fc.check(email_text, sender_email=(sender or None))
                render_factcheck_panel(fc_res)
            except Exception as e:
                st.warning(f"Fact-checking failed: {e}")
                fc_res = None

    # --- No-URL + low-claim dampener (−0.10) ---
    adjust_notes = []
    urls = extract_urls(email_text)
    no_urls = (len(urls) == 0)

    # Prefer fact-checker’s claim_risk if available; else fallback keyword heuristic
    if run_factcheck and fc_res is not None:
        low_claim = (fc_res.components.get("claim_risk", 0.0) < 0.34)
    else:
        # very light fallback: only treat as “high-claim” if strong credentials/payment cues appear
        low_claim_keywords = [
            "verify your account", "reset your password", "confirm your identity",
            "log in to your account", "update your billing", "gift card",
            "urgent wire", "bank transfer immediately", "bitcoin wallet"
        ]
        txt = email_text.lower()
        low_claim = not any(k in txt for k in low_claim_keywords)

    # Apply dampener only when there's no URL and claims are low-risk
    final_p_phish = p_phish
    if no_urls and low_claim:
        final_p_phish = max(0.0, final_p_phish - 0.10)
        adjust_notes.append("Applied no-URL/low-claim dampener (-0.10)")

    # --- Decision with user threshold ---
    pred_is_phish = (final_p_phish >= threshold)
    pred_label = CLASS_NAMES[1] if pred_is_phish else CLASS_NAMES[0]
    conf = final_p_phish if pred_is_phish else (1.0 - final_p_phish)

    # --- Display ---
    st.subheader("Prediction Result")
    st.markdown(
        f"**P(phishing):** raw `{p_phish:.3f}` → adjusted `{final_p_phish:.3f}`  "
        f"(threshold `{threshold:.2f}`)"
    )
    st.markdown(f"**Predicted Class:** `{pred_label}`")
    st.markdown(f"**Confidence:** `{conf*100:.2f}%`")
    if adjust_notes:
        st.caption("Adjustments: " + " • ".join(adjust_notes))
    
    with st.spinner("Generating LIME explanation..."):
        try:
            exp = lime_explain_for_single_text(pipe, email_text, sender or "", num_features)
            st.subheader("Top Contributing Features")
            as_list = exp.as_list()
            
            if as_list:
                df_feat = pd.DataFrame(as_list, columns=["token/feature", "weight"])
                st.dataframe(df_feat, width="stretch")
            try:
                import streamlit.components.v1 as components
                components.html(exp.as_html(), height=500, scrolling=True)
            except Exception:
                st.info("Failed to render full LIME HTML explanation.")
        except Exception as e:
            st.error(f"Failed to generate LIME explanation: {e}")
    
    # Feedback section
    st.subheader("Feedback (Optional)")
    st.caption("Do you agree with the prediction? Your feedback helps us improve the model.")
    agree = st.radio("I agree with the prediction:", options=["Yes", "No"], index=0, horizontal=True)
    user_label, user_comment = "", ""
    
    if agree == "No":
        user_label = st.selectbox("What do you think the correct label should be?", options=CLASS_NAMES, index=0)
        user_comment = st.text_area("Additional comments (optional):", height=80, placeholder="Your comments...")
        
    if st.button("Submit Feedback"):
        row = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sender": sender or "",
            "text": email_text,
            "pred_label": pred_label,
            "p_phishing_raw": round(p_phish, 6),
            "p_phishing_adj": round(final_p_phish, 6),
            "p_legitimate": round(1.0 - final_p_phish, 6),  # from adjusted prob
            "user_agrees": (agree == "Yes"),
            "user_label": user_label,
            "user_comment": user_comment,
            "model_dir": str(model_dir),
            "lime_num_features": num_features,
        }
        
        try:
            save_feedback_row(row)
            st.success(f"Feedback saved to {FEEDBACK_PATH}")
        except Exception as e:
            st.error(f"Could not save feedback: {e}")
            
# --- Footer ---
st.markdown("---")
st.caption(
    "© 2025 Mert Erol, University of Zurich. "
    "This is a prototype application for demonstration purposes only. "
    "Not intended for production use.\n"
    "Privacy notice: inputs and feedback are stored locally (feedback/feedback.csv). "
    "No external transmission occurs. Avoid pasting sensitive personal data."
)