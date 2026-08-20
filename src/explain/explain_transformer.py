#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

"""
Generate a LIME explanation for a Hugging Face transformer text classifier.
This focuses on local explanations (single text) since global SHAP for transformers
is more involved and not wired here.
"""
import argparse
from pathlib import Path
from typing import List
import numpy as np
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def parse_args():
    ap = argparse.ArgumentParser(description="LIME explanation for a HF transformer classifier.")
    ap.add_argument("--model_dir", required=True, help="Directory with saved model/tokenizer.")
    ap.add_argument("--out_dir", required=True, help="Directory to save outputs.")
    ap.add_argument("--text", required=True, help="Text to explain.")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lime_features", type=int, default=10)
    ap.add_argument("--num_samples", type=int, default=1000,
                     help="LIME perturbation samples (LIME's own default is 5000, which is slow for large transformers).")
    return ap.parse_args()


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    def proba_fn(texts: List[str]):
        with torch.no_grad():
            enc = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=args.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits.detach().cpu().numpy()
            if logits.shape[1] == 1:
                # Single-logit: map to 2-class probabilities
                p1 = 1.0 / (1.0 + np.exp(-logits[:, 0]))
                p0 = 1.0 - p1
                return np.vstack([p0, p1]).T
            probs = softmax(logits, axis=1)
            # ensure 2 columns ordering [p0, p1]
            if probs.shape[1] == 2:
                return probs
            # fallback: if >2, return as-is (LIME can handle multi-class)
            return probs

    explainer = LimeTextExplainer(class_names=["ham", "spam"], random_state=42)
    exp = explainer.explain_instance(
        text_instance=args.text,
        classifier_fn=proba_fn,
        num_features=args.lime_features,
        num_samples=args.num_samples,
    )

    html_path = out_dir / "lime_explanation.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(exp.as_html())
    print(f"[LIME] explanation saved to {html_path}")
    print("[LIME] Top features:", exp.as_list())


if __name__ == "__main__":
    main()
