#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

"""
Predict with a fine-tuned Hugging Face transformer on a CSV.
Outputs a CSV with columns: original columns + pred (0/1) + score (P[class=1]).
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    ap = argparse.ArgumentParser(description="Predict with a fine-tuned transformer on a CSV.")
    ap.add_argument("--model_dir", required=True, help="Directory containing the saved HF model/tokenizer.")
    ap.add_argument("--input_csv", required=True, help="Input CSV with at least a 'text' column.")
    ap.add_argument("--out_csv", required=True, help="Output CSV path.")
    ap.add_argument("--max_len", type=int, default=256, help="Max sequence length for tokenization.")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    return ap.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def main():
    args = parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    df = pd.read_csv(args.input_csv)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    texts = df["text"].astype(str).fillna("").tolist()

    all_p1 = []
    all_pred = []

    bs = max(1, int(args.batch_size))
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=args.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits.detach().cpu().numpy()

            if logits.shape[1] == 1:
                # Sigmoid for binary single-logit case
                p1 = sigmoid(logits[:, 0])
            else:
                # Softmax, take probability of label 1
                probs = softmax(logits, axis=1)
                # assume index 1 is the positive class
                p1 = probs[:, 1]

            preds = (p1 >= 0.5).astype(int)
            all_p1.append(p1)
            all_pred.append(preds)

    p1 = np.concatenate(all_p1) if all_p1 else np.array([])
    pred = np.concatenate(all_pred) if all_pred else np.array([])

    out_df = df.copy()
    out_df["pred"] = pred
    out_df["score"] = p1

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Predictions saved to {args.out_csv}")


if __name__ == "__main__":
    main()
