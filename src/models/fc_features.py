#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

# wraps FactCheckcer as an sklearn transformer

from typing import Iterable, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ..fact_checker.checker import FactChecker

# NOTE: Docstrings were refined and reworded using ChatGPT for better clarity and consistency.
# NOTE: ChatGPT was also used to help optimize some functions for clarity and performance.

class FactCheckerFeaturizer(BaseEstimator, TransformerMixin):
    """
    Sklearn compatible transformer:
    input: Dataframe with columns 'text' and 'sender'
    output: numpy array with columns: 'fact_risk', 'brand_mismatch', 'tld_severity', 'url_obfuscation', 'claim_risk'
    """
    def __init__(self, use_sender=True):
        self.use_sender = use_sender
        self.fc = FactChecker()
        
    def fit(self, X: pd.DataFrame, y = None):
        """No fitting needed. Just return self."""
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform input DataFrame to feature array.
        """
    
        if isinstance(X, pd.DataFrame):
            texts = X['text'].tolist()
            senders = X['sender'].tolist() if self.use_sender else [None]*len(texts)
        elif isinstance(X, Iterable):
            texts = list(X)
            senders = [""]*len(texts)
        else:
            raise ValueError("Input should be a pandas DataFrame or an iterable of texts.")
        
        feats = np.zeros((len(texts), 6), dtype=float)
        for i, (t, s) in enumerate(zip(texts, senders)):
            res = self.fc.check(t, sender_email=s)
            feats[i, 0] = res.fact_risk
            feats[i, 1] = res.components.get('brand_mismatch', 0.0)
            feats[i, 2] = res.components.get('tld_risk', 0.0)
            feats[i, 3] = res.components.get('url_obfuscation', 0.0)
            feats[i, 4] = res.components.get('claim_risk', 0.0)
            feats[i, 5] = res.components.get('unverified_entity', 0.0)

        return feats