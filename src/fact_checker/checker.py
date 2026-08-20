#!/usr/bin/env python3
# Copyright (c) 2025 Mert Erol, University of Zurich
# Licensed under the Academic and Educational Use License (AEUL) – see LICENSE file for details.

import re, json, math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlparse, unquote
import idna
import tldextract
from transformers import pipeline
from rapidfuzz import fuzz, process

# NOTE: Docstrings were refined and reworded using ChatGPT for better clarity and consistency.
# NOTE: ChatGPT was also used to help optimize some functions for clarity and performance.

# ---------- Utilities ----------

# NOTE: This regex is intentionally simple and fast. It will catch most HTTP(S) URLs
# but is not a fully RFC-3986 compliant parser (e.g., it may miss edge cases or
# accept some malformed inputs).
URL_REGEX = re.compile(
    r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b"
    r"(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)

def extract_urls(text: str) -> List[str]:
    """Extract de-duplicated HTTP(S) URLs from a block of text.

    This function uses a pragmatic URL regex to find links, then:
        - Ensures each URL has a scheme (http/https). (Regex already enforces it.)
        - Strips common trailing punctuation like ``) . , ; ! ? " '`` that often
        stick to URLs in prose.
        - Preserves first occurrence order while removing duplicates.

    Args:
        text: Arbitrary text to scan.

    Returns:
        A list of unique URLs in the order they first appear.

    Examples:
        >>> extract_urls("see https://example.com, also http://a.co/path.")
        ['https://example.com', 'http://a.co/path']
    """
    urls = URL_REGEX.findall(text or "")

    out: List[str] = []
    for u in urls:
        # Regex already matches http/https, but keep this guard in case when
        # changing URL_REGEX or feeding pre-parsed strings.
        if not u.lower().startswith(("http://", "https://")):
            u = "http://" + u

        # Trim stray punctuation that often trails URLs in natural text.
        u = u.strip(").,;!?\"'")
        out.append(u)

    # De-duplicate while preserving order.
    return list(dict.fromkeys(out))


def tld_extract(domain: str) -> str:
    """Return the registered domain (eTLD+1) component of a host.

    Uses tldextract to robustly split a host into subdomain, domain, and suffix.
    If the input is not recognized as a registered domain (e.g., localhost, IP),
    the original lowercased input is returned.

    Args:
        domain: A hostname (may include subdomains).

    Returns:
        The registered domain (e.g., ``example.co.uk``), or the lowercased input
        if no registered domain is found.

    Examples:
        >>> tld_extract("a.b.example.co.uk")
        'example.co.uk'
        >>> tld_extract("localhost")
        'localhost'
    """
    ext = tldextract.extract(domain)
    if not ext.top_domain_under_public_suffix:
        return domain.lower()
    return ext.top_domain_under_public_suffix.lower()


def domain_from_url(url: str) -> Optional[str]:
    """Extract and normalize the hostname from a URL.

    - Uses ``urllib.parse.urlparse`` to obtain the hostname.
    - Decodes Punycode (``xn--``) into Unicode using ``idna.decode``.
    - Lowercases the result.

    Args:
        url: A URL string (e.g., ``https://sub.xn--d1acpjx3f.xn--p1ai/path``).

    Returns:
        The decoded, lowercased hostname, or ``None`` if parsing fails or the
        URL has no hostname.

    Examples:
        >>> domain_from_url("https://www.Example.com/path")
        'www.example.com'
    """
    try:
        host = urlparse(url).hostname
        if not host:
            return None
        # Decode Punycode if present.
        decoded = idna.decode(host) if host.startswith("xn--") else host
        return decoded.lower()
    except Exception:
        return None


def count_subdomains(domain: str) -> int:
    """Count the number of subdomain labels in a hostname.

    For ``a.b.example.co.uk`` the subdomain part is ``a.b`` → 2 labels.

    Args:
        domain: A hostname string.

    Returns:
        Number of subdomain labels (0 if none).

    Examples:
        >>> count_subdomains("a.b.example.com")
        2
        >>> count_subdomains("example.com")
        0
    """
    ext = tldextract.extract(domain)
    if ext.subdomain:
        return len(ext.subdomain.split("."))
    return 0


def is_ip_literal(domain: str) -> bool:
    """Heuristically check whether a hostname is an IPv4 literal.

    This is a simple pattern check and does **not** validate octet ranges (0–255).

    Args:
        domain: Host portion of a URL or a bare string.

    Returns:
        True if the string looks like ``A.B.C.D`` where each part is 1–3 digits,
        otherwise False.

    Examples:
        >>> is_ip_literal("192.168.0.1")
        True
        >>> is_ip_literal("example.com")
        False
    """
    return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))


def has_hex_encoding(url: str) -> bool:
    """Detect URL-encoded (percent-encoded) bytes within a URL string.

    Checks for common encodings such as ``%3A`` (``:``) and ``%2F`` (``/``),
    as well as any ``%HH`` hex pattern.

    Args:
        url: The URL to inspect.

    Returns:
        True if any percent-encoded bytes are present, else False.

    Examples:
        >>> has_hex_encoding("https://ex.com/a%2Fb")
        True
        >>> has_hex_encoding("https://ex.com/path")
        False
    """
    lower = url.lower()
    return "%3a" in lower or "%2f" in lower or re.search(r"%[0-9a-fA-F]{2}", url) is not None


def has_at_symbol(url: str) -> bool:
    """Check whether the URL's network location contains an '@' sign.

    URLs of the form ``scheme://user@host`` can be misleading to users
    (everything before '@' may be interpreted as credentials or noise).

    Args:
        url: The URL to inspect.

    Returns:
        True if ``@`` appears in the netloc (authority) component, else False.

    Examples:
        >>> has_at_symbol("http://user@evil.com")
        True
        >>> has_at_symbol("http://example.com")
        False
    """
    return "@" in urlparse(url).netloc

# ---------- Knowledge Base Loaders ----------

def load_brands(path="data/fact_checking/brands.csv") -> Dict[str, str]:
    """Load brands from CSV file

    Args:
        path (str): Path to CSV file

    Returns:
        Dict[str, str]: Brands
    """
    base = {}
    
    try:
        import csv
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                base[row["brand"].strip().lower()] = row["official_domain"].strip().lower()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    return base

def load_tld_risks(path="data/fact_checking/clean_tlds.csv") -> Dict[str, str]:
    """Load TLD risks from CSV file

    Args:
        path (str): Path to CSV file

    Returns:
        Dict[str, str]: TLD risks
    """
    risks = {}
    
    try:
        import csv
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                risks[row["metadata_tld"].strip().lower()] = row["metadata_severity"].strip().lower()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    return risks

def load_phrases(path="data/fact_checking/phrases.json") -> Dict[str, List[str]]:
    """Load phrases from JSON file

    Args:
        path (str): Path to JSON file

    Returns:
        Dict[str, List[str]: Phrases
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
SEVERITY_SCORES = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
    "critical": 1.0
}

@dataclass
class Evidence:
    label: str
    details: Dict[str, Any]
    
@dataclass
class FactCheckResult:
    fact_risk: float
    components: Dict[str, float]
    evidence: List[Evidence]
    message: List[str]
    
def etld1(domain: str) -> str:
    """Extract eTLD+1 from a domain, or return the original domain if not possible.

    Args:
        domain (str): The domain to extract from.

    Returns:
        str: The eTLD+1 or the original domain.
    """
    ext = tldextract.extract(domain)
    if not ext.top_domain_under_public_suffix:
        return domain.lower()
    return ext.top_domain_under_public_suffix.lower()

# ---------- Checker ----------

class FactChecker:
    def __init__(self):
        self.brands = load_brands()
        self.tld_risks = load_tld_risks()
        self.phrases = load_phrases()
        self.ner = pipeline("token-classification", model="Davlan/distilbert-base-multilingual-cased-ner-hrl", aggregation_strategy="simple")
        
    def extract_entities_and_claims(self, text:str) -> Tuple[List[str], Dict[str, List[str]], List[Dict[str, Any]], List[str]]:
        # TODO: Add NER with spacy or similar
        # NOTE: For now: Minimal approach with regex and phrase matching
        brands_found = []
        lowered = text.lower()
        
        for brand in self.brands.keys():
            if brand in lowered:
                brands_found.append(brand)
        
        claims = {k: [] for k in self.phrases}
        for cat, plist in self.phrases.items():
            for phrase in plist:
                if phrase.lower() in lowered:
                    claims[cat].append(phrase)
                    
        ner_hits = self.ner(text)
        unmatched_orgs = []
        people_found = []
        
        STRIP_SUFFIXES = re.compile(r"\b(gmbh|ag|mbh|e\.v\.?|inc|llc|ltd|co)\b\.?", re.IGNORECASE)
        
        for hit in ner_hits:
            if hit["score"] < 0.80:
                continue
            span = STRIP_SUFFIXES.sub("", hit["word"]).strip().lower()
            
            if hit["entity_group"] == "ORG":
                match = process.extractOne(span, self.brands.keys(), scorer=fuzz.WRatio, score_cutoff=85)
                if match:
                    brands_found.append(match[0])
                else:
                    unmatched_orgs.append({"text": hit["word"], "score": round(hit["score"], 3)})
            elif hit["entity_group"] == "PER":
                people_found.append(hit["word"])
        
        return list(dict.fromkeys(brands_found)), claims, unmatched_orgs, list(dict.fromkeys(people_found))

    def check_urls(self, urls: List[str]) -> Tuple[float, List[Evidence], Dict[str, float]]:
        obf_score = 0.0
        tld_score = 0.0
        evidence = []
        
        for u in urls:
            d = domain_from_url(u)
            if not d:
                continue
            
            # obfuscation checks
            flags = {
                "punycode": d.startswith("xn--"),
                "ip_literal": is_ip_literal(d),
                "hex_encoding": has_hex_encoding(u),
                "at_symbol": has_at_symbol(u),
                "subdomain_count": count_subdomains(d) > 2
            }
            
            if any(flags.values()):
                triggered = sum(bool(v) for v in flags.values())
                obf_score = max(obf_score, min(1.0, triggered / 5.0)) # finer grained score in [0, 1]
                evidence.append(Evidence("url_obfuscation", {"url": u, **flags}))
            
            # tld risk check
            ext = tldextract.extract(d)
            tld = (ext.suffix or "").lower()
            
            if tld:
                severity = self.tld_risks.get(tld)
                if severity:
                    score = SEVERITY_SCORES.get(severity, 0.0)
                    tld_score = max(tld_score, score)
                    evidence.append(Evidence("tld_risk", {"url": u, "tld": tld, "severity": severity, "score": score}))
                    
        return max(obf_score, 0.0), evidence, {"url_obfuscation": obf_score, "tld_risk": tld_score}
    
    def check_brand_impersonation(self, brands_found: List[str], urls: List[str], sender_email: Optional[str] = None) -> Tuple[float, List[Evidence], Optional[str]]:
        sender_domain_etld1 = None
        if sender_email and "@" in sender_email:
            sender_domain_etld1 = etld1(sender_email.split("@")[-1])
            
        link_domains = [etld1(domain_from_url(u) or "") for u in urls if domain_from_url(u)]
        mismatch = 0.0
        evidence = []
        
        for brand in brands_found:
            officials = self.brands.get(brand)
            if not officials:
                continue
            
            official_etld1 = etld1(officials)
            bad_links = [d for d in link_domains if d and d != official_etld1]
            sender_mismatch = (sender_domain_etld1 and sender_domain_etld1 != official_etld1)
            
            if bad_links or sender_mismatch:
                mismatch = 1.0
                evidence.append(Evidence("brand_impersonation", {
                    "brand": brand,
                    "official_domain": official_etld1,
                    "sender_domain": sender_domain_etld1,
                    "link_domains": list(sorted(set(bad_links))),
                }))

        return mismatch, evidence, sender_domain_etld1
    
    def claim_risk_score(self, claim_hits: Dict[str, List[str]]) -> Tuple[float, List[Evidence]]:
        """
        Weighted claim-risk:
          - High-severity categories (credential/payment/threat) contribute more.
          - If ONLY low-severity categories matched, apply a small dampener.
        """
        # Adjust these to match your phrases.json keys (unknown → medium by default)
        CATEGORY_WEIGHTS = {
            "credential_access": 1.0,       # "verify account", "reset password", "confirm identity"
            "payment_urgency": 0.8,         # "wire immediately", "gift card", "bitcoin wallet"
            "threat_or_consequence": 0.7,   # "account will be deleted", "final notice"
            "neutral_finance_low": 0.25,    # benign finance phrases: "bank account", etc.
            "misc_generic": 0.10            # very weak, generic cues: "click here"
        }

        weighted = 0.0
        total_hits = 0
        any_high = False

        for cat, phrases in claim_hits.items():
            hits = len(phrases)
            total_hits += hits
            w = CATEGORY_WEIGHTS.get(cat, 0.4)  # default medium if unknown category
            weighted += hits * w
            if w >= 0.7:
                any_high = True

        # map weighted sum to [0,1] (≈ 3 strong hits saturate)
        score = min(1.0, weighted / 3.0)

        # dampener if only low-severity categories were hit
        if total_hits > 0 and not any_high:
            score = max(0.0, score - 0.15)

        evidence: List[Evidence] = []
        if total_hits > 0:
            evidence.append(Evidence("claim_risk", claim_hits))

        return score, evidence

    def check(self, email_text: str, sender_email: Optional[str] = None) -> FactCheckResult:
        urls = extract_urls(email_text)
        brands_found, claim_hits, unmatched_orgs, people_found = self.extract_entities_and_claims(email_text)
        
        obf_score, url_evs, url_comps = self.check_urls(urls)
        brand_score, brand_evs, sender_domain = self.check_brand_impersonation(brands_found, urls, sender_email)
        claim_score, claim_evs = self.claim_risk_score(claim_hits)
        
        ner_evs = []
        for org in unmatched_orgs:
            ner_evs.append(Evidence("unverified_entity", {"organization": org["text"], "confidence": org["score"]}))
        for person in people_found:
            ner_evs.append(Evidence("named_individual_referenced", {"person": person}))

        unverified_entity_score = min(1.0, len(unmatched_orgs) / 3.0)  # mirrors the saturation pattern already used in claim_risk_score at checker.py:409

        
        def _sev_label(x: float) -> str:
            if x >= 0.8: return "critical"
            if x >= 0.5: return "high"
            if x >= 0.2: return "medium"
            return "low"
        
        components = {"brand_mismatch": brand_score,
                        "tld_risk": url_comps["tld_risk"],
                        "url_obfuscation": url_comps["url_obfuscation"],
                        "claim_risk": claim_score,
                        "unverified_entity": unverified_entity_score}
        
        fact_risk = (0.35*components["brand_mismatch"] +
                        0.25*components["tld_risk"] +
                        0.20*components["url_obfuscation"] +
                        0.20*components["claim_risk"] +
                        0.15*components["unverified_entity"])
        
        
        msgs = []
        if brand_score > 0:
            msgs.append(f"This email mentiones a brand ({', '.join(brands_found)}) but links to suspicious domains.")
        if components["tld_risk"] > 0:
            msgs.append("This email contains links with risky top-level domains (TLDs).")
        if components["url_obfuscation"] > 0:
            msgs.append("This email contains obfuscated URLs that may be used to mislead users.")
        if claim_score > 0:
            msgs.append(f"This email contains potentially suspicious phrases (claim risk: {_sev_label(claim_score)}).")
        if components["unverified_entity"] > 0:
            orgs_str = ", ".join([org["text"] for org in unmatched_orgs])
            msgs.append(f"This email references unverified organizations ({orgs_str}) or named individuals ({', '.join(people_found)}).")
        evidence = url_evs + brand_evs + claim_evs + ner_evs

        role_words_hit = bool(claim_hits.get("people"))
        if role_words_hit and people_found:
            ner_evs.append(Evidence("impersonation_pattern", {
                "roles_mentioned": claim_hits["people"],
                "people_mentioned": people_found,
            }))
            impersonation_boost = 0.25
        else:
            impersonation_boost = 0.0
        
        return FactCheckResult(fact_risk = min(1.0, fact_risk + impersonation_boost),
                                components = components,
                                evidence = evidence,
                                message = msgs)

# ---------- Runner ----------

if __name__ == "__main__":
        
    # Main checker test
    email_text = """
    Dear user, your PayPal account is locked. Verify your account within 24 hours:
    https://xn--paypa1-secure-login.example.tk/login
    """
    fc = FactChecker()
    res = fc.check(email_text, sender_email="security@paypal-alerts.com")
    
    print("Input Email Text: ")
    print(email_text)
    print("Sender email:")
    print("  security@paypal-alerts.com\n")

    print("Fact Check Result:")
    print(f"  Risk Level: {res.fact_risk}\n")
    print(f"  Components: {res.components}\n")
    print(f"  Evidence: {res.evidence}\n")
    print(f"  Message: {res.message}\n")
