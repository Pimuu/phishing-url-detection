"""Lexical and host-based feature extraction for URLs.

Every feature is computed from the URL *string only*: no DNS lookups, WHOIS
queries, page fetches or reputation services. This keeps the detector fast,
privacy-preserving and usable offline (e.g. in a mail gateway or browser
extension), at the cost of ignoring signals that need network access.

Public entry points:
    parse_url(url)          -> ParsedURL   (host / registered domain / path ...)
    extract_features(url)   -> dict[str, float]
    extract_feature_frame(urls) -> pandas.DataFrame (one row per URL)
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit

import pandas as pd

from .constants import (
    BRAND_DOMAINS,
    FREE_HOSTING_SUFFIXES,
    MULTI_PART_SUFFIXES,
    SENSITIVE_WORDS,
    SHORTENERS,
    SUSPICIOUS_TLDS,
)

# Feature groups, used for the ablation study. "lexical" features look at the
# URL as a whole string; "host" features depend on how the hostname is built
# or on which registered domain it belongs to.
LEXICAL_FEATURES = [
    "url_length",
    "n_dots",
    "n_hyphens",
    "n_digits",
    "has_at",
    "is_https",
    "path_depth",
    "path_length",
    "n_query_params",
    "n_sensitive_words",
    "has_double_slash_path",
    "brand_in_url",
]
HOST_FEATURES = [
    "has_ip_host",
    "n_subdomains",
    "suspicious_tld",
    "domain_entropy",
    "hostname_length",
    "host_n_hyphens",
    "host_n_digits",
    "is_shortener",
    "is_free_hosting",
    "has_port",
    "is_punycode",
    "brand_outside_official_domain",
    "brand_lookalike",
]
FEATURE_NAMES = LEXICAL_FEATURES + HOST_FEATURES

# Digit/letter substitutions commonly used in look-alike domains.
_HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


@dataclass(frozen=True)
class ParsedURL:
    """The pieces of a URL that the features are built from."""

    raw: str
    scheme: str
    host: str               # lower-case hostname, userinfo and port removed
    port: int | None
    path: str
    query: str
    is_ip: bool
    tld: str                # public suffix, e.g. "com", "co.uk", "web.app"
    registered_domain: str  # e.g. "paypal.com", "alice.github.io", or the IP
    subdomain: str          # labels left of the registered domain ("" if none)


def _is_ip(host: str) -> bool:
    """True for dotted IPv4, IPv6 and integer/hex-encoded IPv4 hosts."""
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        pass
    # Obfuscated forms browsers still accept: http://3232235777/ or 0xC0A80001
    return bool(re.fullmatch(r"(0x[0-9a-f]{1,8}|\d{8,10})", host))


def _split_domain(host: str) -> tuple[str, str, str]:
    """Split a hostname into (subdomain, registered_domain, suffix).

    Uses a small built-in subset of the Public Suffix List (see constants.py)
    instead of a third-party package so the project has no extra dependency.
    Unknown suffixes fall back to "last label is the TLD", which is correct
    for the vast majority of hostnames.
    """
    labels = [lab for lab in host.split(".") if lab]
    if len(labels) < 2:
        return "", host, ""
    # Longest matching suffix wins (private suffixes like web.app included).
    for n_suffix_labels in (2, 1):
        if len(labels) <= n_suffix_labels:
            continue
        suffix = ".".join(labels[-n_suffix_labels:])
        if n_suffix_labels == 2 and suffix not in MULTI_PART_SUFFIXES | FREE_HOSTING_SUFFIXES:
            continue
        registered = ".".join(labels[-(n_suffix_labels + 1):])
        subdomain = ".".join(labels[: -(n_suffix_labels + 1)])
        return subdomain, registered, suffix
    return "", host, labels[-1]


def parse_url(url: str) -> ParsedURL:
    """Parse a URL string robustly (never raises on malformed input)."""
    url = url.strip()
    to_split = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url
    try:
        parts = urlsplit(to_split)
        host = (parts.hostname or "").lower().rstrip(".")
        try:
            port = parts.port
        except ValueError:  # e.g. "host:abc"
            port = None
        scheme, path, query = parts.scheme.lower(), parts.path, parts.query
    except ValueError:  # e.g. unbalanced IPv6 brackets
        scheme, host, port, path, query = "http", "", None, "", ""

    is_ip = _is_ip(host)
    if is_ip:
        subdomain, registered, tld = "", host, ""
    else:
        subdomain, registered, tld = _split_domain(host)
    return ParsedURL(url, scheme, host, port, path, query, is_ip, tld, registered, subdomain)


def shannon_entropy(text: str) -> float:
    """Shannon entropy (bits per character) of a string; 0.0 for empty input."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert/delete/substitute), O(len(a)*len(b))."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _domain_label(p: ParsedURL) -> str:
    """The registrable label without its suffix: 'paypal' for 'paypal.com'."""
    if p.is_ip or not p.tld:
        return p.registered_domain
    return p.registered_domain[: -(len(p.tld) + 1)]


def _token_imitates_brand(token: str) -> bool:
    """Is one label token a near-miss of a brand name (but not the brand)?"""
    if len(token) < 5:
        return False
    normalised = token.translate(_HOMOGLYPHS).replace("rn", "m").replace("vv", "w")
    for brand in BRAND_DOMAINS:
        if len(brand) < 5 or token == brand:
            continue
        if normalised == brand:
            return True
        max_dist = 1 if len(brand) <= 6 else 2
        if abs(len(token) - len(brand)) <= max_dist and levenshtein(token, brand) <= max_dist:
            return True
    return False


def _brand_lookalike(label: str) -> bool:
    """Does the registrable label imitate a brand without being it?

    The label is checked as a whole and token by token (split on '-', so
    "paypa1-secure" is caught). A token matches when, for tokens of
    length >= 5,
      * homoglyph normalisation turns it into a brand ("paypa1" -> "paypal",
        "rnicrosoft" -> "microsoft"), or
      * it is within edit distance 1 (brands <= 6 chars) or 2 (longer brands)
        of a brand ("amazom", "conibase").
    A token that *equals* a brand is not a look-alike; that case is covered
    by the brand_outside_official_domain feature.
    """
    tokens = {label.replace("-", "")} | set(label.split("-"))
    return any(_token_imitates_brand(t) for t in tokens)


def extract_features(url: str) -> dict[str, float]:
    """Compute the full feature vector for one URL (see FEATURE_NAMES)."""
    p = parse_url(url)
    lower = url.lower()
    host_no_dots = p.host.replace(".", "")
    path_segments = [seg for seg in p.path.split("/") if seg]

    brands_present = [b for b in BRAND_DOMAINS if b in lower]
    brand_outside = any(p.registered_domain not in BRAND_DOMAINS[b] for b in brands_present)

    features = {
        # ---- lexical (whole-URL) features ----
        "url_length": len(url),
        "n_dots": url.count("."),
        "n_hyphens": url.count("-"),
        "n_digits": sum(ch.isdigit() for ch in url),
        "has_at": int("@" in url),
        "is_https": int(p.scheme == "https"),
        "path_depth": len(path_segments),
        "path_length": len(p.path),
        "n_query_params": len(parse_qsl(p.query, keep_blank_values=True)),
        "n_sensitive_words": sum(lower.count(w) for w in SENSITIVE_WORDS),
        "has_double_slash_path": int("//" in p.path),
        "brand_in_url": int(bool(brands_present)),
        # ---- host-based features ----
        "has_ip_host": int(p.is_ip),
        # "www" is a conventional prefix, not a meaningful subdomain level.
        "n_subdomains": len([s for s in p.subdomain.split(".") if s and s != "www"]),
        "suspicious_tld": int(p.tld.split(".")[-1] in SUSPICIOUS_TLDS),
        "domain_entropy": shannon_entropy(host_no_dots),
        "hostname_length": len(p.host),
        "host_n_hyphens": p.host.count("-"),
        "host_n_digits": 0 if p.is_ip else sum(ch.isdigit() for ch in p.host),
        "is_shortener": int(p.registered_domain in SHORTENERS),
        "is_free_hosting": int(p.tld in FREE_HOSTING_SUFFIXES),
        "has_port": int(p.port is not None),
        "is_punycode": int("xn--" in p.host),
        "brand_outside_official_domain": int(brand_outside),
        "brand_lookalike": int(not p.is_ip and _brand_lookalike(_domain_label(p))),
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}


def extract_feature_frame(urls) -> pd.DataFrame:
    """Vectorise an iterable of URLs into a DataFrame with FEATURE_NAMES columns."""
    return pd.DataFrame([extract_features(u) for u in urls], columns=FEATURE_NAMES)
