"""Adaptive-attacker experiment: how far does detection fall when the attacker
knows which lexical features the detector uses and avoids them?

Each level adds constraints on top of the previous one:

  L0  No adaptation              -- same distribution as the training phishing.
  L1  + HTTPS & mainstream TLD    -- free TLS certificates; buy a .com/.net
                                     instead of a .tk/.xyz.
  L2  + no IP/@/shortener/punycode/port -- only "normal" domain names.
  L3  + no brand names, look-alikes or sensitive words -- e.g. a page at
                                     /documents/view on a neutral domain.
  L4  Full mimicry                -- host and path drawn from the *legitimate*
                                     small-site generator (a hacked or freshly
                                     registered site with an ordinary URL).

At L4 the phishing URL is, by construction, drawn from the same distribution
as part of the legitimate class, so no purely lexical detector can do better
than its false-positive rate on those sites. The experiment quantifies how
quickly performance degrades on the way there.
"""

from __future__ import annotations

import random

import pandas as pd

from .constants import BRAND_DOMAINS, SENSITIVE_WORDS, SUSPICIOUS_TLDS
from .data_gen import (
    _small_site_domain,
    assemble,
    generate_phish_components,
    legit_path,
    legit_query,
)

LEVELS = [
    "L0 No adaptation",
    "L1 + HTTPS & .com",
    "L2 + plain domain names",
    "L3 + no brand/sensitive words",
    "L4 Full mimicry",
]

# Techniques that remain possible once IPs, '@', shorteners, punycode are banned.
_DOMAIN_TECHNIQUES = ["subdomain_chain", "typosquat", "brand_combo", "random_suspicious_tld",
                      "free_hosting", "compromised_site", "clean_mimicry"]
_TECHNIQUE_WEIGHTS = [12, 13, 12, 8, 12, 16, 9]  # same relative weights as training


def _mainstream_tld(host: str, rng: random.Random) -> str:
    labels = host.split(".")
    if labels[-1] in SUSPICIOUS_TLDS:
        labels[-1] = rng.choice(["com", "com", "net", "org"])
    return ".".join(labels)


def _evasive_url(rng: random.Random, level: int) -> str:
    if level >= 4:
        host = _small_site_domain(rng)
        # Plain-word domain: strip hyphens/digits the detector might key on.
        host = "".join(ch for ch in host if not ch.isdigit()).replace("-", "")
        host = _mainstream_tld(host, rng)
        path = legit_path(rng)
        return assemble({"scheme": "https", "host": host, "path": path, "query": legit_query(rng, path)})

    technique = None
    if level >= 2:
        technique = rng.choices(_DOMAIN_TECHNIQUES, weights=_TECHNIQUE_WEIGHTS)[0]
    clean = level >= 3
    c = generate_phish_components(rng, technique=technique, use_brand=not clean, sensitive=not clean)
    if level >= 1:
        c["scheme"] = "https"
        c["host"] = _mainstream_tld(c["host"], rng)
    if level >= 2:
        c["userinfo"], c["port"] = "", None
        if "@" in c.get("query", ""):
            c["query"] = ""
    return assemble(c)


def _has_banned_token(url: str) -> bool:
    lower = url.lower()
    return any(b in lower for b in BRAND_DOMAINS) or any(w in lower for w in SENSITIVE_WORDS)


def _sample(rng: random.Random, level: int) -> str:
    """Draw one URL for a level. At L3 the attacker strictly avoids brand and
    sensitive tokens; ordinary dictionary words can collide with them by
    accident (e.g. "apple"), so such draws are rejected and redrawn."""
    while True:
        url = _evasive_url(rng, level)
        if level != 3 or not _has_banned_token(url):
            return url


def generate_evasion_sets(n_per_level: int = 2000, seed: int = 1234) -> pd.DataFrame:
    """Return url, level (int), level_name for every evasion level."""
    rows = []
    for level, name in enumerate(LEVELS):
        rng = random.Random(seed + level)  # independent of the training corpus seed
        urls: set[str] = set()
        while len(urls) < n_per_level:
            urls.add(_sample(rng, level))
        rows += [(u, level, name) for u in sorted(urls)]
    return pd.DataFrame(rows, columns=["url", "level", "level_name"])
