"""Optional external check on REAL phishing URLs (needs network once).

Source: the Phishing.Database project (github.com/mitchellkrogza/Phishing.Database),
file phishing-links-ACTIVE.txt, pinned to a fixed commit so the sample is
reproducible. The feed contains *only* phishing URLs, so this check can only
measure recall (detection rate) and the distribution shift between our
synthetic phishing URLs and real ones; it cannot measure false positives.

The raw feed (~66 MB of live malicious links) is cached under data/external/
and is git-ignored on purpose: we commit aggregate results only, never the
URLs themselves.
"""

from __future__ import annotations

import random
import urllib.request
from pathlib import Path

FEED_COMMIT = "12a20bf8dd2b7e78d9d38fc12b3e5c970f79d1a7"
FEED_URL = (
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/"
    f"{FEED_COMMIT}/phishing-links-ACTIVE.txt"
)


def load_real_phishing_sample(cache_dir: Path, n: int = 5000, seed: int = 42) -> list[str]:
    """Download (once) the pinned feed and return a deterministic random sample."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"phishing-links-ACTIVE-{FEED_COMMIT[:12]}.txt"
    if not cache.exists():
        print(f"  downloading {FEED_URL}")
        tmp = cache.with_suffix(".part")
        with urllib.request.urlopen(FEED_URL, timeout=120) as resp, open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.rename(cache)
    lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
    urls = sorted({ln.strip() for ln in lines if ln.strip().lower().startswith(("http://", "https://"))})
    return random.Random(seed).sample(urls, min(n, len(urls)))
