"""Deterministic generator for a realistic synthetic phishing-URL corpus.

Why synthetic?  The public phishing/legitimate URL corpora we tried (UCI
PhiUSIIL, PhishTank, Tranco top-1M) were not downloadable from the build
environment, and the only reachable feed (Phishing.Database on GitHub)
contains phishing URLs only. Pairing real phishing URLs with a legitimate set
from a *different* source would let a model learn "which source did this come
from" instead of "is this phishing". So both classes are generated here by
one process, with the same vocabularies and the same randomness, and the real
feed is used only for an external recall check (see external.py).

Design goals for the generator
------------------------------
* Cover the classic attacker tricks (IP hosts, '@' userinfo, long subdomain
  chains, typosquats, suspicious TLDs, hyphen/digit-heavy domains, shorteners,
  punycode, free-hosting platforms, compromised legitimate sites).
* Do NOT make the task trivially separable. Legitimate URLs include hard
  negatives (HTTP sites, hyphenated/digit domains, cheap TLDs, CDN hosts with
  random-looking names, intranet IPs, shorteners, login pages of real brands,
  news articles that *mention* brands). Phishing URLs include hard positives
  (HTTPS, compromised small-business sites, clean dictionary domains).
* Be fully reproducible: one integer seed -> identical CSV.

Each phishing URL is built as a dict of components so that evasion.py can
apply attacker constraints (force HTTPS, no brand tokens, ...) on top.
"""

from __future__ import annotations

import random
import string

import pandas as pd

from .constants import BRAND_DOMAINS, FREE_HOSTING_SUFFIXES, SHORTENERS, SUSPICIOUS_TLDS

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------
WORDS = """
apple river stone green blue harbor maple cedar oak pine sun moon star cloud
rain snow north south east west city town village garden market bakery coffee
book paper studio design craft build home house family health care clinic
dental vision pet farm fresh food kitchen table wood metal glass light bright
smart quick fast true good best prime first royal golden silver urban rural
coast valley mountain lake ocean island bridge road trail path journey travel
tour photo art music sound film media news daily weekly report review guide
learn school college academy science data code tech labs systems network cloud
digital web net online local global united national pacific atlantic summit
peak point field park forest meadow spring summer autumn winter garden rose
lily daisy tiger eagle falcon wolf bear fox lion owl hawk swift bold clear
simple modern classic vintage nordic alpine sunset sunrise horizon vertex pixel
orbit rocket atlas delta omega alpha beta nova core hub base works shop store
supply parts tools motor auto bike wheel sport fitness yoga run swim team club
league games play fun kids baby mom dad friends social local legal law tax
finance money invest capital trust partners group consulting solutions service
""".split()

FIRST_NAMES = "james mary john linda robert sarah michael emma david olivia daniel sofia kevin anna lucas maria noah chloe".split()
LAST_NAMES = "smith johnson brown garcia miller davis wilson moore taylor clark lewis walker hall young king wright".split()

# Real, well-known legitimate registered domains.
POPULAR_DOMAINS = """
google.com youtube.com wikipedia.org amazon.com facebook.com github.com
stackoverflow.com reddit.com nytimes.com bbc.co.uk microsoft.com apple.com
paypal.com linkedin.com netflix.com instagram.com twitter.com yahoo.com
ebay.com spotify.com dropbox.com adobe.com cnn.com theguardian.com
medium.com wordpress.org mozilla.org python.org npmjs.com docker.com
cloudflare.com zoom.us slack.com notion.so salesforce.com shopify.com
etsy.com walmart.com target.com bestbuy.com imdb.com quora.com
booking.com airbnb.com tripadvisor.com espn.com weather.com
chase.com wellsfargo.com bankofamerica.com coinbase.com live.com
office.com outlook.com icloud.com amazon.co.uk fedex.com usps.com
docusign.com hsbc.co.uk theverge.com arstechnica.com nature.com
arxiv.org mit.edu stanford.edu harvard.edu nih.gov irs.gov
""".split()

POPULAR_SUBDOMAINS = ["www", "www", "www", "", "", "blog", "docs", "support",
                      "help", "shop", "news", "en", "m", "mail", "accounts",
                      "login", "developer", "api", "status", "community"]

LEGIT_TLDS = (["com"] * 50 + ["org"] * 10 + ["net"] * 7 + ["io"] * 4 + ["co.uk"] * 4
              + ["de"] * 3 + ["ca"] * 2 + ["com.au"] * 2 + ["fr"] * 2 + ["us"] * 2
              + ["app"] * 1 + ["dev"] * 1 + ["co"] * 2
              # A minority of honest sites on cheap/"suspicious" TLDs.
              + ["info"] * 2 + ["xyz"] * 1 + ["online"] * 1 + ["site"] * 1 + ["live"] * 1)

PHISH_COMMON_TLDS = ["com"] * 6 + ["net"] * 2 + ["org"] + ["co"] + ["us"]
PHISH_SUSPICIOUS_TLDS = sorted(SUSPICIOUS_TLDS)
BRANDS = sorted(BRAND_DOMAINS)
PHISH_WORDS = ["secure", "account", "verify", "login", "update", "support",
               "service", "online", "auth", "signin", "billing", "confirm",
               "alert", "help", "center", "portal", "access", "recovery", "id"]
NEUTRAL_PATH_WORDS = ["documents", "client", "area", "view", "file", "share",
                      "portal", "dashboard", "profile", "session", "access",
                      "member", "invoice", "status", "track", "order"]
FREE_HOSTS = sorted(FREE_HOSTING_SUFFIXES)
SHORTENER_LIST = sorted(SHORTENERS)
CDN_HOSTS = ["cloudfront.net", "akamaihd.net", "fastly.net", "cdn.jsdelivr.net",
             "storage.googleapis.com", "s3.amazonaws.com", "blob.core.windows.net"]


# ---------------------------------------------------------------------------
# Small random helpers
# ---------------------------------------------------------------------------
def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def _alnum(rng: random.Random, n: int, upper: bool = True) -> str:
    alphabet = string.ascii_lowercase + string.digits + (string.ascii_uppercase if upper else "")
    return "".join(rng.choice(alphabet) for _ in range(n))


def _words(rng: random.Random, k: int, sep: str = "-") -> str:
    return sep.join(rng.choice(WORDS) for _ in range(k))


def _random_ip(rng: random.Random) -> str:
    first = rng.choice([23, 45, 62, 77, 91, 103, 104, 141, 172, 185, 192, 194, 203, 212])
    return f"{first}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _typo(rng: random.Random, brand: str) -> str:
    """Produce a look-alike of a brand name using a common typosquat trick."""
    trick = rng.choice(["homoglyph", "homoglyph", "drop", "double", "swap", "rn"])
    if trick == "homoglyph":
        subs = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5"}
        idx = [i for i, ch in enumerate(brand) if ch in subs]
        if idx:
            i = rng.choice(idx)
            return brand[:i] + subs[brand[i]] + brand[i + 1:]
    if trick == "rn" and "m" in brand:
        return brand.replace("m", "rn", 1)
    if trick == "double":
        i = rng.randrange(len(brand))
        return brand[:i] + brand[i] + brand[i:]
    if trick == "swap" and len(brand) > 3:
        i = rng.randrange(1, len(brand) - 2)
        return brand[:i] + brand[i + 1] + brand[i] + brand[i + 2:]
    i = rng.randrange(1, len(brand) - 1)  # "drop" (also the fallback)
    return brand[:i] + brand[i + 1:]


def assemble(c: dict) -> str:
    """Join URL components produced by the generators into a URL string."""
    netloc = c["host"]
    if c.get("userinfo"):
        netloc = f"{c['userinfo']}@{netloc}"
    if c.get("port"):
        netloc = f"{netloc}:{c['port']}"
    url = f"{c['scheme']}://{netloc}{c.get('path', '')}"
    if c.get("query"):
        url += "?" + c["query"]
    return url


# ---------------------------------------------------------------------------
# Legitimate URLs
# ---------------------------------------------------------------------------
def legit_path(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.12:
        return rng.choice(["", "/"])
    if kind < 0.32:   # news / blog article
        return f"/{rng.randint(2015, 2026)}/{rng.randint(1, 12):02d}/{_words(rng, rng.randint(3, 8))}"
    if kind < 0.44:   # product page
        return rng.choice([f"/products/{_words(rng, rng.randint(2, 4))}-{rng.randint(100, 99999)}",
                           f"/dp/B0{_alnum(rng, 8).upper()}",
                           f"/item/{rng.randint(10**5, 10**9)}"])
    if kind < 0.56:   # documentation
        return f"/{rng.choice(['docs', 'en-us', 'help', 'guide', 'manual'])}/{_words(rng, 1)}/{_words(rng, rng.randint(1, 3))}"
    if kind < 0.64:   # wiki-style
        return "/wiki/" + "_".join(w.capitalize() for w in rng.sample(WORDS, rng.randint(1, 3)))
    if kind < 0.72:   # static asset
        return f"/static/{rng.choice(['js', 'css', 'img'])}/{_words(rng, 1)}.{_hex(rng, 8)}.{rng.choice(['js', 'css', 'png'])}"
    if kind < 0.80:   # user profile / repo ('/@user' is common on Medium, Mastodon, TikTok...)
        return f"/{'@' if rng.random() < 0.15 else ''}{rng.choice(FIRST_NAMES)}{rng.choice(['', '_', '.'])}{rng.choice(LAST_NAMES)}{rng.choice(['', str(rng.randint(1, 99))])}" \
               + rng.choice(["", f"/{_words(rng, 2)}"])
    if kind < 0.88:   # category listing
        return f"/{_words(rng, 1)}/{_words(rng, 1)}/"
    if kind < 0.94:   # legitimate account flows also exist!
        return rng.choice(["/login", "/signin", "/account/settings", "/account", "/my-account/orders",
                           "/auth/login", "/user/password/reset", "/billing/invoices"])
    return f"/search"


def legit_query(rng: random.Random, path: str) -> str:
    if path == "/search":
        return f"q={_words(rng, rng.randint(1, 3), '+')}"
    r = rng.random()
    if r < 0.70:
        return ""
    if r < 0.80:
        return f"utm_source={rng.choice(['newsletter', 'twitter', 'google'])}&utm_medium={rng.choice(['email', 'social', 'cpc'])}&utm_campaign={_words(rng, 2, '_')}"
    if r < 0.90:
        return f"id={rng.randint(1, 10**6)}"
    return f"page={rng.randint(1, 30)}&sort={rng.choice(['asc', 'desc', 'new', 'popular'])}"


def _small_site_domain(rng: random.Random) -> str:
    """Long-tail legitimate domain, e.g. 'maplebakery.com', 'city-dental.co.uk'."""
    style = rng.random()
    if style < 0.55:
        label = _words(rng, rng.randint(1, 3), "")
    elif style < 0.80:
        label = _words(rng, rng.randint(2, 3), "-")
    elif style < 0.90:
        label = f"{rng.choice(FIRST_NAMES)}{rng.choice(LAST_NAMES)}"
    else:  # digits in honest names: web2print, 365data, route66diner
        label = rng.choice([f"{_words(rng, 1, '')}{rng.randint(1, 999)}",
                            f"{rng.randint(1, 999)}{_words(rng, 1, '')}",
                            f"{_words(rng, 1, '')}{rng.randint(2, 9)}{_words(rng, 1, '')}"])
    return f"{label}.{rng.choice(LEGIT_TLDS)}"


def generate_legit(rng: random.Random) -> tuple[str, str]:
    """Return (url, category). The category is used only for error analysis."""
    kind = rng.random()
    scheme = "https"
    if kind < 0.33:      # well-known site
        category = "popular"
        sub = rng.choice(POPULAR_SUBDOMAINS)
        host = f"{sub + '.' if sub else ''}{rng.choice(POPULAR_DOMAINS)}"
    elif kind < 0.68:    # long-tail small business / personal / blog site
        category = "small_site"
        host = _small_site_domain(rng)
        if rng.random() < 0.3:
            host = f"www.{host}"
        elif rng.random() < 0.1:
            host = f"{rng.choice(['shop', 'blog', 'mail', 'portal', 'app'])}.{host}"
        if rng.random() < 0.18:
            scheme = "http"
    elif kind < 0.75:    # institutions (deep subdomains are normal here)
        category = "institution"
        dept = rng.choice(["cs", "library", "admissions", "portal", "hr", "students", "it", "research", "mail"])
        org = _words(rng, 1, "") + rng.choice(["university", "college", "county", "city", "state", "institute"])
        host = f"{dept}.{rng.choice(['', 'my.'])}{org}.{rng.choice(['edu', 'gov', 'ac.uk', 'org', 'edu.au'])}"
    elif kind < 0.84:    # CDN / cloud storage with random-looking hosts
        category = "cdn_cloud"
        cdn = rng.choice(CDN_HOSTS)
        if cdn in ("storage.googleapis.com", "s3.amazonaws.com"):
            host = cdn
            path = f"/{_words(rng, 2)}-{rng.choice(['assets', 'backup', 'public', 'media'])}/{_words(rng, 1)}/{_hex(rng, 12)}.{rng.choice(['pdf', 'png', 'zip', 'json'])}"
            return assemble({"scheme": scheme, "host": host, "path": path}), category
        host = f"{rng.choice(['d', 'e', 'a', 'cdn'])}{_alnum(rng, rng.randint(8, 14), upper=False)}.{cdn}"
    elif kind < 0.92:    # legit projects on free hosting platforms
        category = "free_hosting"
        name = rng.choice([f"{rng.choice(FIRST_NAMES)}{rng.choice(LAST_NAMES)}", _words(rng, rng.randint(1, 3)),
                           f"{_words(rng, 2)}-{_alnum(rng, 4, upper=False)}"])
        host = f"{name}.{rng.choice(FREE_HOSTS)}"
    elif kind < 0.95:    # shortened links from marketing / social media
        category = "shortener"
        return f"https://{rng.choice(SHORTENER_LIST)}/{_alnum(rng, rng.randint(5, 8))}", category
    elif kind < 0.97:    # intranet / router admin pages addressed by IP
        category = "intranet_ip"
        host = rng.choice([f"192.168.{rng.randint(0, 5)}.{rng.randint(1, 254)}",
                           f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                           f"172.{rng.randint(16, 31)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"])
        port = rng.choice([None, None, 8080, 8443, 3000, 9090])
        path = rng.choice(["/", "/admin", "/dashboard", "/login", "/status", "/api/v1/health"])
        return assemble({"scheme": rng.choice(["http", "https"]), "host": host, "port": port, "path": path}), category
    else:                # legit site that merely *mentions* a brand in the path
        category = "brand_mention"
        host = rng.choice(["www.theverge.com", "github.com", "www.reddit.com", "news.ycombinator.com",
                           "techcrunch.com", "www.wired.com", "en.wikipedia.org", "stackoverflow.com"])
        brand = rng.choice(BRANDS)
        path = rng.choice([f"/{rng.randint(2018, 2026)}/{rng.randint(1, 12)}/{brand}-{_words(rng, rng.randint(2, 5))}",
                           f"/{brand}/{_words(rng, 1)}", f"/wiki/{brand.capitalize()}",
                           f"/questions/{rng.randint(10**6, 10**8)}/{brand}-{_words(rng, 3)}"])
        return assemble({"scheme": "https", "host": host, "path": path}), category

    path = legit_path(rng)
    return assemble({"scheme": scheme, "host": host, "path": path, "query": legit_query(rng, path)}), category


# ---------------------------------------------------------------------------
# Phishing URLs (component form, so evasion.py can post-process them)
# ---------------------------------------------------------------------------
PHISH_TECHNIQUES = {
    # technique: sampling weight
    "ip_host": 8,
    "at_trick": 4,
    "subdomain_chain": 12,
    "typosquat": 13,
    "brand_combo": 12,
    "random_suspicious_tld": 8,
    "free_hosting": 12,
    "compromised_site": 16,
    "shortener": 4,
    "punycode": 2,
    "clean_mimicry": 9,
}


def phish_path(rng: random.Random, brand: str | None, sensitive: bool = True) -> str:
    """Path of a phishing kit landing page."""
    if not sensitive:  # attacker avoids tell-tale words
        return "/" + "/".join(rng.choice(NEUTRAL_PATH_WORDS) for _ in range(rng.randint(1, 3)))
    b = brand or rng.choice(["portal", "online", "webapp", "cloud"])
    templates = [
        f"/{b}/login.php",
        f"/secure/{b}/signin/",
        f"/{_hex(rng, rng.randint(8, 32))}/{rng.choice(PHISH_WORDS)}/{rng.choice(['index.html', 'login.php', 'verify.php', 'account.html'])}",
        f"/{rng.choice(['wp-content', 'wp-includes', 'wp-admin'])}/{rng.choice(['plugins', 'themes', 'css', 'images'])}/{_words(rng, 1)}/{b}/{rng.choice(['index.html', 'login.php', ''])}",
        f"/{rng.choice(PHISH_WORDS)}/{rng.choice(PHISH_WORDS)}/{rng.choice(['', 'index.php', 'auth.html'])}",
        "/webscr",
        f"/{b}-{rng.choice(PHISH_WORDS)}/{_alnum(rng, 6, upper=False)}/",
        f"/~{_words(rng, 1)}/{b}/{rng.choice(PHISH_WORDS)}.html",
        "/",
        "",
    ]
    path = rng.choice(templates)
    if rng.random() < 0.05:
        path = "/" + path  # '//' trick used by some open-redirect payloads
    return path


def phish_query(rng: random.Random, sensitive: bool = True) -> str:
    if not sensitive:
        return rng.choice(["", "", f"id={rng.randint(1000, 999999)}", f"ref={_alnum(rng, 8)}"])
    r = rng.random()
    if r < 0.45:
        return ""
    if r < 0.60:
        return f"session={_hex(rng, 32)}"
    if r < 0.72:
        return f"email={rng.choice(FIRST_NAMES)}.{rng.choice(LAST_NAMES)}@{rng.choice(['gmail.com', 'outlook.com', 'company.com'])}"
    if r < 0.82:
        return "cmd=_login-submit&dispatch=" + _hex(rng, 24)
    return f"token={_alnum(rng, rng.randint(16, 40))}&{rng.choice(['lang=en', 'step=2', 'redirect=1'])}"


def _pick_scheme(rng: random.Random, p_https: float) -> str:
    return "https" if rng.random() < p_https else "http"


def generate_phish_components(rng: random.Random, technique: str | None = None,
                              use_brand: bool = True, sensitive: bool = True) -> dict:
    """Build one phishing URL as components.

    use_brand=False  -> no brand names or brand look-alikes anywhere.
    sensitive=False  -> no login/verify/account... words in path or query.
    """
    if technique is None:
        names, weights = zip(*PHISH_TECHNIQUES.items())
        technique = rng.choices(names, weights=weights)[0]
    brand = rng.choice(BRANDS) if use_brand else None
    neutral = _words(rng, 1, "")  # stand-in for the brand when brand-less
    word = rng.choice(PHISH_WORDS) if sensitive else rng.choice(WORDS)
    c = {"technique": technique, "scheme": _pick_scheme(rng, 0.65), "userinfo": "", "port": None,
         "path": phish_path(rng, brand, sensitive), "query": phish_query(rng, sensitive)}

    if technique == "ip_host":
        c.update(scheme=_pick_scheme(rng, 0.2), host=_random_ip(rng),
                 port=rng.choice([None, None, None, 8080, 8443, 2083]))
    elif technique == "at_trick":
        # Browser ignores everything before '@': "paypal.com@evil" goes to evil.
        decoy = f"{'www.' if rng.random() < 0.5 else ''}{brand}.com" if brand else f"www.{neutral}.com"
        target = _random_ip(rng) if rng.random() < 0.5 else f"{_words(rng, 1, '')}{rng.randint(1, 99)}.{rng.choice(PHISH_SUSPICIOUS_TLDS)}"
        c.update(userinfo=decoy, host=target)
    elif technique == "subdomain_chain":
        lead = f"{brand}.com" if brand else f"{neutral}.com"
        chain = ".".join(rng.choice([word, f"{word}-{rng.choice(PHISH_WORDS) if sensitive else rng.choice(WORDS)}",
                                     _alnum(rng, 6, upper=False)]) for _ in range(rng.randint(1, 3)))
        base = f"{_words(rng, 1, '')}{rng.choice(['', str(rng.randint(1, 999))])}.{rng.choice(PHISH_SUSPICIOUS_TLDS + PHISH_COMMON_TLDS)}"
        c["host"] = f"{lead}.{chain}.{base}"
    elif technique == "typosquat":
        label = _typo(rng, brand) if brand else _words(rng, 2, "")
        if rng.random() < 0.4:
            label += f"-{word}"
        tld = rng.choice(PHISH_COMMON_TLDS + PHISH_SUSPICIOUS_TLDS)
        c["host"] = f"{rng.choice(['', 'www.', f'{word}.'])}{label}.{tld}"
    elif technique == "brand_combo":
        b = brand or neutral
        parts = [b, word] + ([rng.choice(PHISH_WORDS) if sensitive else rng.choice(WORDS)] if rng.random() < 0.5 else [])
        rng.shuffle(parts)
        label = "-".join(parts)
        if rng.random() < 0.3:
            label += str(rng.randint(1, 9999))
        c["host"] = f"{label}.{rng.choice(PHISH_COMMON_TLDS + PHISH_SUSPICIOUS_TLDS)}"
    elif technique == "random_suspicious_tld":
        label = rng.choice([_alnum(rng, rng.randint(5, 12), upper=False), _words(rng, 2, ""),
                            f"{_words(rng, 1, '')}{rng.randint(10, 9999)}"])
        c["host"] = f"{rng.choice(['', 'www.', 'secure.' if sensitive else 'my.'])}{label}.{rng.choice(PHISH_SUSPICIOUS_TLDS)}"
    elif technique == "free_hosting":
        name = rng.choice([f"{brand or neutral}-{word}-{_alnum(rng, 4, upper=False)}",
                           f"{word}-{brand or neutral}{rng.randint(1, 999)}",
                           f"{_words(rng, 2)}-{_alnum(rng, 4, upper=False)}",
                           _alnum(rng, rng.randint(8, 14), upper=False)])
        c.update(scheme="https", host=f"{name}.{rng.choice(FREE_HOSTS)}")
    elif technique == "compromised_site":
        # A hacked small-business site: the domain itself looks entirely benign.
        c.update(scheme=_pick_scheme(rng, 0.55), host=_small_site_domain(rng))
        if rng.random() < 0.3:
            c["host"] = f"www.{c['host']}"
    elif technique == "shortener":
        c.update(scheme="https", host=rng.choice(SHORTENER_LIST),
                 path=f"/{_alnum(rng, rng.randint(5, 8))}", query="")
    elif technique == "punycode":
        c["host"] = f"xn--{_alnum(rng, rng.randint(4, 8), upper=False)}-{_alnum(rng, 3, upper=False)}.{rng.choice(PHISH_COMMON_TLDS)}"
    elif technique == "clean_mimicry":
        # Newly-registered, professional-looking domain; nothing stands out.
        c.update(scheme="https", host=f"{_words(rng, rng.randint(1, 2), '')}.{rng.choice(['com', 'com', 'net', 'co', 'io'])}")
    else:
        raise ValueError(f"unknown technique {technique!r}")
    return c


def generate_phish(rng: random.Random) -> tuple[str, str]:
    c = generate_phish_components(rng)
    return assemble(c), c["technique"]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def generate_dataset(n_per_class: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Return a shuffled DataFrame with columns url, label (1 = phishing), technique.

    `technique` is the phishing trick used, or "legit:<category>" for
    legitimate URLs (popular, small_site, cdn_cloud, ...).

    Duplicate URLs are discarded so the same string can never sit in both the
    training and the test split.
    """
    rng = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < n_per_class:  # legitimate URLs first
        url, category = generate_legit(rng)
        if url not in seen:
            seen.add(url)
            rows.append((url, 0, f"legit:{category}"))
    n_phish = 0
    while n_phish < n_per_class:
        url, technique = generate_phish(rng)
        if url not in seen:
            seen.add(url)
            rows.append((url, 1, technique))
            n_phish += 1
    df = pd.DataFrame(rows, columns=["url", "label", "technique"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
