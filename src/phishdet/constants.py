"""Shared vocabularies used by both the feature extractor and the data generator.

Keeping these lists in one place makes the threat model explicit: everything the
detector "knows" about brands, risky TLDs, shorteners and free-hosting platforms
lives here and can be audited or replaced.
"""

# ---------------------------------------------------------------------------
# Brands that are frequently impersonated, mapped to their *official*
# registered domains. A brand token appearing anywhere in a URL whose
# registered domain is NOT in this set is a strong phishing indicator.
# Short tokens (< 4 chars, e.g. "ups", "dhl") are deliberately excluded because
# substring matching on them produces too many false hits ("groups", ...).
# ---------------------------------------------------------------------------
BRAND_DOMAINS: dict[str, set[str]] = {
    "paypal": {"paypal.com", "paypal.me"},
    "apple": {"apple.com"},
    "icloud": {"icloud.com", "apple.com"},
    "microsoft": {"microsoft.com", "microsoftonline.com", "live.com", "office.com"},
    "office365": {"office.com", "microsoft.com"},
    "outlook": {"outlook.com", "live.com", "office.com", "microsoft.com"},
    "google": {"google.com", "googleapis.com", "googleusercontent.com", "youtube.com"},
    "gmail": {"gmail.com", "google.com"},
    "amazon": {"amazon.com", "amazon.co.uk", "amazon.de", "amazonaws.com"},
    "netflix": {"netflix.com"},
    "facebook": {"facebook.com", "fb.com"},
    "instagram": {"instagram.com"},
    "whatsapp": {"whatsapp.com"},
    "linkedin": {"linkedin.com"},
    "chase": {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com"},
    "citibank": {"citibank.com", "citi.com"},
    "hsbc": {"hsbc.com", "hsbc.co.uk"},
    "dropbox": {"dropbox.com"},
    "docusign": {"docusign.com", "docusign.net"},
    "adobe": {"adobe.com"},
    "coinbase": {"coinbase.com"},
    "binance": {"binance.com"},
    "metamask": {"metamask.io"},
    "fedex": {"fedex.com"},
    "usps": {"usps.com"},
    "ebay": {"ebay.com"},
    "yahoo": {"yahoo.com"},
    "spotify": {"spotify.com"},
}

# TLDs that are heavily over-represented in phishing/abuse reports
# (free or very cheap registrations). Being on this list is a weak signal,
# not proof: plenty of legitimate sites use .info, .xyz or .online.
SUSPICIOUS_TLDS: set[str] = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "buzz", "click", "icu",
    "cyou", "rest", "fit", "work", "support", "live", "online", "site",
    "info", "zip", "mov", "country", "kim", "loan", "win", "bid",
}

# Public URL-shortening services. The real destination is hidden, so the
# lexical detector can only see the shortener itself.
SHORTENERS: set[str] = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "tiny.cc", "s.id",
}

# Multi-label public suffixes (a small, hand-picked subset of the Public
# Suffix List) so that e.g. "bbc.co.uk" -> registered domain "bbc.co.uk".
MULTI_PART_SUFFIXES: set[str] = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au",
    "co.jp", "co.in", "com.br", "com.tr", "co.za", "go.id", "co.id",
    "com.mx", "com.cn", "co.nz", "com.sg", "edu.au",
}

# "Private" suffixes from the Public Suffix List: free hosting / app
# platforms where anybody can get <name>.<platform>. Each customer name is
# its own registered domain, exactly as browsers treat them.
FREE_HOSTING_SUFFIXES: set[str] = {
    "github.io", "netlify.app", "vercel.app", "web.app", "firebaseapp.com",
    "herokuapp.com", "000webhostapp.com", "wixsite.com", "weebly.com",
    "blogspot.com", "pages.dev", "workers.dev", "glitch.me",
    "azurewebsites.net", "appspot.com",
}

# Words that social-engineering pages use to create urgency or mimic
# account flows.
SENSITIVE_WORDS: tuple[str, ...] = (
    "login", "signin", "logon", "verify", "verification", "secure",
    "account", "update", "banking", "confirm", "password", "wallet",
    "suspend", "unlock", "billing", "webscr", "authenticate", "recover",
)
