"""Unit tests for URL parsing, feature extraction and data generation.

Run with:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishdet.data_gen import generate_dataset  # noqa: E402
from phishdet.evasion import generate_evasion_sets  # noqa: E402
from phishdet.features import (  # noqa: E402
    FEATURE_NAMES,
    extract_features,
    levenshtein,
    parse_url,
    shannon_entropy,
)


class TestParsing(unittest.TestCase):
    def test_registered_domain_simple(self):
        p = parse_url("https://accounts.google.com/signin")
        self.assertEqual(p.registered_domain, "google.com")
        self.assertEqual(p.subdomain, "accounts")
        self.assertEqual(p.tld, "com")

    def test_multi_part_suffix(self):
        self.assertEqual(parse_url("http://www.bbc.co.uk/news").registered_domain, "bbc.co.uk")

    def test_free_hosting_is_its_own_domain(self):
        p = parse_url("https://paypal-verify.web.app/login")
        self.assertEqual(p.registered_domain, "paypal-verify.web.app")
        self.assertEqual(p.tld, "web.app")

    def test_userinfo_is_not_the_host(self):
        # Browsers send this request to 203.0.113.5, not to paypal.com.
        p = parse_url("http://www.paypal.com@203.0.113.5/login")
        self.assertEqual(p.host, "203.0.113.5")
        self.assertTrue(p.is_ip)

    def test_malformed_does_not_raise(self):
        for bad in ["", "http://[::1", "not a url", "http://host:abc/"]:
            parse_url(bad)
            extract_features(bad)


class TestFeatures(unittest.TestCase):
    def test_feature_vector_complete(self):
        f = extract_features("https://example.com/")
        self.assertEqual(list(f), FEATURE_NAMES)

    def test_ip_and_at(self):
        f = extract_features("http://www.paypal.com@203.0.113.5/login")
        self.assertEqual(f["has_ip_host"], 1)
        self.assertEqual(f["has_at"], 1)
        self.assertEqual(f["is_https"], 0)

    def test_brand_on_official_domain_is_not_flagged(self):
        f = extract_features("https://www.paypal.com/signin")
        self.assertEqual(f["brand_in_url"], 1)
        self.assertEqual(f["brand_outside_official_domain"], 0)
        self.assertEqual(f["brand_lookalike"], 0)

    def test_brand_outside_official_domain(self):
        f = extract_features("https://paypal-secure-login.xyz/verify")
        self.assertEqual(f["brand_outside_official_domain"], 1)
        self.assertEqual(f["suspicious_tld"], 1)
        self.assertGreaterEqual(f["n_sensitive_words"], 2)

    def test_lookalikes(self):
        for url in ["http://paypa1.com", "https://rnicrosoft.com/login", "https://amazom-help.net",
                    "https://conibase.top"]:
            self.assertEqual(extract_features(url)["brand_lookalike"], 1, url)

    def test_subdomains_ignore_www(self):
        self.assertEqual(extract_features("https://www.example.com")["n_subdomains"], 0)
        self.assertEqual(extract_features("http://paypal.com.secure.verify.evil.tk")["n_subdomains"], 4)

    def test_path_depth_and_query(self):
        f = extract_features("https://a.com/x/y/z.php?a=1&b=2")
        self.assertEqual(f["path_depth"], 3)
        self.assertEqual(f["n_query_params"], 2)

    def test_entropy_and_levenshtein(self):
        self.assertEqual(shannon_entropy("aaaa"), 0.0)
        self.assertAlmostEqual(shannon_entropy("ab"), 1.0)
        self.assertEqual(levenshtein("kitten", "sitting"), 3)


class TestDataGeneration(unittest.TestCase):
    def test_deterministic_and_balanced(self):
        a = generate_dataset(n_per_class=300, seed=7)
        b = generate_dataset(n_per_class=300, seed=7)
        self.assertTrue(a.equals(b))
        self.assertEqual(a.label.value_counts().to_dict(), {0: 300, 1: 300})
        self.assertFalse(a.url.duplicated().any())

    def test_evasion_levels_respect_constraints(self):
        ev = generate_evasion_sets(n_per_level=200, seed=3)
        for level in (1, 2, 3, 4):
            feats = [extract_features(u) for u in ev[ev.level == level].url]
            self.assertTrue(all(f["is_https"] == 1 for f in feats))
            self.assertTrue(all(f["suspicious_tld"] == 0 for f in feats))
        for level in (2, 3, 4):
            feats = [extract_features(u) for u in ev[ev.level == level].url]
            self.assertTrue(all(f["has_ip_host"] == 0 for f in feats))
            # L4 copies the *legitimate* path distribution, which includes
            # '/@user' profile paths, so '@' is only banned at L2-L3.
            if level < 4:
                self.assertTrue(all(f["has_at"] == 0 for f in feats))
        for u in ev[ev.level == 3].url:
            f = extract_features(u)
            self.assertEqual(f["brand_in_url"], 0, u)
            self.assertEqual(f["n_sensitive_words"], 0, u)


if __name__ == "__main__":
    unittest.main()
