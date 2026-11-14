"""Write-time scrub: .env secrets, token patterns, host identifiers and PII never reach traces/results.

Synthetic values only; identities are injected, nothing is read from the real denylist.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lexsi_unlearn import scrub  # noqa: E402
from lexsi_unlearn import traces  # noqa: E402

SECRET = "zz-env-" + "value-Q9x8W7v6U5"
TOKEN = "hf_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3"
# Built from pieces so this file itself contains no path/IP/email literal for the commit-time redactor to rewrite.
HOME_PATH = "/" + "Users" + "/zed"
IP = "10.1." + "2.3"
MAIL = "a" + "@" + "mail.org"


class ScrubTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / ".env").write_text(f"MY_SERVICE_TOKEN={SECRET}\nPLAIN_SETTING=1\n")
        ids = [scrub.Identity("HOST_GPU", "bare", "gpu-box-1"), scrub.Identity("HOST_SPARK", "bare", "lab-spark-9")]
        scrub._ENGINE_CACHE[str(self.tmp)] = scrub.Engine(ids, scrub.secret_values_from_env(self.tmp), scrub.Allow())
        self.default_key = str(scrub._default_root())
        self.saved_default = scrub._ENGINE_CACHE.get(self.default_key)
        scrub._ENGINE_CACHE[self.default_key] = scrub._ENGINE_CACHE[str(self.tmp)]  # writers use the repo root

    def tearDown(self):
        scrub._ENGINE_CACHE.pop(str(self.tmp), None)
        scrub._ENGINE_CACHE.pop(self.default_key, None)
        if self.saved_default is not None:
            scrub._ENGINE_CACHE[self.default_key] = self.saved_default

    def test_scrub_text_replaces_env_values_tokens_hosts_paths(self):
        text = f"key={SECRET} {TOKEN} ssh lab-spark-9 gpu-box-1 {HOME_PATH}/proj {IP} {MAIL}"
        out = scrub.scrub_text(text, self.tmp)
        for leaked in (SECRET, TOKEN, "lab-spark-9", "gpu-box-1", HOME_PATH, IP, MAIL):
            self.assertNotIn(leaked, out)
        self.assertIn("<REDACTED:MY_SERVICE_TOKEN>", out)
        self.assertIn("<REDACTED:huggingface-token>", out)
        self.assertIn("<HOST_GPU>", out)

    def test_scrub_obj_keeps_structure(self):
        obj = {"a": [1, 2.5, None, True, {"k": f"x {SECRET}"}], "gpu-box-1": ("t", "lab-spark-9")}
        out = scrub.scrub_obj(obj, self.tmp)
        self.assertEqual(out["a"][:4], [1, 2.5, None, True])
        self.assertEqual(out["a"][4], {"k": "x <REDACTED:MY_SERVICE_TOKEN>"})
        self.assertEqual(out["<HOST_GPU>"], ("t", "<HOST_SPARK>"))
        json.dumps(out)

    def test_normal_trace_content_untouched(self):
        row = {"id": "20261113T050438Z-unlearn--151b939a:eval:forget:direct:0", "value": 0.875, "text": "Who wrote it? 2026-11-13 12:30 v1.2.3"}
        self.assertEqual(scrub.scrub_obj(row, self.tmp), row)

    def test_trace_writer_scrubs_at_write_time(self):
        path = self.tmp / "traces" / "r.jsonl"
        traces.append_jsonl(path, [{"id": "r:1", "run_id": "r", "case_id": "c", "model": "m", "note": f"{SECRET} on gpu-box-1"}])
        written = path.read_text()
        self.assertNotIn(SECRET, written)
        self.assertNotIn("gpu-box-1", written)
        self.assertEqual(json.loads(written)["note"], "<REDACTED:MY_SERVICE_TOKEN> on <HOST_GPU>")

    def test_release_scrub_applies_shared_scrub(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import redact_release as R
        out = R.redact_string(f"token {SECRET} on gpu-box-1", [])
        self.assertIn("<REDACTED:MY_SERVICE_TOKEN>", out)
        self.assertNotIn("gpu-box-1", out)

    def test_failure_falls_back_to_token_patterns(self):
        key = "/nonexistent-root"
        scrub._ENGINE_CACHE[key] = None  # engine unusable -> AttributeError inside scrub()
        try:
            self.assertEqual(scrub.scrub_text(f"t {TOKEN}", Path(key)), "t <REDACTED:huggingface-token>")
        finally:
            scrub._ENGINE_CACHE.pop(key, None)


if __name__ == "__main__":
    unittest.main()
