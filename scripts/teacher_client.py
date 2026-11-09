"""Free-only teacher client for SA-2 (research/2026-10-22_sa_plan_v2.md §2 E).

Two providers, free tiers only:
- OpenCode Go (OPENCODE_GO_API_KEY): FREE_MODELS. Its other "-free" ids are unavailable outside the
  OpenCode app (Go: "Model is unavailable"; Zen: 403 FreeTierError), so only these two are listed.
- Gemini API free tier (GEMINI_API_KEY, user-confirmed no billing, 2026-10-22): GEMINI_FREE_MODELS.
  A Gemini 429 must name a FreeTier quota; any other quota id aborts (the project may be billed).

Rules:
- Only allowlisted model ids are ever requested. A reply that reports a different model, any
  billing/credit error, or HTTP 402 aborts the whole process (TeacherAbort). No fallback to paid.
- A model whose daily free quota is spent, or that stays unavailable, raises ModelExhausted so the
  caller can move to the next free model.
- Keys are read from .env or the environment and never printed.
- One request at a time, at least the provider's interval apart, with backoff on 429/5xx.
- Every call is cached in data/teacher/sa2_cache.jsonl, so re-runs cost no requests.

  .venv/Scripts/python.exe scripts/teacher_client.py --ping --model gemini-3.5-flash-lite
"""
import argparse
import hashlib
import json
import os
import pathlib
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_URL = "https://opencode.ai/inference/go/openai/v1/chat/completions"
FREE_MODELS = ("minimax-m3.1-flash-free", "beam-free")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
# Rotation order: flash-lite first (fast, usually the largest free daily quota), then flash.
# Each model has its own per-day free quota, so rotating gives the most free calls per day.
GEMINI_FREE_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite",
                      "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
                      "gemini-2.5-flash")
ALL_FREE_MODELS = FREE_MODELS + GEMINI_FREE_MODELS
USER_AGENT = "nano-slms-datagen/0.1"
MIN_INTERVAL = 1.0
GEMINI_INTERVAL = 6.5  # stays under a 10 RPM free limit; a per-minute 429 waits 60 s anyway
MAX_RETRIES = 5
RATE_RETRIES = 8  # 429 waits: 1, 2, 4, 8, 15, 15, 15, 15 min, then abort
TIMEOUT = 120
CACHE_PATH = ROOT / "data" / "teacher" / "sa2_cache.jsonl"
BILLING_WORDS = ("billing", "credit", "balance", "payment", "insufficient", "extra usage", "quota")


class TeacherAbort(RuntimeError):
    """Stop everything: continuing could bill or break the free-only rule."""


class ModelExhausted(RuntimeError):
    """This model's free quota is spent (or it stays unavailable); another free model may still work."""


def _read_key(name="OPENCODE_GO_API_KEY"):
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    key = os.environ.get(name)
    if not key:
        raise TeacherAbort(f"{name} not found in .env or environment")
    return key


class Teacher:
    def __init__(self, model=GEMINI_FREE_MODELS[0], session_id=None, cache_path=CACHE_PATH):
        if model not in ALL_FREE_MODELS:
            raise TeacherAbort(f"model {model!r} is not on the free allowlist")
        self.model = model
        self.gemini = model in GEMINI_FREE_MODELS
        self.url = GEMINI_URL if self.gemini else BASE_URL
        self.interval = GEMINI_INTERVAL if self.gemini else MIN_INTERVAL
        self.session_id = session_id or f"nano-slms-sa2-{uuid.uuid4().hex[:12]}"
        self._key = _read_key("GEMINI_API_KEY" if self.gemini else "OPENCODE_GO_API_KEY")
        self._last = 0.0
        self._lock = threading.Lock()
        self.cache_path = pathlib.Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row["text"]
        self.calls = 0

    def _cache_key(self, messages, max_tokens, temperature):
        blob = json.dumps([self.model, messages, max_tokens, temperature], sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _post(self, payload):
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                   "User-Agent": USER_AGENT}
        if not self.gemini:
            headers["x-opencode-session"] = self.session_id
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.load(resp)

    def _gemini_429(self, body):
        """Per-minute limit: wait. Per-day free quota: ModelExhausted. A non-free quota: abort."""
        if "freetier" not in body.replace("_", "").replace("-", ""):
            raise TeacherAbort("Gemini 429 does not name a FreeTier quota; the project may be billed, stopping")
        if "perday" in body:
            raise ModelExhausted(f"{self.model}: daily free quota spent")
        time.sleep(60)

    def chat(self, messages, max_tokens=700, temperature=0.2, json_mode=False):
        json_mode = json_mode and self.gemini  # OpenCode models get the plain request
        key = self._cache_key(messages + [{"json_mode": True}] if json_mode else messages, max_tokens, temperature)
        if key in self.cache:
            return self.cache[key]
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if self.gemini:
            payload["reasoning_effort"] = "low"
        if json_mode:
            # Without it Gemini sometimes escapes quotes inside list items and the reply is not valid JSON.
            payload["response_format"] = {"type": "json_object"}
        delay = 4.0
        attempt = 0
        rate_waits = 0
        while attempt < MAX_RETRIES:
            attempt += 1
            with self._lock:
                wait = self._last + self.interval - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            try:
                data = self._post(payload)
            except urllib.error.HTTPError as e:
                full = e.read().decode("utf-8", "replace")
                body = full[:300].lower()
                if self.gemini and e.code == 429 and rate_waits < RATE_RETRIES:
                    # Gemini's 429 text mentions "billing details" even on the free tier, so it is
                    # classified by quota id before the billing-word check.
                    self._gemini_429(full.lower())
                    rate_waits += 1
                    attempt -= 1
                    continue
                if self.gemini and e.code == 503 and attempt >= MAX_RETRIES:
                    raise ModelExhausted(f"{self.model}: still unavailable (503)") from None
                if e.code == 402 or any(w in body for w in BILLING_WORDS):
                    raise TeacherAbort(f"HTTP {e.code}: billing/quota signal, stopping") from None
                if e.code == 429 and rate_waits < RATE_RETRIES:
                    # The free tier rate-limits after a burst; waiting it out does not count as a retry.
                    time.sleep(min(60 * 2 ** rate_waits, 900))
                    rate_waits += 1
                    attempt -= 1
                    continue
                if e.code in (500, 502, 503, 504) and attempt < MAX_RETRIES:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise TeacherAbort(f"HTTP {e.code}: {body[:160]}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise TeacherAbort(f"network error: {e}") from None
            reported = str(data.get("model", self.model)).removeprefix("models/")
            if reported not in ALL_FREE_MODELS:
                raise TeacherAbort(f"reply came from model {reported!r}, not a free model")
            self.calls += 1
            text = data["choices"][0]["message"].get("content") or ""
            if not text.strip():
                # A reasoning model that spends max_tokens thinking returns no content; never cache that.
                return ""
            self.cache[key] = text
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "model": self.model, "text": text}) + "\n")
            return text
        raise TeacherAbort("retries exhausted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true", help="one tiny request; prints the reply and timing")
    ap.add_argument("--model", default=GEMINI_FREE_MODELS[0], choices=ALL_FREE_MODELS)
    args = ap.parse_args()
    if args.ping:
        t = Teacher(args.model)
        began = time.time()
        text = t.chat([{"role": "user", "content": "Write a Python function is_even(n). Code only."}],
                      max_tokens=200, temperature=0.0)
        print(f"model={t.model} calls={t.calls} seconds={time.time() - began:.1f}")
        print(text)


if __name__ == "__main__":
    main()
