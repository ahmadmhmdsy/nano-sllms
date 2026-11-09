"""End-to-end U-line demo: type a task -> a UI spec -> real, rendered HTML.

Five modes, all sharing ui/src/render.py:

  --prompt "a settings page with an email toggle"   generate once, write HTML, open it
  --interactive                                    REPL
  --serve                                          local GUI on 127.0.0.1
  --sample 12                                      render N real corpus specs (no GPU)
  --self-test                                      exercise the server offline, no GPU

BACKENDS. The GUI stays useful while the GPU is busy, and upgrades itself the moment
the card frees:

  auto    use the model if torch sees a CUDA device, else fall back to corpus
  model   real generation (needs the GPU, loads lazily on first request)
  corpus  serve REAL production_50k specs by keyword match, clearly labelled as corpus
          output. Never presented as model output.

The model backend decodes exactly as the U-1b gate does (greedy, raw "task\n" prompt),
so the demo cannot flatter the model with a friendlier decoder.

    .venv/Scripts/python.exe -m ui.scripts.ui_demo --serve --backend auto
    .venv/Scripts/python.exe -m ui.scripts.ui_demo --self-test
"""
import argparse
import json
import pathlib
import random
import re
import sys
import threading
import webbrowser

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ui.src.chain import chain_to_spec, spec_to_chain
from ui.src.render import CSS, render_document
from ui.src.validate import validate_chain

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scratch" / "ui_demo"
WORD = re.compile(r"[a-z0-9]+")


def _has_cuda():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# --------------------------------------------------------------------- backends
class ModelBackend:
    """Real generation. Loads lazily so --serve starts instantly."""

    name = "model"
    label = "MODEL (runs/u1_fused/final)"

    def __init__(self, model_dir, max_new_tokens, ctx=512):
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.ctx = ctx
        self._tok = None
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, dtype=torch.float16
            ).cuda().eval()
        return self._tok, self._model

    def generate(self, task):
        import torch

        with self._lock:  # one generation at a time: 8 GB card, no co-run
            tok, model = self._load()
            ids = tok(task.strip() + "\n", return_tensors="pt").to("cuda")
            budget = max(32, min(self.max_new_tokens, self.ctx - ids.input_ids.shape[1] - 4))
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=budget,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            txt = tok.batch_decode(out[:, ids.input_ids.shape[1]:], skip_special_tokens=True)
        return txt[0].split("<|endoftext|>")[0].strip()

    def chain(self, task):
        return self.generate(task)


class CorpusBackend:
    """Real production_50k specs, matched by keyword overlap. No GPU, no model.

    This exists so the GUI is demonstrable while the GPU is busy. Every response is
    tagged source='corpus' and the banner says so, so it can never be mistaken for
    model output.
    """

    name = "corpus"
    label = "CORPUS (real specs, NOT model output)"

    def __init__(self, corpus, limit=4000, seed=0):
        self.rows = []
        with open(corpus, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= limit:
                    break
                r = json.loads(line)
                if r.get("spec"):
                    self.rows.append(
                        {
                            "task": r["task"],
                            "spec": r["spec"],
                            "chain": r.get("chain", ""),
                            "bag": set(WORD.findall(r["task"].lower())),
                            "arch": r.get("arch_hint") or r.get("arch") or "?",
                            "split": r.get("split", "?"),
                        }
                    )
        if not self.rows:
            raise SystemExit(f"no usable specs in {corpus}")

    def chain(self, task):
        # production_50k/corpus.jsonl stores only {arch_hint, spec, split, task} - there is
        # NO chain field, which is why prepare_u1.py derives one via spec_to_chain.
        # Do the same here so the chain pane shows the real UI1 chain surface.
        want = set(WORD.findall(task.lower()))
        best = (
            max(self.rows, key=lambda r: (len(want & r["bag"]) / max(len(want | r["bag"]), 1)))
            if want
            else self.rows[0]
        )
        c = best.get("chain") or ""
        if not c:
            try:
                c = spec_to_chain(best["spec"])
            except Exception as exc:
                c = f"(chain derivation failed: {type(exc).__name__}: {exc})"
        return c, best["spec"]


def _issue_str(x):
    """validate_chain returns Issue objects, not strings - the HTTP layer needs text."""
    if isinstance(x, str):
        return x
    for attr in ("code", "kind", "type"):
        v = getattr(x, attr, None)
        if v:
            return str(v)
    return str(x)


def describe(chain, limit=40):
    """Run a chain through the real validator. Returns (valid, spec, issues)."""
    try:
        issues = [x for x in validate_chain(chain) if x is not None]
    except Exception as exc:
        return False, None, [f"validator crashed: {type(exc).__name__}: {exc}"]
    spec = None
    try:
        spec, cissues = chain_to_spec(chain)
        issues += [x for x in (cissues or []) if x is not None]
    except Exception as exc:
        issues.append(f"chain_to_spec failed: {type(exc).__name__}: {exc}")
    ok = not issues
    head = [_issue_str(x) for x in issues[:limit]]
    if len(issues) > limit:
        head.append(f"... +{len(issues) - limit} more")
    return ok, spec, head

PAGE = '''<!doctype html><html><head><meta charset=utf-8><title>U-line demo</title>
<style>__CSS__</style><style>
body{margin:0}
.demo{display:grid;grid-template-rows:auto auto 1fr;height:100vh}
.bar{display:flex;gap:10px;padding:12px 16px;background:#161b22;
  border-bottom:1px solid #262d36;align-items:flex-start}
.bar textarea{flex:1;min-height:58px;resize:vertical}
.egrow{padding:0 16px 10px;background:#161b22;display:flex;gap:7px;flex-wrap:wrap;
  border-bottom:1px solid #262d36}
.eg{font-size:12px;padding:4px 10px}
.eg b{color:#79c0ff;font-weight:600}
.eghint{font-size:11px;color:#8b949e;align-self:center;padding-right:4px}
.status{font-size:12px;color:#8b949e;align-self:center;min-width:250px}
.split{display:grid;grid-template-columns:1fr 1fr;min-height:0}
.pane{overflow:auto;padding:14px 18px}
.pane.left{background:#0d1117;border-right:1px solid #262d36}
.pane h4{margin:14px 0 5px;color:#8b949e;font-size:11px;text-transform:uppercase;
  letter-spacing:.06em}
.pane pre{white-space:pre-wrap;word-break:break-word;font-size:12px;color:#c9d1d9;margin:0}
iframe{width:100%;height:100%;border:0;background:#0e1117}
.pass{color:#3fb950}
.fail{color:#f85149}
</style></head><body><div class="demo">
<div class="bar">
  <textarea id="q" placeholder="describe the screen you want, e.g. a settings page with an email notification toggle"></textarea>
  <button class="btn primary" onclick="go()">Generate</button>
  <div class="status" id="st">__LABEL__</div>
</div>
<div class="egrow">__EXAMPLES__</div>
<div class="split">
  <div class="pane left">
    <h4>chain</h4><pre id="chain">-</pre>
    <h4>validation</h4><pre id="issues">-</pre>
    <h4>spec (json)</h4><pre id="spec">-</pre>
  </div>
  <div class="pane"><iframe id="out" sandbox="allow-same-origin"></iframe></div>
</div></div>
<script>
document.querySelectorAll(".eg").forEach(function (b) {
  b.addEventListener("click", function () {
    document.getElementById("q").value = b.getAttribute("data-q");
    go();
  });
});
async function go() {
  var q = document.getElementById("q").value.trim();
  if (!q) return;
  var st = document.getElementById("st");
  st.textContent = "generating...";
  document.getElementById("out").srcdoc = "<p style='padding:20px;color:#8b949e'>generating...</p>";
  var r = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task: q })
  });
  var d = await r.json();
  document.getElementById("chain").textContent = d.chain || "(empty)";
  var v = document.getElementById("issues");
  if (d.valid) { v.className = "pass"; v.textContent = "PASS - 0 issues"; }
  else { v.className = "fail"; v.textContent = "FAIL - " + JSON.stringify((d.issues || []).slice(0, 6)); }
  document.getElementById("spec").textContent = d.spec ? JSON.stringify(d.spec, null, 1) : "-";
  document.getElementById("out").srcdoc = d.html || "<p style='padding:20px'>no html</p>";
  st.textContent = (d.source || "") + " - " + (d.elements || 0) + " elements in " + (d.seconds || 0) + "s";
}
document.getElementById("q").addEventListener("keydown", function (e) {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) go();
});
</script></body></html>'''


def _attr(s):
    s = str(s)
    for a, b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"),
                 (chr(34), "&quot;"), (chr(39), "&#39;")):
        s = s.replace(a, b)
    return s


def diverse_examples(corpus_path, limit=6):
    """One UNSEEN (val/test-split) prompt per archetype.

    production_50k/corpus.jsonl is ordered by archetype, so the first N rows are N
    near-identical prompts from ONE archetype - and they are train-split rows the
    model has MEMORISED (valid 0.56 there vs 0.04 unseen). Offering those made the
    demo look far better than the model is.
    """
    got, out = set(), []
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("spec"):
                continue
            arch = r.get("arch_hint") or r.get("arch") or "?"
            if arch in got:
                continue
            if r.get("split") not in ("val", "validation", "test"):
                continue
            got.add(arch)
            out.append({"task": r["task"], "arch": arch, "split": r.get("split", "?")})
            if len(out) >= limit:
                break
    return out


def gui_page(backend_label, examples):
    # One chip per archetype, each labelled with its split. corpus.jsonl is ordered
    # by archetype, so the old "first 6 rows" chips were all one archetype and all
    # train-split (memorised) - which flattered the model enormously.
    ex = "".join(
        "<button class='btn eg' data-q={q} title={t}><b>{a}</b> {s}</button>".format(
            q=_attr(e["task"]), a=_attr(e.get("arch", "?")),
            s=_attr(e["task"][:44]),
            t=_attr(str(e.get("arch", "?")) + " | split=" + str(e.get("split", "?"))))
        for e in examples
    )
    hint = ("<span class='eghint'>UNSEEN prompts, one per archetype - the hard case:</span>") if examples else ""
    ex = hint + ex
    return (PAGE.replace("__CSS__", CSS)
                .replace("__LABEL__", backend_label)
                .replace("__EXAMPLES__", ex))



def run_one(task, backend, show=True):
    """Generate (or look up) a spec for one task, validate it, and return everything."""
    out = backend.chain(task)
    chain, borrowed = (out if isinstance(out, tuple) else (out, None))
    valid, spec, issues = (
        describe(chain) if chain else (False, None, ["no chain produced"])
    )
    if spec is None and borrowed is not None:
        spec = borrowed
    n_el = len((spec or {}).get("elements") or {})
    if show:
        print("=" * 72)
        print("TASK   :", task)
        print(f"SOURCE : {backend.name.upper()}   VALID: {'PASS' if valid else 'FAIL'}   "
              f"elements={n_el}   chain_chars={len(chain or '')}")
        print("CHAIN  :", (chain or "")[:400] + ("..." if len(chain or "") > 400 else ""))
        if issues:
            print("ISSUES :", json.dumps(issues[:6], ensure_ascii=False)[:500])
    return {"task": task, "chain": chain or "", "spec": spec, "valid": valid,
            "issues": issues, "source": backend.name}


def make_backend(args):
    if args.backend == "corpus":
        return CorpusBackend(args.corpus, limit=args.corpus_limit)
    if args.backend == "model" or _has_cuda():
        return ModelBackend(args.model, args.max_new_tokens)
    print("no CUDA device -> falling back to the corpus backend "
          "(real specs, clearly labelled; not model output)", flush=True)
    return CorpusBackend(args.corpus, limit=args.corpus_limit)


def serve(args):
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    backend = make_backend(args)
    cb = CorpusBackend(args.corpus, limit=200) if backend.name == "corpus" else None
    try:
        examples = diverse_examples(args.corpus, limit=6)
    except Exception as exc:
        examples = []
        print("WARN diverse_examples failed: " + str(exc))
    page = gui_page(backend.label, examples)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            raw = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page)
            elif self.path == "/api/health":
                self._send(200, json.dumps({"backend": backend.name, "label": backend.label}),
                           "application/json")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/generate":
                return self._send(404, "not found", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            if n > 64_000:
                return self._send(413, "too big", "text/plain")
            try:
                task = json.loads(self.rfile.read(n) or b"{}").get("task", "")
            except Exception:
                return self._send(400, "bad json", "text/plain")
            task = str(task)[:2000].strip()
            if not task:
                return self._send(400, "empty task", "text/plain")
            t0 = time.time()
            try:
                rec = run_one(task, backend, show=False)
            except Exception as exc:
                return self._send(
                    500,
                    json.dumps({"error": f"{type(exc).__name__}: {exc}",
                                "hint": "model backend needs a free GPU; try --backend corpus"}),
                    "application/json",
                )
            spec = rec["spec"] or {}
            meta = f"{'PASS' if rec['valid'] else 'FAIL'} - {backend.name} - {task[:110]}"
            html = render_document(spec, title=task[:80], meta=meta) if spec else                 "<p style='padding:20px;color:#8b949e'>no spec produced</p>"
            self._send(200, json.dumps({
                "chain": rec["chain"][:4000], "valid": rec["valid"], "issues": rec["issues"],
                "spec": spec, "html": html, "source": backend.name,
                "elements": len(spec.get("elements") or {}),
                "seconds": round(time.time() - t0, 1),
            }), "application/json")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", args.port), H)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"U-line demo GUI: {url}", flush=True)
    print(f"backend: {backend.label}   (ctrl+enter to generate)", flush=True)
    if args.open:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


def self_test(args):
    """Exercise the whole HTTP path offline. Needs no GPU and no browser."""
    import time
    import urllib.request
    from http.server import BaseHTTPRequestHandler, HTTPServer

    backend = CorpusBackend(args.corpus, limit=300)
    page = gui_page(backend.label, [{"task": "a", "arch": "x", "split": "test"},
                                       {"task": "b", "arch": "y", "split": "test"}])

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            raw = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self._send(200, page, "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            task = json.loads(self.rfile.read(n) or b"{}").get("task", "")
            if not str(task).strip():
                return self._send(400, json.dumps({"error": "empty task"}))
            rec = run_one(str(task), backend, show=False)
            spec = rec["spec"] or {}
            self._send(200, json.dumps({
                "chain": rec["chain"][:4000], "valid": rec["valid"], "issues": rec["issues"],
                "spec": spec, "source": backend.name,
                "html": render_document(spec, title=str(task)[:80]),
                "elements": len(spec.get("elements") or {}),
            }))

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)  # port 0 -> OS picks a free one
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    checks = []

    r = urllib.request.urlopen(base + "/")
    body = r.read().decode("utf-8")
    checks.append(("GET / returns the GUI", r.status == 200 and "<textarea" in body))
    checks.append(("CSS is injected", ".metric .val" in body and "<style>" in body))
    checks.append(("example chips present", "data-q=" in body))
    checks.append(("iframed render pane", "iframe" in body and "srcdoc" in body))

    req = urllib.request.Request(
        base + "/api/generate",
        data=json.dumps({"task": backend.rows[0]["task"]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    d = json.loads(urllib.request.urlopen(req).read().decode())
    checks.append(("POST returns 1 spec", d["elements"] > 0))
    checks.append(("POST returns html", "<!doctype html>" in d["html"] and len(d["html"]) > 2000))
    checks.append(("html contains a real component", "class='card'" in d["html"]
                   or 'class="card"' in d["html"] or "badge" in d["html"]))
    checks.append(("spec echoes elements", isinstance(d["spec"].get("elements"), dict)))
    checks.append(("labelled as corpus", d["source"] == "corpus"))
    checks.append(("chain is non-empty", len(d.get("chain", "")) > 0))

    req2 = urllib.request.Request(base + "/api/generate", data=b"{}",
                                  headers={"Content-Type": "application/json"})
    got = False
    try:
        urllib.request.urlopen(req2)
    except urllib.error.HTTPError as e:
        got = e.code == 400
    checks.append(("empty task rejected 400", got))

    srv.shutdown()
    n_ok = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


def sample(args):
    """Render real corpus specs - no model, no GPU."""
    rows = []
    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    pick = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = [
        "<!doctype html><meta charset=utf-8><title>U-line corpus render</title>",
        f"<style>{CSS}</style><div class='wrap'><h1>U-line: corpus spec &rarr; UI</h1>"
        f"<p class='text muted'>{len(pick)} real specs from {args.corpus}</p>",
    ]
    for i, r in enumerate(pick):
        try:
            sp, ci = chain_to_spec(r["chain"])
        except Exception as exc:
            sp, ci = r["spec"], [str(exc)]
        html = render_document(sp, title=r["task"][:80], meta=r["task"][:120])
        fn = f"s{i:02d}.html"
        (OUT_DIR / fn).write_text(html, encoding="utf-8")
        idx.append(
            f'<div class="card" style="margin-bottom:10px"><div class="hd">'
            f'<div class="t"><a style="color:#58a6ff" href="{fn}">{r["task"][:80]}</a></div>'
            f'<div class="d">{len(sp.get("elements") or {})} elements - {fn}</div></div></div>'
        )
    (OUT_DIR / "index.html").write_text("".join(idx) + "</div>", encoding="utf-8")
    print(f"wrote {len(pick)} pages + index.html to {OUT_DIR}")
    print(f"open: {OUT_DIR / 'index.html'}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="runs/u1_fused/final")
    ap.add_argument("--prompt")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--serve", action="store_true", help="local GUI on 127.0.0.1")
    ap.add_argument("--self-test", action="store_true", help="offline end-to-end check")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--sample", type=int, default=0, help="render N real corpus specs")
    ap.add_argument("--backend", choices=["auto", "model", "corpus"], default="auto")
    ap.add_argument("--corpus", default="ui/data/space_bunny/production_50k/corpus.jsonl")
    ap.add_argument("--corpus-limit", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--out", default=str(OUT_DIR / "demo.html"))
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test(args))
    if args.sample:
        sample(args)
        return
    if args.serve:
        serve(args)
        return

    backend = make_backend(args)
    print(f"backend: {backend.label}", flush=True)
    if args.interactive:
        print("type a task, 'q' to quit.", flush=True)
        while True:
            try:
                task = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if task.lower() in ("q", "quit", "exit"):
                break
            if task:
                run_one(task, backend)
        return
    if args.prompt:
        rec = run_one(args.prompt, backend)
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = f"{'PASS' if rec['valid'] else 'FAIL'} - {backend.name} - {args.prompt[:110]}"
        p.write_text(render_document(rec["spec"] or {}, title=args.prompt[:80], meta=meta),
                     encoding="utf-8")
        print("\nwrote", p)
        if args.open:
            webbrowser.open(p.as_uri())
        return
    ap.print_help()


if __name__ == "__main__":
    main()
