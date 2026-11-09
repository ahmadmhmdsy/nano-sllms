"""FastAPI layer for the flow/ MVP (Task 7).

A THIN wrapper around the Task 2-6 modules (graph_schema, nodes, flows,
config_gen, runner): every route only parses the request, calls one
already-tested function and maps a typed error to an HTTP status. All
real logic lives in those imported modules (or in this module's pure
helpers below, which the contract tests exercise directly).

Error -> HTTP status mapping (also codified in status_for_error()):
- runner.JobRunningError / runner.NoJobRunningError (typed runner errors) -> 409
- flows/config_gen validation ValueError                                 -> 400 {"errors": [...]}
- flows.load FileNotFoundError (no such flow)                            -> 404 {"detail": ...}

Port resolution: --port wins when given; otherwise FLOW_PORT env; default 3010.
Prod static serving: mounts flow/dist at "/" when NOT --dev and dist exists
(dev mode needs nothing - the frontend dev server proxies /api).
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from flow.server import config_gen, flows, nodes, runner

FLOW_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = FLOW_ROOT / "dist"

# Model-graph store (F2 Task 4): flow/models/<slug>.modelgraph.json,
# git-tracked like flow/flows/ (same directory-store pattern; tests pass
# a models_dir and never touch the repo one).
MODELS_DIR = FLOW_ROOT / "models"
_MODEL_SUFFIX = ".modelgraph.json"
_MODEL_FORMAT = "model/0.1"

DEFAULT_PORT = 3010
# CORS is for the frontend dev server only (Vite on 5174, historical 5173).
DEV_ORIGINS = ("http://localhost:5174", "http://localhost:5173")

DEFAULT_HOST = "127.0.0.1"


# ---------------------------------------------------------------------------
# Pure helpers (contract-tested)
# ---------------------------------------------------------------------------

def resolve_port(port_arg, env=None):
    """--port wins when present; else FLOW_PORT env; else DEFAULT_PORT.

    env overrides os.environ for tests (None = use os.environ). A
    non-integer FLOW_PORT falls back to the default port, never crashes.
    """
    if port_arg is not None:
        return int(port_arg)
    if env is None:
        env = os.environ
    raw = env.get("FLOW_PORT")
    if raw:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return DEFAULT_PORT
    return DEFAULT_PORT


def error_list(exc):
    """Split a validator ValueError (newline-joined reasons) into a list."""
    return [line for line in str(exc).split("\n") if line]


def status_for_error(exc):
    """HTTP status for an exception of a backend module (409/404/400).

    runner.JobRunningError subclasses ValueError, so the typed runner
    errors are checked FIRST (409 wins over the generic 400).
    """
    if isinstance(exc, (runner.JobRunningError, runner.NoJobRunningError)):
        return 409           # typed runner errors: busy / no live job
    if isinstance(exc, FileNotFoundError):
        return 404           # no such flow
    if isinstance(exc, ValueError):
        return 400           # flows/config_gen validation reasons
    return 500


def validate_preflight(g):
    """Collect every validation error WITHOUT writing into the repo.

    Runs graph_schema.validate and the per-node registry prop checks (via
    flows' dual-validator collector), then a config_gen.generate dry-run
    through a process-local tempfile out_dir so NO repo file is touched by
    validation (reserved names / rows cap surface from there too).
    Returns a list of human-readable reason strings; [] means valid.
    """
    errors = list(flows.collect_validation_errors(g))
    try:
        with tempfile.TemporaryDirectory(prefix="flow-validate-") as out_dir:
            config_gen.generate(g, out_dir=out_dir)
    except ValueError as exc:
        errors.extend(error_list(exc))
    return errors


def flows_validation_response(exc):
    """400 JSONResponse carrying the newline-split validation reasons."""
    return JSONResponse(status_code=400, content={"errors": error_list(exc)})


def model_slug(name):
    """Validate a model name with the SAME rule as flows.save/load.

    Delegates to flows._validate_slug (deliberate private reuse: flow
    names and model names must share ONE slug validation) and relabels
    the error's "flow name:" prefix to "model name:" so the 400 message
    names what the client actually sent.
    """
    try:
        return flows._validate_slug(name)
    except ValueError as exc:
        raise ValueError(str(exc).replace("flow name:", "model name:", 1)) from exc


def validate_model_graph(body):
    """Honest model/0.1 body validation; returns [] when valid.

    Checks exactly what Task 4 pins: format == "model/0.1"; nodes is a
    list of objects carrying id/kind/props/position; edges is a list of
    objects carrying id/from/to/fromPort/toPort. The name is validated
    separately (model_slug) because it arrives in the URL, and "meta"
    stays free-form (backward-open format, like flow/0.1).
    """
    if not isinstance(body, dict):
        return ["body: expected a model/0.1 JSON object, got %s"
                % type(body).__name__]
    errors = []
    fmt = body.get("format")
    if fmt != _MODEL_FORMAT:
        errors.append("format: expected %r, got %r" % (_MODEL_FORMAT, fmt))
    node_list = body.get("nodes")
    if not isinstance(node_list, list):
        errors.append("nodes: expected a list, got %r" % (node_list,))
    else:
        for i, item in enumerate(node_list):
            if not isinstance(item, dict):
                errors.append("nodes[%d]: expected an object, got %s"
                              % (i, type(item).__name__))
                continue
            missing = [k for k in ("id", "kind", "props", "position")
                       if k not in item]
            if missing:
                errors.append("nodes[%d]: missing %s"
                              % (i, ", ".join(repr(k) for k in missing)))
    edge_list = body.get("edges")
    if not isinstance(edge_list, list):
        errors.append("edges: expected a list, got %r" % (edge_list,))
    else:
        for i, item in enumerate(edge_list):
            if not isinstance(item, dict):
                errors.append("edges[%d]: expected an object, got %s"
                              % (i, type(item).__name__))
                continue
            missing = [k for k in ("id", "from", "to", "fromPort", "toPort")
                       if k not in item]
            if missing:
                errors.append("edges[%d]: missing %s"
                              % (i, ", ".join(repr(k) for k in missing)))
    return errors


def list_models(models_dir=None):
    """Sorted model slugs from <models_dir|MODELS_DIR>; [] when absent.

    Mirrors flows.list_flows for the models directory: slugs (without
    the .modelgraph.json suffix), sorted; the GET /api/models route
    wraps this list in {"models": [...]} exactly like the /api/flows
    list response shape.
    """
    target = Path(models_dir) if models_dir is not None else MODELS_DIR
    if not target.is_dir():
        return []
    return sorted(
        p.name[: -len(_MODEL_SUFFIX)]
        for p in target.iterdir()
        if p.is_file() and p.name.endswith(_MODEL_SUFFIX)
    )


def save_model(name, body, models_dir=None):
    """Validate then write <models_dir|MODELS_DIR>/<slug>.modelgraph.json.

    Same contract as flows.save: a ValueError carrying the newline-joined
    reason list (body validation, or the shared slug error) is raised
    WITHOUT writing anything on violation; on success the document is
    written atomically via flows.atomic_write and the written Path is
    returned.
    """
    slug = model_slug(name)
    errors = validate_model_graph(body)
    if errors:
        raise ValueError("\n".join(errors))
    target = Path(models_dir) if models_dir is not None else MODELS_DIR
    path = target / (slug + _MODEL_SUFFIX)
    payload = json.dumps(body, indent=2, ensure_ascii=False) + "\n"
    return flows.atomic_write(path, payload)


def read_model(name, models_dir=None):
    """Load one model doc or return a 404 JSONResponse — mirrors
    flows.load for the models directory (honest detail message; never
    invents a doc)."""
    slug = model_slug(name)
    target = Path(models_dir) if models_dir is not None else MODELS_DIR
    path = target / (slug + _MODEL_SUFFIX)
    if not path.is_file():
        return JSONResponse(status_code=404,
                            content={"detail": "model '%s' does not exist" % slug})
    return json.loads(path.read_text(encoding="utf-8"))


def nodes_snapshot():
    """Registry view for the frontend (kinds/ports/props/gates)."""
    return {
        "valid_kinds": sorted(nodes.VALID_KINDS),
        "nodes": nodes.NODES,
        "gates": dict(nodes.GATES),
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(dev=False):
    """Build the FastAPI app. dev=True adds CORS for the Vite dev server
    and skips the flow/dist static mount; prod mounts dist when present."""
    app = FastAPI(title="flow server", docs_url=None, redoc_url=None,
                  openapi_url=None)
    if dev:
        app.add_middleware(
            CORSMiddleware, allow_origins=list(DEV_ORIGINS),
            allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/nodes")
    def get_nodes():
        return nodes_snapshot()

    @app.get("/api/flows")
    def list_all_flows():
        return flows.list_flows()

    @app.get("/api/flows/{name}")
    def read_flow(name: str):
        try:
            return flows.load(name)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"detail": str(exc)})
        except ValueError as exc:
            return flows_validation_response(exc)

    @app.put("/api/flows/{name}")
    async def write_flow(name: str, request: Request):
        try:
            graph = await request.json()
        except Exception:
            return JSONResponse(status_code=400,
                                content={"errors": ["body: invalid JSON"]})
        try:
            flows.save(name, graph)
        except ValueError as exc:
            return flows_validation_response(exc)
        return {"ok": True, "name": name}

    @app.delete("/api/flows/{name}")
    def remove_flow(name: str):
        try:
            flows.delete_flow(name)
        except (FileNotFoundError, ValueError) as exc:
            return JSONResponse(status_code=status_for_error(exc),
                                content={"detail": str(exc)})
        return {"ok": True}

    @app.get("/api/models")
    def list_all_models():
        return {"models": list_models()}

    @app.get("/api/models/{name}")
    def get_single_model(name: str):
        """Load one model graph doc; 404 with an honest detail message.
        Mirrors GET /api/flows/{name} for the models directory."""
        res = read_model(name)
        if isinstance(res, JSONResponse):
            return res
        return res

    @app.post("/api/models/{name}")
    async def write_model(name: str, request: Request):
        try:
            doc = await request.json()
        except Exception:
            return JSONResponse(status_code=400,
                                content={"errors": ["body: invalid JSON"]})
        try:
            save_model(name, doc)
        except ValueError as exc:
            return flows_validation_response(exc)
        return {"ok": True, "name": name}

    @app.post("/api/validate")
    async def validate(request: Request):
        try:
            graph = await request.json()
        except Exception:
            return JSONResponse(status_code=400,
                                content={"errors": ["body: invalid JSON"]})
        errors = validate_preflight(graph)
        return {"ok": not errors, "errors": errors}

    @app.post("/api/run")
    async def run_flow(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400,
                                content={"errors": ["body: invalid JSON"]})
        name = body.get("name") if isinstance(body, dict) else None
        try:
            graph = flows.load(name)
        except ValueError as exc:
            return flows_validation_response(exc)
        except FileNotFoundError as exc:
            return JSONResponse(status_code=404, content={"detail": str(exc)})
        # Same preflight the validate endpoint runs: an invalid graph is a
        # 400 with reasons (never a started job generated from junk).
        errors = validate_preflight(graph)
        if errors:
            return JSONResponse(status_code=400, content={"errors": errors})
        try:
            # Deliverable configs/flow_<slug>.yaml first, then start.
            config_path = config_gen.generate(graph)
            pid = runner.start(config_path)
        except runner.JobRunningError as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            return JSONResponse(status_code=status_for_error(exc),
                                content={"detail": str(exc)})
        return {"ok": True, "pid": pid}

    @app.get("/api/run/status")
    def run_status():
        return runner.status()

    @app.post("/api/run/stop")
    def run_stop():
        try:
            stop_flag = runner.request_stop()
        except runner.NoJobRunningError as exc:
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        except RuntimeError as exc:
            return JSONResponse(status_code=status_for_error(exc),
                                content={"detail": str(exc)})
        return {"ok": True, "stop_flag": stop_flag}

    if not dev and DIST_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True),
                  name="static")
    return app


app = create_app(dev=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="flow MVP FastAPI server")
    parser.add_argument("--port", type=int, default=None,
                        help="server port (default "
                             + str(DEFAULT_PORT) + "; FLOW_PORT env wins "
                             "when --port is absent)")
    parser.add_argument("--dev", action="store_true",
                        help="dev mode: CORS for the Vite dev server, no "
                             "flow/dist static mount")
    parser.add_argument("--host", default=DEFAULT_HOST)
    args = parser.parse_args(argv)
    port = resolve_port(args.port, os.environ)
    import uvicorn
    uvicorn.run(create_app(dev=args.dev), host=args.host, port=port)


if __name__ == "__main__":
    main()
