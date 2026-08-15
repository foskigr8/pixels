#!/usr/bin/env python3
"""
krea_server.py -- warm Krea 2 Turbo Edit server for Kaggle T4 x2.

Loads the model once and holds it on GPU0, then serves generations over HTTP so
the agent can call it repeatedly without paying the 3-5 minute load each time.

    GET  /health    -> liveness + what actually got loaded
    GET  /stats     -> rolling generate_s history (use this before optimising)
    POST /generate  -> one image
    POST /shutdown  -> free the GPU

Two things about this file are worth understanding before you change it.

1. T4 is Turing (SM75). It has no bf16. Krea 2 ships bf16 weights and the
   Wan2GP loaders assume bf16 in several places, so every bf16 tensor has to
   become fp16 before anything touches a kernel. That happens in two layers:

     - source patches from scripts/patches/t4_fp16.json, applied to the Wan2GP
       checkout before import. Fast path, but they are string matches against
       upstream, and upstream drifts. When a patch stops matching, this server
       prints "!! NO MATCH" and refuses to start unless you override.

     - a post-load sweep (force_fp16_inplace) that walks every module and casts
       any bf16 parameter or buffer that survived. This does not depend on
       upstream source at all, so it is the actual guarantee. The patches are
       an optimisation; the sweep is the safety net. The sweep's converted
       count tells you whether the patches did their job -- a large count means
       patches are missing and you are casting late.

2. The Wan2GP generate signature drifts too. Rather than hardcode it, this
   server introspects the resolved function and passes only the kwargs it
   actually accepts, logging anything it had to drop. Run

       python krea_server.py --probe

   to dump the real API surface of your pinned checkout without loading weights.

Pin WAN2GP_COMMIT to a SHA. Every silent-failure story in this project traces
back to running against a moving main.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# configuration (all overridable from the environment)
# ---------------------------------------------------------------------------

WAN2GP_DIR = Path(os.environ.get("WAN2GP_DIR", "/kaggle/working/Wan2GP"))
OUT_DIR = Path(os.environ.get("KREA_OUT_DIR", "/kaggle/working/repo/shots"))
PATCH_FILE = Path(
    os.environ.get(
        "KREA_PATCH_FILE",
        str(Path(__file__).resolve().parent / "patches" / "t4_fp16.json"),
    )
)

HOST = os.environ.get("KREA_HOST", "127.0.0.1")
PORT = int(os.environ.get("KREA_PORT", "8711"))

MODEL_TYPE = os.environ.get("KREA_MODEL_TYPE", "krea2_turbo_edit")
DEVICE = os.environ.get("KREA_DEVICE", "cuda:0")

# Refuse to start when a required patch no longer matches upstream. Set to 1
# only when you have confirmed the post-load sweep is covering for it.
ALLOW_UNPATCHED = os.environ.get("KREA_ALLOW_UNPATCHED", "0") == "1"

# Defaults tuned for iteration speed on a single T4, not for final quality.
DEF_WIDTH = int(os.environ.get("KREA_WIDTH", "1024"))
DEF_HEIGHT = int(os.environ.get("KREA_HEIGHT", "576"))
DEF_STEPS = int(os.environ.get("KREA_STEPS", "8"))
DEF_GUIDANCE = float(os.environ.get("KREA_GUIDANCE", "1.0"))

MAX_PIXELS = int(os.environ.get("KREA_MAX_PIXELS", str(1536 * 1536)))
MAX_REFS = 2

_LOG_LOCK = threading.Lock()


def log(msg: str) -> None:
    with _LOG_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# layer 1: source patches
# ---------------------------------------------------------------------------


def apply_source_patches() -> dict[str, Any]:
    """Apply the fp16 patch table to the Wan2GP checkout.

    Each patch is idempotent: if `replace` is already present the patch counts
    as applied. A patch whose `find` string is absent (and whose `replace` is
    also absent) means upstream moved -- that is reported loudly, because a
    silently no-op patch is exactly how the original notebook broke.
    """
    if not PATCH_FILE.exists():
        log(f"!! patch file missing: {PATCH_FILE}")
        return {"applied": 0, "total": 0, "missing": ["<patch file not found>"]}

    patches = json.loads(PATCH_FILE.read_text())["patches"]
    applied, already, missing = [], [], []

    for p in patches:
        target = WAN2GP_DIR / p["file"]
        if not target.exists():
            missing.append(f"{p['name']} (no such file: {p['file']})")
            continue

        text = target.read_text()
        if p["find"] not in text:
            # No work to do. Either it was already applied (replacement present)
            # or upstream moved and this patch is now dead weight.
            (already if p["replace"] in text else missing).append(p["name"])
            continue

        # count -1 means replace every occurrence in the file.
        target.write_text(text.replace(p["find"], p["replace"], p.get("count", -1)))
        applied.append(p["name"])

    total = len(patches)
    ok = len(applied) + len(already)
    log(f"[patch] applied {ok}/{total}  (fresh={len(applied)} already={len(already)})")
    for name in applied:
        log(f"[patch]   + {name}")
    for name in already:
        log(f"[patch]   = {name} (already present)")
    for name in missing:
        log(f"[patch]   !! NO MATCH: {name}")

    if missing:
        log("[patch] !! upstream drifted. Fix scripts/patches/t4_fp16.json against")
        log("[patch] !! your pinned WAN2GP_COMMIT before debugging anything else.")
        required = {p["name"] for p in patches if p.get("required", True)}
        if required & set(missing) and not ALLOW_UNPATCHED:
            raise SystemExit(
                "refusing to start: required fp16 patch did not match upstream.\n"
                "Set KREA_ALLOW_UNPATCHED=1 to start anyway and rely on the "
                "post-load fp16 sweep (slower load, but it does work)."
            )

    return {"applied": ok, "total": total, "missing": missing}


# ---------------------------------------------------------------------------
# layer 2: post-load fp16 sweep -- the actual guarantee
# ---------------------------------------------------------------------------


def force_fp16_inplace(obj: Any, _seen: set[int] | None = None) -> dict[str, int]:
    """Cast every surviving bf16 parameter/buffer to fp16, in place.

    Walks nn.Module attributes recursively because the pipeline holds several
    sibling models (transformer, text encoder, vision tower, VAE) that are not
    children of a single root module. Quantised (int8/quanto) tensors are left
    alone -- only bf16 is touched.
    """
    import torch
    import torch.nn as nn

    if _seen is None:
        _seen = set()
    stats = {"converted": 0, "failed": 0, "modules": 0}

    def convert_module(m: "nn.Module") -> None:
        stats["modules"] += 1
        for name, p in list(m.named_parameters(recurse=False)):
            if getattr(p, "dtype", None) is torch.bfloat16:
                try:
                    p.data = p.data.to(torch.float16)
                    stats["converted"] += 1
                except Exception:
                    stats["failed"] += 1
        for name, b in list(m.named_buffers(recurse=False)):
            if getattr(b, "dtype", None) is torch.bfloat16:
                try:
                    setattr(m, name, b.to(torch.float16))
                    stats["converted"] += 1
                except Exception:
                    stats["failed"] += 1

    def walk(o: Any, depth: int) -> None:
        if depth > 4 or id(o) in _seen:
            return
        _seen.add(id(o))
        if isinstance(o, nn.Module):
            for sub in o.modules():
                if id(sub) not in _seen:
                    _seen.add(id(sub))
                    convert_module(sub)
            return
        if isinstance(o, (list, tuple)):
            for item in o:
                walk(item, depth + 1)
            return
        if isinstance(o, dict):
            for item in o.values():
                walk(item, depth + 1)
            return
        if hasattr(o, "__dict__"):
            for item in list(vars(o).values()):
                walk(item, depth + 1)

    walk(obj, 0)
    return stats


# ---------------------------------------------------------------------------
# Wan2GP adapter
# ---------------------------------------------------------------------------


class Wan2GPAdapter:
    """Thin, drift-tolerant wrapper around the Wan2GP entry points.

    Everything upstream-specific is isolated here. If a boot fails, it fails
    with the list of names that were searched, not with an AttributeError.
    """

    def __init__(self) -> None:
        self.wgp = None
        self.model = None
        self.gen_fn = None
        self.gen_params: set[str] = set()
        self.load_report: dict[str, Any] = {}
        self.lock = threading.Lock()

    # -- import -------------------------------------------------------------

    def import_wgp(self):
        if str(WAN2GP_DIR) not in sys.path:
            sys.path.insert(0, str(WAN2GP_DIR))
        # Wan2GP resolves model/config paths relative to its own root.
        os.chdir(WAN2GP_DIR)
        import wgp  # type: ignore

        self.wgp = wgp
        return wgp

    @staticmethod
    def _resolve(obj: Any, names: list[str]):
        for n in names:
            fn = getattr(obj, n, None)
            if callable(fn):
                return n, fn
        return None, None

    # -- probe --------------------------------------------------------------

    def probe(self) -> None:
        """Dump the real API surface of the pinned checkout. No weights loaded."""
        wgp = self.import_wgp()
        print(f"\nWan2GP at {WAN2GP_DIR}")
        print(f"module file: {getattr(wgp, '__file__', '?')}\n")

        interesting = [
            n
            for n in dir(wgp)
            if not n.startswith("_")
            and callable(getattr(wgp, n, None))
            and any(
                k in n.lower()
                for k in ("model", "load", "generate", "config", "offload", "setup")
            )
        ]
        print("-- candidate callables in wgp --")
        for n in sorted(interesting):
            try:
                sig = inspect.signature(getattr(wgp, n))
            except (TypeError, ValueError):
                sig = "(?)"
            print(f"  {n}{sig}")

        krea = WAN2GP_DIR / "models" / "krea2" / "krea2_main.py"
        print(f"\n-- models/krea2/krea2_main.py present: {krea.exists()} --")
        if krea.exists():
            text = krea.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                s = line.strip()
                if s.startswith("def ") or s.startswith("class ") or "video_prompt_type" in s:
                    print(f"  {i:5d}: {s[:140]}")

        print("\n-- patch table dry run --")
        for p in json.loads(PATCH_FILE.read_text())["patches"]:
            t = WAN2GP_DIR / p["file"]
            if not t.exists():
                state = "FILE MISSING"
            else:
                body = t.read_text()
                if p["find"] in body:
                    state = "matches"
                elif p["replace"] in body:
                    state = "already applied"
                else:
                    state = "!! NO MATCH"
            print(f"  [{state}] {p['name']}  ({p['file']})")

    # -- load ---------------------------------------------------------------

    def load(self) -> None:
        import torch

        t0 = time.time()
        patch_report = apply_source_patches()

        log(f"[load] importing wgp from {WAN2GP_DIR}")
        wgp = self.import_wgp()

        loader_name, loader = self._resolve(
            wgp, ["load_models", "load_model", "get_model", "setup_model"]
        )
        if loader is None:
            raise RuntimeError(
                "could not find a model loader in wgp. Searched: load_models, "
                "load_model, get_model, setup_model. Run --probe and update "
                "Wan2GPAdapter._resolve candidates."
            )
        log(f"[load] loader = wgp.{loader_name}{inspect.signature(loader)}")
        log(f"[load] model_type = {MODEL_TYPE}  (must be an _edit variant for refs)")
        if not MODEL_TYPE.endswith("_edit"):
            log("[load] !! WARNING: non-_edit model type. Reference images will be")
            log("[load] !! silently ignored -- the vision tower is not loaded.")

        result = loader(MODEL_TYPE)
        # Loaders return either the model or a (model, offloadobj, ...) tuple.
        self.model = result[0] if isinstance(result, tuple) else result
        log(f"[load] model object: {type(self.model).__name__}")

        gen_name, gen_fn = self._resolve(
            self.model, ["generate", "__call__", "generate_image", "run"]
        )
        if gen_fn is None:
            raise RuntimeError(
                f"no generate entry point on {type(self.model).__name__}. "
                "Run --probe and inspect models/krea2/krea2_main.py."
            )
        self.gen_fn = gen_fn
        try:
            sig = inspect.signature(gen_fn)
            self.gen_params = set(sig.parameters)
            has_kwargs = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            self.gen_params, has_kwargs = set(), True
        log(f"[load] generate = {type(self.model).__name__}.{gen_name}, "
            f"{len(self.gen_params)} params, **kwargs={has_kwargs}")
        self._accepts_kwargs = has_kwargs

        log("[load] sweeping for surviving bf16 tensors...")
        sweep = force_fp16_inplace(self.model)
        log(f"[load] fp16 sweep: converted={sweep['converted']} "
            f"failed={sweep['failed']} modules={sweep['modules']}")
        if sweep["converted"] > 64:
            log("[load] !! large sweep count -- source patches are probably not")
            log("[load] !! matching. Load is correct but slower than it should be.")

        torch.cuda.empty_cache()
        self.load_report = {
            "patches": patch_report,
            "fp16_sweep": sweep,
            "loader": loader_name,
            "generate": gen_name,
            "generate_params": sorted(self.gen_params),
            "load_s": round(time.time() - t0, 1),
        }
        log(f"[load] ready in {self.load_report['load_s']}s")

    # -- generate -----------------------------------------------------------

    def call_generate(self, wanted: dict[str, Any]) -> Any:
        """Pass only kwargs the resolved signature accepts; log the rest."""
        if self._accepts_kwargs or not self.gen_params:
            return self.gen_fn(**wanted)
        accepted = {k: v for k, v in wanted.items() if k in self.gen_params}
        dropped = sorted(set(wanted) - set(accepted))
        if dropped:
            log(f"[gen] dropped unsupported kwargs: {dropped}")
            if "video_prompt_type" in dropped:
                log("[gen] !! video_prompt_type was dropped -- references WILL be")
                log("[gen] !! ignored. Check the signature with --probe.")
        return self.gen_fn(**accepted)


ADAPTER = Wan2GPAdapter()
HISTORY: deque = deque(maxlen=200)


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------


def to_pil(out: Any):
    """Normalise whatever generate returned into a single PIL image."""
    from PIL import Image
    import numpy as np
    import torch

    if isinstance(out, dict):
        for key in ("images", "image", "samples", "frames", "output"):
            if key in out:
                return to_pil(out[key])
        raise TypeError(f"dict output with no known image key: {list(out)}")

    if isinstance(out, (list, tuple)):
        if not out:
            raise TypeError("generate returned an empty sequence")
        return to_pil(out[0])

    if isinstance(out, Image.Image):
        return out

    if isinstance(out, torch.Tensor):
        t = out.detach().float().cpu()
        while t.ndim > 3:  # strip batch / frame dims
            t = t[0]
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.shape[0] in (1, 3, 4):  # CHW -> HWC
            t = t.permute(1, 2, 0)
        arr = t.numpy()
        if arr.min() < -0.05:  # [-1,1] -> [0,1]
            arr = (arr + 1.0) / 2.0
        if arr.max() <= 1.05:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype("uint8")
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        return Image.fromarray(arr)

    if isinstance(out, np.ndarray):
        return to_pil(torch.from_numpy(out))

    raise TypeError(f"cannot convert {type(out)} to an image")


def load_refs(paths: list[str]):
    from PIL import Image

    refs = []
    for p in paths:
        fp = Path(p)
        if not fp.is_absolute():
            # Relative paths are resolved against the repo, so the agent can say
            # "refs/hero.png" the same way it would in the editor.
            fp = OUT_DIR.parent / p
        if not fp.exists():
            raise FileNotFoundError(f"reference image not found: {p} (looked at {fp})")
        refs.append(Image.open(fp).convert("RGB"))
    return refs


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

app = FastAPI(title="krea-shot-factory", version="1.0")


class GenRequest(BaseModel):
    prompt: str
    out_name: str = Field(..., description="filename, e.g. shot_003.png")
    negative_prompt: str = ""
    width: int = DEF_WIDTH
    height: int = DEF_HEIGHT
    steps: int = DEF_STEPS
    guidance: float = DEF_GUIDANCE
    seed: int = -1
    ref_paths: list[str] = []
    # "KI" = first reference is the scene/base, stretched to canvas.
    # "I"  = every reference is a subject/object, aspect preserved.
    ref_mode: str = "KI"


@app.get("/health")
def health():
    import torch

    free = used = None
    if torch.cuda.is_available():
        f, t = torch.cuda.mem_get_info(0)
        free, used = round(f / 2**30, 2), round((t - f) / 2**30, 2)
    return {
        "status": "ready" if ADAPTER.model is not None else "loading",
        "model_type": MODEL_TYPE,
        "device": DEVICE,
        "edit_variant": MODEL_TYPE.endswith("_edit"),
        "out_dir": str(OUT_DIR),
        "vram_free_gb": free,
        "vram_used_gb": used,
        "generated": len(HISTORY),
        **ADAPTER.load_report,
    }


@app.get("/stats")
def stats():
    """Measure before you optimise. Compute floor at 1024x576/8 steps is 18-22s."""
    times = [h["generate_s"] for h in HISTORY]
    if not times:
        return {"n": 0, "note": "no generations yet"}
    s = sorted(times)
    return {
        "n": len(s),
        "min_s": s[0],
        "median_s": s[len(s) // 2],
        "max_s": s[-1],
        "mean_s": round(sum(s) / len(s), 1),
        "with_refs": sum(1 for h in HISTORY if h["refs"]),
        "recent": list(HISTORY)[-10:],
    }


@app.post("/generate")
def generate(req: GenRequest):
    import torch

    if ADAPTER.model is None:
        raise HTTPException(503, "model still loading -- poll /health")

    # -- validation, with the reasons the pipeline actually cares about ------
    if req.width % 16 or req.height % 16:
        raise HTTPException(
            400, f"width/height must be divisible by 16 (got {req.width}x{req.height})"
        )
    if req.width * req.height > MAX_PIXELS:
        raise HTTPException(400, f"{req.width}x{req.height} exceeds the pixel cap")
    if len(req.ref_paths) > MAX_REFS:
        raise HTTPException(
            400, f"max {MAX_REFS} reference images (Wan2GP enforces this too); "
            f"got {len(req.ref_paths)}"
        )
    if req.ref_paths and not MODEL_TYPE.endswith("_edit"):
        raise HTTPException(
            400,
            f"references passed but model_type is {MODEL_TYPE}. The vision tower "
            "is only loaded by the _edit variants; references would be silently "
            "dropped.",
        )
    if not req.out_name.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "out_name must end in .png or .jpg")
    if "/" in req.out_name or ".." in req.out_name:
        raise HTTPException(400, "out_name must be a bare filename")

    seed = req.seed if req.seed >= 0 else int(time.time() * 1000) % (2**31)
    refs = load_refs(req.ref_paths) if req.ref_paths else None

    # video_prompt_type MUST contain "I" or Wan2GP drops references on the
    # floor without warning:  reference_images = input_ref_images if "I" in ...
    vpt = req.ref_mode if refs else ""
    if refs and "I" not in vpt:
        vpt = vpt + "I"

    kwargs = {
        "input_prompt": req.prompt,
        "prompt": req.prompt,
        "n_prompt": req.negative_prompt,
        "negative_prompt": req.negative_prompt,
        "width": req.width,
        "height": req.height,
        "sampling_steps": req.steps,
        "num_inference_steps": req.steps,
        "guide_scale": req.guidance,
        "guidance_scale": req.guidance,
        "seed": seed,
        "input_ref_images": refs,
        "video_prompt_type": vpt,
        "batch_size": 1,
        "VAE_tile_size": 0,
    }

    with ADAPTER.lock:  # one generation at a time; 16 GB has no room for two
        t0 = time.time()
        try:
            out = ADAPTER.call_generate(kwargs)
        except Exception as e:
            log("[gen] FAILED:\n" + traceback.format_exc())
            torch.cuda.empty_cache()
            raise HTTPException(500, f"{type(e).__name__}: {e}")
        generate_s = round(time.time() - t0, 1)

        img = to_pil(out)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / req.out_name
        img.save(path)
        torch.cuda.empty_cache()

    rec = {
        "path": str(path),
        "generate_s": generate_s,
        "seed": seed,
        "size": f"{img.width}x{img.height}",
        "steps": req.steps,
        "refs": len(req.ref_paths),
        "ref_mode": vpt,
    }
    HISTORY.append(rec)
    log(f"[gen] {req.out_name}  {generate_s}s  seed={seed}  refs={len(req.ref_paths)}")
    return {"ok": True, **rec}


@app.post("/shutdown")
def shutdown():
    import torch

    ADAPTER.model = None
    torch.cuda.empty_cache()
    return {"ok": True, "note": "model dropped; restart the server to reload"}


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--probe",
        action="store_true",
        help="dump the pinned checkout's API surface and patch status, then exit",
    )
    args = ap.parse_args()

    if args.probe:
        ADAPTER.probe()
        return

    import uvicorn

    log(f"[boot] WAN2GP_DIR={WAN2GP_DIR}")
    log(f"[boot] OUT_DIR={OUT_DIR}")
    threading.Thread(target=_load_in_background, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def _load_in_background() -> None:
    try:
        ADAPTER.load()
    except SystemExit as e:
        log(f"[load] FATAL: {e}")
        os._exit(1)
    except Exception:
        log("[load] FATAL:\n" + traceback.format_exc())
        os._exit(1)


if __name__ == "__main__":
    main()
