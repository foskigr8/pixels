#!/usr/bin/env python3
"""
Studio server -- Krea 2 Turbo Edit on a Kaggle T4 x2.

Serves the UI and the API from one process on one port, so there is a single
thing to start and a single tunnel to open.

    /                 the UI
    /api/health       status, VRAM, what loaded
    /api/generate     make an image
    /api/shots        what has been generated

Two things worth knowing before changing this file.

1. T4 is Turing: no bf16. Krea ships bf16 weights, so two patches in
   patches/t4_fp16.json rewrite the loader before import, and a post-load
   sweep casts anything they missed. The sweep is the guarantee; the patches
   are the fast path.

2. Speed comes from mmgp, and mmgp is bounded by host RAM, not by the GPU.
   18 GB of weights stream through 30 GB of system RAM. If the pinned working
   set does not fit, mmgp silently drops to partial pinning and you lose async
   transfers -- worth roughly 30s/image. Watch the load log for "partial
   pinning" -- if it is gone, the transfers are async and you are at speed.

3. mmgp 3.7.12 is single-GPU by construction. Read its source: it hardcodes
   `torch.cuda.get_device_properties(0)` for VRAM capacity, sets every hooked
   model's `_force_device` to the bare string "cuda", and every weight transfer
   is `p.to("cuda")` -- the *current* device. There is no device argument, no
   device map, no per-model placement. So the only way to use the second T4 is
   to keep a model OUT of the dict handed to offload.profile() and place it on
   cuda:1 ourselves. Anything moved to cuda:1 *after* profile() is silently
   dragged back to cuda:0 by the first gpu_load_blocks() call.

   That is what gpu1_text_tower() does, and it fixes both problems at once:
   the Qwen3-VL text+vision tower (~4.3 GB of the 18.1 GB pin requirement)
   stops being mmgp's problem, which drops the requirement under the reservable
   ceiling and restores FULL pinning, and GPU1 finally does real work.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("STUDIO_DIR", Path(__file__).resolve().parent.parent))
WAN2GP = Path(os.environ.get("WAN2GP_DIR", "/kaggle/working/Wan2GP"))
SHOTS = Path(os.environ.get("SHOTS_DIR", REPO / "shots"))
REFS = REPO / "refs"
FRAMES = REPO / "frames"
UI = REPO / "ui"
PATCHES = Path(__file__).resolve().parent / "patches" / "t4_fp16.json"

HOST, PORT = os.environ.get("HOST", "0.0.0.0"), int(os.environ.get("PORT", "8711"))
MODEL_TYPE = os.environ.get("MODEL_TYPE", "krea2_turbo_edit")
DEVICE, DEVICE2 = os.environ.get("DEVICE", "cuda:0"), os.environ.get("DEVICE2", "cuda:1")

TRANSFORMER = "Krea2Turbo_quanto_bf16_int8.safetensors"
TE_DIR = "Qwen3-VL-4B-Instruct"
TE_FILE = f"{TE_DIR}_quanto_bf16_int8.safetensors"

# Transformer budget is exposed so a session with VRAM headroom can push it up
# toward the model size (12,863 MiB at int8) to keep more of it resident and
# shave a few seconds per image. The 0.8*VRAM cap (~13,107 MiB) is mmgp's own
# safety ceiling and clamps anything above it. The default 11000 leaves room
# for activations; raising it risks CUDA OOM if activations spill.
TRANSFORMER_BUDGET = int(os.environ.get("TRANSFORMER_BUDGET_MB", "11000"))
# mmgp caps any single budget at 80% of VRAM (~12.8 GiB on a 16 GiB T4), so a
# larger number here is silently clamped and just prints a warning.
BUDGETS = {"transformer": TRANSFORMER_BUDGET, "text_encoder": 4000, "vae": 1500, "*": 800}

# mmgp reserves at most perc_reserved_mem_max of system RAM for pinned weights.
# VERIFIED against mmgp 3.7.12 source: offload.all() declares it as a real
# keyword argument, and offload.profile() forwards every **overrideKwargs
# straight into all(), so passing it to profile() is correct. At the 0.50
# default the ceiling is ~16 GB of Kaggle's 31 GB. Belt-and-braces headroom on
# top of the GPU1 split below. If the session gets OOM-killed, lower it.
# (mmgp also reads a lowercase `perc_reserved_mem_max` OS env var as a
# fallback, so `export perc_reserved_mem_max=0.65` is an equivalent lever.)
PERC_RESERVED = float(os.environ.get("PERC_RESERVED_MEM", "0.65"))

# Run the Qwen3-VL text+vision tower on the second T4 instead of streaming it
# through GPU0. This is done by REMOVING it from the dict given to
# offload.profile(), so mmgp never hooks it and never fights the placement --
# see note 3 in the module docstring. Two wins: GPU1 stops being idle, and
# mmgp's pin requirement drops by ~4.3 GB (3467 MB text_encoder + 792 MB
# vision_encoder), which is what pushes it back under the reservable ceiling
# and restores full pinning. Set SPLIT_GPUS=0 to fall back to single-GPU.
SPLIT = os.environ.get("SPLIT_GPUS", "1") == "1"
# Keys in Wan2GP's krea2 `pipe` dict that belong to that one tower object
# (pipe["text_encoder"] is model.text_encoder.language_model and
#  pipe["vision_encoder"] is model.text_encoder.visual -- both submodules of
#  model.text_encoder, which is what we relocate). The VAE stays on GPU0: it is
# small, it runs once, and its call sites take no device argument to hook.
AUX = ("text_encoder", "vision_encoder")

STYLE = ("minimal hand-drawn stickman on a pure white background, thick 6px solid "
         "black monoline ink strokes, flat saturated color fills, no gradients, "
         "no shadows, whiteboard explainer illustration")
NEGATIVE = ("photorealistic, 3d render, gradient, drop shadow, soft shading, "
            "textured paper, grey background, watermark, blurry")
# Optional, opt-in only. Nothing here is applied unless explicitly selected.
PRESETS = {
    "House style": STYLE,
    "Hook": "a relatable everyday situation, one or two literal props, lots of white space",
    "Cross-section": "anatomical or mechanical cross-section, interior revealed as cartoon machinery with gears",
    "Data / HUD": "a semicircular gauge with a needle, simple bar readout, hand-lettered uppercase label",
    "Metaphor": "an abstract concept as a physical contraption, stickman reacting, motion arrows",
    "Climax": "the consequence landing, exaggerated reaction, radiating emphasis lines, bold uppercase text",
}

HISTORY: deque = deque(maxlen=200)
LOCK = threading.Lock()
STATE: dict[str, Any] = {"ready": False, "report": {}}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def patch_source() -> dict:
    """Rewrite the Wan2GP loader for fp16. Idempotent."""
    if not PATCHES.exists():
        return {"applied": 0, "total": 0, "missing": ["patch file not found"]}
    entries = json.loads(PATCHES.read_text())["patches"]
    ok, missing = 0, []
    for p in entries:
        f = WAN2GP / p["file"]
        if not f.exists():
            missing.append(p["name"])
            continue
        body = f.read_text()
        if p["replace"] in body:
            ok += 1
        elif p["find"] in body:
            f.write_text(body.replace(p["find"], p["replace"], p.get("count", 1)))
            ok += 1
        else:
            missing.append(p["name"])
            if p.get("required"):
                raise SystemExit(
                    f"required fp16 patch '{p['name']}' no longer matches upstream.\n"
                    f"Fix the find-string in {PATCHES}, or set ALLOW_UNPATCHED=1."
                    if os.environ.get("ALLOW_UNPATCHED") != "1" else ""
                )
    log(f"[patch] {ok}/{len(entries)} applied" + (f", missing {missing}" if missing else ""))
    return {"applied": ok, "total": len(entries), "missing": missing}


def _is_quantized(t) -> bool:
    """True for quanto/GGUF-style packed weights.

    These report their DEQUANTIZED dtype -- an int8 quanto weight says
    `bfloat16` -- but raise "The dtype of a weights Tensor cannot be changed"
    if you try to cast them. They are already the compact form we want, so
    there is nothing to do; recognising them is purely about not touching them.
    """
    return "q" in type(t).__name__.lower() and hasattr(t, "__torch_dispatch__")


def sweep_fp16(model) -> int:
    """Cast any bf16 tensor the patches missed.

    Best-effort by design: this is the backstop behind the source patches and
    the dtype= arguments, so a tensor it cannot convert is a skip, never a
    failure. Letting it raise would abort a model that had already loaded.
    """
    import torch
    import torch.nn as nn

    n, seen, skipped = 0, set(), 0

    def walk(o, d=0):
        nonlocal n, skipped
        if d > 4 or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, nn.Module):
            for m in o.modules():
                for _, p in list(m.named_parameters(recurse=False)):
                    if p.dtype is not torch.bfloat16 or _is_quantized(p.data):
                        continue
                    try:
                        p.data = p.data.to(torch.float16); n += 1
                    except Exception:
                        skipped += 1
                for name, b in list(m.named_buffers(recurse=False)):
                    if b.dtype is not torch.bfloat16 or _is_quantized(b):
                        continue
                    try:
                        setattr(m, name, b.to(torch.float16)); n += 1
                    except Exception:
                        skipped += 1
            return
        if isinstance(o, (list, tuple)):
            [walk(i, d + 1) for i in o]
        elif isinstance(o, dict):
            [walk(v, d + 1) for v in o.values()]
        elif hasattr(o, "__dict__"):
            [walk(v, d + 1) for v in vars(o).values()]

    walk(model)
    if skipped:
        log(f"[fp16] {skipped} tensor(s) could not be cast (already quantized) -- fine")
    return n


def self_test(model, split: dict) -> dict:
    """Prove the loaded pipeline works BEFORE the UI is told it is ready.

    Runs two tiny generations through the exact path /api/generate uses -- one
    text-only and one with a real corpus frame as the reference image -- so the
    dual-GPU split and the img2img/reference path are verified at boot instead
    of being discovered by the user's first shot. Each probe times a couple of
    denoising steps, which also warms the transformer into VRAM so the first
    real shot does not pay the cold-start cost.

    A failed probe is logged loudly but does not stop the server: a broken
    probe is better than a silent text-only generation later, and the health
    endpoint surfaces the verdict so a returning UI can see it.
    """
    import torch
    from PIL import Image

    # Divisible by 16 (and by 32/64 for safety against the mmdit patch size)
    # so the pipeline's width%align check can never reject the probe.
    W, H = 512, 256
    out: dict[str, Any] = {"res": f"{W}x{H}", "steps": 2}

    tower = getattr(model, "text_encoder", None)
    if isinstance(tower, torch.nn.Module):
        try:
            dev = next(tower.parameters()).device
            out["text_tower_device"] = str(dev)
            out["on_gpu1"] = bool(split.get("enabled")) and str(dev) == str(DEVICE2)
        except StopIteration:
            out["text_tower_device"] = "empty"
    else:
        out["text_tower_device"] = "missing"

    def probe(tag: str, **extra) -> None:
        t0 = time.time()
        try:
            res = model.generate(
                seed=7,
                input_prompt="minimal hand-drawn stickman on a pure white background",
                n_prompt="", sampling_steps=2, width=W, height=H, guide_scale=0.0,
                batch_size=1, loras_slists={"phase1": []}, **extra)
            out[tag] = {"ok": bool(res is not None), "s": round(time.time() - t0, 1)}
        except Exception as e:
            torch.cuda.empty_cache()
            out[tag] = {"ok": False, "s": round(time.time() - t0, 1),
                        "error": f"{type(e).__name__}: {e}"}
        torch.cuda.empty_cache()

    probe("text")
    ref = None
    if FRAMES.exists():
        for d in sorted(FRAMES.iterdir()):
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    ref = Image.open(p).convert("RGB")
                    break
            if ref is not None:
                break
    if ref is not None:
        probe("with_ref", input_ref_images=[ref], video_prompt_type="KI")
    else:
        out["with_ref"] = {"ok": None, "note": "no corpus frame found"}

    out["ok"] = bool(out["text"].get("ok")) and bool(out["with_ref"].get("ok"))
    log(f"[selftest] text tower on {out.get('text_tower_device')} "
        f"({'GPU1' if out.get('on_gpu1') else 'GPU0/CPU'})")
    log(f"[selftest] text-only probe {out['text'].get('s')}s -> "
        f"{'OK' if out['text'].get('ok') else 'FAILED: ' + (out['text'].get('error') or '?')}")
    if out["with_ref"].get("ok") is None:
        log(f"[selftest] reference probe skipped ({out['with_ref'].get('note')})")
    else:
        log(f"[selftest] reference probe {out['with_ref'].get('s')}s -> "
            f"{'OK' if out['with_ref'].get('ok') else 'FAILED: ' + (out['with_ref'].get('error') or '?')}")
    if not out["ok"]:
        log("[selftest] !!! probe failed -- the dual-GPU split or reference path "
            "is broken; the UI will show the verdict on /api/health")
    return out


def gpu1_text_tower(model, pipe: dict) -> dict:
    """Pin the Qwen3-VL text+vision tower to GPU1. MUST run before offload.profile().

    Mutates `pipe` in place, removing the entries mmgp would otherwise hook, so
    that model stays entirely ours: mmgp never pins it to host RAM, never
    streams it, and never rewrites its device.

    The bridge is a one-line seam that Wan2GP already provides for us. The
    conditioner's signature is

        Qwen3VLConditioner.forward(self, text, device, images=None)

    and everything it builds -- token ids, masks, position ids, pixel values --
    is explicitly placed on that `device` argument. So passing cuda:1 moves the
    whole encode onto GPU1 with no tensor-chasing. Results come back on their
    own: the caller (Krea2Pipeline._encode_prompts) finishes with an explicit
    `.to(device=<cuda:0>)` on the returned hiddens and masks, and the
    TextEncoderCache stores on CPU and re-materialises on the device it was
    handed. Nothing downstream ever sees a cuda:1 tensor.
    """
    import torch

    if not SPLIT:
        return {"enabled": False, "reason": "SPLIT_GPUS=0", "moved": []}
    if torch.cuda.device_count() < 2:
        return {"enabled": False, "reason": "one GPU", "moved": []}

    # model is Wan2GP's krea2 `model_factory`. It exposes the tower directly as
    # .text_encoder, but the conditioner that wraps it hangs off the inner
    # Krea2Pipeline as .pipeline.encoder -- NOT .encoder. Check both so a
    # refactor upstream degrades to single-GPU instead of crashing.
    tower = getattr(model, "text_encoder", None)                    # Krea2TextEncoder
    enc = getattr(getattr(model, "pipeline", None), "encoder", None)
    if enc is None:
        enc = getattr(model, "encoder", None)                       # Qwen3VLConditioner
    if not isinstance(tower, torch.nn.Module) or enc is None or not callable(getattr(enc, "forward", None)):
        return {"enabled": False, "reason": "krea2 pipeline shape changed", "moved": []}
    if getattr(enc, "qwen", tower) is not tower:
        return {"enabled": False, "reason": "conditioner wraps a different tower", "moved": []}

    # Take them out of mmgp's hands first; put them back if anything goes wrong,
    # because a half-applied split is worse than no split.
    held = {k: pipe.pop(k) for k in AUX if k in pipe}
    if not held:
        return {"enabled": False, "reason": "no aux entries in pipe", "moved": []}

    try:
        tower.to(DEVICE2)
        if not getattr(enc, "_studio_on_gpu1", False):
            orig = enc.forward

            def encode_on_gpu1(text, device=None, images=None, **kw):
                # torch.cuda.device() so that any bare .cuda()/"cuda" inside the
                # tower resolves to GPU1 too -- mmgp leaves the global default
                # device set to 'cuda', which is GPU0.
                with torch.cuda.device(DEVICE2), torch.device(DEVICE2):
                    return orig(text, device=DEVICE2, images=images, **kw)

            enc.forward = encode_on_gpu1
            enc._studio_on_gpu1 = True
    except Exception as e:
        pipe.update(held)
        try:
            tower.to(DEVICE)
        except Exception:
            pass
        log(f"[split] failed ({type(e).__name__}: {e}) -- staying on one GPU")
        return {"enabled": False, "reason": f"{type(e).__name__}", "moved": []}

    return {"enabled": True, "target": DEVICE2, "moved": sorted(held)}


def load() -> None:
    import torch

    t0 = time.time()
    patches = patch_source()

    sys.path.insert(0, str(WAN2GP))
    os.chdir(WAN2GP)
    from mmgp import offload
    from shared.utils import files_locator as fl
    from models.krea2.krea2_handler import family_handler

    fl.set_checkpoints_paths(["models", "ckpts", "."])
    tf = os.path.join("models", TRANSFORMER)
    te = os.path.join("models", TE_DIR, TE_FILE)
    for label, p in (("transformer", tf), ("text encoder", te)):
        if not os.path.exists(p):
            raise SystemExit(f"{label} missing: {WAN2GP}/{p} -- re-run the weights cell")

    log(f"[load] {MODEL_TYPE}")
    model, pipe = family_handler.load_model(
        model_filename=tf, model_type=MODEL_TYPE, base_model_type=MODEL_TYPE,
        model_def=family_handler.query_model_def(MODEL_TYPE, {}),
        quantizeTransformer=False, dtype=torch.float16, VAE_dtype=torch.float16,
        text_encoder_filename=te,
    )

    # Order matters. The tower has to leave `pipe` BEFORE profile() runs, or
    # mmgp hooks it, pins it, and owns its device for the rest of the process.
    split = gpu1_text_tower(model, pipe)
    if split["enabled"]:
        log(f"[split] text tower -> {DEVICE2}, removed {split['moved']} from mmgp; "
            "pin requirement should drop ~4.3 GB")
    else:
        log(f"[split] single-GPU ({split.get('reason')}) -- GPU1 will stay idle")

    # mmgp resolves every weight transfer against the *current* CUDA device, so
    # make sure that is GPU0 before it builds its streams and hooks.
    torch.cuda.set_device(0)

    log(f"[load] mmgp budgets={BUDGETS}")
    # perc_reserved_mem_max is a genuine offload.all() kwarg in 3.7.12 and
    # profile() forwards **overrideKwargs into all() verbatim -- no fallback
    # needed, and a TypeError here would mean the mmgp version changed.
    import contextlib

    profile_buf = io.StringIO()
    with contextlib.redirect_stdout(profile_buf):
        offload.profile(pipe, profile_no=2, quantizeTransformer=False,
                        convertWeightsFloatTo=torch.float16, pinnedMemory=True,
                        asyncTransfers=True, budgets=dict(BUDGETS),
                        perc_reserved_mem_max=PERC_RESERVED)
    offload.shared_state["_attention"] = "sdpa"
    profile_log = profile_buf.getvalue()
    # mmgp can report partial pinning two ways: the estimate-level check in
    # all() before pinning, or inside _pin_to_memory() under RAM pressure.
    # Catch both so health never claims FULL when transfers are actually off.
    partial_markers = ("Switching to partial pinning",
                       "The model was partially pinned",
                       "Unable to pin more tensors")
    pinning = "PARTIAL" if any(m in profile_log for m in partial_markers) else "FULL"
    if pinning == "PARTIAL":
        log("[load] PINNING=PARTIAL -- async transfers are OFF; shots will be slow "
            "(the split is on, so the fix is a higher PERC_RESERVED_MEM or more RAM)")
    else:
        log("[load] pinning=FULL -- async transfers on, at speed")

    try:
        cast = sweep_fp16(model)
    except Exception as e:
        cast = 0
        log(f"[fp16] sweep failed ({type(e).__name__}) -- continuing; "
            "the dtype= arguments and source patches already cover the load")
    log(f"[load] fp16 sweep cast {cast} tensors | perc_reserved_mem_max={PERC_RESERVED} "
        f"| gpu1={split.get('moved') or 'none'}")

    torch.cuda.empty_cache()
    STATE["model"] = model
    STATE["report"] = {"patches": patches, "fp16_cast": cast, "device_split": split,
                       "pinning": pinning, "budgets": BUDGETS,
                       "load_s": round(time.time() - t0, 1)}
    # Prove the split + img2img path before telling the UI we are ready. This
    # costs ~15-30 s of boot but verifies the two things that decide whether
    # shots work and how fast they are, and it warms the transformer into VRAM.
    # Set SELF_TEST=0 to skip (e.g. while iterating on load errors).
    if os.environ.get("SELF_TEST", "1") == "1":
        st0 = time.time()
        STATE["report"]["selftest"] = self_test(model, split)
        log(f"[load] selftest done in {round(time.time() - st0, 1)}s")
    STATE["ready"] = True
    log(f"[load] ready in {STATE['report']['load_s']}s")


# ---------------------------------------------------------------------------
# api
# ---------------------------------------------------------------------------

import base64  # noqa: E402
import io  # noqa: E402
import re  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="studio")

SAFE = re.compile(r"[^a-zA-Z0-9 _-]")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
MAX_UPLOAD = int(float(os.environ.get("MAX_UPLOAD_MB", "32")) * 2**20)
# How many /api/generate calls may sit waiting on LOCK before we start refusing.
# Requests block a thread from the (finite) request threadpool while they wait,
# so an unbounded queue eventually starves /api/health and /api/shots -- exactly
# the endpoints a returning UI needs. Refusing early keeps the studio answerable.
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "3"))

# What generate() is doing right now, so a UI that was closed mid-render can come
# back and see that its work is still in flight instead of assuming it was lost.
# Guarded by GEN_LOCK (cheap, never held across a generation).
GEN: dict[str, Any] = {"busy": False, "waiting": 0, "project": None, "prompt": "", "started": 0.0}
GEN_LOCK = threading.Lock()


# --- errors ----------------------------------------------------------------
# The UI does `r.json().then(e => Promise.reject(e.detail))`, so every failure
# has to be JSON with a *string* `detail`. Bare tracebacks (plain-text 500) and
# FastAPI's default 422 (detail is a list of dicts) both render as noise.


@app.exception_handler(RequestValidationError)
def _bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    parts = []
    for e in exc.errors()[:4]:
        loc = ".".join(str(x) for x in e.get("loc", ())[1:]) or "body"
        parts.append(f"{loc}: {e.get('msg', 'invalid')}")
    return JSONResponse(status_code=422, content={"detail": "; ".join(parts) or "invalid request"})


@app.exception_handler(Exception)
def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    log(f"[api] unhandled on {request.url.path}\n" + traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


# --- names -----------------------------------------------------------------


def _clean(name: str) -> str:
    return SAFE.sub("", (name or "").strip())[:48].strip()


def slug(name: str) -> str:
    """Project name -> directory name. Never empty, never a path."""
    return _clean(name) or "untitled"


def safe_file(name: str) -> str:
    """A single file name inside a project directory. Rejects traversal.

    `..`, `/`, `\\`, NUL and absolute paths never survive: what comes back is a
    bare basename or a 400. Callers must use this, not Path(name).name, which
    silently turns `..` into `..`.
    """
    raw = (name or "").strip().replace("\\", "/")
    if "\x00" in raw:
        raise HTTPException(400, "invalid file name")
    n = Path(raw).name
    if not n or n in (".", "..") or "/" in n:
        raise HTTPException(400, "invalid file name")
    if len(n) > 128:
        raise HTTPException(400, "file name too long")
    return n


def pdirs(project: str, create: bool = True) -> tuple[Path, Path]:
    """(shots, refs) for a project. Created on demand unless create=False.

    Read endpoints pass create=False: a GET should never conjure a directory,
    or a typo'd project name litters the studio with empty projects.
    """
    p = slug(project)
    a, b = SHOTS / p, REFS / p
    if create:
        try:
            a.mkdir(parents=True, exist_ok=True)
            b.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise HTTPException(500, f"could not create project directory: {e}")
    return a, b


def _mtime(p: Path) -> float:
    """stat() that tolerates the file being deleted underneath the listing."""
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def atomic_write(path: Path, data: bytes) -> None:
    """Write via a temp file in the same directory + os.replace.

    Readers of the studio directory only ever see the complete file or no file
    at all -- never the half of it that made it to disk before the session was
    killed. The temp name deliberately ends in `.tmp<pid>` so it matches neither
    *.png nor *.json nor the ref extensions.
    """
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class Req(BaseModel):
    prompt: str
    project: str = "untitled"
    preset: str | None = None
    negative: str = ""
    width: int = 1024
    height: int = 576
    steps: int = 8
    seed: int = -1
    ref_paths: list[str] = []


class NewProject(BaseModel):
    name: str


class Rename(BaseModel):
    project: str
    name: str


class Upload(BaseModel):
    project: str
    name: str
    data: str          # data: URL or bare base64


@app.get("/", response_class=HTMLResponse)
def index():
    try:
        return (UI / "index.html").read_text(encoding="utf-8")
    except OSError as e:
        raise HTTPException(500, f"cannot read {UI / 'index.html'}: {e}")


@app.get("/api/config")
def config():
    return {"presets": PRESETS, "style": STYLE}


@app.get("/api/health")
def health():
    free, per_gpu = None, []
    try:
        import torch

        if torch.cuda.is_available():
            # per-GPU used, so you can see at a glance whether GPU1 is actually
            # holding the text tower (~4 GB) or still sitting at ~0.
            for i in range(torch.cuda.device_count()):
                f, t = torch.cuda.mem_get_info(i)
                per_gpu.append({"gpu": i, "free_gb": round(f / 2**30, 2),
                                "used_gb": round((t - f) / 2**30, 2)})
            free = per_gpu[0]["free_gb"]
    except Exception as e:
        # VRAM reporting is a nicety; never let it take health down with it,
        # health is what a returning UI polls first.
        log(f"[health] vram read failed ({type(e).__name__}: {e})")
    with GEN_LOCK:
        gen = {"busy": GEN["busy"], "queued": GEN["waiting"], "project": GEN["project"],
               "prompt": GEN["prompt"],
               "elapsed_s": round(time.time() - GEN["started"], 1) if GEN["busy"] else 0.0}
    split = STATE["report"].get("device_split", {})
    pinning = STATE["report"].get("pinning", "")
    selftest = STATE["report"].get("selftest")
    speed = None
    if STATE["ready"]:
        if selftest and selftest.get("ok") is False:
            speed = {"flag": "selftest-failed",
                     "msg": "boot self-test failed -- check the server log; "
                            "the split or reference path is broken"}
        elif not split.get("enabled"):
            speed = {"flag": "single-gpu",
                     "msg": f"dual-GPU split off ({split.get('reason') or 'not configured'}) -- expect ~55-60 s/image"}
        elif not split.get("moved"):
            speed = {"flag": "single-gpu",
                     "msg": "split enabled but moved nothing -- GPU1 idle, expect ~55-60 s/image"}
        elif pinning == "PARTIAL":
            speed = {"flag": "partial-pinning",
                     "msg": "text tower on GPU1 but host-RAM pinning is PARTIAL -- async transfers off, expect slow images"}
        elif selftest:
            tt = selftest.get("text", {}).get("s")
            rf = selftest.get("with_ref", {}).get("s")
            speed = {"flag": "ok",
                     "msg": f"verified: text tower on {split.get('target')}, pinning {pinning}; "
                            f"probe text-only {tt}s / with-ref {rf}s "
                            f"(2 steps at {selftest.get('res')}) -- expect ~20-40 s/image at full res"}
        else:
            speed = {"flag": "ok",
                     "msg": f"text tower on {split.get('target')}, pinning {pinning} -- expect ~20-46 s/image"}
    return {"status": "ready" if STATE["ready"] else "loading",
            "vram_free_gb": free, "gpus": per_gpu,
            "generated": len(HISTORY),
            # A UI that was closed mid-render polls this on return: busy=true
            # means the shot it thinks it lost is still being made.
            "busy": gen["busy"], "queued": gen["queued"], "generating": gen,
            "speed": speed,
            **STATE["report"]}


@app.get("/api/stats")
def stats():
    t = sorted(h["generate_s"] for h in HISTORY)
    return {"n": len(t), **({"min_s": t[0], "median_s": t[len(t) // 2], "max_s": t[-1]} if t else {})}


@app.get("/api/projects")
def projects():
    try:
        SHOTS.mkdir(parents=True, exist_ok=True)
        entries = sorted(SHOTS.iterdir())
    except OSError as e:
        raise HTTPException(500, f"cannot read shots directory {SHOTS}: {e}")
    out = []
    for d in entries:
        if d.is_dir():
            out.append({"name": d.name,
                        "shots": len(list(d.glob("*.png"))),
                        "refs": len([p for p in (REFS / d.name).glob("*")
                                     if p.suffix.lower() in IMG_EXT]) if (REFS / d.name).exists() else 0})
    # No implicit project: an empty studio starts empty. Directories are created
    # on first generate or upload, not on first page load.
    return {"projects": out}


@app.post("/api/projects")
def new_project(p: NewProject):
    name = slug(p.name)
    pdirs(name)
    return {"name": name}


@app.post("/api/rename")
def rename_project(r: Rename):
    old, base = slug(r.project), _clean(r.name)
    # _clean, not slug: slug() turns "" and "///" into "untitled", which would
    # silently rename the project rather than telling the user to type a name.
    if not base:
        raise HTTPException(400, "name required")
    if base == old:
        return {"name": base}
    if not (SHOTS / old).exists() and not (REFS / old).exists():
        raise HTTPException(404, f"project '{old}' not found")
    with GEN_LOCK:
        if GEN["busy"] and GEN["project"] == old:
            raise HTTPException(409, "a generation is running in this project -- "
                                     "rename it once the shot finishes")
    # keep the target free instead of overwriting
    new, i = base, 2
    while (SHOTS / new).exists() or (REFS / new).exists():
        new = f"{base} {i}"
        i += 1
        if i > 999:
            raise HTTPException(409, f"too many projects named '{base}'")
    done: list[tuple[Path, Path]] = []
    try:
        for src, dst in ((SHOTS / old, SHOTS / new), (REFS / old, REFS / new)):
            if src.exists():
                os.replace(src, dst)
                done.append((src, dst))
    except OSError as e:
        for src, dst in reversed(done):   # half a rename is worse than none
            try:
                os.replace(dst, src)
            except OSError:
                pass
        raise HTTPException(500, f"rename failed: {e}")
    return {"name": new}


@app.get("/api/frames")
def frames():
    """The committed style-reference corpus in frames/, grouped by video."""
    if not FRAMES.exists():
        return {"videos": []}
    out = []
    for d in sorted(p for p in FRAMES.iterdir() if p.is_dir()):
        imgs = sorted(p.name for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        if imgs:
            out.append({"video": d.name, "frames": imgs})
    return {"videos": out}


@app.get("/api/shots")
def shots(project: str = "untitled"):
    d, _ = pdirs(project)
    ps = sorted(d.glob("*.png"), key=lambda p: _mtime(p), reverse=True)
    out = []
    for p in ps[:80]:
        meta = {}
        side = p.with_suffix(".json")
        if side.exists():
            try:
                meta = json.loads(side.read_text())
            except Exception:
                pass
        out.append({"name": p.name, "prompt": meta.get("prompt", ""),
                    "seed": meta.get("seed"), "secs": meta.get("generate_s")})
    return {"shots": out}


@app.get("/api/refs")
def refs(project: str = "untitled"):
    """A project's own reference images, newest first."""
    _, d = pdirs(project)
    ps = sorted((p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")),
                key=lambda p: _mtime(p), reverse=True)
    return {"refs": [p.name for p in ps]}


@app.post("/api/upload")
def upload(u: Upload):
    """Drag-and-drop target. Base64 in JSON, so no multipart dependency."""
    _, d = pdirs(u.project)
    raw = u.data.split(",", 1)[-1]
    if len(raw) > MAX_UPLOAD * 4 // 3 + 16:
        raise HTTPException(413, f"image too large (max {MAX_UPLOAD // 2**20} MB)")
    try:
        blob = base64.b64decode(raw)
    except Exception:
        raise HTTPException(400, "could not decode image data")
    stem = slug(Path(u.name).stem) or "ref"
    ext = (Path(u.name).suffix.lower() or ".png")[:8]
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(400, f"unsupported file type {ext}")
    out = d / f"{stem}{ext}"
    i = 1
    while out.exists():
        out = d / f"{stem}_{i}{ext}"
        i += 1
    atomic_write(out, blob)
    return {"name": out.name}


@app.get("/api/img")
def img(name: str, kind: str = "shot", project: str = "untitled"):
    if kind == "frame":
        cand = (FRAMES / name).resolve()
        if FRAMES.resolve() in cand.parents and cand.is_file():
            return FileResponse(cand)
        raise HTTPException(404, "not found")
    f = safe_file(name)
    d = (pdirs(project)[0] if kind == "shot" else pdirs(project)[1]) / f
    if not d.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(d)


@app.delete("/api/img")
def delete(name: str, kind: str = "shot", project: str = "untitled"):
    # The frames corpus is committed reference material, never deletable. Without
    # this, a frame delete falls through and removes a same-named project ref.
    if kind == "frame":
        raise HTTPException(403, "corpus frames cannot be deleted")
    f = safe_file(name)
    d = (pdirs(project)[0] if kind == "shot" else pdirs(project)[1]) / f
    if d.is_file():
        d.unlink()
        d.with_suffix(".json").unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/generate")
def generate(r: Req):
    import torch
    from PIL import Image

    if not STATE["ready"]:
        raise HTTPException(503, "model still loading")
    if r.width % 16 or r.height % 16:
        raise HTTPException(400, "width and height must be divisible by 16")

    shots_dir, refs_dir = pdirs(r.project)
    # Send the prompt as written. Nothing is prepended, ever -- a house style
    # baked in here silently overrides what you asked for ("a Ferrari" came
    # back as a stickman). A preset is appended ONLY when you pick one.
    prompt = r.prompt.strip()
    if r.preset and r.preset in PRESETS:
        prompt = f"{prompt}. {PRESETS[r.preset]}"
    seed = r.seed if r.seed >= 0 else int(time.time() * 1000) % (2**31)

    # Wan2GP accepts at most 2 reference images; more are dropped upstream
    # without a word, so cap here and say so in the response. A requested ref
    # that is NOT on disk is an error, never a silent text-only generation:
    # the UI uploads every ref before attaching it, so a miss means the file
    # went away and the user should hear about it instead of wondering why the
    # output ignores the image they picked.
    wanted = r.ref_paths[:]
    used = wanted[:2]
    imgs, missing = [], []
    for n in used:
        p = refs_dir / Path(n).name
        if p.is_file():
            imgs.append(Image.open(p).convert("RGB"))
        else:
            missing.append(Path(n).name)
    if missing:
        raise HTTPException(400, "reference image(s) not found in this project: "
                                 + ", ".join(missing))

    kw = dict(seed=seed, input_prompt=prompt, n_prompt=(r.negative or None), sampling_steps=r.steps,
              width=r.width, height=r.height, guide_scale=0.0, batch_size=1,
              loras_slists={"phase1": []})
    if imgs:
        kw["input_ref_images"] = imgs
        kw["video_prompt_type"] = "KI"   # must contain "I" or refs are ignored

    # Announce the job before waiting on LOCK so a UI that reloads mid-render
    # can see the work is still in flight (health.busy) instead of assuming it
    # died, and so we can refuse to pile unlimited requests onto the threadpool.
    with GEN_LOCK:
        # waiting counts the running request too (it is bumped before LOCK is
        # taken), so the ceiling is 1 running + MAX_QUEUE sitting on LOCK.
        if GEN["waiting"] >= 1 + MAX_QUEUE:
            raise HTTPException(429, "the model is busy and the queue is full -- "
                                     "wait for the current shot to finish")
        GEN["waiting"] += 1

    try:
        with LOCK:
            with GEN_LOCK:
                GEN["busy"] = True
                GEN["project"] = slug(r.project)
                GEN["prompt"] = prompt
                GEN["started"] = time.time()
            t0 = time.time()
            try:
                out = STATE["model"].generate(**kw)
            except Exception as e:
                log("[gen] failed\n" + traceback.format_exc())
                torch.cuda.empty_cache()
                hint = ""
                if "device" in str(e).lower() and STATE["report"].get("device_split", {}).get("moved"):
                    hint = "  (cross-device -- try SPLIT_GPUS=0)"
                raise HTTPException(500, f"{type(e).__name__}: {e}{hint}")
            secs = round(time.time() - t0, 1)

            t = out.detach().cpu() if hasattr(out, "detach") else out
            if hasattr(t, "ndim") and t.ndim == 4:      # [C, 1, H, W]
                t = t[:, 0]
            img_ = Image.fromarray(t.permute(1, 2, 0).numpy()) if hasattr(t, "permute") else t

            n = max([int(m.group(1)) for p in shots_dir.glob("shot_*.png")
                     if (m := re.match(r"shot_(\d+)", p.name))] or [0]) + 1
            name = f"shot_{n:03d}.png"
            buf = io.BytesIO()
            img_.save(buf, format="PNG")
            atomic_write(shots_dir / name, buf.getvalue())
            atomic_write((shots_dir / name).with_suffix(".json"),
                         json.dumps({"prompt": r.prompt.strip(), "seed": seed, "steps": r.steps,
                                     "generate_s": secs, "refs": used}).encode())
            torch.cuda.empty_cache()
    finally:
        with GEN_LOCK:
            GEN["busy"] = False
            GEN["started"] = 0.0
            GEN["waiting"] = max(0, GEN["waiting"] - 1)

    rec = {"name": name, "generate_s": secs, "seed": seed, "project": slug(r.project),
           "refs_used": len(imgs), "refs_dropped": max(0, len(wanted) - 2)}
    HISTORY.append(rec)
    log(f"[gen] {slug(r.project)}/{name} {secs}s seed={seed} refs={len(imgs)}")
    return rec


if __name__ == "__main__":
    import socket
    import uvicorn

    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((HOST, PORT))
    except OSError:
        raise SystemExit(f"port {PORT} in use -- pkill -9 -f server.py")
    finally:
        s.close()

    def boot():
        try:
            load()
        except BaseException:
            log("[load] FATAL\n" + traceback.format_exc())
            os._exit(1)

    threading.Thread(target=boot, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
