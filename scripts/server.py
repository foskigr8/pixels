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
   the Qwen3-VL text+vision tower (~4.0 GB of the 18.1 GB pin requirement)
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

# mmgp caps any single budget at 80% of VRAM (~11.9 GB on a 16 GB T4), so a
# larger number here is silently clamped and just prints a warning.
BUDGETS = {"transformer": 11000, "text_encoder": 4000, "vae": 1500, "*": 800}

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
# mmgp's pin requirement drops by ~4.0 GB (3467 MB text_encoder + 577 MB
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
            "pin requirement should drop ~4.0 GB")
    else:
        log(f"[split] single-GPU ({split.get('reason')}) -- GPU1 will stay idle")

    # mmgp resolves every weight transfer against the *current* CUDA device, so
    # make sure that is GPU0 before it builds its streams and hooks.
    torch.cuda.set_device(0)

    log(f"[load] mmgp budgets={BUDGETS}")
    # perc_reserved_mem_max is a genuine offload.all() kwarg in 3.7.12 and
    # profile() forwards **overrideKwargs into all() verbatim -- no fallback
    # needed, and a TypeError here would mean the mmgp version changed.
    offload.profile(pipe, profile_no=2, quantizeTransformer=False,
                    convertWeightsFloatTo=torch.float16, pinnedMemory=True,
                    asyncTransfers=True, budgets=dict(BUDGETS),
                    perc_reserved_mem_max=PERC_RESERVED)
    offload.shared_state["_attention"] = "sdpa"

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
                       "budgets": BUDGETS, "load_s": round(time.time() - t0, 1)}
    STATE["ready"] = True
    log(f"[load] ready in {STATE['report']['load_s']}s")


# ---------------------------------------------------------------------------
# api
# ---------------------------------------------------------------------------

import base64  # noqa: E402
import re  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="studio")

SAFE = re.compile(r"[^a-zA-Z0-9 _-]")


def slug(name: str) -> str:
    s = SAFE.sub("", (name or "").strip())[:48].strip()
    return s or "untitled"


def pdirs(project: str) -> tuple[Path, Path]:
    """(shots, refs) for a project. Created on demand."""
    p = slug(project)
    a, b = SHOTS / p, REFS / p
    a.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)
    return a, b


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
    return (UI / "index.html").read_text()


@app.get("/api/config")
def config():
    return {"presets": PRESETS, "style": STYLE}


@app.get("/api/health")
def health():
    import torch

    free, per_gpu = None, []
    if torch.cuda.is_available():
        # per-GPU used, so you can see at a glance whether GPU1 is actually
        # holding the text tower (~4 GB) or still sitting at ~0.
        for i in range(torch.cuda.device_count()):
            f, t = torch.cuda.mem_get_info(i)
            per_gpu.append({"gpu": i, "free_gb": round(f / 2**30, 2),
                            "used_gb": round((t - f) / 2**30, 2)})
        free = per_gpu[0]["free_gb"]
    return {"status": "ready" if STATE["ready"] else "loading",
            "vram_free_gb": free, "gpus": per_gpu,
            "generated": len(HISTORY), **STATE["report"]}


@app.get("/api/stats")
def stats():
    t = sorted(h["generate_s"] for h in HISTORY)
    return {"n": len(t), **({"min_s": t[0], "median_s": t[len(t) // 2], "max_s": t[-1]} if t else {})}


@app.get("/api/projects")
def projects():
    SHOTS.mkdir(parents=True, exist_ok=True)
    out = []
    for d in sorted(SHOTS.iterdir()):
        if d.is_dir():
            out.append({"name": d.name,
                        "shots": len(list(d.glob("*.png"))),
                        "refs": len(list((REFS / d.name).glob("*"))) if (REFS / d.name).exists() else 0})
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
    old, new = slug(r.project), slug(r.name)
    if not new:
        raise HTTPException(400, "name required")
    if new == old:
        return {"name": new}
    # keep the target free instead of overwriting
    i = 2
    while (SHOTS / new).exists() or (REFS / new).exists():
        new = f"{slug(r.name)} {i}"
        i += 1
    for src, dst in ((SHOTS / old, SHOTS / new), (REFS / old, REFS / new)):
        if src.exists():
            src.rename(dst)
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
    ps = sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
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
                key=lambda p: p.stat().st_mtime, reverse=True)
    return {"refs": [p.name for p in ps]}


@app.post("/api/upload")
def upload(u: Upload):
    """Drag-and-drop target. Base64 in JSON, so no multipart dependency."""
    _, d = pdirs(u.project)
    raw = u.data.split(",", 1)[-1]
    try:
        blob = base64.b64decode(raw)
    except Exception:
        raise HTTPException(400, "could not decode image data")
    ext = Path(u.name).suffix.lower() or ".png"
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(400, f"unsupported file type {ext}")
    stem = slug(Path(u.name).stem) or "ref"
    out = d / f"{stem}{ext}"
    i = 1
    while out.exists():
        out = d / f"{stem}_{i}{ext}"
        i += 1
    out.write_bytes(blob)
    return {"name": out.name}


@app.get("/api/img")
def img(name: str, kind: str = "shot", project: str = "untitled"):
    if kind == "frame":
        cand = (FRAMES / name).resolve()
        if FRAMES.resolve() in cand.parents and cand.is_file():
            return FileResponse(cand)
        raise HTTPException(404, "not found")
    d = (pdirs(project)[0] if kind == "shot" else pdirs(project)[1]) / Path(name).name
    if not d.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(d)


@app.delete("/api/img")
def delete(name: str, kind: str = "shot", project: str = "untitled"):
    # The frames corpus is committed reference material, never deletable. Without
    # this, a frame delete falls through and removes a same-named project ref.
    if kind == "frame":
        raise HTTPException(403, "corpus frames cannot be deleted")
    d = (pdirs(project)[0] if kind == "shot" else pdirs(project)[1]) / Path(name).name
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
    # without a word, so cap here and say so in the response.
    wanted = r.ref_paths[:]
    used = wanted[:2]
    imgs = []
    for n in used:
        p = refs_dir / Path(n).name
        if p.is_file():
            imgs.append(Image.open(p).convert("RGB"))

    kw = dict(seed=seed, input_prompt=prompt, n_prompt=(r.negative or None), sampling_steps=r.steps,
              width=r.width, height=r.height, guide_scale=0.0, batch_size=1,
              loras_slists={"phase1": []})
    if imgs:
        kw["input_ref_images"] = imgs
        kw["video_prompt_type"] = "KI"   # must contain "I" or refs are ignored

    with LOCK:
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
        img_.save(shots_dir / name)
        (shots_dir / name).with_suffix(".json").write_text(json.dumps(
            {"prompt": r.prompt.strip(), "seed": seed, "steps": r.steps,
             "generate_s": secs, "refs": used}))
        torch.cuda.empty_cache()

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
