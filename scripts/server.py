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
   transfers -- worth ~30s/image. Putting the text encoder and VAE on the
   second (otherwise idle) GPU is what keeps that working set small enough.
   Watch the load log for "partial pinning"; that is the tell.
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

# Move the auxiliary towers to the second GPU. On by default because they run
# once per image while the transformer runs every step, and because it takes
# ~6 GB out of the host-RAM pinning requirement, which is the real bottleneck.
SPLIT = os.environ.get("SPLIT_GPUS", "1") == "1"
AUX = ("text_encoder", "vae", "vision_encoder", "vision_tower", "image_encoder")

STYLE = ("minimal hand-drawn stickman on a pure white background, thick 6px solid "
         "black monoline ink strokes, flat saturated color fills, no gradients, "
         "no shadows, whiteboard explainer illustration")
NEGATIVE = ("photorealistic, 3d render, gradient, drop shadow, soft shading, "
            "textured paper, grey background, watermark, blurry")
PRESETS = {
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


def sweep_fp16(model) -> int:
    """Cast any bf16 tensor the patches missed. Depends on nothing upstream."""
    import torch
    import torch.nn as nn

    n, seen = 0, set()

    def walk(o, d=0):
        nonlocal n
        if d > 4 or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, nn.Module):
            for m in o.modules():
                for _, p in list(m.named_parameters(recurse=False)):
                    if p.dtype is torch.bfloat16:
                        p.data = p.data.to(torch.float16); n += 1
                for name, b in list(m.named_buffers(recurse=False)):
                    if b.dtype is torch.bfloat16:
                        setattr(m, name, b.to(torch.float16)); n += 1
            return
        if isinstance(o, (list, tuple)):
            [walk(i, d + 1) for i in o]
        elif isinstance(o, dict):
            [walk(v, d + 1) for v in o.values()]
        elif hasattr(o, "__dict__"):
            [walk(v, d + 1) for v in vars(o).values()]

    walk(model)
    return n


def split_gpus(model) -> dict:
    """Put the auxiliary towers on GPU1 and bridge devices at their boundary."""
    import torch

    if not SPLIT:
        return {"enabled": False, "moved": []}
    if torch.cuda.device_count() < 2:
        return {"enabled": False, "reason": "one GPU", "moved": []}

    def bridge(mod, dev, back):
        if getattr(mod, "_bridged", False):
            return
        orig = mod.forward

        def move(x, t):
            if isinstance(x, torch.Tensor):
                return x.to(t, non_blocking=True)
            if isinstance(x, (list, tuple)):
                return type(x)(move(i, t) for i in x)
            if isinstance(x, dict):
                return {k: move(v, t) for k, v in x.items()}
            return x

        mod.forward = lambda *a, **k: move(orig(*move(a, dev), **move(k, dev)), back)
        mod._bridged = True

    moved = []
    for name in dir(model):
        if not any(a in name.lower() for a in AUX):
            continue
        try:
            sub = getattr(model, name)
        except Exception:
            continue
        if isinstance(sub, torch.nn.Module):
            try:
                sub.to(DEVICE2)
                bridge(sub, DEVICE2, DEVICE)
                moved.append(name)
            except Exception as e:
                log(f"[split] {name} stayed put: {type(e).__name__}")
    return {"enabled": True, "target": DEVICE2, "moved": moved}


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

    log(f"[load] mmgp budgets={BUDGETS}")
    offload.profile(pipe, profile_no=2, quantizeTransformer=False,
                    convertWeightsFloatTo=torch.float16, pinnedMemory=True,
                    asyncTransfers=True, budgets=dict(BUDGETS))
    offload.shared_state["_attention"] = "sdpa"

    cast = sweep_fp16(model)
    split = split_gpus(model)
    log(f"[load] fp16 sweep cast {cast} tensors | split: {split.get('moved') or 'none'}")

    torch.cuda.empty_cache()
    STATE["model"] = model
    STATE["report"] = {"patches": patches, "fp16_cast": cast, "device_split": split,
                       "budgets": BUDGETS, "load_s": round(time.time() - t0, 1)}
    STATE["ready"] = True
    log(f"[load] ready in {STATE['report']['load_s']}s")


# ---------------------------------------------------------------------------
# api
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="studio")


class Req(BaseModel):
    prompt: str
    preset: str | None = None
    width: int = 1024
    height: int = 576
    steps: int = 8
    seed: int = -1
    ref_paths: list[str] = []


@app.get("/", response_class=HTMLResponse)
def index():
    return (UI / "index.html").read_text()


@app.get("/api/config")
def config():
    vids = sorted(p.name for p in (REPO / "frames").iterdir() if p.is_dir()) \
        if (REPO / "frames").exists() else []
    return {"presets": PRESETS, "videos": vids, "style": STYLE}


@app.get("/api/health")
def health():
    import torch

    free = None
    if torch.cuda.is_available():
        free = round(torch.cuda.mem_get_info(0)[0] / 2**30, 2)
    return {"status": "ready" if STATE["ready"] else "loading",
            "model": MODEL_TYPE, "vram_free_gb": free,
            "generated": len(HISTORY), **STATE["report"]}


@app.get("/api/stats")
def stats():
    t = sorted(h["generate_s"] for h in HISTORY)
    if not t:
        return {"n": 0}
    return {"n": len(t), "min_s": t[0], "median_s": t[len(t) // 2], "max_s": t[-1]}


@app.get("/api/refs")
def refs(video: str = ""):
    d = REPO / "frames" / video
    return {"refs": sorted(str(p.relative_to(REPO)) for p in d.glob("*.jpg"))
            if d.exists() else []}


@app.get("/api/file")
def file(path: str):
    p = (REPO / path).resolve()
    if not p.is_file() or REPO.resolve() not in p.parents:
        raise HTTPException(404, "not found")
    return FileResponse(p)


@app.get("/api/shots")
def shots():
    if not SHOTS.exists():
        return {"shots": []}
    ps = sorted(SHOTS.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"shots": [p.name for p in ps[:40]]}


@app.get("/api/shot")
def shot(name: str):
    p = SHOTS / Path(name).name
    if not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(p)


@app.post("/api/generate")
def generate(r: Req):
    import torch
    from PIL import Image

    if not STATE["ready"]:
        raise HTTPException(503, "model still loading")
    if r.width % 16 or r.height % 16:
        raise HTTPException(400, "width and height must be divisible by 16")

    parts = [STYLE]
    if r.preset and r.preset in PRESETS:
        parts.append(PRESETS[r.preset])
    parts.append(r.prompt.strip())
    prompt = ". ".join(parts)

    seed = r.seed if r.seed >= 0 else int(time.time() * 1000) % (2**31)
    refs_ = []
    for rp in r.ref_paths[:2]:
        p = (REPO / rp).resolve()
        if p.is_file():
            refs_.append(Image.open(p).convert("RGB"))

    kw = dict(seed=seed, input_prompt=prompt, n_prompt=NEGATIVE,
              sampling_steps=r.steps, width=r.width, height=r.height,
              guide_scale=0.0, batch_size=1, loras_slists={"phase1": []})
    if refs_:
        # "I" must be present or Wan2GP drops references without warning.
        kw["input_ref_images"] = refs_
        kw["video_prompt_type"] = "KI"

    with LOCK:
        t0 = time.time()
        try:
            out = STATE["model"].generate(**kw)
        except Exception as e:
            log("[gen] failed\n" + traceback.format_exc())
            torch.cuda.empty_cache()
            hint = ""
            if "device" in str(e).lower() and STATE["report"].get("device_split", {}).get("moved"):
                hint = "  (looks cross-device -- try SPLIT_GPUS=0)"
            raise HTTPException(500, f"{type(e).__name__}: {e}{hint}")
        dt = round(time.time() - t0, 1)

        # Krea returns [C, 1, H, W] uint8 -- a frame axis after the channels.
        t = out.detach().cpu() if hasattr(out, "detach") else out
        if hasattr(t, "ndim") and t.ndim == 4:
            t = t[:, 0]
        img = Image.fromarray(t.permute(1, 2, 0).numpy()) if hasattr(t, "permute") else t

        SHOTS.mkdir(parents=True, exist_ok=True)
        n = max([int(p.stem.split("_")[1]) for p in SHOTS.glob("shot_*.png")
                 if p.stem.split("_")[1].isdigit()] or [0]) + 1
        name = f"shot_{n:03d}.png"
        img.save(SHOTS / name)
        torch.cuda.empty_cache()

    rec = {"name": name, "generate_s": dt, "seed": seed,
           "size": f"{img.width}x{img.height}", "refs": len(refs_)}
    HISTORY.append(rec)
    log(f"[gen] {name} {dt}s seed={seed} refs={len(refs_)}")
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
