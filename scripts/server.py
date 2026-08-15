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

    free = round(torch.cuda.mem_get_info(0)[0] / 2**30, 2) if torch.cuda.is_available() else None
    return {"status": "ready" if STATE["ready"] else "loading",
            "vram_free_gb": free, "generated": len(HISTORY), **STATE["report"]}


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
    if not out:
        pdirs("untitled")
        out = [{"name": "untitled", "shots": 0, "refs": 0}]
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
def img(project: str, name: str, kind: str = "shot"):
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
def delete(project: str, name: str, kind: str = "shot"):
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
    parts = [STYLE]
    if r.preset and r.preset in PRESETS:
        parts.append(PRESETS[r.preset])
    parts.append(r.prompt.strip())
    prompt = ". ".join(parts)
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

    kw = dict(seed=seed, input_prompt=prompt, n_prompt=NEGATIVE, sampling_steps=r.steps,
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
