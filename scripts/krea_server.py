#!/usr/bin/env python3
"""
krea_server.py -- warm Krea 2 Turbo Edit server for Kaggle T4 x2.

Loads the model once and holds it on GPU0, then serves generations over HTTP so
you can iterate without paying the 3-5 minute load on every image.

    GET  /health    -> liveness + what actually got loaded
    GET  /stats     -> rolling generate_s history
    POST /generate  -> one image
    POST /shutdown  -> drop the model, free VRAM

Two things about this file are worth understanding before you change it.

1. T4 is Turing (SM75) and has no bf16. Krea 2 ships bf16 weights and the
   Wan2GP loaders assume bf16 in several places, so every bf16 tensor has to
   become fp16 before it reaches a kernel. That happens in two layers:

     - source patches from scripts/patches/t4_fp16.json, applied to the Wan2GP
       checkout before import. Fast path, but they are string matches against
       upstream, and upstream drifts. When a patch stops matching, this server
       prints "!! NO MATCH" and refuses to start rather than half-working.

     - a post-load sweep (force_fp16_inplace) that walks every module and casts
       any bf16 parameter or buffer that survived. This depends on nothing
       upstream, so it is the actual guarantee; the patches are the
       optimisation. A large sweep count means the patches stopped matching --
       the load is still correct, just slower.

2. The Wan2GP generate signature drifts too, so this server introspects the
   resolved function and passes only the kwargs it accepts, logging what it
   dropped. To see the real API surface of your pinned checkout without
   loading any weights:

       python scripts/krea_server.py --probe

Pin WAN2GP_COMMIT. Every silent-failure story in this project traces back to
running against a moving main.
"""

from __future__ import annotations

import argparse
import inspect
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

REPO_DIR = Path(os.environ.get("STICKMAN_DIR", Path(__file__).resolve().parent.parent))
WAN2GP_DIR = Path(os.environ.get("WAN2GP_DIR", "/kaggle/working/Wan2GP"))
OUT_DIR = Path(os.environ.get("KREA_OUT_DIR", REPO_DIR / "shots"))
PATCH_FILE = Path(
    os.environ.get("KREA_PATCH_FILE", Path(__file__).resolve().parent / "patches" / "t4_fp16.json")
)

HOST = os.environ.get("KREA_HOST", "127.0.0.1")
PORT = int(os.environ.get("KREA_PORT", "8711"))

# Model identifiers, from a verified working run.
#   krea2_turbo      = text-to-image only
#   krea2_turbo_edit = adds the vision tower, i.e. reference images
MODEL_TYPE = os.environ.get("KREA_MODEL_TYPE", "krea2_turbo_edit")
BASE_MODEL_TYPE = os.environ.get("KREA_BASE_MODEL_TYPE", MODEL_TYPE)
DEVICE = os.environ.get("KREA_DEVICE", "cuda:0")

TRANSFORMER_FILE = os.environ.get(
    "KREA_TRANSFORMER", "Krea2Turbo_quanto_bf16_int8.safetensors")
TE_DIR = os.environ.get("KREA_TE_DIR", "Qwen3-VL-4B-Instruct")
TE_FILE = os.environ.get(
    "KREA_TE_FILE", "Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors")

# THE speed lever. 18 GB of weights do not fit in 16 GB of VRAM, so mmgp
# streams them; how well it does that is what separates ~60 s from 20-46 s.
#   pinnedMemory   - page-locked host memory, so transfers can be async at all
#   asyncTransfers - overlap the next module's copy with the current compute
#   budgets (MB)   - how much of each component to keep resident on the GPU
#   sdpa           - the only attention backend that builds on SM75
# These values come from a working run; do not "tidy" them without measuring.
PROFILE_NO = int(os.environ.get("KREA_PROFILE_NO", "2"))
ATTENTION = os.environ.get("KREA_ATTENTION", "sdpa")
BUDGETS = {
    "transformer": 13000,
    "text_encoder": 4500,
    "vae": 2000,
    "*": 1000,
}
if MODEL_TYPE.endswith("_edit"):
    # The vision tower is only resident on the _edit variants.
    BUDGETS["vision_encoder"] = 2000

# Refuse to start when a required patch no longer matches upstream. Set to 1
# only once you have confirmed the post-load sweep is covering for it.
ALLOW_UNPATCHED = os.environ.get("KREA_ALLOW_UNPATCHED", "0") == "1"

# Defaults tuned for iteration speed on one T4, not for final quality.
DEF_WIDTH = int(os.environ.get("KREA_WIDTH", "1024"))
DEF_HEIGHT = int(os.environ.get("KREA_HEIGHT", "576"))
DEF_STEPS = int(os.environ.get("KREA_STEPS", "8"))
# Turbo is distilled: CFG is disabled, not merely low.
DEF_GUIDANCE = float(os.environ.get("KREA_GUIDANCE", "0.0"))

MAX_PIXELS = int(os.environ.get("KREA_MAX_PIXELS", str(1536 * 1536)))
MAX_REFS = 2  # Wan2GP enforces this too; failing here gives a better message.

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
    as applied. A patch whose `find` string is absent means upstream moved --
    reported loudly, because a silently no-op patch is exactly the failure mode
    that wastes an afternoon.
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
        if p["replace"] in text:
            already.append(p["name"])
            continue
        if p["find"] not in text:
            missing.append(p["name"])
            continue

        target.write_text(text.replace(p["find"], p["replace"], p.get("count", 1)))
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
                "refusing to start: a required fp16 patch did not match upstream.\n"
                "Set KREA_ALLOW_UNPATCHED=1 to start anyway and rely on the "
                "post-load fp16 sweep (slower load, but it does work)."
            )

    return {"applied": ok, "total": total, "missing": missing}


# ---------------------------------------------------------------------------
# layer 2: post-load fp16 sweep -- the actual guarantee
# ---------------------------------------------------------------------------


def force_fp16_inplace(obj: Any) -> dict[str, int]:
    """Cast every surviving bf16 parameter/buffer to fp16, in place.

    Walks attributes recursively because the pipeline holds several sibling
    models (transformer, text encoder, vision tower, VAE) that are not children
    of one root module. Quantised (int8/quanto) tensors are left alone; only
    bf16 is touched.
    """
    import torch
    import torch.nn as nn

    seen: set[int] = set()
    stats = {"converted": 0, "failed": 0, "modules": 0}

    def convert_module(m: nn.Module) -> None:
        stats["modules"] += 1
        for _, p in list(m.named_parameters(recurse=False)):
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
        if depth > 4 or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, nn.Module):
            for sub in o.modules():
                if id(sub) not in seen:
                    seen.add(id(sub))
                    convert_module(sub)
            return
        if isinstance(o, (list, tuple)):
            for item in o:
                walk(item, depth + 1)
        elif isinstance(o, dict):
            for item in o.values():
                walk(item, depth + 1)
        elif hasattr(o, "__dict__"):
            for item in list(vars(o).values()):
                walk(item, depth + 1)

    walk(obj, 0)
    return stats


# ---------------------------------------------------------------------------
# layer 3: dual-GPU split -- EXPERIMENTAL, off by default
# ---------------------------------------------------------------------------
#
# Moving the text encoder and VAE to cuda:1 sounds like free speed, since mmgp
# is single-GPU and GPU1 idles. It is off by default because mmgp already owns
# device placement: it decides what is resident and when, per the budgets
# above. Relocating modules behind its back fights that logic and can cost
# more than it saves.
#
# The measured path is the mmgp profile (pinned + async + budgets + sdpa).
# Try this only after /stats shows a stable median, and compare directly.

SPLIT_DEVICES = os.environ.get("KREA_SPLIT_DEVICES", "0") == "1"
SECOND_DEVICE = os.environ.get("KREA_SECOND_DEVICE", "cuda:1")

# Auxiliary towers: each runs once per image, so the PCIe hop costs far less
# than the offload thrash it removes. The transformer/DiT is deliberately
# absent -- it runs every step and must stay on the primary device.
AUX_PATTERNS = (
    "text_encoder", "text_enc", "t5", "clip",
    "vae", "vision_encoder", "vision_tower", "image_encoder",
)


def _wrap_forward_for_device(mod, dev: str, back: str) -> None:
    """Let a module live on `dev` while its call sites still pass `back` tensors.

    Wan2GP assumes one device throughout. Rather than patch every call site,
    wrap forward so inputs hop to the module's device and outputs hop back.
    """
    import torch

    if getattr(mod, "_krea_wrapped", False):
        return
    orig = mod.forward

    def move(x, target):
        if isinstance(x, torch.Tensor):
            return x.to(target, non_blocking=True)
        if isinstance(x, (list, tuple)):
            return type(x)(move(i, target) for i in x)
        if isinstance(x, dict):
            return {k: move(v, target) for k, v in x.items()}
        return x

    def forward(*a, **kw):
        return move(orig(*move(a, dev), **move(kw, dev)), back)

    mod.forward = forward
    mod._krea_wrapped = True


def split_devices(model: Any) -> dict[str, Any]:
    """Move the auxiliary towers to the second GPU. Returns what actually moved."""
    import torch

    if not SPLIT_DEVICES:
        return {"enabled": False, "reason": "KREA_SPLIT_DEVICES=0", "moved": []}
    if torch.cuda.device_count() < 2:
        return {"enabled": False, "reason": "only one GPU visible", "moved": []}

    moved, failed = [], []
    for name in dir(model):
        if name.startswith("__") or not any(p in name.lower() for p in AUX_PATTERNS):
            continue
        try:
            sub = getattr(model, name)
        except Exception:
            continue
        if not isinstance(sub, torch.nn.Module):
            continue
        try:
            sub.to(SECOND_DEVICE)
            _wrap_forward_for_device(sub, SECOND_DEVICE, DEVICE)
            moved.append(name)
        except Exception as e:
            failed.append(f"{name}: {type(e).__name__}")

    return {
        "enabled": True,
        "target": SECOND_DEVICE,
        "moved": moved,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Wan2GP adapter -- everything upstream-specific lives here
# ---------------------------------------------------------------------------


class Wan2GPAdapter:
    def __init__(self) -> None:
        self.model = None
        self.pipe = None
        self.gen_fn = None
        self.gen_params: set[str] = set()
        self.accepts_kwargs = True
        self.load_report: dict[str, Any] = {}
        self.lock = threading.Lock()

    def import_wgp(self):
        if str(WAN2GP_DIR) not in sys.path:
            sys.path.insert(0, str(WAN2GP_DIR))
        os.chdir(WAN2GP_DIR)  # Wan2GP resolves model/config paths relative to its root
        import wgp  # type: ignore

        self.wgp = wgp
        return wgp

    @staticmethod
    def _resolve(obj: Any, names: list[str]):
        """Find the first callable among `names`. Returns (name, fn) or (None, None)."""
        for n in names:
            fn = getattr(obj, n, None)
            if callable(fn):
                return n, fn
        return None, None

    # -- probe --------------------------------------------------------------

    def probe(self) -> None:
        """Dump the real API surface and patch status. No weights loaded."""
        import inspect as _i

        if str(WAN2GP_DIR) not in sys.path:
            sys.path.insert(0, str(WAN2GP_DIR))
        os.chdir(WAN2GP_DIR)

        print(f"\nWan2GP at {WAN2GP_DIR}\n")
        print("-- patch table dry run --")
        for p in json.loads(PATCH_FILE.read_text())["patches"]:
            f = WAN2GP_DIR / p["file"]
            if not f.exists():
                state = "FILE MISSING"
            else:
                body = f.read_text()
                state = ("already applied" if p["replace"] in body
                         else ("matches" if p["find"] in body else "!! NO MATCH"))
            print(f"  [{state:16}] {p['name']}")

        print("\n-- weights on disk --")
        for label, rel in (("transformer", os.path.join("models", TRANSFORMER_FILE)),
                           ("text encoder", os.path.join("models", TE_DIR, TE_FILE)),
                           ("vision tower", os.path.join(
                               "models", TE_DIR, f"{TE_DIR}_vision_bf16.safetensors"))):
            print(f"  {label:13} {'OK ' if os.path.exists(rel) else 'MISSING'}  {rel}")

        print("\n-- handler API --")
        try:
            from models.krea2.krea2_handler import family_handler
            for name in ("query_model_def", "load_model"):
                fn = getattr(family_handler, name, None)
                print(f"  family_handler.{name}{_i.signature(fn) if fn else ' MISSING'}")
        except Exception as e:
            print(f"  could not import handler: {type(e).__name__}: {e}")

        print("\n-- generate signature (from source) --")
        krea = WAN2GP_DIR / "models" / "krea2" / "krea2_main.py"
        if krea.exists():
            src = krea.read_text().splitlines()
            for i, line in enumerate(src):
                if line.strip().startswith("def generate"):
                    for j in range(i, min(i + 30, len(src))):
                        print(f"  {src[j].rstrip()}")
                        if src[j].rstrip().endswith(":") and j > i:
                            break
                    break

    def load(self) -> None:
        import torch

        t0 = time.time()
        patch_report = apply_source_patches()

        if str(WAN2GP_DIR) not in sys.path:
            sys.path.insert(0, str(WAN2GP_DIR))
        os.chdir(WAN2GP_DIR)  # Wan2GP resolves its paths relative to its root

        from mmgp import offload
        from shared.utils import files_locator as fl
        from models.krea2.krea2_handler import family_handler

        # Where Wan2GP looks for weights. Without this it re-downloads its own
        # copies next to yours and fills the disk.
        fl.set_checkpoints_paths(["models", "ckpts", "."])

        transformer_path = os.path.join("models", TRANSFORMER_FILE)
        te_path = os.path.join("models", TE_DIR, TE_FILE)
        for label, path in (("transformer", transformer_path),
                            ("text encoder", te_path)):
            if not os.path.exists(path):
                raise SystemExit(
                    f"{label} not found at {WAN2GP_DIR}/{path}\n"
                    "Re-run the weights cell in the notebook.")

        # The _edit variants need the vision tower; without it the model loads
        # fine and silently ignores every reference image you pass.
        if MODEL_TYPE.endswith("_edit"):
            vis = os.path.join("models", TE_DIR, f"{TE_DIR}_vision_bf16.safetensors")
            if not os.path.exists(vis):
                raise SystemExit(
                    f"vision tower missing at {WAN2GP_DIR}/{vis}\n"
                    f"{MODEL_TYPE} needs it for references. Re-run the weights cell, "
                    "or set KREA_MODEL_TYPE=krea2_turbo for text-only.")

        log(f"[load] model_type={MODEL_TYPE} base={BASE_MODEL_TYPE}")
        model_def = family_handler.query_model_def(BASE_MODEL_TYPE, {})

        self.model, self.pipe = family_handler.load_model(
            model_filename=transformer_path,
            model_type=MODEL_TYPE,
            base_model_type=BASE_MODEL_TYPE,
            model_def=model_def,
            quantizeTransformer=False,   # weights are already quanto int8
            dtype=torch.float16,         # T4: no bf16
            VAE_dtype=torch.float16,
            text_encoder_filename=te_path,
        )
        log(f"[load] loaded {type(self.model).__name__}")

        # The measured speed lever -- see BUDGETS at the top of this file.
        log(f"[load] mmgp profile {PROFILE_NO}, budgets={BUDGETS}")
        offload.profile(
            self.pipe,
            profile_no=PROFILE_NO,
            quantizeTransformer=False,
            convertWeightsFloatTo=torch.float16,
            pinnedMemory=True,
            asyncTransfers=True,
            budgets=dict(BUDGETS),
        )
        offload.shared_state["_attention"] = ATTENTION
        log(f"[load] attention={ATTENTION}")

        self.gen_fn = self.model.generate
        try:
            sig = inspect.signature(self.gen_fn)
            self.gen_params = set(sig.parameters)
            self.accepts_kwargs = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        except (TypeError, ValueError):
            self.gen_params, self.accepts_kwargs = set(), True
        log(f"[load] generate params: {sorted(self.gen_params)}")
        if MODEL_TYPE.endswith("_edit") and "input_ref_images" not in self.gen_params:
            log("[load] !! generate() has no input_ref_images -- references will")
            log("[load] !! be ignored. Check --probe output.")

        sweep = force_fp16_inplace(self.model)
        log(f"[load] fp16 sweep: converted={sweep['converted']} "
            f"failed={sweep['failed']} modules={sweep['modules']}")
        if sweep["converted"] > 64:
            log("[load] !! large sweep -- source patches likely not matching.")

        split = split_devices(self.model)
        if split["enabled"]:
            log(f"[split] EXPERIMENTAL: moved {split['moved']} -> {split['target']}")

        torch.cuda.empty_cache()
        self.load_report = {
            "patches": patch_report,
            "fp16_sweep": sweep,
            "device_split": split,
            "mmgp": {"profile": PROFILE_NO, "attention": ATTENTION, "budgets": BUDGETS},
            "generate_params": sorted(self.gen_params),
            "load_s": round(time.time() - t0, 1),
        }
        log(f"[load] ready in {self.load_report['load_s']}s")

    def call_generate(self, wanted: dict[str, Any]) -> Any:
        """Pass only kwargs the resolved signature accepts; log what was dropped."""
        if self.accepts_kwargs or not self.gen_params:
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
        t = out.detach().cpu()
        # Krea returns [C, 1, H, W] -- a frame axis after the channel axis, not
        # a leading batch dim. Squeezing dim 0 would hand back a 1-channel
        # image, so drop the frame axis specifically.
        if t.ndim == 4 and t.shape[0] in (1, 3, 4) and t.shape[1] == 1:
            t = t[:, 0]
        was_uint8 = t.dtype == torch.uint8
        t = t.float()
        while t.ndim > 3:
            t = t[0]
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.shape[0] in (1, 3, 4):  # CHW -> HWC
            t = t.permute(1, 2, 0)
        arr = t.numpy()
        if not was_uint8:
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
    """Resolve reference paths against the repo, so 'frames/01_fan_death/x.jpg' works."""
    from PIL import Image

    refs = []
    for p in paths:
        fp = Path(p)
        if not fp.is_absolute():
            fp = REPO_DIR / p
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
    out_name: str = Field(..., description="bare filename, e.g. shot_003.png")
    negative_prompt: str = ""
    width: int = DEF_WIDTH
    height: int = DEF_HEIGHT
    steps: int = DEF_STEPS
    guidance: float = DEF_GUIDANCE
    seed: int = -1
    ref_paths: list[str] = []
    # "KI" = first reference is the scene/base, stretched to canvas.
    # "I"  = every reference is a subject/object, aspect ratio preserved.
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

    if req.width % 16 or req.height % 16:
        raise HTTPException(
            400, f"width/height must be divisible by 16 (got {req.width}x{req.height})"
        )
    if req.width * req.height > MAX_PIXELS:
        raise HTTPException(400, f"{req.width}x{req.height} exceeds the pixel cap")
    if len(req.ref_paths) > MAX_REFS:
        raise HTTPException(400, f"max {MAX_REFS} reference images, got {len(req.ref_paths)}")
    if req.ref_paths and not MODEL_TYPE.endswith("_edit"):
        raise HTTPException(
            400,
            f"references passed but model_type is {MODEL_TYPE}. The vision tower is "
            "only loaded by _edit variants, so they would be silently dropped.",
        )
    if not req.out_name.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "out_name must end in .png or .jpg")
    if "/" in req.out_name or ".." in req.out_name:
        raise HTTPException(400, "out_name must be a bare filename")

    seed = req.seed if req.seed >= 0 else int(time.time() * 1000) % (2**31)
    refs = load_refs(req.ref_paths) if req.ref_paths else None

    # video_prompt_type MUST contain "I" or Wan2GP drops references without
    # warning:  reference_images = input_ref_images if "I" in video_prompt_type
    vpt = req.ref_mode if refs else ""
    if refs and "I" not in vpt:
        vpt += "I"

    kwargs = {
        "seed": seed,
        "input_prompt": req.prompt,
        "n_prompt": req.negative_prompt if req.negative_prompt.strip() else None,
        "sampling_steps": req.steps,
        "width": req.width,
        "height": req.height,
        "guide_scale": req.guidance,
        "batch_size": 1,
        # Required by the handler even when unused; omitting it raises.
        "loras_slists": {"phase1": []},
    }
    if refs:
        kwargs["input_ref_images"] = refs
        kwargs["video_prompt_type"] = vpt

    with ADAPTER.lock:  # one at a time; 16 GB has no room for two
        t0 = time.time()
        try:
            out = ADAPTER.call_generate(kwargs)
        except Exception as e:
            log("[gen] FAILED:\n" + traceback.format_exc())
            # The dual-GPU split is the one change likely to surface as a device
            # mismatch, so name it rather than making you guess.
            if "device" in str(e).lower() and ADAPTER.load_report.get(
                "device_split", {}
            ).get("moved"):
                log("[gen] !! looks like a cross-device error from the GPU split.")
                log("[gen] !! Restart with KREA_SPLIT_DEVICES=0 to confirm, then")
                log("[gen] !! narrow AUX_PATTERNS to the module that broke.")
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


def _load_in_background() -> None:
    try:
        ADAPTER.load()
    except SystemExit as e:
        log(f"[load] FATAL: {e}")
        os._exit(1)
    except Exception:
        log("[load] FATAL:\n" + traceback.format_exc())
        os._exit(1)


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

    import socket
    import uvicorn

    # Claim the port BEFORE starting the load. uvicorn only binds after the
    # background load is already running, so without this a second process
    # loads a whole model onto the GPU and only then discovers it lost the
    # race -- which is how three copies end up competing for 16 GB.
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((HOST, PORT))
    except OSError:
        raise SystemExit(
            f"port {PORT} is already in use -- another krea_server is running.\n"
            f"Use it, or stop it first:  pkill -9 -f krea_server.py"
        )
    finally:
        probe.close()

    log(f"[boot] REPO_DIR={REPO_DIR}")
    log(f"[boot] WAN2GP_DIR={WAN2GP_DIR}")
    log(f"[boot] OUT_DIR={OUT_DIR}")
    threading.Thread(target=_load_in_background, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
