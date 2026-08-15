#!/usr/bin/env python3
"""
shot_ui.py -- Whymentary shot factory UI.

A Gradio front end for krea_server.py, streamlined for this pipeline rather than
for generic image generation. It is a thin client: it holds no model, so it can
be restarted freely without touching the warm model on GPU0.

What it does that a stock UI does not:

  - The style block is never retyped. Presets encode the five recurring shot
    types from the dissections, and the canonical style clause is prepended
    automatically so it can't drift shot to shot.
  - References are picked from the committed corpus (frames/, dense_frames/)
    with thumbnails, instead of uploading files by hand. Matching an existing
    keyframe is far more reliable than describing the style in words.
  - The semantic palette is on screen. Colors carry fixed meaning in this style,
    so a click appends the right phrasing rather than leaving the model to guess.
  - Output is named and filed into shots/ on the naming convention, and timing
    is shown per generation so a speed regression is visible immediately.

    python scripts/shot_ui.py            # serves on 0.0.0.0:7860
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

import gradio as gr

REPO_DIR = Path(os.environ.get("STICKMAN_DIR", Path(__file__).resolve().parent.parent))
KREA_URL = os.environ.get("KREA_URL", "http://127.0.0.1:8711")
SHOTS_DIR = Path(os.environ.get("KREA_OUT_DIR", REPO_DIR / "shots"))
UI_PORT = int(os.environ.get("SHOT_UI_PORT", "7860"))

# The canonical style clause. Every prompt gets this, so it cannot drift.
# Black is the linework only -- the fills are flat saturated color.
STYLE = (
    "minimal hand-drawn stickman on a pure white background, thick 6px solid black "
    "monoline ink strokes, flat saturated color fills, no gradients, no shadows, "
    "no texture, whiteboard explainer illustration"
)

NEGATIVE = (
    "photorealistic, 3d render, gradient, drop shadow, soft shading, textured paper, "
    "grey background, sketchy hatching, watermark, signature, blurry"
)

# The five recurring shot types, named as the dissections name them.
PRESETS: dict[str, str] = {
    "Hook (0-15s)": (
        "a relatable everyday situation, single stickman, one or two literal props, "
        "lots of empty white space"
    ),
    "Cross-section": (
        "anatomical or mechanical cross-section, interior revealed as cartoon "
        "machinery with gears and valves, labelled parts"
    ),
    "Data / HUD": (
        "a semicircular gauge with a needle, simple bar readout, hand-lettered "
        "uppercase label underneath"
    ),
    "Visual metaphor": (
        "an abstract concept rendered as a physical contraption, stickman reacting "
        "to it, motion arrows"
    ),
    "Climax": (
        "the consequence landing, exaggerated stickman reaction, radiating emphasis "
        "lines, bold hand-lettered uppercase text"
    ),
    "(none)": "",
}

# Semantic palette. In this style colour is meaning, not decoration.
PALETTE: dict[str, str] = {
    "red — heat / danger / failure": "red squiggly heat lines and red warning marks",
    "orange — fire / thermal": "orange and yellow layered flame shapes",
    "green — biology / money": "bright green organic elements",
    "blue — water / cool": "saturated blue airflow curves and water",
    "pink — flesh / organs": "soft pink organ and tissue shapes",
}


# ---------------------------------------------------------------------------
# server calls
# ---------------------------------------------------------------------------


def _get(path: str, timeout: int = 5):
    with urllib.request.urlopen(f"{KREA_URL}{path}", timeout=timeout) as r:
        return json.load(r)


def server_status() -> str:
    try:
        h = _get("/health")
    except Exception as e:
        return f"### 🔴 server unreachable\n`{KREA_URL}` — {type(e).__name__}"

    if h.get("status") != "ready":
        return f"### 🟡 loading\nstatus `{h.get('status')}` — generation returns 503 until ready."

    split = h.get("device_split", {}) or {}
    if split.get("moved"):
        speed = f"🟢 GPU split active → `{split['target']}`: {', '.join(split['moved'])}"
    elif split.get("enabled"):
        speed = "🔴 **split enabled but nothing moved — expect ~60s/image, not 20-46s**"
    else:
        speed = f"🟡 split off ({split.get('reason', '?')}) — expect ~60s/image"

    sweep = h.get("fp16_sweep", {}) or {}
    patches = h.get("patches", {}) or {}
    return (
        f"### 🟢 ready · {h.get('model_type')}\n"
        f"{speed}\n\n"
        f"VRAM free **{h.get('vram_free_gb')} GB** · loaded in {h.get('load_s')}s · "
        f"{h.get('generated', 0)} generated\n\n"
        f"patches {patches.get('applied')}/{patches.get('total')} · "
        f"fp16 sweep cast {sweep.get('converted', 0)} tensors"
    )


def timing_line() -> str:
    try:
        s = _get("/stats")
    except Exception:
        return ""
    if not s.get("n"):
        return "_no generations yet_"
    med = s["median_s"]
    flag = "🟢" if med <= 46 else ("🟡" if med <= 60 else "🔴")
    return (
        f"{flag} median **{med}s** · min {s['min_s']}s · max {s['max_s']}s · n={s['n']}  \n"
        f"_target is 20-46s; above 60s means the GPU split isn't working_"
    )


# ---------------------------------------------------------------------------
# reference corpus
# ---------------------------------------------------------------------------


def list_videos() -> list[str]:
    d = REPO_DIR / "frames"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def list_refs(video: str, dense: bool) -> list[str]:
    root = REPO_DIR / ("dense_frames" if dense else "frames") / (video or "")
    if not root.exists():
        return []
    return sorted(str(p.relative_to(REPO_DIR)) for p in root.glob("*.jpg"))


def ref_gallery(video: str, dense: bool):
    return [(str(REPO_DIR / p), Path(p).stem) for p in list_refs(video, dense)]


def next_shot_name() -> str:
    """shot_001.png, continuing past whatever is already in shots/."""
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    used = [
        int(m.group(1))
        for p in SHOTS_DIR.glob("shot_*.png")
        if (m := re.match(r"shot_(\d+)", p.name))
    ]
    return f"shot_{max(used, default=0) + 1:03d}.png"


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def build_prompt(subject: str, preset: str, colors: list[str]) -> str:
    parts = [STYLE]
    if PRESETS.get(preset):
        parts.append(PRESETS[preset])
    if subject.strip():
        parts.append(subject.strip())
    parts.extend(PALETTE[c] for c in colors or [] if c in PALETTE)
    return ". ".join(parts)


def generate(subject, preset, colors, ref, ref_mode, out_name,
             width, height, steps, guidance, seed, progress=gr.Progress()):
    if not subject.strip():
        return None, "Describe the shot first.", "", gr.update()

    prompt = build_prompt(subject, preset, colors)
    body = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE,
        "out_name": out_name.strip() or next_shot_name(),
        "width": int(width), "height": int(height),
        "steps": int(steps), "guidance": float(guidance), "seed": int(seed),
        "ref_paths": [ref] if ref else [],
        "ref_mode": ref_mode,
    }

    progress(0.1, desc="generating…")
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{KREA_URL}/generate", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=900) as r:
            res = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:400]
        return None, f"**{e.code}** — {detail}", timing_line(), gr.update()
    except Exception as e:
        return None, f"**{type(e).__name__}** — {e}", timing_line(), gr.update()

    info = (
        f"**{Path(res['path']).name}** · {res['generate_s']}s · seed `{res['seed']}` · "
        f"{res['size']}" + (f" · ref `{res['ref_mode']}`" if res["refs"] else "")
        + f"\n\n<details><summary>full prompt</summary>\n\n```\n{prompt}\n```\n</details>"
    )
    _ = time.time() - t0
    return res["path"], info, timing_line(), gr.update(value=next_shot_name())


def recent_shots():
    if not SHOTS_DIR.exists():
        return []
    ps = sorted(SHOTS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [(str(p), p.name) for p in ps[:24]]


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

CSS = """
.status-box { border-radius: 8px; padding: 10px 14px; background: var(--block-background-fill); }
footer { display: none !important; }
"""

with gr.Blocks(title="Stickman Shot Factory", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown("# Stickman Shot Factory\nKrea 2 Turbo Edit · Whymentary style")
    status = gr.Markdown(server_status, every=15, elem_classes="status-box")

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=3):
                subject = gr.Textbox(
                    label="Shot", lines=3,
                    placeholder="a stickman sitting on a chair sweating beside a desk fan",
                    info="Describe only the subject — the style block is added for you.",
                )
                preset = gr.Radio(
                    list(PRESETS), value="(none)", label="Shot type",
                    info="The five recurring types from the dissections.",
                )
                colors = gr.CheckboxGroup(
                    list(PALETTE), label="Palette",
                    info="Colour is meaning in this style. Naming it beats hoping for it.",
                )

                with gr.Accordion("Reference image", open=True):
                    gr.Markdown(
                        "Matching a real keyframe is far more reliable than describing "
                        "the style. Leave empty for text-only."
                    )
                    with gr.Row():
                        video = gr.Dropdown(
                            list_videos(), label="Video",
                            value=(list_videos() or [None])[0], scale=2)
                        dense = gr.Checkbox(label="Dense frames", scale=1)
                    gallery = gr.Gallery(
                        ref_gallery((list_videos() or [""])[0], False),
                        label="Click to select", columns=7, height=190,
                        allow_preview=False)
                    ref = gr.Textbox(label="Selected", interactive=False)
                    ref_mode = gr.Radio(
                        ["KI", "I"], value="KI", label="Reference mode",
                        info="KI = scene/base stretched to canvas · I = subject, aspect kept",
                    )

            with gr.Column(scale=2):
                out_name = gr.Textbox(next_shot_name, label="Save as")
                go = gr.Button("Generate", variant="primary", size="lg")
                image = gr.Image(label="Result", height=340, show_download_button=True)
                info = gr.Markdown()
                timing = gr.Markdown(timing_line, every=20)

                with gr.Accordion("Settings", open=False):
                    gr.Markdown(
                        "Defaults are the measured fast path: **8 steps at 1024x576**. "
                        "Raising either costs time roughly linearly."
                    )
                    with gr.Row():
                        width = gr.Number(1024, label="Width", precision=0)
                        height = gr.Number(576, label="Height", precision=0)
                    gr.Markdown("_Both must be divisible by 16._")
                    steps = gr.Slider(4, 30, 8, step=1, label="Steps")
                    guidance = gr.Slider(0.0, 7.0, 1.0, step=0.1, label="Guidance")
                    seed = gr.Number(-1, label="Seed (-1 = random)", precision=0)

    with gr.Tab("Shots"):
        gr.Markdown(
            f"Everything in `{SHOTS_DIR.relative_to(REPO_DIR) if SHOTS_DIR.is_relative_to(REPO_DIR) else SHOTS_DIR}`, "
            "newest first. `shots/` is gitignored — move keepers to `shots/final/` to commit them."
        )
        refresh = gr.Button("Refresh")
        shots_gallery = gr.Gallery(recent_shots, columns=6, height=560, label=None)
        refresh.click(recent_shots, outputs=shots_gallery)

    with gr.Tab("Style"):
        gr.Markdown(f"""
### Always applied
```
{STYLE}
```
### Negative
```
{NEGATIVE}
```
### Palette — colour is fixed meaning, not decoration
| Role | Hex | Used for |
|---|---|---|
| Canvas | `#FFFFFF` | every scene, always pure white |
| Ink | `#000000` | 4–8px linework **only** — this is not a monochrome style |
| Danger / heat | `#E51919` | heat, failure, acidity, anger |
| Thermal | `#FF7A00` | fire layers, heat waves, coins |
| Biology | `#7ED957` | gastric acid, leaves, microbes, money |
| Water / cool | `#2F54EB` | airflow, water, baseline states |
| Flesh | `#F8C8DC` | organs, stomach lining, tissue |

Full reference: `WHYMENTARY_COMPLETE_MASTER_DISSECTION.md`.
""")

    # -- wiring --------------------------------------------------------------
    def _select(video, dense, evt: gr.SelectData):
        refs = list_refs(video, dense)
        return refs[evt.index] if evt.index < len(refs) else ""

    video.change(ref_gallery, [video, dense], gallery).then(lambda: "", None, ref)
    dense.change(ref_gallery, [video, dense], gallery).then(lambda: "", None, ref)
    gallery.select(_select, [video, dense], ref)

    go.click(
        generate,
        [subject, preset, colors, ref, ref_mode, out_name,
         width, height, steps, guidance, seed],
        [image, info, timing, out_name],
    ).then(recent_shots, outputs=shots_gallery)


if __name__ == "__main__":
    print(f"krea server : {KREA_URL}")
    print(f"repo        : {REPO_DIR}")
    print(f"shots       : {SHOTS_DIR}")
    demo.queue().launch(server_name="0.0.0.0", server_port=UI_PORT, show_api=False)
