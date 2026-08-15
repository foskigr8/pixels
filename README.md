# Stickman

Reference corpus and production setup for making [Whymentary](https://www.youtube.com/@Whymentary)-style
stickman explainer videos: flat black monoline characters on a pure white canvas,
mechanical metaphors for scientific concepts, a visual beat every ~2 seconds.

The repo holds two things — **the style bible**, derived from frame-by-frame analysis of
all 6 published channel videos, and **a Kaggle notebook** that puts VS Code and an image
model on a free GPU so you can produce shots against that reference.

## What's here

| Path | What it is |
|---|---|
| [WHYMENTARY_COMPLETE_MASTER_DISSECTION.md](WHYMENTARY_COMPLETE_MASTER_DISSECTION.md) | The main reference. Script-to-visual translation hierarchy, art direction standards, pop-in animation spec, verbatim per-video breakdowns, consistency checklist. Start here. |
| [WHYMENTARY_STYLE_AND_PRODUCTION_DISSECTION.md](WHYMENTARY_STYLE_AND_PRODUCTION_DISSECTION.md) | The shorter companion. Exact color palette with hex values, character anatomy, pacing matrix, and the Illustrator → After Effects → mix pipeline. |
| [frames/](frames/) | 42 hand-picked keyframes, 7 per video, at the narrative beats the dissections cite by filename (`01_hook_10s.jpg`, `04_scientific_cross_section_120s.jpg`, …). |
| [dense_frames/](dense_frames/) | 222 evenly-sampled frames, ~40 per video. Use these for motion and pacing questions the 7 keyframes can't answer. |
| [transcripts/](transcripts/) | Verbatim `.vtt` subtitles with timestamps for all 6 videos. Pair with `dense_frames/` to see exactly which visual lands on which spoken clause. |
| [notebooks/kaggle_vscode.ipynb](notebooks/kaggle_vscode.ipynb) | Boots VS Code in the browser on a Kaggle T4 x2, plus a warm Krea 2 image model. |
| [scripts/krea_server.py](scripts/krea_server.py) | The model server the notebook runs. HTTP on `127.0.0.1:8711`. |
| [shots/](shots/) | Generated output. Keepers go in `shots/final/`, which is the only part that's committed. |

The palette, in short — everything else is in the dissections:

| Role | Hex | Used for |
|---|---|---|
| Canvas | `#FFFFFF` | Every scene. Always pure white. |
| Ink | `#000000` | 4–8px organic hand-drawn linework |
| Danger / heat | `#E51919` | Heat, failure, acidity, anger |
| Thermal | `#FF7A00` | Fire layers, heat waves, coins |
| Biology | `#7ED957` | Gastric acid, leaves, microbes, money |
| Water / cool | `#2F54EB` | Airflow, water, baseline states |
| Flesh | `#F8C8DC` | Organs, stomach lining, tissue |

## Running it on Kaggle

Open `notebooks/kaggle_vscode.ipynb` on Kaggle with **Accelerator: GPU T4 x2** and
**Internet: On**, then run cells 1–5. You get a public HTTPS URL for a full VS Code
session with this repo open. Install whatever extensions you want from the Extensions
panel — nothing in this repo assumes a particular editor setup or agent.

Cells 6–9 are optional and bring up [Krea 2 Turbo Edit](https://github.com/DeepBeepMeep/Wan2GP)
for generating shots. It's a separate step because the editor is useful on its own and
the model takes 10–20 minutes to download the first time.

Everything you need to know to run or debug it lives in the notebook's own markdown
cells, including a troubleshooting table — deliberately not duplicated here, so there's
one source of truth to keep current.

Two things that will bite you if you skip them:

- **Pin `WAN2GP_COMMIT`.** It starts as `"main"`. Cell 6 prints the resolved SHA; paste
  it back into cell 2. Upstream moves, and the fp16 patches are string matches against it.
- **Change `VSCODE_PASSWORD`.** The cloudflared tunnel is a public URL. The password is
  the only thing between it and a shell on your session.

### Why the fp16 machinery exists

Kaggle's T4 is Turing (SM75), which has no bf16. Krea 2 ships bf16 weights and Wan2GP
assumes bf16 in several places, so every bf16 tensor has to become fp16 before it reaches
a kernel. `scripts/krea_server.py` does this twice over: string patches applied to the
Wan2GP checkout before import (fast, but they drift with upstream), and a post-load sweep
that casts anything the patches missed (slow, but depends on nothing upstream).

The sweep is the correctness guarantee; the patches are just an optimisation. So a
`!! NO MATCH` warning means a slower load, not wrong output. The patch strings in
`scripts/patches/t4_fp16.json` are **unverified templates** — run
`python scripts/krea_server.py --probe` against your pinned checkout to check them, fix
the find-strings, then flip `required` to `true` so a future drift fails loudly instead
of silently costing you load time.

## What's deliberately not in the repo

`videos/` (58 MB of source `.mp4`s), `scratch/`, and the root-level `.py`/`.sh` extraction
scripts are all gitignored. Those scripts were one-off local tools that *produced* the
committed frames and transcripts — they depend on a local ffmpeg/yt-dlp setup and aren't
part of the production pipeline. They're still on disk locally, just untracked.
