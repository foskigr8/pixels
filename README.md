# Studio

Reference corpus and production setup for making [Whymentary](https://www.youtube.com/@Whymentary)-style
stickman explainer videos: black monoline characters on a pure white canvas, filled with
flat saturated color that carries fixed meaning — red is heat and failure, green is
biology, blue is water and cool. Mechanical metaphors for scientific concepts, and a
visual beat every ~2 seconds.

The engine is **Krea 2 Turbo Edit** running on a free Kaggle T4 x2, driven by a UI built
for this pipeline. The repo also carries **the style bible** — frame-by-frame analysis of
all 6 published channel videos — which is what the generated shots are matched against.

## What's here

| Path | What it is |
|---|---|
| [WHYMENTARY_COMPLETE_MASTER_DISSECTION.md](WHYMENTARY_COMPLETE_MASTER_DISSECTION.md) | The main reference. Script-to-visual translation hierarchy, art direction standards, pop-in animation spec, verbatim per-video breakdowns, consistency checklist. Start here. |
| [WHYMENTARY_STYLE_AND_PRODUCTION_DISSECTION.md](WHYMENTARY_STYLE_AND_PRODUCTION_DISSECTION.md) | The shorter companion. Exact color palette with hex values, character anatomy, pacing matrix, and the Illustrator → After Effects → mix pipeline. |
| [frames/](frames/) | 42 hand-picked keyframes, 7 per video, at the narrative beats the dissections cite by filename (`01_hook_10s.jpg`, `04_scientific_cross_section_120s.jpg`, …). |
| [dense_frames/](dense_frames/) | 222 evenly-sampled frames, ~40 per video. Use these for motion and pacing questions the 7 keyframes can't answer. |
| [transcripts/](transcripts/) | Verbatim `.vtt` subtitles with timestamps for all 6 videos. Pair with `dense_frames/` to see exactly which visual lands on which spoken clause. |
| [notebooks/studio.ipynb](notebooks/studio.ipynb) | Boots the whole thing on Kaggle: weights, model server, Studio UI, and VS Code. |
| [scripts/server.py](scripts/server.py) | Warm Krea 2 model server. HTTP on `127.0.0.1:8711`. Holds the model, does the fp16 work and the dual-GPU split. |
| [ui/index.html](ui/index.html) | The Studio UI. Style presets, reference picker over `frames/`, palette, shot naming, live timing. |
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
**Internet: On**, then run the cells in order. You end up with two public HTTPS URLs:
the **Studio UI**, which is where the work happens, and **VS Code**, for editing
prompts and scripts. Both go through cloudflared, no account needed.

The UI is built for this pipeline rather than for generic image generation. The style
clause is applied automatically so it can't drift between shots; references are picked
from the committed `frames/` and `dense_frames/` corpus with thumbnails rather than
uploaded by hand; the semantic palette is on screen; shots are named and filed into
`shots/`; and median generation time is displayed while you work, so a speed regression
is visible immediately instead of three sessions later.

Everything needed to run or debug it lives in the notebook's own markdown cells,
including a troubleshooting table — deliberately not duplicated here so there's one
source of truth.

Three things that will bite you if you skip them:

- **Run the benchmark cell.** It's the only thing that catches a silent regression from
  20–46 s back to 60 s per image. See below for why that happens.
- **Pin `WAN2GP_COMMIT`.** It starts as `"main"`. Cell 3 prints the resolved SHA; paste
  it back into cell 2. Upstream moves, and the fp16 patches are string matches against it.
- **Change `VSCODE_PASSWORD`.** The cloudflared tunnel is a public URL. The password is
  the only thing between it and a shell on your session.

### Speed: 20–46 s/image, not 60

A naive setup gives ~60 s per image. The compute floor on one T4 at 1024×576 / 8 steps
is roughly 18–22 s, so most of that 60 s isn't compute — it's mmgp thrashing weights
between GPU0 and CPU, because **mmgp is single-GPU and GPU1 sits completely idle.**

Three things close the gap, and all three are set explicitly in the notebook rather than
left to defaults that could quietly change:

1. **8 steps at 1024×576.** The model is turbo-distilled; more steps buy almost nothing.
2. **int8 quantized weights** for the transformer and text encoder — half the weight
   volume for mmgp to move.
3. **Text encoder and VAE on `cuda:1`.** The big one, worth ~15–25 s/image. The
   transformer stays on GPU0 because it runs on every step; the auxiliary towers run
   once per image, so the PCIe hop costs far less than the offload thrash it removes.

`krea_server.py` implements the split by wrapping each moved module's `forward` so
inputs hop to `cuda:1` and outputs hop back — Wan2GP assumes one device throughout, and
this avoids patching every call site. `/health` reports what actually moved. If
`device_split.moved` is empty you are running single-GPU speeds; the attribute names in
`AUX_PATTERNS` didn't match your build.

What does *not* help, already ruled out: DDP, `device_map="auto"`, and PCIe
tensor-parallel. Diffusion steps are sequential, so splitting one step across two T4s
over PCIe is a net loss.

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
