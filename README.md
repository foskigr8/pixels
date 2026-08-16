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

Open [notebooks/studio.ipynb](notebooks/studio.ipynb) with **Accelerator: GPU T4 x2**
and **Internet: On**, then run the cells top to bottom. Five steps: setup, weights,
start, push, keepalive. You get two public HTTPS URLs — **Studio** and **VS Code** —
both through cloudflared, no account needed.

Change `PASSWORD` in cell 1 before running. The tunnel is public and that password is
the only thing between it and a shell on your session.

First run downloads ~18 GB and takes roughly 20 minutes. Later runs in the same session
are instant: every step checks disk before doing anything.

The UI applies the style clause automatically so it can't drift between shots, picks
references from the committed `frames/` corpus with thumbnails, files shots into
`shots/`, and shows median generation time as you work.

### Speed

18 GB of weights do not fit in 16 GB of VRAM, so mmgp streams them from host RAM. That
streaming — not compute — is the bottleneck. The compute floor at 1024×576 / 8 steps is
roughly 18–22 s; a naive setup gives ~55–60 s.

What moves the number:

1. **8 steps, guidance 0.0.** Turbo is distilled. Upstream defaults (28 steps,
   guide_scale 4.5) are for the non-distilled model and cost ~3× for nothing.
2. **int8 quantized weights** — half the volume for mmgp to move.
3. **mmgp profile 2** with pinned memory, async transfers, per-component budgets,
   and sdpa attention (the only backend that builds on SM75).
4. **Text encoder and VAE on `cuda:1`.** The second GPU is otherwise idle. Beyond
   freeing VRAM, this takes ~6 GB out of the *host RAM* pinning requirement, which is
   what actually matters.

That last point is the one to watch. If the load log says:

```
Switching to partial pinning since full requirements ... is 18107.2 MB
  while estimated available reservable RAM is 16050.0 MB
```

then async transfers are effectively off and you are paying for it. Two places make
the diagnosis a one-liner instead of reading logs:

- `/api/health` now reports `speed` — a `{flag, msg}` pair that reads
  `single-gpu`, `partial-pinning`, or `ok` with the expected seconds per image. The
  load log also prints `pinning=FULL` or `PINNING=PARTIAL` explicitly.
- The notebook's Start cell ends with a speed self-check that warns out loud if the
  split did not engage or pinning is partial, so a slow session explains itself
  before you start generating.

`device_split.moved` still tells you the raw split state; if that list is empty you
are running single-GPU. If a session was started before the split commit and never
restarted, re-run the Start cell — refreshing the browser does not reload the server.

Budgets are capped by mmgp at 80% of VRAM (~11.9 GB on a T4), so setting a larger
number is silently clamped.

Ruled out, don't retry: DDP, `device_map="auto"`, PCIe tensor-parallel. Diffusion steps
are sequential, so splitting one step across two T4s over PCIe is a net loss.

### The fp16 machinery

T4 is Turing (SM75) — no bf16 — and Krea ships bf16 weights. Two patches in
[scripts/patches/t4_fp16.json](scripts/patches/t4_fp16.json) rewrite the Wan2GP loader
before import, and a post-load sweep casts whatever they missed. The sweep depends on
nothing upstream, so it is the correctness guarantee; the patches are the fast path.

Two further patches are marked optional because `load_model` accepts `dtype` and
`VAE_dtype` directly and we pass fp16 for both. The two required ones have no
argument-level equivalent, so the server refuses to boot if they stop matching.

## What's deliberately not in the repo

`videos/` (58 MB of source `.mp4`s), `scratch/`, and the root-level `.py`/`.sh` extraction
scripts are all gitignored. Those scripts were one-off local tools that *produced* the
committed frames and transcripts — they depend on a local ffmpeg/yt-dlp setup and aren't
part of the production pipeline. They're still on disk locally, just untracked.
