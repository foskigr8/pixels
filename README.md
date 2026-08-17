# Studio

Production setup for making explainer-video shots with a flat-vector illustration
style: thin uneven black outline, flat bucket-fill saturated color, no gradients or
shading, coherent human figures (not stickmen). Color carries fixed meaning — red is
heat and failure, green is biology, blue is water and cool. Mechanical metaphors for
abstract concepts, and a visual beat every ~2 seconds.

This is our **own house style**, not [Whymentary](https://www.youtube.com/@Whymentary)'s —
it was chosen because it's what actually renders well through Krea 2 (Whymentary's
monoline stickman look came out warped/uncanny in testing; this flat-vector look didn't).
The Whymentary reference corpus below is kept for the *structural* side only — how to
translate a script beat into a visual, pacing, mechanism-metaphor thinking — not for
literal visual style. See [AGENTS.md](AGENTS.md) for the full house-style spec and the
winning prompt clause.

The engine is **Krea 2 Turbo** (text-to-image only, no references/img2img) running on a
free Kaggle T4 x2, driven by a UI built for this pipeline.

## What's here

| Path | What it is |
|---|---|
| [NICHE.md](NICHE.md) | What we're actually making: an AI-focused sub-niche using Whymentary's script blueprint. Content modes, the extracted structural rules, AI-specific mechanical-analogy starting points, shot-list workflow. Start here for scripting. |
| [WHYMENTARY_COMPLETE_MASTER_DISSECTION.md](WHYMENTARY_COMPLETE_MASTER_DISSECTION.md) | Script-to-visual translation hierarchy, pacing, verbatim per-video breakdowns. Structural reference only — visual-style detail in here is Whymentary's, not ours. |
| [WHYMENTARY_STYLE_AND_PRODUCTION_DISSECTION.md](WHYMENTARY_STYLE_AND_PRODUCTION_DISSECTION.md) | The shorter companion. Same caveat: useful for pacing matrix and how they think about a shot, not for anatomy/linework/lettering, which is now ours instead of theirs. |
| [frames/](frames/) | 42 hand-picked keyframes, 7 per video, at the narrative beats the dissections cite by filename (`01_hook_10s.jpg`, `04_scientific_cross_section_120s.jpg`, …). |
| [dense_frames/](dense_frames/) | 222 evenly-sampled frames, ~40 per video. Use these for motion and pacing questions the 7 keyframes can't answer. |
| [transcripts/](transcripts/) | Verbatim `.vtt` subtitles with timestamps for all 6 videos. Pair with `dense_frames/` to see exactly which visual lands on which spoken clause. |
| [notebooks/studio.ipynb](notebooks/studio.ipynb) | Boots the whole thing on Kaggle: weights, model server, Studio UI, and VS Code. |
| [scripts/server.py](scripts/server.py) | Warm Krea 2 model server. HTTP on `127.0.0.1:8711`. Holds the model, does the fp16 work and the dual-GPU split. |
| [ui/index.html](ui/index.html) | The Studio UI. Style presets, palette, shot naming, live timing. Text prompt → shot. Polls every 5s, so any generation — from the UI or the API — appears automatically. |
| [shots/](shots/) | Generated output. Gitignored — survives kernel restarts within the same Kaggle session, but not a session reset. Keepers go in `shots/final/`, the only part that's committed and durable. |

The palette, in short — everything else is in [AGENTS.md](AGENTS.md):

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

The workflow is deliberately single-path: type a beat, hit Generate, get a shot. The
UI applies the style clause automatically so it can't drift between shots, files shots
into `shots/`, and shows median generation time as you work. No reference slots, no
image-to-image — the model is the plain Turbo, which is smaller and faster to load
(no vision encoder) than the edit variant.

### Speed

18 GB of weights do not fit in 16 GB of VRAM, so mmgp streams them from host RAM. That
streaming — not compute — is the bottleneck. The compute floor at 1024×576 / 8 steps is
roughly 18–22 s; a naive setup gives ~55–60 s.

**Measured reality on Kaggle T4 x2, live-monitored with `nvidia-smi dmon`:** even with
`device_split` and pinning both fully healthy, real generations land at ~55-60 s, not
the 20-40 s the optimizations below target. GPU0 sits at 100% utilization but
power-throttles at the T4's 70 W cap the whole time — SM clock oscillates 585-1020 MHz
and never approaches the 1590 MHz boost. That's the card hitting its power ceiling under
sustained diffusion compute, not a broken split or bad settings. Don't chase the 20-40s
number by re-tuning `device_split`/pinning if `/api/health` already shows both healthy —
this session's actual floor is the naive number. Fewer steps is the only lever that
still moves it, roughly linearly (the UI's 4-step "Max speed" preset ≈ half the time).

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
- On boot the server runs a **self-test**: a tiny generation through the exact
  `/api/generate` path, so the dual-GPU split is *proven* working (or loudly
  failed) before the UI is marked ready. It also warms the transformer into
  VRAM, so the first real shot does not pay the cold-start cost. Set
  `SELF_TEST=0` to skip it while iterating on load errors.

`device_split.moved` still tells you the raw split state; if that list is empty you
are running single-GPU. If a session was started before the split commit and never
restarted, re-run the Start cell — refreshing the browser does not reload the server.

Budgets are capped by mmgp at 80% of VRAM (~12.8 GiB on a 16 GiB T4), so setting a
larger number is silently clamped. The transformer (12.86 GiB at int8) sits just
under that cap, so `TRANSFORMER_BUDGET_MB` is exposed to push it up toward fully
resident if a session has the VRAM headroom — the default 11000 leaves room for
activations; raising it risks CUDA OOM.

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
