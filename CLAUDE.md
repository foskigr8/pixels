# CLAUDE.md — project context

Handoff from a design session (Aug 2026). Read this before suggesting changes;
several obvious-looking ideas were already tried and rejected for concrete
reasons documented below.

## What this is

Agent-driven image generation for video. Not a chatbot, not a UI project.

```
DeepSeek (Roo Code, in browser VS Code on Kaggle)
   ├── reads style.md + shots.md
   ├── generate_image()  ──▶ Krea 2 Turbo Edit, warm on GPU0, localhost:8711
   ├── critique_image()  ──▶ Gemini Flash  (DeepSeek has NO vision)
   └── commits to this repo ──▶ user git pulls ──▶ Remotion compiles
```

**Quality > speed.** Images are generated one at a time with a critique loop.
Do not propose batching for throughput; it defeats the purpose.

## Hardware reality — Kaggle GPU T4 x2

These constraints drove most decisions. Don't re-litigate them.

| Constraint | Value | Consequence |
|---|---|---|
| GPU | 2× T4, 16 GB each | Turing (SM75) |
| **bf16** | **not supported** | Everything must be cast to fp16. This is why the patches exist. |
| FlashAttention-2 | not supported (needs SM80+) | `sdpa` attention only |
| fp8 / NVFP4 | not supported | int8 (quanto) or fp16 only |
| Interconnect | PCIe, no NVLink | Tensor-parallel across GPUs is slower than single-GPU. Don't. |
| System RAM | 31.3 GB | The real bottleneck. mmgp pins ~19.5 GB of it. |
| Disk | `/kaggle/working` 20 GB persistent; use `/kaggle/tmp` + symlinks for the rest | Weights total ~19.5 GB |
| Session | 12 h max, ~30 h/week | Commit often; everything outside the repo dies |
| Inbound network | none | Tunnel out via cloudflared (Gradio's `share=True` proves it works) |

## Krea 2 / Wan2GP — hard-won source findings

Source: `github.com/DeepBeepMeep/Wan2GP`, `models/krea2/krea2_main.py` + `krea2_handler.py`.

- **Model type must be `krea2_turbo_edit`, not `krea2_turbo`.** The `_edit`
  variants load the Qwen3-VL vision tower, which is what enables reference
  images, inpainting, and outpainting. This was the original blocker.
- **`video_prompt_type` must contain `"I"` or references are silently dropped:**
  `reference_images = input_ref_images if "I" in video_prompt_type else None`.
  Silent, no warning. `"KI"` = first ref is scene/base (stretched to canvas);
  `"I"` = all refs are subjects/objects (aspect preserved).
- **Max 2 reference images** (1 when inpainting). Enforced in
  `validate_generative_settings`.
- **References are consumed twice:** a ≤768px copy goes to the Qwen3-VL vision
  tower for semantics; the full-res version goes through the VAE as latent
  tokens appended to the sequence. Both matter.
- **The vision tower load has no `preprocess_sd`,** so it stays bf16 unless
  patched. Without a fix, passing a reference crashes on dtype mismatch.
- **Upstream `main` drifts and patches silently no-op.** The original notebook's
  text-encoder patch had already stopped matching. **Pin `WAN2GP_COMMIT` to a
  SHA.**
- `TextEncoderCache` is already in the pipeline — repeated/similar prompts skip
  text encoding. Retry loops benefit for free.
- Width/height must be divisible by 16 or the pipeline raises.
- Krea 2's text encoder is Qwen3-VL-4B, but it's a `Qwen3VLTextModel` with **no
  LM head** — it cannot generate text. Don't try to reuse it as an agent.

## How the fp16 problem is handled here

Two layers, because one was not trustworthy on its own:

1. **Source patches** — `scripts/patches/t4_fp16.json`, applied to the Wan2GP
   checkout before import. Fast, but they are string matches against a moving
   upstream. `krea_server.py` prints `applied N/4` and `!! NO MATCH` per entry,
   and can refuse to boot (`required: true`) once you've verified them against
   your pinned SHA.
2. **Post-load sweep** — `force_fp16_inplace()` walks every module and casts any
   surviving bf16 parameter or buffer. This depends on no upstream strings at
   all, so it is the actual guarantee. The patches are the optimisation; the
   sweep is the safety net. A high `converted=` count in the boot log means the
   patches are not covering.

The patches ship with `required: false` so a first boot comes up regardless.
Verify with `--probe`, then harden.

`krea_server.py --probe` also introspects the loader/generate signatures in the
pinned checkout and dumps them, because those drift too. The server passes only
kwargs the resolved signature accepts and logs anything it dropped — with a
loud warning if `video_prompt_type` is among them, since that silently disables
references.

## Decisions made, with reasons

- **Qwen3.8-27B local agent: rejected.** Released 2026-08-14, 27B dense
  multimodal, Apache 2.0. A 4-bit GGUF (~15 GB) would fit on the idle GPU1 via
  llama.cpp with mmap. But it competes for the GPU we want for placement
  optimization, day-old vision (`mmproj`) support in llama.cpp is unproven, and
  DeepSeek is smarter and faster. Dropped in favour of API.
- **DeepSeek + Gemini, not Gemini alone.** User is on Gemini free tier
  (15 RPM / 1,500 RPD). DeepSeek is paid, so it carries the reasoning load;
  Gemini is used only for vision critique, which is cheap in request count.
- **Gemini exposed as an MCP tool, not as a second Cline model.** DeepSeek never
  needs vision — it calls `critique_image()` and gets prose back. This is the
  key architectural move; don't replace it with multi-model routing.
- **code-server on Kaggle, not local VS Code + tunnel.** Local was recommended
  and the user chose remote. Verified Cline (`saoudrizwan.claude-dev`) and Roo
  Code (`RooVeterinaryInc.roo-cline`) are both on Open VSX, so code-server can
  install them. Cline needs VS Code ≥1.101, Roo only ≥1.84 — **prefer Roo.**
- **Git for code/prompts, not for every image.** `.gitignore` excludes
  `shots/*.png`, allows `shots/final/*.png`. User may flip this; if so, update
  `AGENT.md` to match.
- **code-server uses password auth, not `--auth none`.** The cloudflared URL is
  public, and `--auth none` behind it hands anyone with the link a shell plus the
  GitHub PAT and both API keys.

## Performance — current state

**Baseline: 60 s/image** at 8 steps (Gradio notebook, `krea2_turbo`, resolution
unconfirmed — probably 1024×576).

Estimated compute floor on one T4 (~25–30 effective fp16 TFLOPS):

| Resolution | Image tokens | Floor, 8 steps |
|---|---|---|
| 1024×576 | ~2,300 | 18–22 s |
| 1024×1024 | ~4,100 | 35–40 s |
| 1536×864 | ~5,200 | 50–60 s |

**Unresolved: nobody has measured where the 60 s actually goes.** An earlier
claim that mmgp thrashes the transformer every step was wrong — the budget is
13,000 MB against a 12.5 GB transformer with `asyncTransfers=True`, so transfer
overlaps compute and TE/VAE swap once per image, not per step.

**Do this before optimizing anything:** notebook Cell 7 runs ten generations and
prints the median, and `/stats` keeps a rolling history. If it's near the floor,
stop. If there's 30 s+ of overhead, the candidate fix is moving text encoder +
VAE to `cuda:1` (mmgp is single-GPU — `runtime_device` returns bare
`torch.device("cuda")`, so GPU1 is idle). Expected win is 60 s → 35–45 s.
**Not "seconds"** — a YouTube comment claimed DDP/`device_map` would do that; it
won't. DDP is a training construct, diffusion steps are sequential, and PCIe
tensor-parallel on T4 is a net loss.

## Files

- `notebooks/shot_factory_kaggle.ipynb` — the Kaggle notebook; upload and run top to bottom
- `scripts/krea_server.py` — warm model server, fp16 patches + sweep, `--probe`, `/generate`
- `scripts/patches/t4_fp16.json` — the fp16 patch table, with verification instructions
- `scripts/mcp_shotfactory.py` — MCP tools: `generate_image`, `critique_image`, `krea_health`, `generation_stats`
- `docs/KAGGLE_SETUP.md` — cell-by-cell reasoning and troubleshooting table
- `docs/WORKFLOW.md` — run order, timings, reference rules, retry policy
- `AGENT.md` — operating instructions the runtime agent follows
- `style.md` / `shots.md` — user content; style.md is quoted into critique criteria

## Current state

Repo scaffold is complete and pushed. Nothing has been run on Kaggle yet, so
**none of the Wan2GP-facing code has executed against real weights.** The parts
most likely to need adjustment on first boot, in order:

1. the patch table's `find` strings (verify with `--probe`)
2. the loader/generate names resolved in `Wan2GPAdapter` (also printed by `--probe`)
3. the exact kwarg names in `/generate` (the adapter drops unknown ones and logs it)

That is why the probe exists. Everything else — HTTP surface, MCP tools, the
critique prompt, the notebook flow — is independent of upstream.

## Next steps

1. Create the fine-grained PAT (Contents: read/write) and add the three Kaggle secrets
2. Upload the notebook, set GPU T4 x2 + Internet On, run cells 0–5 (~25–35 min)
3. Run `--probe` (Cell 3) and fix any `!! NO MATCH` before anything else
4. Smoke test with a reference (Cell 6) — confirm the vision tower path works
5. Measure `generate_s` × 10 (Cell 7), then decide on GPU1 placement
6. Pin `WAN2GP_COMMIT` to the SHA that worked, and flip the patches to `required: true`

## Known unknowns

- code-server's bundled VS Code version vs Roo's `^1.84.0` requirement
- Whether the patch `find` strings match current upstream (unverified — no
  network access to Wan2GP when this repo was written)
- The exact Wan2GP loader/generate signatures on the pinned SHA
- Actual per-image time with a reference attached (refs add tokens; attention is
  quadratic — expect 20–40% slower than text-only)
