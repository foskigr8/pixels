# MUST READ — project state, findings, bugs, plans

Read this before starting any new session on `pixels/`. Kept identical across
`main`, `video-pipeline`, and `wan2gp-archive` so it's visible no matter which
branch is checked out. Last updated 2026-08-19.

## Where things stand

- **Krea2 stills**: working, fast, on `main`. See below.
- **Krea2 img2img (identity edit)**: working, fixed after a long dead-end
  chase. See below.
- **LTX-2.5 video**: built, benchmarked, then parked — moved to branch
  `video-pipeline` for a bigger-GPU studio. Not usable on a 23GB card without
  the compromises documented there.
- **Wan2GP**: retired in favor of ComfyUI. Archived on branch `wan2gp-archive`.
  Zero local code changes existed — fully recoverable from upstream.
- **Next phase**: write the real production script, add motion (60-second
  scenes now, not the earlier 10s/interval plan), user is sourcing an
  animation LoRA (not yet acquired — ask what it targets before designing
  the motion pipeline around it).

---

## Krea2 text-to-image

**Model: `krea2_turbo_int8_convrot.safetensors`** (not `fp8_scaled` — switched
2026-08-19). Same 13GB size, same quality on direct A/B, **22-27% faster**
because it hits comfy-kitchen's native fused kernels instead of a slower
native-but-not-fused path. 8 steps, CFG 1, euler/simple. **~7.4-8.6s/image.**

**This required upgrading torch to 2.13.0+cu130.** The int8-convrot /
convrot_w4a4 quantization format's fast kernels live in comfy-kitchen's
`cuda` backend, which is gated behind `torch.version.cuda >= 13`
(`comfy/quant_ops.py`). On cu128 it silently fell back to slow emulation —
this was the actual bottleneck the whole time, on both Krea2 and LTX-2.5.
Confirm the upgrade took by checking the boot log for
`Found comfy_kitchen backend cuda: {... 'disabled': False ...}`.

Launch flags: `--enable-triton-backend` (still worth ~5-13% even with cuda13
active), **no `--gpu-only`** (see img2img section — it breaks that pipeline
and costs <1s on t2i, not worth running two servers over).

## Krea2 img2img / character consistency (identity edit)

**User's workflow rule:** img2img is the priority path for any character that
recurs in the video — it's how identity stays consistent. Plain t2i is only
for one-off subjects that won't reappear (e.g. two unrelated characters in a
single throwaway scene).

### The dead end — do not revisit

`TextEncodeQwenImageEditPlus` + `FluxKontextMultiReferenceLatentMethod`
(`index_timestep_zero` / `offset` / `index`), whether via the node's built-in
`vae`+`image1` inputs or a hand-built `ImageScale -> VAEEncode ->
ReferenceLatent` graph. **This was never clean, on any config tried** —
not before optimization, not after, not on fp8_scaled, not on int8_convrot,
not with triton on or off. Symptoms: reference-background ghosting, or (with
offset/index) a hard two-tone canvas split plus checkerboard/moire corruption
on fine detail. The official Comfy-Org "Image Style Reference" template
(`pixels_character_consistency.json`) uses this same broken mechanism — it is
not a fix either.

### The actual fix

`ComfyUI/custom_nodes/comfyui-krea2edit` (github.com/lbouaraba/comfyui-krea2edit)
+ the **`krea2_identity_edit_v1_2.safetensors`** LoRA
(`ComfyUI/models/loras/krea2/`, 1.8GB — do not confuse with
`krea2_style_reference.safetensors`, a different, unrelated LoRA).

Proof this works: `pixels/notebooks/shots/IDENTITY_EDIT_RESULTS.png` — four
completely different scenes, identity held, zero ghosting/checkerboard/split.
**Look at this file first**, before debugging any future generation-quality
issue, to know what "working" looks like.

Two nodes, both required (stock CLIPTextEncode silently drops quality):
- `Krea2EditModelPatch` — source image as in-context latent tokens.
  Wire `vae` + `source_image` + `target_latent` (same latent that feeds
  `KSampler.latent_image`) for the pixel path — pre-encodes before sampling
  starts instead of mid-sampling, which otherwise evicts part of the
  diffusion model and slows every remaining step.
- `Krea2EditGroundedEncode` — image-grounded instruction encoding via
  Qwen3-VL. Used twice: real instruction -> positive, empty prompt on the
  same image -> negative.

Saved, ready-to-run workflow: **`ComfyUI/user/default/workflows/pixels_identity_edit.json`**
(not in this git repo — lives on the studio's ComfyUI install).

**Settings, all confirmed by testing, not guessed:**
- `ref_boost = 2` (NOT the template's shipped default of 4 — that value
  visibly suppresses expression and pose/profile-turn variation; this is why
  early test generations "all looked the same")
- **Single-figure guard, mandatory in every instruction**: *"A single figure
  only, not two, no duplicate, no before-and-after, one continuous character
  in one final pose."* Without it, pose-change instructions ("turn to
  profile") sometimes rendered as two panels side by side with mismatched
  skin tone between them. Same root cause as the t2i "unnamed word" bug in
  `PROMPT_PATTERNS.md` §2 — an instruction implying change without an
  explicit single-subject guard invites multiple states.
- 8 steps, CFG 1, 1024x576 (matches every other still in the pipeline —
  the template's own default is a 1:1 square, don't use it for this project).
- `fit_mode: "fit"` (needs `vae` + `source_image` wired — resamples
  mismatched aspect ratios instead of degrading).
- **~20s/image** steady state (~55s cold first run).

**Server config is critical: NO `--gpu-only`.** Confirmed OOM 3 times,
reproducibly, with both `--gpu-only` alone and `--gpu-only
--enable-triton-backend`. Root cause: the four models in play (UNET 13GB +
LoRA 1.8GB + text encoder 4.9GB + VAE 0.24GB = **19.9GB static**) already fill
almost the whole 22GB card; `--gpu-only` pins all of them permanently and
prevents the eviction this node pack's own code relies on between the encode
phase and the sampling phase. `--highvram` (a gentler "keep cached after use"
flag) hits the identical ceiling — fails the same way, even faster. There is
no flag-only fix; it's a genuine VRAM-budget conflict. Running without the
flag costs t2i <1s and is the only thing that works for img2img — use one
server, no `--gpu-only`, for both workloads.

Launcher: `start_comfy_krea2.sh` (on the studio disk, not this repo) — no
`--gpu-only`, `--enable-triton-backend` on.

---

## LTX-2.5 video (parked — see branch `video-pipeline` for full detail)

Headline finding, applies before benchmarking anything on new hardware:
**upgrade to cu130 first.** Sampling went 7.4s/step -> 1.73s/step (4.3x) the
moment the CUDA-13 kernel gate opened for the int8-convrot checkpoint — same
mechanism as the Krea2 finding above. A bigger GPU on cu128 would still be
crippled by this.

Other verified findings on `video-pipeline`: `mix4x8` checkpoint beats
official `int8-convrot` (17GB vs 21.5GB, same speed, native kernels — unlike
GGUF, which dequantizes to bf16 and loses the speedup); VAE decode tile size
matters a lot (512px default = 41s decode, 2048px = 5s, but needs ~3GB free
VRAM at decode time); 10-second clips are far more efficient than 5-second
ones (fixed per-clip overhead dominates at short lengths); `--gpu-only`
crashed differently here too (breaks `--enable-triton-backend`'s rms_rope
kernel — `Triton Error [CUDA]: invalid argument`); the tutorial's "~20s per
clip" was on unstated (almost certainly much larger) hardware, not something
to treat as a target on this L4.

Best result reached on the L4 before parking: **~100-130s per 5s clip**, down
from a 304s cold start. Full setup, download script, and six benchmark tools
are on `video-pipeline` under `video/`.

---

## Wan2GP (retired — see branch `wan2gp-archive`)

Replaced by ComfyUI because of unpredictable `torch.compile` cost on first
generation (40s-436s across runs, vs ComfyUI's flat ~34s cold load) and
because ComfyUI's HTTP/websocket API is what made every benchmark in this
project scriptable — that's the difference that actually mattered, more than
raw steady-state speed (which was close either way, ~10.5s vs ~11.5s/image
pre-optimization). Deleted from disk to reclaim 17GB; zero local code changes
existed, so nothing was lost — see `wan2gp-archive`'s own README for the
restore command and full model manifest.

---

## Known bugs / gotchas worth not re-discovering

- **`pgrep -f`/`pkill -f` on a launch command self-matches.** A restart
  script containing the literal string `"main.py --listen"` will match its
  own wrapper process and kill itself before or during launch (manifests as
  mysterious `exit 144`s). Use `pgrep -af "[m]ain\.py"` (bracket trick) or
  kill by exact PID, never by matching the command you're about to run.
- **Chaining `sleep N; grep ...` after a backgrounded launch in one Bash
  call gets the whole thing torn down** once the foreground part returns,
  taking the background server with it despite `disown`. Launch with the
  tool's actual background mode instead of manual `&`/`disown`/`setsid`.
- **`ResolutionSelector` -> `EmptySD3LatentImage` wiring can silently swap
  width/height** — a `16:9` selection produced a 768x1376 *portrait* latent,
  invisible in the identity-edit template because its own default (`1:1
  Square`) can't expose an orientation swap. Caused a silent kernel OOM-kill
  (no Python traceback at all) from generating at ~2x the intended pixel
  count. Fixed in `pixels_identity_edit.json` by hardcoding 1024x576 instead
  of routing through the selector. If copying that template again for a new
  aspect ratio, verify the actual resulting width/height in the boot log,
  don't trust the selector name.
- **A background-task "completed" notification does not mean the run
  succeeded.** A node-timing script that only watches websocket node
  transitions (not final status) will happily report clean timings for a run
  that OOM'd and produced no file. Always check `status_str == "success"`
  and that an output file actually exists before trusting a benchmark number.
- **Check `notebooks/shots/` (or equivalent) and already-installed
  `custom_nodes/` for a purpose-built solution before reaching for a
  generic/official ComfyUI template.** The identity-edit dead end above was
  reached three separate times — raw FluxKontext, a hand-optimized version of
  it, then the official Comfy-Org template — before the actual answer (a
  specific custom node already installed for exactly this purpose, with proof
  of it working already sitting on disk) was found.

---

## Prompting rules

Full detail in `PROMPT_PATTERNS.md` (t2i-focused; still applies as the base
style). Additions from img2img work, not yet merged into that file:
- The single-figure guard above is required for identity-edit pose-change
  instructions specifically — `PROMPT_PATTERNS.md` §2's "never leave a word
  unnamed" rule is the same underlying principle, applied to a different
  symptom (duplicate figures instead of stray text).
- identity-edit prompts are **instructions**, not scene descriptions:
  "change the scene to: X", not a from-scratch description of the whole
  frame. The source image supplies everything not explicitly changed.
