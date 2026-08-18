# Video pipeline — LTX-2.5 (parked 2026-08-18)

Interval B-roll for the pixels pipeline: short image-to-video clips seeded from
Krea2 stills, inserted roughly every 30s of finished video. Parked on the L4
studio (23GB VRAM / 31GB RAM) and moving to a bigger-GPU space.

## Read this first: upgrade CUDA before benchmarking anything

The ComfyUI checkpoint is `int8-convrot` quantized. Those kernels live in
comfy-kitchen's **cuda** backend, which is gated behind `torch.version.cuda >= 13`
(`ComfyUI/comfy/quant_ops.py`). On cu128 the model silently runs on the `eager`
fallback, emulating the quantization in torch ops.

| | step time |
|---|---|
| torch 2.8.0+cu128 (eager fallback) | 7.40-7.60 s/step |
| torch 2.13.0+cu130 (fused kernels) | **1.72-1.79 s/step** |

**4.3x, same seed, same workflow.** Confirm it worked by checking the boot log for
`Found comfy_kitchen backend cuda: {... 'disabled': False ...}`.

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.13.0+cu130 torchvision torchaudio
```
Keep torch/torchvision/torchaudio versions matched or ComfyUI dies at import with
`undefined symbol: torch_library_impl`. comfy_kitchen's `_C.abi3.so` links no
libtorch, so any torch version is fine as long as the CUDA runtime is 13+.

## Setup

```bash
./download_models.sh /path/to/ComfyUI/models   # ~36GB, HF gated repo
./start_comfy_ltx.sh                           # serves :8188 + cloudflare tunnel
```
Then load `workflows/video_ltx2_5_i2v.json` in the UI, or drive it headlessly
with `workflows/ltx_api.json` (already flattened to API format — see Gotchas).

## Settings that matter

- **Prompt enhancer OFF.** The stock template runs `TextGenerateLTX2Prompt`,
  which loads a third 10.3GB model *and* rewrites your prompt. The tutorial
  disables it in all three workflows. `ltx_api.json` has the nodes removed.
- **mix4x8 transformer** (17GB) over the official int8-convrot (21.5GB): same
  sampling speed, ~8s less staging per clip. GGUF builds dequantize to bf16 and
  probably re-enter the slow path - untested, treat with suspicion.
- **VAE decode tiling** is the second-biggest lever. Template default
  (512/64/64/16) = 41s. `2048/32/256/8` = **5s**. Big tiles need ~3.25GB free
  VRAM at decode or they OOM; on a card where the transformer stays resident,
  add `--reserve-vram 4` so ComfyUI evicts it first. (Untested - work stopped here.)
- **Generate 10s clips, not 5s.** Fixed per-run overhead dominates:
  5s clip = 34.3s per second of video, 10s clip = 19.6s per second. Model does 20s.
- Distilled model = 8 + 3 steps. The `dev` model needs 20-30 and is for LoRA
  training, not generation.

## Why it was slow on the L4 (probably gone on a bigger card)

Transformer 17-21.5GB + text encoder 15.4GB + VAEs 2GB = ~34GB against 23GB of
VRAM, so ComfyUI evicts and reloads **every clip**. 40-60s of each clip was model
shuffling; real compute was ~45s. Two things made it worse and are studio-specific:
`/teamspace/studios/this_studio` is a `lightning` network mount reading at
**343 MB/s** (local /tmp does 739 MB/s), and 31GB RAM can't cache 34GB of models,
so every reload hit the network filesystem. `stage_models_local.sh` copies weights
to local disk; it was built but never measured.

Best verified on the L4: **~110-130s per 5s clip**, down from 304s.

## Gotchas (hours lost to these)

- **`--enable-triton-backend` breaks LTX.** Its `rms_rope` kernel dies with
  `Triton Error [CUDA]: invalid argument` in the video VAE decode. Krea2 *keeps*
  that flag (13% faster there). Never merge the two launchers. Worth retesting on
  triton 3.7.1, which we never got to.
- **No `--gpu-only`** here (that's Krea2's flag) - the models must load
  sequentially.
- `--fast-disk` / `--cache-none` / `--disable-pinned-memory` were added for an OOM
  that did not exist and made things worse; `--disable-dynamic-vram` forces full
  residency and then OOMs the decode. Measure before adding memory flags.
- **The built-in templates wrap everything in a 47-node subgraph**, which the
  /prompt API cannot accept. `tools/flatten.py` expands it; `workflows/ltx_api.json`
  is the output. The widget-to-input mapping is positional and silently
  misassigns: `batch_size` got 768 and 97 instead of 1, `bit_depth` swallowed the
  frame rate, and dynamic-combo nodes (`ResizeImageMaskNode`) need nested keys -
  replaced with a plain `ImageScale`.
- **Always check `status_str == "success"`.** A websocket node-timing loop reports
  timings happily for runs that OOMed and produced no file.

## Tools

`tools/nodetime.py` per-node timing over the websocket · `tools/final.py` two-clip
steady-state test with success checks · `tools/mixtest.py` A/B two checkpoints ·
`tools/vaesweep.py` decode tile sweep · `tools/duration.py` clip-length scaling ·
`tools/flatten.py` subgraph -> API format.
