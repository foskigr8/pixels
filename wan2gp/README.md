# Wan2GP archive — retired 2026-08-18

The Wan2GP notebook pipeline, retired in favour of ComfyUI. Deleted from the
L4 studio to get under Lightning's 50GB studio-size limit (it was 17GB of
weights). Nothing here was lost: the checkout had **zero local code changes**.

## Restore

```bash
git clone https://github.com/DeepBeepMeep/Wan2GP.git
cd Wan2GP && git checkout 7f06022   # the commit this studio ran
```

Weights came from the HF repo `DeepBeepMeep/krea-2` into `Wan2GP/models/`.

## Model manifest as it existed

```
   13.49 GB  notebooks/Wan2GP/models/Krea2Turbo_quanto_bf16_int8.safetensors
    4.41 GB  notebooks/Wan2GP/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors
    0.25 GB  notebooks/Wan2GP/models/qwen_vae.safetensors
    0.01 GB  notebooks/Wan2GP/models/Qwen3-VL-4B-Instruct/tokenizer.json
    0.00 GB  notebooks/Wan2GP/models/wan/camera_extrinsics.json
    0.00 GB  notebooks/Wan2GP/models/ltx2/ltx-2.3-22b_PrunaAI_vae.json
    0.00 GB  notebooks/Wan2GP/models/_settings.json
    0.00 GB  notebooks/Wan2GP/models/Qwen3-VL-4B-Instruct/config.json
    0.00 GB  notebooks/Wan2GP/models/qwen_vae_config.json
    0.00 GB  notebooks/Wan2GP/models/Qwen3-VL-4B-Instruct/tokenizer_config.json
    0.00 GB  notebooks/Wan2GP/models/Qwen3-VL-4B-Instruct/preprocessor_config.json
```

## Why it was retired

ComfyUI was chosen over Wan2GP for the pixels pipeline, mainly because of
Wan2GP's unpredictable first-generation `torch.compile` cost (~40s to 436s
across runs) versus ComfyUI's flat ~34s cold load. Steady-state speed was
close either way (Wan2GP ~10.5s, ComfyUI ~11.5s per image on an L4, both fp8).
ComfyUI also exposes an HTTP/websocket API, which is what made every benchmark
in this repo scriptable — that is what finally settled it.

Related: `scripts/patches/t4_fp16.json` is a Wan2GP-era config kept on main.
The measured findings live in KREA2_L4_ATTENTION_BENCHMARK.md and
KREA2_L4_FP8_RESULT.md.
