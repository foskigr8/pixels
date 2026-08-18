#!/usr/bin/env bash
# Restore the LTX-2.5 weights. Requires `hf auth login` with access granted at
# https://huggingface.co/Lightricks/LTX-2.5 (gated, auto-approve).
set -e
M=${1:-$HOME/ComfyUI/models}
mkdir -p "$M"/{diffusion_models,text_encoders,vae,latent_upscale_models}
dl(){ hf download "$1" "$2" --local-dir /tmp/_dl && cp /tmp/_dl/"$2" "$M/$3/" && rm -rf /tmp/_dl; }

# Transformer: mix4x8 is FASTER-EQUAL and 4.5GB smaller than the official int8.
# It hits comfy-kitchen's native asym_w4a8_int8 kernels (needs CUDA >= 13).
dl joeygambino/LTX-2.5-Quantized LTX25-distilled-DiT-comfy-mix4x8-17GB.safetensors diffusion_models
# Official alternative (21.5GB): Lightricks/LTX-2.5
#   diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
dl Lightricks/LTX-2.5 text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors text_encoders
dl Lightricks/LTX-2.5 vae/ltx-2.5-video-vae-bf16.safetensors vae
dl Lightricks/LTX-2.5 vae/ltx-2.5-audio-vae-bf16.safetensors vae
dl Lightricks/LTX-2.5 latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors latent_upscale_models
# NOT needed: gemma4_e2b_it_bf16 (10.3GB) is only for the prompt enhancer, which
# we disable - the tutorial disables it too, and it rewrites your prompt.
echo "done"; du -sh "$M"/*
