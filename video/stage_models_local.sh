#!/usr/bin/env bash
# Copy the LTX weights onto local disk (~2.2x faster reads than the studio mount).
set -e
M=/teamspace/studios/this_studio/ComfyUI/models
mkdir -p /tmp/models/{diffusion_models,text_encoders,vae,latent_upscale_models}
cp -n $M/diffusion_models/LTX25-distilled-DiT-comfy-mix4x8-17GB.safetensors /tmp/models/diffusion_models/ 2>/dev/null || true
cp -n $M/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors /tmp/models/text_encoders/ 2>/dev/null || true
cp -n $M/vae/ltx-2.5-video-vae-bf16.safetensors /tmp/models/vae/ 2>/dev/null || true
cp -n $M/vae/ltx-2.5-audio-vae-bf16.safetensors /tmp/models/vae/ 2>/dev/null || true
cp -n $M/latent_upscale_models/*.safetensors /tmp/models/latent_upscale_models/ 2>/dev/null || true
du -sh /tmp/models/*
