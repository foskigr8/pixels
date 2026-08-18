#!/usr/bin/env bash
# LTX-2.5 video stage.
#
# Memory notes (this box: 23GB VRAM, 31GB system RAM, ~37GB of weights):
#   NO --gpu-only     : transformer (21.5GB) + text encoder (15.4GB) must load
#                       sequentially and offload. --gpu-only forces residency = OOM.
#   --disable-dynamic-vram : REQUIRED. Without it comfy "stages" the 20.5GB
#                       transformer and force-preloads only ~3.3MB of it, then
#                       streams the rest per step -> 7.45s/step, ~300s/clip.
#                       Same pathology --gpu-only fixed for Krea2.
#   Do NOT add --fast-disk/--cache-none/--disable-pinned-memory here: they were
#   added for an OOM that did not exist, and made the streaming disk-backed.
#   Measured peak with streaming: 10GB RAM / 21GB VRAM, so there is headroom.
#   NO --enable-triton-backend : its rms_rope kernel dies with "Triton Error
#                       [CUDA]: invalid argument" in the LTX video VAE decode.
#                       Krea2 KEEPS that flag (13% faster there). Do not merge these two.
set -u
cd /teamspace/studios/this_studio/ComfyUI
LOG=/teamspace/studios/this_studio/comfy_ltx.log
: > "$LOG"
python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header '*' \
  --reserve-vram 4 \
  "$@" >> "$LOG" 2>&1 &
COMFY_PID=$!
echo "ComfyUI pid $COMFY_PID  (log: $LOG)"

# public URL (cloudflared lives in ./bin so it survives /tmp wipes)
CF=/teamspace/studios/this_studio/bin/cloudflared
CFLOG=/teamspace/studios/this_studio/cloudflared.log
if [ -x "$CF" ]; then
  : > "$CFLOG"
  "$CF" tunnel --url http://localhost:8188 --no-autoupdate >> "$CFLOG" 2>&1 &
  echo "waiting for tunnel..."
  for i in $(seq 1 20); do
    URL=$(grep -o "https://[a-z0-9-]*\.trycloudflare\.com" "$CFLOG" 2>/dev/null | head -1)
    [ -n "${URL:-}" ] && break
    sleep 2
  done
  echo "PUBLIC URL: ${URL:-<tunnel failed, see $CFLOG>}"
fi
echo "Tail the log with:  tail -f $LOG"
wait $COMFY_PID
