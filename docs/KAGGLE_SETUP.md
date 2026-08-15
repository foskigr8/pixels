# Shot factory on Kaggle — setup

VS Code in the browser on a Kaggle T4 x2, with DeepSeek driving Krea 2 Turbo Edit
and Gemini acting as its eyes. Files live in this repo; images land in `shots/`
and you `git pull` locally for Remotion.

**The fastest path is to upload `notebooks/shot_factory_kaggle.ipynb` to Kaggle
and run it top to bottom.** This document explains what each cell does and why,
so you can debug when something goes wrong. The notebook is the source of truth
for the commands; this is the source of truth for the reasoning.

**Kaggle settings:** Accelerator `GPU T4 x2`, Internet `On`.
**Session budget:** 12 h max, ~30 h/week. Commit early and often — everything
outside the repo dies with the session.

---

## Before you start

Put three secrets in Kaggle **Add-ons → Secrets**:

| secret | what it is |
|---|---|
| `GH_PAT` | fine-grained GitHub PAT, **Contents: read and write**, scoped to this repo only |
| `DEEPSEEK_API_KEY` | the agent's brain |
| `GEMINI_API_KEY` | the agent's eyes (free tier is fine) |

Then edit `GH_USER` and `GH_REPO` at the top of Cell 1.

## Cold start budget

| step | time |
|---|---|
| pip install | 3–6 min |
| weight downloads (~19.5 GB) | 10–20 min |
| model load | 3–5 min |
| code-server + tunnel | 2–3 min |
| **total** | **~25–35 min** |

Cells 4 and 5 (code-server, MCP config) can run while the model is loading.

---

## Cell 0 — preflight

Confirms you got two T4s and prints `bf16 supported: False`. That False is the
single fact that shapes this whole setup: T4 is Turing (SM75), it has no bf16, no
FlashAttention-2, no fp8. Krea 2 ships bf16 weights, so everything has to be cast
down to fp16 before it touches a kernel.

## Cell 1 — secrets + clone

Clones this repo to `/kaggle/working/repo`. The scripts come with it, so there is
nothing to copy by hand.

## Cell 2 — Wan2GP + weights

**Pin the commit.** The cell prints the SHA it checked out — put that in
`WAN2GP_COMMIT` once you have a working run. Upstream `main` drifts, and when it
does the fp16 patches stop matching and silently do nothing. That is exactly how
the notebook this replaces broke.

Weights land in two places because `/kaggle/working` is only 20 GB persistent and
the weights total ~19.5 GB: the transformer goes there, everything else goes to
`/kaggle/tmp` and gets symlinked back into the Wan2GP models directory.

`Qwen3-VL-4B-Instruct_vision_bf16.safetensors` is the file that matters most.
That is the vision tower. Without it you have `krea2_turbo`, not
`krea2_turbo_edit`, and no reference images — which was the original blocker
behind "there is no img to img".

## Cell 3 — the warm model server

Run the probe first:

```bash
python /kaggle/working/repo/scripts/krea_server.py --probe
```

It imports Wan2GP without loading weights and prints, per patch, whether it
`matches`, is `already applied`, or is a `!! NO MATCH`. Thirty seconds here saves
an hour of debugging a dtype crash later. It also dumps the real signatures of
the loader and generate functions in your pinned checkout — useful when
something has moved.

Then start the server and poll `/health` until `status: ready` (3–5 min).

**Read the boot log.** Two numbers matter:

- `[patch] applied N/4` — how many source patches took.
- `[load] fp16 sweep: converted=N` — how many bf16 tensors the post-load sweep
  had to fix up afterwards. A large number means the patches are not covering,
  and while the load is still *correct*, it is slower than it should be.

The sweep exists so a first boot works even when the patches miss. It is a safety
net, not a substitute — fix the patch table when it drifts.

## Cell 4 — code-server + tunnel

Kaggle accepts no inbound connections, so the editor tunnels out through
cloudflared.

Roo Code needs VS Code ≥1.84; Cline needs ≥1.101. code-server usually lags behind
upstream VS Code, so **Roo is the safer bet** and is the one installed.

The notebook sets a generated password rather than `--auth none`. The tunnel URL
is public: anyone who sees it would otherwise get a shell on your session, your
GitHub token, and your API keys. Treat the URL and password as secrets, and do
not leave the session running unattended.

## Cell 5 — wire up the agent

In code-server: **Roo Code → Settings → Provider: DeepSeek**, paste your key.

The cell writes the MCP config to
`~/.local/share/code-server/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json`.
Reload the code-server window afterwards, then confirm Roo lists four tools:
`krea_health`, `generate_image`, `critique_image`, `generation_stats`.

## Cell 6 — smoke test

Generate one text-only frame and one with a reference, and **look at both**.

If the reference frame does not visibly inherit palette and texture from
`refs/`, the reference was dropped. Wan2GP does this silently:

```python
reference_images = input_ref_images if "I" in video_prompt_type else None
```

No warning, no error, no log line. Check that the model type is
`krea2_turbo_edit` and that `/health` reports `edit_variant: true`.

## Cell 7 — measure before optimising

Baseline was 60 s/image. The compute floor on one T4 is roughly:

| resolution | image tokens | floor, 8 steps |
|---|---|---|
| 1024×576 | ~2,300 | 18–22 s |
| 1024×1024 | ~4,100 | 35–40 s |
| 1536×864 | ~5,200 | 50–60 s |

Run ten generations and look at the median. Near the floor means stop. 50 s+
means there is real overhead, and the candidate fix is moving the text encoder
and VAE to `cuda:1` — mmgp is single-GPU (`runtime_device` returns a bare
`torch.device("cuda")`), so GPU1 sits idle. Expect 60 s → 35–45 s.

Not "seconds". DDP is a training construct, diffusion steps are sequential, and
PCIe tensor-parallel on T4 without NVLink is a net loss.

---

## Troubleshooting

| symptom | cause |
|---|---|
| `!! NO MATCH` on a patch | upstream drifted; fix `scripts/patches/t4_fp16.json` against your pinned SHA |
| dtype mismatch on the first reference image | vision tower still bf16 — the sweep did not reach it; check the boot log |
| references seem ignored | `video_prompt_type` missing `"I"`, or model type is not `_edit` |
| `RuntimeError` about divisibility | width/height must both be divisible by 16 |
| server never reports ready | read `/kaggle/working/krea.log`; usually OOM during load or a missing weight file |
| Roo shows no MCP tools | reload the code-server window; check the JSON path and that `mcp[cli]` installed |
| Gemini 429 | free tier is 15 req/min, 1500/day — wait a minute |
| out of disk | `/kaggle/working` is 20 GB; check the symlinks into `/kaggle/tmp` actually resolved |

## Ending a session

1. Move keepers into `shots/final/`.
2. `git add -A && git commit && git push` (Cell 8 does this).
3. Locally: `git pull`, then Remotion.

Everything not pushed is gone.
