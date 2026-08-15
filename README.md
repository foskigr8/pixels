# pixels — a shot factory

Agent-driven image generation for video. An LLM reads a style guide and a shot
list, renders frames with **Krea 2 Turbo Edit** on a Kaggle T4, critiques its own
output through a vision model, retries what fails, and commits the keepers here.
You pull and compile with Remotion.

Not a chatbot. Not a UI project. The point is getting 40 frames that look like
they belong to the same video, without you prompting each one by hand.

```
DeepSeek  (Roo Code, in browser VS Code on Kaggle)
   ├── reads style.md + shots.md
   ├── generate_image()  ──▶ Krea 2 Turbo Edit, warm on GPU0
   ├── critique_image()  ──▶ Gemini Flash        (DeepSeek has no vision)
   └── git push          ──▶ you git pull        ──▶ Remotion
```

The agent has no eyes. That is deliberate: vision is a *tool* it calls, not a
second model to route between. See `CLAUDE.md` for why.

## Layout

```
notebooks/shot_factory_kaggle.ipynb   upload this to Kaggle, run top to bottom
scripts/krea_server.py                warm model server on localhost:8711
scripts/mcp_shotfactory.py            the agent's four tools
scripts/patches/t4_fp16.json          bf16 -> fp16 patches (T4 has no bf16)
docs/KAGGLE_SETUP.md                  cell-by-cell reasoning + troubleshooting
docs/WORKFLOW.md                      run order, timings, reference rules
AGENT.md                              instructions the runtime agent follows
CLAUDE.md                             design context: what was tried and rejected
style.md                              your visual direction  <- fill this in
shots.md                              your shot list         <- fill this in
refs/                                 reference images (tracked)
shots/                                output (gitignored; shots/final/ is tracked)
```

## Quick start

**On Kaggle:**

1. Settings → Accelerator **GPU T4 x2**, Internet **On**
2. Add-ons → Secrets: `GH_PAT` (fine-grained, Contents read/write),
   `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`
3. Upload `notebooks/shot_factory_kaggle.ipynb`, edit `GH_USER` / `GH_REPO` in
   Cell 1, run cells 0–5 — about **25–35 minutes** cold
4. Run Cell 6 and **look at both smoke-test images**. If the reference one does
   not visibly inherit the look of `refs/`, stop and read the troubleshooting
   table — Wan2GP drops references silently
5. Open the tunnel URL, log into code-server, point Roo Code at DeepSeek
6. Tell it: *"read AGENT.md, style.md and shots.md, then start on sequence A"*

**Locally:** `git pull`, then Remotion.

Fill in `style.md` and `shots.md` first. They are the input; everything else is
plumbing.

## Timing

| | |
|---|---|
| cold start, once per session | 25–35 min |
| one clean shot | 40–75 s |
| one shot with a retry | 90–140 s |
| 40-shot video | 1–1.5 h |

## Things that will bite you

- **Pin `WAN2GP_COMMIT`.** Upstream drifts; the fp16 patches then silently stop
  applying. This already broke the notebook this replaces.
- **`krea2_turbo_edit`, not `krea2_turbo`.** Only the `_edit` variants load the
  vision tower, and only the vision tower gives you reference images.
- **Max 2 references, and one strong one beats two mediocre ones.** Same
  reference across every shot is what makes the set look like one video.
- **Both dimensions divisible by 16**, always.
- **12 h session cap, and it can die earlier.** Everything outside this repo goes
  with it. Push often.
- **The tunnel URL is public.** It carries your GitHub token and API keys behind
  it. Treat it as a secret; don't leave the session unattended.

## Status

Scaffold complete, nothing run on Kaggle yet — so the Wan2GP-facing code has not
executed against real weights. `python scripts/krea_server.py --probe` exists for
exactly this: it reports patch status and the real function signatures in your
pinned checkout without loading anything. Run it first.

`CLAUDE.md` has the full design record, including several plausible-looking ideas
that were already tried and rejected with reasons.
