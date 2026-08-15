# AGENT.md — operating instructions

You are the shot factory. You turn a shot list into finished frames for a video.

Read this file, then `style.md`, then `shots.md` before generating anything.

## What you are working with

You have four tools from the `shotfactory` MCP server:

| tool | what it does |
|---|---|
| `krea_health()` | is the image model up, and did its fp16 patches apply |
| `generate_image()` | render one frame into `shots/` |
| `critique_image()` | **your eyes** — a vision model describes and judges a frame |
| `generation_stats()` | timing history |

**You cannot see images.** Not the ones you generate, not the ones in `refs/`.
`critique_image()` is the only channel. When it tells you what is in a frame,
that is ground truth — do not argue with it, and do not assume a frame is fine
because the prompt was good.

## Start of session

1. Call `krea_health()`. If `status` is not `"ready"`, wait and retry — the model
   takes 3–5 minutes to load. Do not start generating before it is ready.
2. Check the health response:
   - `edit_variant` must be `true`. If it is false, reference images will be
     silently ignored. Stop and tell the user.
   - `patches.missing` should be empty. If it is not, say so before generating —
     an unpatched load is the most common cause of failures in this setup.
3. Read `style.md` and `shots.md`.

## The loop, per shot

```
1. build the prompt   -- shot intent from shots.md + style language from style.md
2. generate_image(prompt, out_name="shot_007.png", ref_paths=[...])
3. critique_image("shots/shot_007.png", criteria=<style excerpt + shot intent>)
4. PASS -> next shot
   FAIL -> regenerate with the REVISED PROMPT from the critique, same out_name
5. hard cap: 3 attempts. Then keep the best, note why in your summary, move on.
```

**The 3-attempt cap is not a suggestion.** You cannot see, so if you do not
understand a critique you will chase it forever and burn an hour on one frame.
Three attempts, then move on and flag it for the user.

Reuse the same `out_name` when retrying. Rejects should not accumulate.

## Writing prompts

- Long and concrete beats short and clever. Subject, action, camera and lens,
  lighting, palette, mood.
- Style vocabulary comes from `style.md`. Do not invent a new visual direction
  per shot — consistency across shots is the entire point.
- Do not describe things the reference already carries (palette, texture,
  rendering). Describe what changes: composition, subject, action.
- Negative prompts do very little at Turbo's low guidance. Usually leave empty.

## Reference images

- **Hard cap of 2.** One inpainting. Enforced upstream; more will error.
- **One strong reference beats two mediocre ones.**
- `ref_mode="KI"` — the first reference is the scene/base and is stretched to the
  canvas. Use it for establishing the look, palette, and environment.
- `ref_mode="I"` — every reference is a subject/object, aspect preserved. Use it
  for a character or prop that must stay recognisably itself.
- References carry palette, texture, and rendering. **Composition follows the
  prompt, not the reference.** If you want a different framing, say so in words.
- **Using the same reference across every shot is what makes 40 separate frames
  look like one video.** Change it only when the scene genuinely changes.
- Expect 20–40% slower generation with a reference attached. That is normal —
  attention is quadratic and references add latent tokens.

## Resolution and steps

- Iterate at **1024×576**. Both dimensions must be divisible by 16.
- 1024×1024 costs roughly **2×**, not 1.8×. Upscale only the keepers.
- 8 steps is the Turbo default. Above ~12 you are mostly buying time.

## Writing critique criteria

The quality of the critique depends entirely on the criteria you pass. Include:

1. The relevant excerpt from `style.md` — quote it, do not paraphrase.
2. What this specific shot is supposed to show, from `shots.md`.
3. Anything the previous attempt got wrong, if this is a retry.

Vague criteria produce vague critiques and waste your three attempts.

Gemini's free tier is 15 requests/minute, 1500/day. If `critique_image()` returns
a rate-limit error, wait a minute. Do not retry in a tight loop.

## Committing

- Commit every 3–5 finished shots: `git add -A && git commit -m "shots: 007-011"`
  then `git push`. The Kaggle session dies without warning and takes everything
  outside the repo with it.
- `shots/*.png` is gitignored; `shots/final/*.png` is not. Move frames you are
  confident in into `shots/final/`. Do not commit rejects.
- Keep a short log in `shots.md` next to each shot: seed, attempts, and one line
  on what worked. It is what makes the next session cheaper.

## Do not

- Do not batch generations for throughput. Quality over speed is the whole
  design; one at a time with a critique loop is deliberate.
- Do not edit `scripts/` without asking. The fp16 patches and reference plumbing
  are load-bearing and were hard to get right — see `CLAUDE.md`.
- Do not change `ref_mode` mid-sequence to "fix" a shot. It changes the look of
  the whole set.
- Do not suggest running a second model on GPU1, moving to a bigger resolution,
  or re-architecting the pipeline. Those were considered and rejected with
  reasons in `CLAUDE.md`.
