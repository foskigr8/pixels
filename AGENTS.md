# Working in this repo

Context for any coding agent or assistant used here. Editor-agnostic — nothing below
assumes a particular extension, model, or harness.

## What this repo is

A style reference corpus for Whymentary-style stickman explainer videos, plus a Kaggle
notebook that runs VS Code and an image model on a free T4 GPU. Read
[README.md](README.md) for the file map.

The two `WHYMENTARY_*_DISSECTION.md` files are the source of truth for anything
stylistic. They were derived from actual footage, not from memory — when a question is
about how the channel looks or paces, quote them rather than generalising about
"whiteboard animation".

## House style for generated shots

Non-negotiables, all from the dissections:

- Pure white `#FFFFFF` background. Never a gradient, never a drop shadow, never a
  textured paper background.
- Solid black `#000000` monoline linework, 4–8px, uniform weight.
- Flat color fills only, from the palette in the README. Colors carry fixed meaning —
  red is heat/danger/failure, green is biology/nutrients/money, blue is cool/water,
  pink is flesh/organs. Don't use a color decoratively against its assigned role.
- Stickman anatomy: circle head with thick contour, dot or slit eyes, single-stroke
  body and limbs, angular joint bends, 3-prong hands, flat feet.
- Text on canvas is hand-lettered irregular uppercase marker. Black for explanation,
  red with an underline stroke for warnings and failure states.
- Abstract concepts get a **mechanical analogy**, never a realistic rendering.
  Thermoregulation is a combustion engine inside the torso. Dopamine adaptation is a
  dangling carrot on a treadmill. Never anatomically accurate illustration.

When writing an image prompt, lead with the style clause, then the subject. The style
block that works:

```
minimal hand-drawn stickman on a pure white background, thick 6px solid black
monoline ink strokes, flat color fills, no gradients, no shadows, whiteboard
explainer illustration
```

Pass a real keyframe from `frames/` as a reference image whenever the shot needs to
match an existing look — it's far more reliable than describing the style in words.

## Generating images

The model server is HTTP on `127.0.0.1:8711`, started by the notebook. It is not an
MCP server and needs no special client — plain POST works from a terminal or a script.

```bash
curl -s localhost:8711/health          # status, VRAM, what actually loaded
curl -s localhost:8711/stats           # timing history — check before optimising

curl -s localhost:8711/generate -H 'Content-Type: application/json' -d '{
  "prompt": "minimal hand-drawn stickman ... sweating beside a desk fan",
  "out_name": "shot_012.png",
  "ref_paths": ["frames/01_fan_death/01_hook_10s.jpg"],
  "ref_mode": "KI"
}'
```

Things the API will reject, and why:

- **Width/height must be divisible by 16.** Latent-space constraint; other sizes fail
  deep in the pipeline with an unhelpful error, so the server checks up front.
- **Max 2 reference images.** Wan2GP's own limit.
- `ref_mode` `"KI"` treats the first reference as the scene/base, stretched to canvas.
  `"I"` treats every reference as a subject, preserving aspect ratio. The mode string
  must contain `I` or references are silently ignored upstream — the server enforces this.

Expect ~20–25 s per image at 1024x576 / 8 steps on a T4. If you're seeing much worse,
check `/stats` before changing parameters.

## Conventions

- Generated shots land in `shots/`, which is gitignored. Move keepers to `shots/final/`
  to commit them.
- Don't commit anything to `videos/`, `scratch/`, or the root-level `.py`/`.sh` scripts —
  all intentionally gitignored, see the README.
- `dense_frames/` and `transcripts/` share an index: frame N of a video lines up with the
  transcript timestamps for the same video. Use both together for pacing questions.
- The notebook's markdown cells are the setup documentation. Update them there rather
  than starting a parallel doc that will drift.
