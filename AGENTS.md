# Working in this repo

Context for any coding agent or assistant used here. Editor-agnostic — nothing below
assumes a particular extension, model, or harness.

## What this repo is

A Kaggle notebook that runs VS Code and an image model (Krea 2 Turbo) on a free T4 GPU,
plus a Whymentary reference corpus, for making explainer-video shots. Read
[README.md](README.md) for the file map.

The house **visual style is our own** — a flat-vector illustration look (see below),
not Whymentary's monoline stickman. The two `WHYMENTARY_*_DISSECTION.md` files are kept
and still useful, but only for the *structural* side: script-to-visual translation
hierarchy, pacing (a visual beat every ~2s), and how the channel turns an abstract
concept into a mechanical analogy. Don't pull visual-style detail (anatomy, linework,
lettering) from them anymore — that part is superseded by the spec below.

The content niche itself — AI explainers/absurdist takes in the Whymentary script
format — is in [NICHE.md](NICHE.md). Read that before writing a script or planning a
shot list; this file is about how to render a shot once you know what it is.

## House style for generated shots

Locked in from a head-to-head style test (5 prompts, same subjects, rendered in the old
monoline stickman spec vs. this one) — this one won clearly: coherent anatomy, no
warped hands, mechanism metaphors read instantly.

- Pure white / pale flat background. No gradients, no drop shadows, no texture.
- Thin, slightly uneven black outline (not thick monoline) — closer to a clean flat-vector
  illustration than a whiteboard-marker sketch. Started life as an "MS Paint style"
  prompt; Krea interprets that as clean flat-vector rather than literally crude/jaggy —
  that's the actual output to expect, not a naive kid-drawing look.
  Anatomically coherent people, not stickmen. Real proportions, real faces, ordinary
  clothing — the style comes from the flat rendering, not from simplified anatomy.
- Flat color fills only, from the palette in the README. Colors carry fixed meaning —
  red is heat/danger/failure, orange is thermal, green is biology/nutrients/success,
  blue is cool/water/calm, pink is flesh/organs. Don't use a color decoratively against
  its assigned role.
- Abstract concepts get a **mechanical analogy**, never a realistic rendering.
  Thermoregulation is a combustion engine inside the torso. Dopamine adaptation is a
  dangling carrot on a treadmill. This part carries over unchanged from Whymentary's
  approach — it's the mechanism-metaphor thinking that's worth keeping, just rendered
  in our own style.

When writing an image prompt, lead with the style clause, then the subject. The style
block that won the test (use this one, not the old monoline block):

```
MS Paint style drawing, thin uneven black outline, flat bucket-fill saturated
colors, no shading, no gradients, naive simple computer-paint-program look
```

Black is the *linework only* — it is not a monochrome style. Scenes carry flat
saturated color, and it's meaningful: name the specific colors the shot needs
(`red squiggly heat lines`, `bright green acid`, `pink stomach lining`) rather than
relying on "flat color fills" to produce them. A prompt that omits color tends to
come back black-and-white.

**Character consistency, without references.** The reference/img2img path is gone from
this pipeline (see below) — there's no way to pin a character via an image anymore. The
fix is text discipline, not tooling: reuse the same character description, verbatim,
in every prompt for a given video — hair, clothing color, build. E.g. "a person with
short brown hair, wearing a solid blue long-sleeve top." Keep the style clause and the
character description identical across shots and viewers won't notice the model isn't
actually the same seed/reference underneath.

## Generating images

Most of the time, use the **Studio UI** (same port as the API, 8711) — it
applies the style clause and names shots for you. It polls `/api/health` every 5s and
reloads the gallery whenever a generation finishes, from *any* source — the UI, a
script, another tab — so anything landing in the currently-selected project shows up
within a few seconds without a manual refresh. It will not show a shot generated into a
*different* project until you switch to it.

For scripted work, the model server is plain HTTP on `127.0.0.1:8711`. Not an MCP
server, no special client — a POST from a terminal works. Text-to-image only — there is
no reference/img2img path anymore (removed entirely; see `Req` in server.py for the
full field list).

```bash
curl -s localhost:8711/api/health          # status, VRAM, what actually loaded
curl -s localhost:8711/api/stats           # timing history — check before optimising

curl -s localhost:8711/api/generate -H 'Content-Type: application/json' -d '{
  "prompt": "MS Paint style drawing, thin uneven black outline, flat bucket-fill
             saturated colors, no shading, no gradients, naive simple
             computer-paint-program look. a person with short brown hair, wearing a
             solid blue long-sleeve top, sweating beside a desk fan",
  "project": "my_video",
  "steps": 8
}'
```

Things the API will reject, and why:

- **Width/height must be divisible by 16.** Latent-space constraint; other sizes fail
  deep in the pipeline with an unhelpful error, so the server checks up front.

Expect **~55-60 s** per image at 1024x576 / 8 steps on this Kaggle T4 x2, even with
`device_split`/pinning fully healthy. Measured live with `nvidia-smi dmon`: GPU0 sits at
100% utilization but power-throttles at the T4's 70W cap (clock oscillates 585-1020 MHz,
never near the 1590 MHz boost) — the bottleneck is the power budget, not the dual-GPU
split or prompt complexity. The README's older 20-40s figure assumed less throttling
than this session actually shows; don't chase that number by re-tuning `device_split` or
pinning, both are already confirmed healthy via `/api/health`. Fewer steps (the UI's
4-step "Max speed" preset) is the lever that actually moves it, roughly linearly.

## Conventions

- Generated shots land in `shots/`, which is gitignored. Move keepers to `shots/final/`
  to commit them. This is not just curation — it's the *only* thing guaranteed to
  survive a Kaggle session getting reset or forked, since nothing else here is
  version-controlled. Raw generations in `shots/<project>/` persist across restarts of
  the same live Kaggle session (that's why the notebook's disk-caching works at all),
  but there's no guarantee beyond that session.
- Don't commit anything to `videos/`, `scratch/`, or the root-level `.py`/`.sh` scripts —
  all intentionally gitignored, see the README.
- `dense_frames/` and `transcripts/` share an index: frame N of a video lines up with the
  transcript timestamps for the same video. Use both together for pacing questions —
  this is still useful for beat-matching even though visual style has diverged.
- The notebook's markdown cells are the setup documentation. Update them there rather
  than starting a parallel doc that will drift.
