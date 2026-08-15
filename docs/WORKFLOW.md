# Workflow — what you actually do, and how long it takes

This answers the practical questions: what is the run order, what do *you* do
versus what the agent does, how long does each part take, and what are the rules
for references.

## The two machines

| | where | what runs there |
|---|---|---|
| **Kaggle** | remote, T4 x2 | Krea 2 model server, code-server, Roo Code + DeepSeek, MCP tools |
| **Your laptop** | local | `git pull`, Remotion, final compile |

Git is the only link between them. Images move as commits, not as a live stream.

## Run order

```
1. Kaggle: upload notebooks/shot_factory_kaggle.ipynb, set GPU T4 x2 + Internet On
2. Kaggle: run cells 0-5                                  ~25-35 min, once per session
3. Kaggle: run cell 6 (smoke test) and LOOK at the images  ~3 min
4. Browser: open the tunnel URL, log into code-server
5. Roo Code: "read AGENT.md, style.md and shots.md, then start on sequence A"
6. Agent works. You watch, correct, and answer questions.
7. Kaggle: run cell 8 to commit + push, or let the agent do it
8. Laptop: git pull, run Remotion
```

Steps 5–7 loop. Steps 1–4 are once per Kaggle session.

## Timing

| thing | time |
|---|---|
| cold start (once per session) | 25–35 min |
| one clean shot (generate + critique) | 40–75 s |
| one shot needing a retry | 90–140 s |
| 40-shot video, mostly clean | 1–1.5 h |
| generation alone at 1024×576 / 8 steps | 20–60 s |
| same, with a reference attached | +20–40% |

The critique adds 5–15 s per attempt. It is worth it: without it the agent is
generating blind and you find out at compile time.

## Who does what

**You:**
- write `style.md` and `shots.md`
- put reference images in `refs/`
- run the notebook cells
- look at the smoke test — this is the one place your eyes are irreplaceable
- spot-check frames the agent passed, especially early on
- decide what moves to `shots/final/`

**The agent:**
- builds prompts from shot intent + style
- generates, critiques, retries up to 3 times
- logs seeds and notes back into `shots.md`
- commits and pushes

**Neither:** nobody is watching for a session timeout. 12 h max, and it can die
earlier. Push often.

## Reference images — the rules that matter

These come from Wan2GP's actual behaviour, not from general diffusion advice.

1. **Hard limit of 2 references.** One when inpainting. Enforced upstream in
   `validate_generative_settings`; passing three is an error, not a truncation.

2. **`video_prompt_type` must contain `"I"`** or references are dropped with no
   warning at all. `krea_server.py` adds the letter for you, but if you ever call
   Wan2GP directly, this is the trap.

3. **`"KI"` vs `"I"`:**
   - `"KI"` — the first reference is the **scene/base**, stretched to the canvas.
     Use for establishing the look, palette, environment.
   - `"I"` — every reference is a **subject/object**, aspect preserved. Use for a
     character or prop that must stay recognisably itself.

4. **References are consumed twice.** A ≤768px copy goes to the Qwen3-VL vision
   tower for semantics; the full-resolution version goes through the VAE as
   latent tokens appended to the sequence. Both paths matter, which is why a
   reference that is sharp and well-composed beats one that merely has the right
   colours.

5. **One strong reference beats two mediocre ones.** Two references split the
   conditioning and usually produce a muddier result than either alone.

6. **References carry palette, texture, and rendering. Composition follows the
   prompt.** If you want different framing, write it in words — do not expect the
   reference to control it.

7. **The same reference across every shot is what makes 40 frames look like one
   video.** This is the single highest-leverage habit in the whole workflow.
   Change references only when the scene genuinely changes.

8. **Expect 20–40% slower with a reference.** Attention is quadratic in sequence
   length and references add latent tokens. Not a bug.

## Resolution policy

- Iterate at **1024×576**. Both dimensions divisible by 16, always.
- 1024×1024 costs roughly **2×**, not 1.8× — quadratic attention again.
- Upscale only the keepers, at the end, as a separate pass.

## Retry policy

Three attempts per shot, hard cap. The agent has no vision; if it does not
understand a critique it will chase it indefinitely. Three attempts, keep the
best, flag it, move on. A human glance resolves in five seconds what twenty
generations will not.

## Committing

`.gitignore` excludes `shots/*.png` and allows `shots/final/*.png`. Rejects stay
local; keepers get promoted into `shots/final/` and committed.

This is deliberate. Git stores every version of every binary forever — 200
rejects at 1–3 MB permanently adds a few hundred MB to every future clone. If you
want everything tracked instead, flip the `.gitignore` rules and set up git-lfs
first; then update `AGENT.md` so the agent's instructions match.
