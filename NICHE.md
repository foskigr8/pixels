# Niche & Script Blueprint

## The niche

Same question-format explainer as Whymentary — "Why ___" — same script DNA, but the
subject is AI instead of biology/physics/economics. Three content modes, same rules
across all three:

- **Evergreen concept explainers** — "Why AI Can't Actually Think", "Why AI Can't
  Replace Everyone", "Why Everyone Is Suddenly Building AI Agents".
- **Absurdist / hyperbolic takes** — "Why AI Will Kill Everyone" and similar. Same
  deadpan-authoritative-voiceover-vs-slapstick-visual contrast the source channel uses
  for its wealth/mosquito videos, aimed at AI doom tropes instead.
- **News-reactive** — pegged to a real, current AI story instead of a general everyday
  observation. Same blueprint below; only the hook changes, from a relatable observation
  to a specific event. Fact-check the underlying claim before scripting it — that step
  is outside this pipeline.

This is a sub-niche, not a copy — see [WHYMENTARY_COMPLETE_MASTER_DISSECTION.md](WHYMENTARY_COMPLETE_MASTER_DISSECTION.md)
for the full source analysis this is extracted from.

## What carries over from Whymentary, unchanged

Pulled straight from the dissection's Section 1 and Section 6:

1. **The 5-Second Inverted Hook.** Open with a relatable, everyday observation,
   immediately inverted by a paradox. Source example: "It's hot → your fan is killing
   you." AI equivalent: "You just asked it to write your email → it doesn't know what
   an email *is*."
2. **3-Tier Script-to-Visual Translation Hierarchy.**
   - Tier 1 (Literal): everyday relatable premise — a person at a laptop, a phone
     notification.
   - Tier 2 (Metaphor): the abstract mechanism, materialized as a physical contraption
     — gears, conveyor belts, machinery.
   - Tier 3 (Abstract): the underlying process laid bare — particles, flows,
     cross-sections.
3. **Mechanical Analogy Rule.** Never explain the concept in dry text or a diagram.
   Materialize it as a physical contraption, the same way thermoregulation became a
   combustion engine in the torso. See the starter kit below for AI-specific ones.
4. **Deadpan vs. slapstick tone.** Calm, authoritative narration; comedic,
   exaggerated character reactions underneath it.
5. **One visual beat every ~1.9–2.5s.** Elements stack onto the canvas clause-by-clause
   as the voiceover names them, not all at once. For this pipeline (stills, not
   animation) that means: **one generated shot per script clause**, not one shot per
   scene. A 10-minute video runs ~250-300 beats/shots at that rate — plan the shot list
   accordingly, it's a lot of individual generations, not a handful of hero images.
6. **Color as a fixed semantic anchor** — see below for how the existing palette maps
   onto AI subject matter specifically.

## Visual style: no change needed

The flat-vector house style in [AGENTS.md](AGENTS.md) (thin outline, flat bucket-fill,
coherent human figures, the MS-Paint-style prompt clause, character-consistency via
repeated text description) is domain-agnostic — it applies to AI content exactly as
tested, no rework. This doc is about *script and shot-planning* structure, not
rendering.

## Extending the color-semantic map for AI subject matter

The original palette was built for biology/physics content. Some roles carry over
directly; others need reinterpreting since there's no literal biology in this niche.
Treat this as a first pass to confirm once we actually shoot a script, not a locked
spec:

| Color | Old role | This niche |
|---|---|---|
| Red | Heat, danger, failure | Unchanged — AI screwing up, a doom beat, hype-driven panic |
| Green | Biology, nutrients, money | Unchanged for money/success (funding, a working demo); drop the biology sense |
| Blue | Water, cool, baseline | Reassigned — logic, code, the inside of the machine, screens |
| Orange | Fire, mid-heat | Unchanged sense, new target — hype/virality, an overheating GPU |
| Pink | Flesh, organs | Mostly unused here — reserve for the rare literal human-brain-vs-AI comparison shot |

## A starter kit of AI mechanical analogies

The equivalent of "thermoregulation → combustion engine." Starting hypotheses, meant to
be tested against Krea 2 and replaced if they don't render or read clearly — same
process as the visual-style test that picked the current house look.

- **Next-token prediction / "AI doesn't think"** → a crank-operated fortune-teller
  machine, spitting the statistically likely next word out of a slot-machine reel — no
  brain inside, just gears matching patterns.
- **Hallucination** → a broken photocopier confidently printing garbage as if it were
  the original.
- **AI agents multiplying** → an assembly line stamping out identical small robots.
- **Training on data** → a giant funnel pouring the entire internet into a grinder.
- **"AI takes your job" fear** → a conveyor belt swapping a human figure out for a
  robot arm mid-frame.
- **Doomer / "AI kills everyone" beat** → an oversized red lever in a control room, or
  a paperclip factory running away unattended (the actual AI-safety thought experiment
  this trope comes from).

## Shot-list workflow

1. Write the script in the Whymentary voice — inverted hook in the first 5 seconds,
   warm-authoritative narration throughout.
2. Break it into clauses at noun/verb boundaries, additive-stacking style. Each clause
   = one shot.
3. Tag each clause's Tier (1/2/3) and write its prompt: the house style clause from
   AGENTS.md + the fixed character description + the clause's specific content, colored
   per the map above.
4. Generate via the Studio UI or API into a project named for the video.
5. Curate into `shots/final/` once the shot list is locked.

The pipeline generates stills only — pop-in timing, easing, and Foley are a downstream
editing step this repo doesn't do yet.
