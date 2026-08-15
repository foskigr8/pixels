# shots.md — the shot list

One row per frame. The agent reads `intent` to build the prompt and to write the
critique criteria, so intent should say *what the shot must show*, not how to
prompt it.

Fill in `seed`, `attempts`, and `notes` as you go — that log is what makes the
next session cheaper. Keep `ref` the same across a sequence; that consistency is
what makes separate frames read as one video.

Replace the examples below with your own.

---

## Sequence A — the street

| # | file | intent | ref | mode | seed | attempts | notes |
|---|---|---|---|---|---|---|---|
| 001 | shot_001.png | Wide establishing: empty wet street, sodium lights receding into fog. No people. | refs/street_base.png | KI | | | |
| 002 | shot_002.png | Medium: lone figure from behind, centre-left, walking away. Face not visible. | refs/street_base.png | KI | | | |
| 003 | shot_003.png | Close: rain hitting a puddle, neon reflection breaking up. No subject. | refs/street_base.png | KI | | | |

## Sequence B — interior

| # | file | intent | ref | mode | seed | attempts | notes |
|---|---|---|---|---|---|---|---|
| 010 | shot_010.png | Wide: cramped apartment, single desk lamp as the only key light. | refs/room_base.png | KI | | | |
| 011 | shot_011.png | Insert: hands on a desk, papers, lamp glare across the surface. | refs/room_base.png | KI | | | |

---

## Status

- **Not started:** everything
- **Blocked:** —
- **Needs a human look:** —

## Running notes

Anything learned that should survive the session — prompt phrasings that worked,
seeds worth reusing, things the critique kept flagging.
