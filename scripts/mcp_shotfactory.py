#!/usr/bin/env python3
"""
mcp_shotfactory.py -- the agent's hands and eyes.

Roo Code runs DeepSeek, and DeepSeek has no vision. Rather than bolt a second
model into the editor and route between them, vision is exposed as a *tool*:
the agent calls critique_image() and gets prose back. It never has to see
anything. That is the central design move in this project -- if you are
tempted to replace it with multi-model routing, read CLAUDE.md first.

Tools
    krea_health()      -- is the model up, and what did it actually load
    generate_image()   -- one image via the warm server on localhost:8711
    critique_image()   -- Gemini Flash looks at a PNG and returns PASS/FAIL + fixes
    generation_stats() -- timing history; measure before optimising

Environment
    KREA_URL          default http://127.0.0.1:8711
    GEMINI_API_KEY    required for critique_image only
    GEMINI_MODEL      default gemini-flash-latest
    REPO_DIR          default /kaggle/working/repo
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

KREA_URL = os.environ.get("KREA_URL", "http://127.0.0.1:8711").rstrip("/")
REPO_DIR = Path(os.environ.get("REPO_DIR", "/kaggle/working/repo"))
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# A generation at 1024x576/8 steps runs 20-60s; with a reference, 20-40% longer
# because attention is quadratic in sequence length and refs add latent tokens.
GEN_TIMEOUT = float(os.environ.get("KREA_TIMEOUT", "600"))

mcp = FastMCP("shotfactory")


def _resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_DIR / rel


# ---------------------------------------------------------------------------


@mcp.tool()
def krea_health() -> str:
    """Check whether the Krea 2 image server is loaded and ready.

    Call this once at the start of a session. If status is not "ready", wait --
    the model takes 3-5 minutes to load. Do not start generating until it is.

    The response also reports whether the fp16 source patches applied. If it
    shows missing patches, tell the user before generating anything; a bad
    patch state is the single most common cause of failures here.
    """
    try:
        r = httpx.get(f"{KREA_URL}/health", timeout=10)
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return (
            f"ERROR: cannot reach the Krea server at {KREA_URL} ({e}).\n"
            "It is probably still loading or it crashed on boot. "
            "Ask the user to check /kaggle/working/krea.log."
        )


@mcp.tool()
def generate_image(
    prompt: str,
    out_name: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 576,
    steps: int = 8,
    guidance: float = 1.0,
    seed: int = -1,
    ref_paths: list[str] | None = None,
    ref_mode: str = "KI",
) -> str:
    """Generate one image with Krea 2 Turbo Edit and save it into the repo.

    Args:
        prompt: Full scene description. Krea 2 rewards long, concrete prompts --
            subject, action, camera/lens, lighting, palette, mood. Style words
            should come from style.md rather than being invented.
        out_name: Bare filename, e.g. "shot_003.png". Saved into shots/.
            Reuse the same name when retrying a shot so rejects do not pile up.
        negative_prompt: Usually leave empty; Turbo runs at low guidance where
            negatives do very little.
        width, height: Both MUST be divisible by 16 or the pipeline raises.
            Iterate at 1024x576. 1024x1024 costs roughly 2x, not 1.8x.
        steps: 8 is the Turbo default. Above ~12 you are mostly buying time.
        guidance: ~1.0 for Turbo. Raising it tends to scorch contrast.
        seed: -1 for random. Pass a previous seed to vary a prompt while
            holding composition roughly steady.
        ref_paths: Up to 2 reference images, repo-relative (e.g. "refs/hero.png").
            The hard cap is enforced upstream. One strong reference beats two
            mediocre ones. Using the SAME reference across every shot is what
            makes a set of shots look like one video.
        ref_mode: "KI" -- the first reference is the scene/base and is stretched
            to the canvas (use for establishing look, palette, environment).
            "I" -- every reference is a subject/object and aspect is preserved
            (use for a character or prop that must stay itself).
            The "I" letter is what actually switches references on; without it
            Wan2GP drops them silently, with no warning and no error.

    Returns JSON with the saved path, seed, and generate_s. Feed the path
    straight into critique_image().
    """
    payload = {
        "prompt": prompt,
        "out_name": out_name,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance,
        "seed": seed,
        "ref_paths": ref_paths or [],
        "ref_mode": ref_mode,
    }
    try:
        r = httpx.post(f"{KREA_URL}/generate", json=payload, timeout=GEN_TIMEOUT)
    except httpx.TimeoutException:
        return (
            f"ERROR: generation exceeded {GEN_TIMEOUT}s. The server may still be "
            "working. Do NOT immediately retry -- check krea_health() first, then "
            "look for the file on disk."
        )
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

    if r.status_code != 200:
        detail = r.json().get("detail", r.text) if r.text else r.status_code
        return f"ERROR ({r.status_code}): {detail}"

    data = r.json()
    rel = os.path.relpath(data["path"], REPO_DIR)
    return json.dumps({**data, "repo_path": rel}, indent=2)


@mcp.tool()
def critique_image(image_path: str, criteria: str) -> str:
    """Look at a generated image and judge it. This is your only way of seeing.

    You have no vision. This tool sends the image to a vision model and returns
    a written verdict, so treat the returned prose as ground truth about what
    is actually in the picture -- not as a suggestion.

    Args:
        image_path: Repo-relative path, e.g. "shots/shot_003.png".
        criteria: What this shot is supposed to be. Include (a) the relevant
            excerpt from style.md and (b) the shot's intent from shots.md.
            Vague criteria produce vague critiques and burn retries.

    Returns a verdict block: VERDICT (PASS/FAIL), WHAT I SEE, ISSUES, and
    REVISED PROMPT. On FAIL, regenerate using the revised prompt. Cap at 3
    attempts per shot, then keep the best one and move on -- an agent that
    cannot see will otherwise chase a critique it does not understand forever.
    """
    if not GEMINI_KEY:
        return "ERROR: GEMINI_API_KEY is not set in the MCP server environment."

    p = _resolve(image_path)
    if not p.exists():
        return f"ERROR: no such image: {image_path} (looked at {p})"

    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode()

    instruction = f"""You are a critical art director reviewing one frame from a video.

The frame must satisfy these criteria:
---
{criteria}
---

Answer in exactly this format:

VERDICT: PASS or FAIL
WHAT I SEE: 2-4 sentences describing the image literally -- subject, composition,
lighting, palette, and any visible artifacts. The person reading this cannot see
the image, so be concrete.
ISSUES: bullet list of concrete deviations from the criteria. Write "none" if it
passes. Do not invent problems to seem thorough.
REVISED PROMPT: if FAIL, a complete rewritten image prompt that fixes the issues
while keeping everything that already worked. If PASS, write "n/a".

Be strict about anatomy, text rendering, and consistency with the stated style.
Be forgiving about micro-detail that will not read at video playback speed."""

    body = {
        "contents": [
            {
                "parts": [
                    {"text": instruction},
                    {"inline_data": {"mime_type": mime, "data": b64}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }

    try:
        r = httpx.post(
            f"{GEMINI_URL}/{GEMINI_MODEL}:generateContent",
            headers={"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
    except Exception as e:
        return f"ERROR contacting Gemini: {type(e).__name__}: {e}"

    if r.status_code == 429:
        return (
            "ERROR: Gemini rate limit hit (free tier is 15 req/min, 1500/day). "
            "Wait a minute before critiquing again. Do not retry in a tight loop."
        )
    if r.status_code != 200:
        return f"ERROR from Gemini ({r.status_code}): {r.text[:500]}"

    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return f"ERROR: unexpected Gemini response shape: {r.text[:500]}"


@mcp.tool()
def generation_stats() -> str:
    """Timing history for generations so far (min/median/max seconds).

    Useful when the user asks whether things are slow. The compute floor on one
    T4 at 1024x576/8 steps is roughly 18-22s, so a median near that means there
    is nothing left to win without changing resolution or step count.
    """
    try:
        r = httpx.get(f"{KREA_URL}/stats", timeout=10)
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


if __name__ == "__main__":
    mcp.run()
