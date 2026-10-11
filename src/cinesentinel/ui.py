"""Shared look-and-feel for the Streamlit dashboard (theme CSS, cards, charts).

Visual direction: cinematic, Spider-Man-inspired. Deep black, crimson,
charcoal and silver, with the user's own background images used in the
hero banners and the sidebar only (never behind charts or dense text).

Motion is CSS-first and limited to transform / opacity so it stays on the
compositor. One small script adds pointer parallax and card tilt. All of it
is switched off under `prefers-reduced-motion`.
"""

import base64
import io
import math
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from cinesentinel.config import APP_NAME, APP_TAGLINE, MODEL_COLORS, SENTIMENT_COLORS

CRIMSON = "#E11D2E"
CRIMSON_DEEP = "#8B0F1A"
SILVER = "#C9CED6"
EMBER = "#F59E0B"
STEEL = "#7C8798"

ACCENT = CRIMSON
ACCENT_2 = SILVER
TEXT = "#ECEDEF"
MUTED = "#9BA1AC"
GRID = "rgba(201,206,214,0.12)"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The user's images, used as-is from disk. They are looked for in
# app/background-img/ first and then background-img/ at the project root.
IMAGE_DIRS = (PROJECT_ROOT / "app" / "background-img", PROJECT_ROOT / "background-img")

# name -> (file name, crop box). The files are screenshots, so the crop only
# trims the screenshot's own edges (rounded corners, viewer labels, black
# bars). Cropping happens in memory; the files on disk are never changed.
IMAGES = {
    "emblem": ("Screenshot 2026-10-11 141722.png", (10, 10, 726, 372)),
    "descent": ("Screenshot 2026-10-11 141822.png", (0, 0, 176, 416)),
    "mask": ("Screenshot 2026-10-11 142313.png", (0, 0, 500, 255)),
}


def image_path(name: str) -> Path | None:
    """Path of one of the supplied background images, or None if absent."""
    file_name = IMAGES[name][0]
    for directory in IMAGE_DIRS:
        candidate = directory / file_name
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=None)
def image_data_uri(name: str) -> str:
    """The image as a compact JPEG data URI ("" if the file is missing).

    Streamlit cannot serve arbitrary project files by URL, so the image is
    read from disk and inlined. Re-encoding the PNG screenshots as JPEG in
    memory cuts the page weight roughly tenfold.
    """
    path = image_path(name)
    if path is None:
        return ""
    try:
        from PIL import Image

        with Image.open(path) as source:
            image = source.convert("RGB")
            box = IMAGES[name][1]
            if box[2] <= image.width and box[3] <= image.height:
                image = image.crop(box)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=86, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001 - a broken image must never break the app
        return ""


def _web_svg_uri() -> str:
    """A spider-web corner (radial threads plus sagging rings) as an SVG data URI."""
    size, spokes, rings = 600, 9, 7
    angles = [math.radians(90 + 90 * i / (spokes - 1)) for i in range(spokes)]  # fan from the top-right corner
    origin = (size, 0)

    def point(angle: float, radius: float) -> tuple[float, float]:
        return origin[0] + radius * math.cos(angle), origin[1] + radius * math.sin(angle)

    parts = []
    for angle in angles:
        x, y = point(angle, size * 1.5)
        parts.append(f"M{origin[0]} {origin[1]}L{x:.0f} {y:.0f}")
    for ring in range(1, rings + 1):
        radius = 70 * ring + 6 * ring * ring
        for a, b in zip(angles, angles[1:]):
            x1, y1 = point(a, radius)
            x2, y2 = point(b, radius)
            # Control point pulled toward the corner so each strand sags.
            cx, cy = point((a + b) / 2, radius * 0.86)
            parts.append(f"M{x1:.0f} {y1:.0f}Q{cx:.0f} {cy:.0f} {x2:.0f} {y2:.0f}")
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {size} {size}'>"
        f"<path d='{''.join(parts)}' fill='none' stroke='rgb(214,219,228)' stroke-width='1' "
        "stroke-linecap='round'/></svg>"
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --cs-black: #060607; --cs-char: #121215; --cs-char-2: #1A1A1F;
    --cs-crimson: #E11D2E; --cs-crimson-deep: #8B0F1A; --cs-silver: #C9CED6;
    --cs-line: rgba(201,206,214,0.13);
}
html, body, [class*="css"], .stMarkdown, .stButton button {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
.stApp { background: var(--cs-black); }
[data-testid="stHeader"] { background: transparent; }
.block-container {
    padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px;
    position: relative; z-index: 0; isolation: isolate;
}
/* Wrappers of purely decorative elements take no layout space. */
[data-testid="stElementContainer"]:has(.cs-fx), .element-container:has(.cs-fx),
[data-testid="stElementContainer"]:has(iframe[height="0"]), .element-container:has(iframe[height="0"]) {
    position: absolute; width: 0; height: 0; margin: 0; overflow: visible;
}

/* ------------------------------------------------------------ ambient layer */
.cs-ambient { position: fixed; inset: 0; z-index: -1; pointer-events: none; overflow: hidden; }
.cs-ambient::before {
    content: ""; position: absolute; inset: 0;
    background:
        radial-gradient(900px 520px at 88% -8%, rgba(225,29,46,0.20), transparent 62%),
        radial-gradient(760px 520px at 4% 108%, rgba(139,15,26,0.22), transparent 60%),
        radial-gradient(1200px 700px at 50% 40%, rgba(26,26,31,0.9), transparent 75%);
}
.cs-plx {
    position: absolute; inset: 0;
    transform: translate3d(calc(var(--px, 0) * var(--depth, 10) * -1px), calc(var(--py, 0) * var(--depth, 10) * -1px), 0);
    transition: transform .35s cubic-bezier(.2,.7,.2,1);
}
.cs-web {
    position: absolute; width: min(40vh, 380px); aspect-ratio: 1; opacity: var(--web-o, .10);
    background: __WEB__ center / contain no-repeat;
    -webkit-mask-image: radial-gradient(circle at 100% 0%, #000 35%, transparent 78%);
    mask-image: radial-gradient(circle at 100% 0%, #000 35%, transparent 78%);
    animation: cs-web-draw 2.2s cubic-bezier(.3,.7,.2,1) both, cs-web-breathe 9s ease-in-out 2.2s infinite;
}
.cs-web-tr { top: 0; right: 0; }
.cs-web-bl { bottom: 0; left: 0; transform: rotate(180deg); --web-o: .06; animation-delay: .5s, 2.7s; }
@keyframes cs-web-draw {
    from { clip-path: circle(0% at 100% 0%); }
    to { clip-path: circle(150% at 100% 0%); }
}
@keyframes cs-web-breathe { 0%, 100% { opacity: var(--web-o, .10); } 50% { opacity: calc(var(--web-o, .10) * 1.7); } }

/* A thread is a hairline from the top-right corner; a glint travels down it. */
.cs-thread {
    position: absolute; top: 0; right: 0; width: 120vh; height: 1px;
    transform-origin: 100% 50%; transform: rotate(var(--a));
    background: linear-gradient(270deg, rgba(214,219,228,0.14), rgba(214,219,228,0) 42%);
}
.cs-thread > i {
    position: absolute; top: -1px; right: 0; width: 90px; height: 3px; border-radius: 3px;
    background: linear-gradient(270deg, transparent, rgba(255,90,100,0.95), transparent);
    opacity: 0; animation: cs-glint var(--d, 7s) linear var(--delay, 0s) infinite;
}
@keyframes cs-glint {
    0% { transform: translateX(0); opacity: 0; }
    8% { opacity: .9; }
    45% { opacity: .5; }
    60%, 100% { transform: translateX(-46vh); opacity: 0; }
}

.cs-embers { position: absolute; inset: 0; }
.cs-embers > span {
    position: absolute; bottom: -12px; left: var(--x); width: var(--s); height: var(--s); border-radius: 50%;
    background: radial-gradient(circle, #FFB4A8 0%, #E11D2E 45%, rgba(225,29,46,0) 72%);
    opacity: 0; animation: cs-ember var(--d) linear var(--delay) infinite;
}
@keyframes cs-ember {
    0% { transform: translate3d(0, 0, 0) scale(1); opacity: 0; }
    10% { opacity: var(--o, .7); }
    80% { opacity: calc(var(--o, .7) * .5); }
    100% { transform: translate3d(var(--drift), -104vh, 0) scale(.4); opacity: 0; }
}

/* ------------------------------------------------------------------ sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111114 0%, #09090B 100%);
    border-right: 1px solid rgba(225,29,46,0.22);
}
[data-testid="stSidebar"]::after {
    content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: min(46%, 400px); pointer-events: none;
    background: __SIDEBAR_IMG__ center bottom / auto 100% no-repeat; opacity: .5;
    -webkit-mask-image: radial-gradient(ellipse 46% 62% at 50% 62%, #000 30%, transparent 78%);
    mask-image: radial-gradient(ellipse 46% 62% at 50% 62%, #000 30%, transparent 78%);
}
[data-testid="stSidebar"] > div { position: relative; z-index: 1; }
[data-testid="stSidebarNav"] a { border-radius: 10px; transition: background .2s ease, transform .2s ease; }
[data-testid="stSidebarNav"] a:hover { background: rgba(225,29,46,0.14); transform: translateX(3px); }
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: linear-gradient(90deg, rgba(225,29,46,0.30), rgba(225,29,46,0.05));
    box-shadow: inset 3px 0 0 var(--cs-crimson);
}

/* --------------------------------------------------------------------- hero */
.cs-hero {
    position: relative; overflow: hidden; isolation: isolate;
    padding: 2.2rem 2.4rem; border-radius: 22px; margin-bottom: 1.4rem; min-height: 190px;
    display: flex; align-items: center;
    background: linear-gradient(135deg, #17171B 0%, #0B0B0D 100%);
    border: 1px solid rgba(225,29,46,0.28);
    box-shadow: 0 30px 70px -38px rgba(225,29,46,0.75), inset 0 1px 0 rgba(255,255,255,0.05);
    animation: cs-rise .7s cubic-bezier(.2,.7,.2,1) both;
}
.cs-hero-xl { min-height: 380px; padding: 3rem 3rem; }
.cs-hero-layer { position: absolute; inset: -18px; z-index: -3; }
.cs-hero-img {
    position: absolute; inset: 0 0 0 40%; background-size: var(--size, cover); background-position: var(--pos, 50% 42%);
    background-repeat: no-repeat;
    opacity: var(--img-o, .5); animation: cs-kenburns 34s ease-in-out infinite alternate;
    -webkit-mask-image: linear-gradient(90deg, transparent 0%, #000 34%);
    mask-image: linear-gradient(90deg, transparent 0%, #000 34%);
}
.cs-hero-img.cs-alt { animation: cs-kenburns 34s ease-in-out infinite alternate-reverse, cs-crossfade 26s ease-in-out infinite; }
.cs-hero-img.cs-strip {
    left: auto; right: 9%; width: 190px; background-size: auto 100% !important; background-position: center !important;
    -webkit-mask-image: linear-gradient(90deg, transparent, #000 28%, #000 72%, transparent);
    mask-image: linear-gradient(90deg, transparent, #000 28%, #000 72%, transparent);
    animation: none;
}
@keyframes cs-kenburns {
    from { transform: scale(1.04) translate3d(0, 0, 0); }
    to { transform: scale(1.14) translate3d(-1.5%, -1.2%, 0); }
}
@keyframes cs-crossfade { 0%, 38% { opacity: 0; } 50%, 88% { opacity: var(--img-o, .5); } 100% { opacity: 0; } }
/* Shade: near-solid behind the text column, clearing toward the right. */
.cs-hero-shade {
    position: absolute; inset: 0; z-index: -2;
    background:
        linear-gradient(90deg, rgba(6,6,7,0.97) 0%, rgba(6,6,7,0.92) 46%, rgba(6,6,7,0.40) 72%, rgba(6,6,7,0.12) 100%),
        linear-gradient(0deg, rgba(6,6,7,0.80) 0%, rgba(6,6,7,0) 38%);
}
.cs-hero-web {
    position: absolute; top: 0; right: 0; height: 105%; aspect-ratio: 1; z-index: -1; opacity: .22;
    background: __WEB__ center / contain no-repeat;
    -webkit-mask-image: radial-gradient(circle at 100% 0%, #000 30%, transparent 72%);
    mask-image: radial-gradient(circle at 100% 0%, #000 30%, transparent 72%);
    animation: cs-web-draw 1.8s cubic-bezier(.3,.7,.2,1) .25s both;
}
.cs-hero-sweep {
    position: absolute; top: -20%; bottom: -20%; left: 0; width: 34%; z-index: -1;
    background: linear-gradient(100deg, transparent 0%, rgba(225,29,46,0.0) 20%, rgba(255,70,84,0.30) 50%, rgba(225,29,46,0.0) 80%, transparent 100%);
    mix-blend-mode: screen; transform: translateX(-140%) skewX(-16deg);
    animation: cs-sweep 8s cubic-bezier(.4,0,.2,1) 1.1s infinite;
}
@keyframes cs-sweep {
    0% { transform: translateX(-140%) skewX(-16deg); }
    38%, 100% { transform: translateX(440%) skewX(-16deg); }
}
.cs-hero-content { position: relative; max-width: min(640px, 58%); }
.cs-eyebrow {
    font-size: 0.74rem; letter-spacing: 0.24em; text-transform: uppercase;
    color: #FF6B76; font-weight: 700; margin-bottom: 0.6rem;
    animation: cs-fade-up .6s ease-out .15s both;
}
.cs-eyebrow::before {
    content: ""; display: inline-block; width: 26px; height: 2px; margin-right: .6rem; vertical-align: middle;
    background: var(--cs-crimson); transform-origin: left; animation: cs-grow-x .6s ease-out .3s both;
}
.cs-hero h1 {
    font-size: 2.6rem; font-weight: 800; line-height: 1.08; margin: 0 0 0.6rem 0; padding: 0;
    color: #F5F6F8; letter-spacing: -0.01em;
    animation: cs-title-in .9s cubic-bezier(.2,.7,.2,1) .2s both, cs-glitch .55s steps(1, end) 1.15s 1;
}
.cs-hero-xl h1 { font-size: 3.5rem; }
@keyframes cs-title-in {
    from { opacity: 0; transform: translate3d(0, 26px, 0); letter-spacing: 0.08em; clip-path: inset(0 0 100% 0); }
    to { opacity: 1; transform: none; letter-spacing: -0.01em; clip-path: inset(-10% -4% -18% -4%); }
}
@keyframes cs-glitch {
    0% { transform: translate3d(0, 0, 0); text-shadow: none; }
    14% { transform: translate3d(-3px, 1px, 0) skewX(-4deg); text-shadow: 4px 0 #E11D2E, -4px 0 rgba(201,206,214,.7); }
    28% { transform: translate3d(3px, -1px, 0) skewX(3deg); text-shadow: -5px 0 #E11D2E, 3px 0 rgba(201,206,214,.6); }
    42% { transform: translate3d(-1px, 0, 0); text-shadow: 2px 0 #E11D2E; clip-path: inset(18% 0 34% 0); }
    56% { transform: translate3d(2px, 0, 0); text-shadow: -3px 0 #E11D2E; clip-path: inset(52% 0 8% 0); }
    70% { transform: translate3d(-1px, 0, 0); text-shadow: 1px 0 #E11D2E; clip-path: inset(-10% -4% -18% -4%); }
    100% { transform: none; text-shadow: none; }
}
.cs-hero p {
    color: #D2D6DD; font-size: 1.05rem; margin: 0; max-width: 720px; line-height: 1.6;
    animation: cs-fade-up .7s ease-out .55s both;
}
.cs-badges { margin-top: 1.1rem; animation: cs-fade-up .7s ease-out .75s both; }
.cs-badge {
    display: inline-block; padding: 0.3rem 0.8rem; margin: 0 0.4rem 0.4rem 0;
    border-radius: 999px; font-size: 0.78rem; font-weight: 500; color: #ECEDEF;
    background: rgba(10,10,12,0.66); border: 1px solid rgba(201,206,214,0.24);
    transition: border-color .2s ease, background .2s ease;
}
.cs-badge:hover { border-color: rgba(225,29,46,0.8); background: rgba(225,29,46,0.16); }

@keyframes cs-rise { from { opacity: 0; transform: translate3d(0, 14px, 0); } to { opacity: 1; transform: none; } }
@keyframes cs-fade-up { from { opacity: 0; transform: translate3d(0, 12px, 0); } to { opacity: 1; transform: none; } }
@keyframes cs-grow-x { from { transform: scaleX(0); } to { transform: scaleX(1); } }

/* -------------------------------------------------------------------- cards */
.cs-grid { display: grid; gap: 0.9rem; margin: 0.2rem 0 1.2rem 0; perspective: 1100px; }
.cs-card {
    position: relative; overflow: hidden;
    padding: 1.05rem 1.2rem; border-radius: 16px;
    background: linear-gradient(180deg, rgba(30,30,35,0.92), rgba(15,15,18,0.92));
    border: 1px solid var(--cs-line);
    transform: perspective(900px) rotateX(calc(var(--ry, 0) * -7deg)) rotateY(calc(var(--rx, 0) * 7deg)) translate3d(0, var(--lift, 0px), 0);
    transition: transform .22s cubic-bezier(.2,.7,.2,1), border-color .22s ease, box-shadow .22s ease;
    animation: cs-card-in .55s ease-out backwards;
}
@keyframes cs-card-in { from { opacity: 0; transform: translate3d(0, 14px, 0); } }
.cs-grid > .cs-card:nth-child(2) { animation-delay: .06s; }
.cs-grid > .cs-card:nth-child(3) { animation-delay: .12s; }
.cs-grid > .cs-card:nth-child(4) { animation-delay: .18s; }
.cs-grid > .cs-card:nth-child(5) { animation-delay: .24s; }
.cs-grid > .cs-card:nth-child(6) { animation-delay: .30s; }
/* Red highlight that follows the pointer. */
.cs-card::before {
    content: ""; position: absolute; inset: 0; pointer-events: none; opacity: 0; transition: opacity .25s ease;
    background: radial-gradient(260px circle at var(--mx, 50%) var(--my, 0%), rgba(225,29,46,0.22), transparent 70%);
}
.cs-card:hover {
    --lift: -5px; border-color: rgba(225,29,46,0.6);
    box-shadow: 0 22px 44px -24px rgba(225,29,46,0.7), 0 10px 22px -16px rgba(0,0,0,0.9);
}
.cs-card:hover::before { opacity: 1; }
.cs-card > * { position: relative; }
.cs-card .cs-label {
    font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: #9BA1AC; font-weight: 600;
}
.cs-card .cs-value { font-size: 1.75rem; font-weight: 700; color: #F8F9FA; line-height: 1.25; margin-top: 0.2rem; }
.cs-card .cs-sub { font-size: 0.82rem; color: #9BA1AC; margin-top: 0.15rem; }
.cs-card .cs-body { font-size: 0.88rem; color: #BEC3CC; margin-top: 0.45rem; line-height: 1.5; }
.cs-card .cs-title { font-size: 1.02rem; font-weight: 650; color: #F1F2F4; }
.cs-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 0.5rem; }
.cs-bar { height: 6px; border-radius: 999px; background: rgba(201,206,214,0.16); margin-top: 0.7rem; overflow: hidden; }
.cs-bar > span { display: block; height: 100%; border-radius: 999px; transform-origin: left; animation: cs-grow-x 1s cubic-bezier(.2,.7,.2,1) .35s both; }

.cs-section { margin: 1.6rem 0 0.7rem 0; }
.cs-section h3 {
    font-size: 1.28rem; font-weight: 700; color: #F1F2F4; margin: 0; padding: 0 0 0 .75rem;
    border-left: 3px solid var(--cs-crimson);
}
.cs-section p { color: #9BA1AC; font-size: 0.9rem; margin: 0.25rem 0 0 0; padding-left: .95rem; }

/* ------------------------------------------------------------------ verdict */
.cs-verdict {
    position: relative; overflow: hidden;
    padding: 1.4rem 1.6rem; border-radius: 18px; margin: 0.4rem 0 1rem 0;
    border: 1px solid var(--cs-line);
    box-shadow: 0 0 0 0 var(--glow);
    animation: cs-verdict-in .7s cubic-bezier(.2,.8,.2,1) both, cs-verdict-pulse 1.6s ease-out .55s 1;
}
.cs-verdict::after {
    content: ""; position: absolute; top: 0; bottom: 0; left: 0; width: 40%; pointer-events: none;
    background: linear-gradient(100deg, transparent, rgba(255,255,255,0.14), transparent);
    transform: translateX(-130%) skewX(-16deg); animation: cs-verdict-sweep 1.1s ease-out .35s 1 both;
}
.cs-verdict .cs-big {
    font-size: 2.3rem; font-weight: 800; line-height: 1.15;
    animation: cs-verdict-word .6s cubic-bezier(.2,.9,.2,1.2) .25s both;
}
.cs-verdict .cs-detail { color: #D2D6DD; margin-top: 0.3rem; animation: cs-fade-up .5s ease-out .5s both; }
@keyframes cs-verdict-in {
    from { opacity: 0; transform: translate3d(0, 16px, 0) scale(.97); }
    to { opacity: 1; transform: none; }
}
@keyframes cs-verdict-word {
    from { opacity: 0; transform: translate3d(-14px, 0, 0) scale(1.12); letter-spacing: .12em; }
    to { opacity: 1; transform: none; letter-spacing: 0; }
}
@keyframes cs-verdict-pulse {
    0% { box-shadow: 0 0 0 0 var(--glow); }
    100% { box-shadow: 0 0 0 22px rgba(0,0,0,0); }
}
@keyframes cs-verdict-sweep { to { transform: translateX(330%) skewX(-16deg); } }
.cs-step {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 9px; margin-right: 0.55rem;
    background: linear-gradient(135deg, #E11D2E, #8B0F1A); color: #fff; font-weight: 800; font-size: 0.85rem;
}

/* ------------------------------------------------------------------ buttons */
.stButton > button, .stDownloadButton > button {
    position: relative; overflow: hidden; border-radius: 12px; font-weight: 600;
    transition: transform .18s ease, box-shadow .22s ease, border-color .22s ease, background .22s ease;
}
.stButton > button[kind="primary"], .stDownloadButton > button {
    background: linear-gradient(100deg, #B3121F 0%, #E11D2E 55%, #F0444F 100%); color: #fff; border: 0;
    padding: 0.55rem 1.3rem; box-shadow: 0 12px 28px -14px rgba(225,29,46,0.95);
}
/* Sheen that crosses the button on hover. */
.stButton > button::after, .stDownloadButton > button::after {
    content: ""; position: absolute; top: 0; bottom: 0; left: 0; width: 45%; pointer-events: none;
    background: linear-gradient(100deg, transparent, rgba(255,255,255,0.32), transparent);
    transform: translateX(-140%) skewX(-18deg); transition: transform .6s cubic-bezier(.2,.7,.2,1);
}
.stButton > button:hover::after, .stDownloadButton > button:hover::after { transform: translateX(320%) skewX(-18deg); }
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px); color: #fff; border-color: rgba(225,29,46,0.8);
    box-shadow: 0 18px 34px -14px rgba(225,29,46,1);
}
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0) scale(.98); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 2px solid #FF6B76; outline-offset: 2px;
}
.stButton > button:disabled { filter: grayscale(.7); opacity: .55; transform: none; box-shadow: none; }

[data-testid="stPageLink"] a {
    border: 1px solid rgba(225,29,46,0.5); border-radius: 10px; padding: 0.35rem 0.8rem;
    background: rgba(225,29,46,0.10); transition: background .2s ease, transform .2s ease, box-shadow .2s ease;
}
[data-testid="stPageLink"] a:hover {
    background: rgba(225,29,46,0.26); transform: translateX(3px); box-shadow: 0 10px 22px -14px rgba(225,29,46,0.9);
}

/* ------------------------------------------------- inputs, tables and charts */
.stTextArea textarea {
    background: rgba(18,18,21,0.9); border-radius: 14px;
    border: 1px solid rgba(201,206,214,0.2); color: #ECEDEF;
    transition: border-color .2s ease, box-shadow .2s ease;
}
.stTextArea textarea:focus { border-color: rgba(225,29,46,0.8); box-shadow: 0 0 0 3px rgba(225,29,46,0.18); }
[data-testid="stFileUploaderDropzone"] {
    background: rgba(18,18,21,0.9); border: 1.5px dashed rgba(225,29,46,0.6); border-radius: 16px;
    transition: border-color .2s ease, background .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(225,29,46,0.08); border-color: #E11D2E; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--cs-line); }
[data-testid="stExpander"] {
    border-radius: 14px; border: 1px solid var(--cs-line); background: rgba(18,18,21,0.88);
    transition: border-color .2s ease;
}
[data-testid="stExpander"]:hover { border-color: rgba(225,29,46,0.45); }
/* Charts sit on a solid charcoal panel so nothing shows through them, and
   wipe in from the left when they appear (including on a tab switch). */
[data-testid="stPlotlyChart"] {
    background: linear-gradient(180deg, rgba(22,22,26,0.96), rgba(12,12,14,0.96));
    border: 1px solid var(--cs-line); border-radius: 16px; overflow: hidden;
    animation: cs-chart-in .9s cubic-bezier(.2,.7,.2,1) both;
}
@keyframes cs-chart-in {
    from { opacity: 0; clip-path: inset(0 100% 0 0 round 16px); transform: translate3d(0, 10px, 0); }
    to { opacity: 1; clip-path: inset(0 0 0 0 round 16px); transform: none; }
}
[data-testid="stColumn"]:nth-of-type(2) [data-testid="stPlotlyChart"] { animation-delay: .12s; }
[data-testid="stColumn"]:nth-of-type(3) [data-testid="stPlotlyChart"] { animation-delay: .24s; }
[data-testid="stMetric"] { animation: cs-fade-up .5s ease-out both; }
[data-testid="stAlert"] { animation: cs-fade-up .45s ease-out both; }
.stTabs [data-baseweb="tab-list"] { gap: 0.35rem; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 0.5rem 1rem; font-weight: 600; transition: color .2s ease; }
.stTabs [data-baseweb="tab"]:hover { color: #FF6B76; }
.cs-pill {
    display: inline-block; padding: 0.12rem 0.6rem; border-radius: 999px; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.04em; border: 1px solid rgba(201,206,214,0.3); color: #D2D6DD;
}
.cs-note {
    padding: 0.85rem 1.1rem; border-radius: 12px; margin: 0.5rem 0 1rem 0;
    background: rgba(225,29,46,0.09); border: 1px solid rgba(225,29,46,0.32);
    color: #D2D6DD; font-size: 0.9rem; line-height: 1.55; animation: cs-fade-up .5s ease-out .2s both;
}
.cs-note b { color: #F5F6F8; }
.cs-foot { color: #6B7280; font-size: 0.78rem; text-align: center; margin-top: 2.2rem; }

/* ------------------------------------------------------------ accessibility */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.001ms !important; animation-delay: 0s !important;
        animation-iteration-count: 1 !important; transition-duration: 0.001ms !important;
    }
    .cs-embers, .cs-thread > i, .cs-hero-sweep, .cs-hero-img.cs-alt { display: none !important; }
    .cs-plx, .cs-card { transform: none !important; }
}
@media (max-width: 900px) {
    .cs-hero-xl { min-height: 300px; padding: 2rem 1.6rem; }
    .cs-hero-content { max-width: 100%; }
    .cs-hero-img { inset: 0; opacity: calc(var(--img-o, .5) * .45); }
    .cs-hero-xl h1 { font-size: 2.4rem; }
    .cs-embers, .cs-web-bl { display: none; }
}
</style>
"""

# Pointer parallax and card tilt. Runs once per browser tab (the flag lives on
# the top window, which survives page switches), does its work inside a single
# requestAnimationFrame, and does nothing under prefers-reduced-motion.
_FX_SCRIPT = """
<script>
(function () {
  var w = window.parent, d = w.document;
  if (w.__cineSentinelFx) { return; }
  w.__cineSentinelFx = true;
  var reduce = w.matchMedia('(prefers-reduced-motion: reduce)');
  var frame = 0, last = null, tilted = null;

  function apply() {
    frame = 0;
    if (!last || reduce.matches) { return; }
    var px = (last.clientX / w.innerWidth - 0.5).toFixed(3);
    var py = (last.clientY / w.innerHeight - 0.5).toFixed(3);
    var layers = d.querySelectorAll('.cs-plx');
    for (var i = 0; i < layers.length; i++) {
      layers[i].style.setProperty('--px', px);
      layers[i].style.setProperty('--py', py);
    }
    var card = last.target && last.target.closest ? last.target.closest('.cs-card') : null;
    if (tilted && tilted !== card) {
      tilted.style.removeProperty('--rx');
      tilted.style.removeProperty('--ry');
    }
    tilted = card;
    if (card) {
      var r = card.getBoundingClientRect();
      var x = (last.clientX - r.left) / r.width, y = (last.clientY - r.top) / r.height;
      card.style.setProperty('--rx', (x - 0.5).toFixed(3));
      card.style.setProperty('--ry', (y - 0.5).toFixed(3));
      card.style.setProperty('--mx', (x * 100).toFixed(1) + '%');
      card.style.setProperty('--my', (y * 100).toFixed(1) + '%');
    }
  }

  d.addEventListener('pointermove', function (event) {
    last = event;
    if (!frame) { frame = w.requestAnimationFrame(apply); }
  }, { passive: true });
})();
</script>
"""

# (left %, size px, duration s, delay s, sideways drift px, peak opacity)
_EMBERS = [
    (4, 3, 19, -2, 40, 0.55), (11, 2, 26, -11, -30, 0.45), (19, 4, 22, -7, 55, 0.6),
    (27, 2, 31, -19, 20, 0.4), (36, 3, 24, -4, -45, 0.5), (44, 2, 29, -15, 35, 0.4),
    (53, 4, 21, -9, -25, 0.6), (61, 2, 33, -23, 50, 0.4), (69, 3, 25, -1, -40, 0.5),
    (77, 2, 28, -13, 30, 0.45), (85, 4, 23, -17, -55, 0.6), (93, 3, 30, -6, 25, 0.5),
]
# (angle from the top-right corner in degrees, glint duration s, delay s)
_THREADS = [(-14, 9, 0.5), (-33, 11, 3.5), (-52, 10, 6.0), (-71, 12, 1.8)]


def _variable_css() -> str:
    """Per-element CSS variables for embers, threads and verdict glows.

    Streamlit's HTML sanitizer drops custom properties from inline `style`
    attributes, so they are set here, from stylesheet rules keyed on class.
    """
    rules = [".cs-ambient .cs-plx{--depth:14}", ".cs-hero-layer{--depth:18}"]
    for index, (x, s, d, delay, drift, o) in enumerate(_EMBERS, start=1):
        rules.append(
            f".cs-embers>span:nth-child({index})"
            f"{{--x:{x}%;--s:{s}px;--d:{d}s;--delay:{delay}s;--drift:{drift}px;--o:{o}}}"
        )
    for index, (a, d, delay) in enumerate(_THREADS, start=1):
        rules.append(f".cs-thread-{index}{{--a:{a}deg;--d:{d}s;--delay:{delay}s}}")
    for label, color in SENTIMENT_COLORS.items():
        rules.append(f".cs-verdict-{label}{{--glow:{color}88}}")
    return "<style>" + "".join(rules) + "</style>"


def _ambient_html() -> str:
    """Fixed decorative layer: glows, web corners, travelling glints, embers."""
    embers = "<span></span>" * len(_EMBERS)
    threads = "".join(
        f"<div class='cs-thread cs-thread-{index}'><i></i></div>" for index in range(1, len(_THREADS) + 1)
    )
    return (
        "<div class='cs-fx cs-ambient' aria-hidden='true'>"
        "<div class='cs-plx'>"
        "<div class='cs-web cs-web-tr'></div><div class='cs-web cs-web-bl'></div>"
        f"{threads}</div>"
        f"<div class='cs-embers'>{embers}</div>"
        "</div>"
    )


def _css_url(uri: str) -> str:
    return f"url('{uri}')" if uri else "none"


def inject_theme() -> None:
    """Add the theme CSS, the ambient layer and the parallax script to a page."""
    css = _CSS.replace("__WEB__", _css_url(_web_svg_uri())).replace(
        "__SIDEBAR_IMG__", _css_url(image_data_uri("descent"))
    )
    st.markdown(css + _variable_css() + _ambient_html(), unsafe_allow_html=True)
    components.html(_FX_SCRIPT, height=0)


def sidebar_brand() -> None:
    with st.sidebar:
        st.markdown(
            "<div style='padding:0.2rem 0 0.9rem 0;'>"
            f"<div style='font-size:1.3rem;font-weight:800;color:#F5F6F8;'>{APP_NAME}</div>"
            "<div style='font-size:0.74rem;letter-spacing:0.14em;text-transform:uppercase;"
            f"color:#FF6B76;margin-top:0.2rem;'>{APP_TAGLINE}</div></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "UE24CS352A — Machine Learning mini-project. All metrics are measured on "
            "the official 25,000-review Stanford IMDb test split."
        )


def hero(
    title: str,
    subtitle: str,
    badges=(),
    eyebrow: str = APP_NAME,
    image: str | None = None,
    alt_image: str | None = None,
    large: bool = False,
    position: str = "50% 42%",
    opacity: float = 0.5,
    size: str = "cover",
) -> None:
    """Page banner.

    `image` / `alt_image` name entries of IMAGES. The banner cross-fades
    between them when both are given. "descent" is a narrow image and is
    shown as a strip on the right instead of covering the banner. `position`
    and `size` are CSS background-position / background-size values, used to
    choose which part of an image shows.
    """
    layers = ""
    for index, name in enumerate([image, alt_image]):
        uri = image_data_uri(name) if name else ""
        if not uri:
            continue
        classes = "cs-hero-img" + (" cs-alt" if index == 1 else "") + (" cs-strip" if name == "descent" else "")
        layers += f"<div class='{classes}' style=\"background-image:url('{uri}');\"></div>"
    if layers:
        # Set from a stylesheet rule: inline custom properties are stripped.
        layers = (
            f"<style>.cs-hero-img{{--pos:{position};--img-o:{opacity};--size:{size}}}</style>"
            f"<div class='cs-hero-layer cs-plx'>{layers}</div>"
        )

    badge_html = "".join(f"<span class='cs-badge'>{escape(b)}</span>" for b in badges)
    st.markdown(
        f"<div class='cs-hero{' cs-hero-xl' if large else ''}'>"
        f"{layers}"
        "<div class='cs-hero-shade'></div><div class='cs-hero-web'></div><div class='cs-hero-sweep'></div>"
        "<div class='cs-hero-content'>"
        f"<div class='cs-eyebrow'>{escape(eyebrow)}</div>"
        f"<h1>{escape(title)}</h1>"
        f"<p>{escape(subtitle)}</p>"
        + (f"<div class='cs-badges'>{badge_html}</div>" if badge_html else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    st.markdown(
        f"<div class='cs-section'><h3>{escape(title)}</h3>"
        + (f"<p>{escape(caption)}</p>" if caption else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", accent: str = ACCENT) -> str:
    return (
        f"<div class='cs-card' style='border-top:3px solid {accent};'>"
        f"<div class='cs-label'>{escape(label)}</div>"
        f"<div class='cs-value'>{escape(value)}</div>"
        + (f"<div class='cs-sub'>{escape(sub)}</div>" if sub else "")
        + "</div>"
    )


def info_card(title: str, body: str, accent: str = ACCENT, footer: str = "", bar: float | None = None) -> str:
    """Card with a colored dot title, body text and an optional 0-1 progress bar."""
    bar_html = ""
    if bar is not None:
        width = max(0.0, min(1.0, bar)) * 100
        bar_html = f"<div class='cs-bar'><span style='width:{width:.1f}%;background:{accent};'></span></div>"
    return (
        "<div class='cs-card'>"
        f"<div class='cs-title'><span class='cs-dot' style='background:{accent};'></span>{escape(title)}</div>"
        f"<div class='cs-body'>{escape(body)}</div>"
        + (f"<div class='cs-sub' style='margin-top:0.55rem;color:#D2D6DD;'>{escape(footer)}</div>" if footer else "")
        + bar_html
        + "</div>"
    )


def card_grid(cards: list, columns: int = 4) -> None:
    """Render pre-built card HTML snippets in a responsive CSS grid."""
    st.markdown(
        f"<div class='cs-grid' style='grid-template-columns:repeat({columns}, minmax(0, 1fr));'>"
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def verdict_banner(label: str, headline: str, detail: str, eyebrow: str = "Verdict") -> None:
    """Animated result banner; it replays whenever a new prediction is shown."""
    color = SENTIMENT_COLORS.get(label, SENTIMENT_COLORS["neutral"])
    st.markdown(
        f"<div class='cs-verdict cs-verdict-{escape(label)}' style='"
        f"background:linear-gradient(135deg,{color}30,rgba(18,18,21,0.94) 62%);"
        f"border-left:5px solid {color};'>"
        "<div class='cs-label' style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;"
        f"color:#9BA1AC;font-weight:600;'>{escape(eyebrow)}</div>"
        f"<div class='cs-big' style='color:{color};'>{escape(headline)}</div>"
        f"<div class='cs-detail'>{escape(detail)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def note(html: str) -> None:
    """Highlighted explanatory note. `html` is trusted, app-authored markup."""
    st.markdown(f"<div class='cs-note'>{html}</div>", unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        f"<div class='cs-foot'>{APP_NAME} — {APP_TAGLINE} · scikit-learn · TF-IDF · Streamlit · Plotly — "
        "reference: Wu &amp; Shin, “Machine Learning based classification for Sentimental "
        "analysis of IMDb reviews”</div>",
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 420, title: str | None = None, legend_bottom: bool = False):
    """Apply the dashboard's dark, transparent Plotly styling to a figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=13),
        height=height,
        margin=dict(t=64 if title else 28, b=36, l=24, r=24),
        hoverlabel=dict(bgcolor="#17171B", bordercolor="rgba(225,29,46,0.55)", font_color=TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        # Smooth interpolation when a chart's data changes (metric switches).
        transition=dict(duration=450, easing="cubic-in-out"),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16, color="#F1F2F4"), x=0.01))
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.16, x=0))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def model_color(model_key: str) -> str:
    return MODEL_COLORS.get(model_key, ACCENT)


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
