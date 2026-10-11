"""
Start the dashboard, drive it in headless Microsoft Edge, and save screenshots.

This is both a browser-level smoke test and the source of the screenshots
used in the presentation. It:
  - starts `streamlit run app/Home.py` on a spare port and stops it afterwards;
  - opens every page and counts rendered charts and error boxes;
  - checks the visual effects on the Home page: background images present,
    parallax and card tilt respond to the pointer, idle CPU cost, and that
    animations stop under prefers-reduced-motion;
  - types a review and runs the live analyzer;
  - uploads tests/fixtures/sample_batch.csv and runs a batch prediction;
  - writes submission/screenshots/<page>.png (16:9 top of page, for slides).
    With --all it also writes whole-page and per-tab captures for review;
    those are git-ignored.

Exit code is 0 only if no page showed a Streamlit exception.

Requires Microsoft Edge. Uses the DevTools protocol through tornado, which
ships with Streamlit, so no extra packages are needed.

Usage:  python scripts\\capture_screenshots.py [--all]
"""

import asyncio
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from tornado.websocket import websocket_connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "submission" / "screenshots"
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "sample_batch.csv"
APP_PORT = 8765
DEBUG_PORT = 9333
BASE = f"http://localhost:{APP_PORT}/"
WIDTH, SLIDE_HEIGHT = 1440, 900
REVIEW_CAPTURES = "--all" in sys.argv[1:]

EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

REVIEW = (
    "The visuals are gorgeous and the lead actor is excellent, but the script is a mess "
    "and the second half is painfully slow. Not terrible, not great."
)

CLICK = (
    "(() => {{ const b=[...document.querySelectorAll('button')]"
    ".find(b=>b.innerText.includes({text!r})); if(b){{b.click();return true}} return false }})()"
)
STATE = (
    "JSON.stringify({plots: document.querySelectorAll('.js-plotly-plot').length,"
    "exceptions: document.querySelectorAll('[data-testid=stException]').length,"
    "alerts: [...document.querySelectorAll('[data-testid=stAlert]')].map(e=>e.innerText.slice(0,90)),"
    "cards: document.querySelectorAll('.cs-card').length,"
    "hero: !!document.querySelector('.cs-hero')})"
)


class Browser:
    def __init__(self, ws):
        self.ws, self.next_id, self.exceptions = ws, 0, 0

    async def call(self, method, **params):
        self.next_id += 1
        message_id = self.next_id
        await self.ws.write_message(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            message = json.loads(await self.ws.read_message())
            if message.get("id") == message_id:
                return message.get("result", message)

    async def js(self, expression):
        result = await self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        return result.get("result", {}).get("value")

    async def resize(self, height):
        await self.call(
            "Emulation.setDeviceMetricsOverride", width=WIDTH, height=height, deviceScaleFactor=1, mobile=False
        )

    async def settle(self, limit=40):
        """Wait until the element counts stop changing and Streamlit is idle."""
        probe = (
            "JSON.stringify([document.querySelectorAll('.js-plotly-plot').length,"
            "document.querySelectorAll('.cs-card').length,"
            "document.querySelectorAll('[data-testid=stMarkdown]').length,"
            "!!document.querySelector('[data-testid=stStatusWidget]')])"
        )
        started, previous, stable = time.time(), None, 0
        while time.time() - started < limit:
            await asyncio.sleep(1)
            current = await self.js(probe)
            running = current is None or current.endswith("true]")
            stable = stable + 1 if (current == previous and not running) else 0
            previous = current
            if stable >= 3:
                break
        return time.time() - started

    async def goto(self, page):
        await self.call("Page.navigate", url=BASE + page)
        await asyncio.sleep(1.5)
        seconds = await self.settle()
        print(f"  /{page} settled after {seconds:.0f}s", flush=True)

    async def click(self, text, wait=3):
        clicked = await self.js(CLICK.format(text=text))
        await asyncio.sleep(1)
        await self.settle(limit=max(wait, 20))
        return clicked

    async def report(self, name):
        state = json.loads(await self.js(STATE))
        self.exceptions += state["exceptions"]
        print(f"{name}: {state}", flush=True)
        return state

    async def shoot(self, name, full=True):
        await self.js("document.querySelector('[data-testid=stMain]')?.scrollTo(0, 0)")
        await self.resize(SLIDE_HEIGHT)
        await asyncio.sleep(1.5)
        shot = await self.call("Page.captureScreenshot", format="png")
        (OUT_DIR / f"{name}.png").write_bytes(base64.b64decode(shot["data"]))
        if full and REVIEW_CAPTURES:
            height = await self.js(
                "Math.max(document.querySelector('[data-testid=stMain]')?.scrollHeight||0, 900)"
            )
            await self.resize(int(min(height + 40, 7000)))
            await asyncio.sleep(2.5)
            shot = await self.call("Page.captureScreenshot", format="png")
            (OUT_DIR / f"{name}_full.png").write_bytes(base64.b64decode(shot["data"]))
            await self.resize(SLIDE_HEIGHT)


EFFECTS = """
(async () => {
  const pause = ms => new Promise(r => setTimeout(r, ms));
  const card = document.querySelector('.cs-card');
  const box = card.getBoundingClientRect();
  card.dispatchEvent(new PointerEvent('pointermove', {
    bubbles: true, clientX: box.left + box.width * 0.8, clientY: box.top + box.height * 0.2 }));
  await pause(250);
  const layer = document.querySelector('.cs-plx');
  const names = new Set(document.getAnimations().map(a => a.animationName).filter(Boolean));
  return JSON.stringify({
    script_installed: window.__cineSentinelFx === true,
    parallax_x: layer.style.getPropertyValue('--px'),
    card_tilt_x: card.style.getPropertyValue('--rx'),
    card_highlight_x: card.style.getPropertyValue('--mx'),
    running_animations: document.getAnimations().filter(a => a.playState === 'running').length,
    animation_names: [...names].sort(),
    hero_images: document.querySelectorAll('.cs-hero-img').length,
    embers: document.querySelectorAll('.cs-embers > span').length,
    embers_display: getComputedStyle(document.querySelector('.cs-embers')).display,
    ember_animation: getComputedStyle(document.querySelector('.cs-embers > span')).animationName,
    thread_rotated: getComputedStyle(document.querySelector('.cs-thread')).transform !== 'none',
    hero_image_opacity: getComputedStyle(document.querySelector('.cs-hero-img')).opacity,
    sidebar_image: getComputedStyle(document.querySelector('[data-testid=stSidebar]'), '::after')
      .backgroundImage.startsWith('url("data:image/jpeg'),
  });
})()
"""


async def effects_check(browser: Browser) -> None:
    """Confirm the effects are live, measure their idle cost, and test reduced motion."""
    result = await browser.call(
        "Runtime.evaluate", expression=EFFECTS, returnByValue=True, awaitPromise=True
    )
    normal = json.loads(result["result"]["value"])
    print(f"effects: {normal}", flush=True)
    assert normal["script_installed"], "parallax script did not install"
    assert normal["parallax_x"] and normal["card_tilt_x"], "pointer effects did not respond"
    assert normal["hero_images"] >= 1 and normal["sidebar_image"], "background images missing"
    assert normal["ember_animation"] == "cs-ember", "embers are not animating"
    assert normal["thread_rotated"], "web threads are not positioned"

    # Main-thread time used while the page just sits there animating.
    await browser.call("Performance.enable")

    async def task_seconds() -> float:
        metrics = (await browser.call("Performance.getMetrics"))["metrics"]
        return next(m["value"] for m in metrics if m["name"] == "TaskDuration")

    before = await task_seconds()
    await asyncio.sleep(5)
    busy = (await task_seconds() - before) / 5
    print(f"  idle main-thread load with animations running: {busy * 100:.1f}%", flush=True)

    await browser.call(
        "Emulation.setEmulatedMedia", features=[{"name": "prefers-reduced-motion", "value": "reduce"}]
    )
    await asyncio.sleep(1)
    result = await browser.call(
        "Runtime.evaluate", expression=EFFECTS, returnByValue=True, awaitPromise=True
    )
    reduced = json.loads(result["result"]["value"])
    print(
        f"  reduced motion: running animations {reduced['running_animations']}, "
        f"embers display '{reduced['embers_display']}'",
        flush=True,
    )
    assert reduced["embers_display"] == "none", "embers still shown under reduced motion"
    assert reduced["running_animations"] == 0, "animations still running under reduced motion"
    await browser.call("Emulation.setEmulatedMedia", features=[])


async def drive(browser: Browser) -> None:
    await browser.call("Page.enable")
    await browser.call("DOM.enable")
    await browser.resize(SLIDE_HEIGHT)

    await browser.goto("")
    await browser.report("home")
    await effects_check(browser)
    await browser.shoot("home")

    await browser.goto("Model_Comparison_Lab")
    await browser.report("comparison lab")
    await browser.shoot("comparison")
    for tab in ["Confusion matrices", "ROC and cost", "Model details", "Reference paper", "Experiments"]:
        print(f"  open tab '{tab}': {await browser.click(tab)}")
        await browser.report(f"comparison lab / {tab}")
        if REVIEW_CAPTURES:
            await browser.shoot("comparison_" + tab.lower().replace(" ", "_"))

    await browser.goto("Live_Analyzer")
    await browser.js("document.querySelector('textarea').focus()")
    await browser.call("Input.insertText", text=REVIEW)
    await browser.js("document.querySelector('textarea').blur()")
    await asyncio.sleep(3)
    print(f"  click 'Analyze sentiment': {await browser.click('Analyze sentiment', wait=12)}")
    state = await browser.report("live analyzer (after prediction)")
    has_verdict = await browser.js("!!document.querySelector('.cs-verdict')")
    print(f"  verdict banner rendered: {has_verdict}")
    assert state["plots"] >= 2, "live analyzer did not render its charts"
    await browser.shoot("live")

    await browser.goto("Batch_Studio")
    document = await browser.call("DOM.getDocument")
    node = await browser.call(
        "DOM.querySelector", nodeId=document["root"]["nodeId"], selector="input[type=file]"
    )
    await browser.call("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[str(FIXTURE)])
    await asyncio.sleep(5)
    await browser.report("batch studio (file uploaded)")
    print(f"  click 'Run batch prediction': {await browser.click('Run batch prediction', wait=10)}")
    state = await browser.report("batch studio (scored)")
    downloads = await browser.js(
        "[...document.querySelectorAll('button')].filter(b=>b.innerText.includes('Download predictions')).length"
    )
    print(f"  download button present: {downloads == 1}")
    assert state["plots"] >= 2 and downloads == 1, "batch studio did not produce results"
    await browser.shoot("batch")

    await browser.goto("Methodology")
    await browser.report("methodology")
    await browser.shoot("methodology")


def wait_for(url: str, seconds: int) -> bool:
    for _ in range(seconds * 2):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return False


async def main() -> int:
    edge = next((p for p in EDGE_CANDIDATES if p.exists()), None)
    if edge is None:
        print("Microsoft Edge not found; cannot run the browser check.")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="cinesentinel-edge-"))
    server = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app/Home.py",
            "--server.headless", "true", "--server.port", str(APP_PORT),
        ],
        cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    edge_process = None
    try:
        if not wait_for(BASE + "_stcore/health", 40):
            print("Streamlit did not start.")
            return 2
        print(f"Streamlit is serving on {BASE}")
        for page in ["", "Live_Analyzer", "Model_Comparison_Lab", "Batch_Studio", "Methodology"]:
            status = urllib.request.urlopen(BASE + page, timeout=10).status
            print(f"  GET /{page} -> {status}")

        edge_process = subprocess.Popen(
            [
                str(edge), "--headless=new", "--disable-gpu", f"--remote-debugging-port={DEBUG_PORT}",
                "--remote-allow-origins=*", f"--user-data-dir={profile}",
                f"--window-size={WIDTH},{SLIDE_HEIGHT}", "about:blank",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not wait_for(f"http://127.0.0.1:{DEBUG_PORT}/json", 20):
            print("Edge did not open its debugging port.")
            return 2
        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json"))
        page_target = next(t for t in targets if t["type"] == "page")
        ws = await websocket_connect(page_target["webSocketDebuggerUrl"], max_message_size=200 * 1024 * 1024)
        browser = Browser(ws)
        await drive(browser)
        print(f"\nStreamlit exceptions seen across all pages: {browser.exceptions}")
        print(f"Screenshots saved to {OUT_DIR}")
        return 0 if browser.exceptions == 0 else 1
    finally:
        for process in (edge_process, server):
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
