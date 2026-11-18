"""Capture the actual UI and Folium map through a running Firefox geckodriver."""
import argparse
import base64
from pathlib import Path
from time import monotonic, sleep
from urllib.parse import urlencode

import httpx


def require_script(driver, session, script):
    deadline = monotonic() + 45
    while monotonic() < deadline:
        result = driver.post(f"{session}/execute/sync", json={"script": script, "args": []})
        result.raise_for_status()
        if result.json()["value"]:
            return
        sleep(0.4)
    raise RuntimeError(f"Map did not render within 45 seconds. Page check: {script}")


def capture(driver, session, url, condition, output, width, height):
    response = driver.post(f"{session}/window/rect", json={"width": width, "height": height})
    response.raise_for_status()
    response = driver.post(f"{session}/url", json={"url": url})
    response.raise_for_status()
    require_script(driver, session, condition)
    sleep(2)
    response = driver.get(f"{session}/screenshot")
    response.raise_for_status()
    output.write_bytes(base64.b64decode(response.json()["value"]))


def run(output, ui, api, trip, webdriver):
    output = output.resolve()
    with httpx.Client(base_url=webdriver, timeout=90) as driver:
        response = driver.post("/session", json={"capabilities": {"alwaysMatch": {
            "browserName": "firefox", "pageLoadStrategy": "eager",
            "moz:firefoxOptions": {"args": ["-headless"]},
        }}})
        response.raise_for_status()
        session = "/session/" + response.json()["value"]["sessionId"]
        try:
            capture(driver, session, ui + "?" + urlencode({"api": api, "trip": trip}),
                    "return document.querySelectorAll('#trip option').length === 11 && "
                    "document.querySelector('#accuracy').textContent.includes('%') && "
                    "document.querySelector('#map-empty').hidden;",
                    output / "demo_ui.png", 1280, 880)
            capture(driver, session, (output / "trip_map.html").as_uri(),
                    "return document.querySelector('.leaflet-container') && "
                    "document.querySelector('.leaflet-control-layers-list');",
                    output / "folium_map.png", 1280, 850)
        finally:
            driver.delete(session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("report"))
    parser.add_argument("--ui", default="http://127.0.0.1:8001")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--trip", default="00000076")
    parser.add_argument("--webdriver", default="http://127.0.0.1:4444")
    args = parser.parse_args()
    run(args.output, args.ui, args.api, args.trip, args.webdriver)
