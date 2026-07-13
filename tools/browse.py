"""Headless-browser fetch/interact via Playwright — for JS-heavy pages and file
exports that plain WebFetch/curl can't handle.

Motivation: interactive spectral-data sites (LSPDD-style query builders, LICA-UCM
export dialogs) and bot-walled pages (ResearchGate) render their content with
JavaScript and gate downloads behind clicks. This drives a real Chromium so we can:
  * render the page and read the *real* DOM (``--text`` / ``--html`` / ``--links``),
  * fill fields and click buttons to run a query (``--fill`` / ``--click``),
  * screenshot a JS-rendered chart so it can be fed to tools/plot_digitizer.py
    (``--screenshot``),
  * capture files a click triggers (``--download-dir``).

Setup (one-time):  python -m pip install playwright && python -m playwright install chromium

Examples:
  # render a JS page and dump visible text + links
  python tools/browse.py https://guaix.fis.ucm.es/lamps_spectra --text --links

  # run a query builder then grab the resulting chart image
  python tools/browse.py https://example.org/db \
      --fill "#lamp=Xenon short arc" --click "#search" \
      --wait-selector "svg.chart" --screenshot scraps/xenon_chart.png

  # capture a CSV/PDF that a download button produces
  python tools/browse.py https://example.org/lamp/42 \
      --click "a.export-csv" --download-dir data_raw/

Selectors are Playwright selectors (CSS, ``text=...``, ``xpath=...``). ``--fill``
takes ``SELECTOR=VALUE``. ``--click`` and ``--fill`` may be repeated and run in the
order given, before the final capture.
"""
import argparse
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Headless-browser fetch/interact (Playwright).")
    ap.add_argument("url")
    ap.add_argument("--wait-until", default="networkidle",
                    choices=["load", "domcontentloaded", "networkidle", "commit"])
    ap.add_argument("--wait-selector", help="wait for this selector before capture")
    ap.add_argument("--wait-after-ms", type=int, default=0,
                    help="extra sleep (ms) after actions, for late JS")
    ap.add_argument("--timeout", type=int, default=45000, help="nav/selector timeout ms")
    ap.add_argument("--fill", action="append", default=[], metavar="SELECTOR=VALUE",
                    help="fill an input (repeatable, run in order)")
    ap.add_argument("--click", action="append", default=[], metavar="SELECTOR",
                    help="click an element (repeatable, run in order)")
    ap.add_argument("--actions-order", choices=["fill-first", "click-first"],
                    default="fill-first", help="whether fills or clicks run first")
    ap.add_argument("--text", action="store_true", help="print visible innerText")
    ap.add_argument("--html", metavar="PATH", help="save rendered HTML to file")
    ap.add_argument("--links", action="store_true", help="print anchor hrefs + text")
    ap.add_argument("--screenshot", metavar="PATH", help="save full-page PNG")
    ap.add_argument("--download-dir", metavar="DIR",
                    help="capture files triggered by clicks into this dir")
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--stealth", action="store_true",
                    help="apply playwright-stealth to reduce headless fingerprinting "
                         "(helps light bot-detection; will NOT beat hard Cloudflare)")
    ap.add_argument("--user-agent", default=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"))
    args = ap.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed: python -m pip install playwright && "
              "python -m playwright install chromium", file=sys.stderr)
        return 2

    if args.stealth:
        try:
            from playwright_stealth import Stealth
            pw_ctx = Stealth().use_sync(sync_playwright())
        except ImportError:
            print("playwright-stealth not installed; run: "
                  "python -m pip install playwright-stealth", file=sys.stderr)
            return 2
    else:
        pw_ctx = sync_playwright()

    downloads = []
    with pw_ctx as p:
        browser = p.chromium.launch(headless=not args.headed)
        ctx = browser.new_context(user_agent=args.user_agent,
                                   accept_downloads=bool(args.download_dir))
        page = ctx.new_page()
        page.set_default_timeout(args.timeout)
        try:
            page.goto(args.url, wait_until=args.wait_until, timeout=args.timeout)
        except Exception as e:
            print(f"WARN: goto: {e}", file=sys.stderr)

        def do_fills():
            for spec in args.fill:
                sel, _, val = spec.partition("=")
                try:
                    page.fill(sel, val)
                except Exception as e:
                    print(f"WARN: fill {sel!r}: {e}", file=sys.stderr)

        def do_clicks():
            for sel in args.click:
                try:
                    if args.download_dir:
                        with page.expect_download(timeout=args.timeout) as di:
                            page.click(sel)
                        d = di.value
                        dest = Path(args.download_dir) / d.suggested_filename
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        d.save_as(str(dest))
                        downloads.append(str(dest))
                    else:
                        page.click(sel)
                except Exception as e:
                    print(f"WARN: click {sel!r}: {e}", file=sys.stderr)

        if args.actions_order == "fill-first":
            do_fills(); do_clicks()
        else:
            do_clicks(); do_fills()

        if args.wait_selector:
            try:
                page.wait_for_selector(args.wait_selector, timeout=args.timeout)
            except Exception as e:
                print(f"WARN: wait_selector: {e}", file=sys.stderr)
        if args.wait_after_ms:
            time.sleep(args.wait_after_ms / 1000.0)

        if args.screenshot:
            Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=args.screenshot, full_page=True)
            print(f"screenshot -> {args.screenshot}")
        if args.html:
            Path(args.html).write_text(page.content(), encoding="utf-8")
            print(f"html -> {args.html}")
        if args.links:
            hrefs = page.eval_on_selector_all(
                "a[href]", "els => els.map(e => e.href + '\\t' + (e.innerText||'').trim())")
            print("=== LINKS ===")
            for h in hrefs:
                print(h)
        if args.text:
            print("=== TEXT ===")
            print(page.inner_text("body"))
        for d in downloads:
            print(f"downloaded -> {d}")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
