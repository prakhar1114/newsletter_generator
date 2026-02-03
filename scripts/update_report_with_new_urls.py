from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from summariser.incremental import incremental_update_and_generate_report
from summariser.reporting import generate_compiled_report


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(message)s")
    else:
        logging.getLogger().setLevel(level)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incrementally update report with new URLs.")
    p.add_argument("--urls", nargs="*", default=None, help="URLs to ingest (space-separated)")
    p.add_argument("--urls-file", type=Path, default=None, help="Text file with one URL per line")
    p.add_argument("--bootstrap", action="store_true", help="Run batch clustering once and persist cluster metadata")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args(argv)


def _load_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []
    if args.urls:
        urls.extend([str(u).strip() for u in args.urls if str(u).strip()])
    if args.urls_file:
        text = args.urls_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            u = line.strip()
            if u and not u.startswith("#"):
                urls.append(u)
    # Dedup but keep order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


async def _main_async(argv: list[str]) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    if args.bootstrap:
        logging.info("[script] bootstrap: generating batch report and persisting cluster metadata ...")
        _md, path = generate_compiled_report(use_stored_centroids=False)
        logging.info("[script] bootstrap report written: %s", path)

    urls = _load_urls(args)
    if not urls:
        logging.info("[script] no urls provided; nothing to do")
        return 0

    logging.info("[script] incremental update starting; urls=%s", len(urls))
    res = await incremental_update_and_generate_report(urls=urls)
    if res.report_path:
        logging.info("[script] incremental update done; report=%s", res.report_path)
    else:
        logging.info("[script] incremental update done; no centroid changes so report not regenerated")
    logging.info("[script] touched_clusters=%s noise=%s", len(res.touched_clusters), len(res.noise_point_ids))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_main_async(sys.argv[1:])))


if __name__ == "__main__":
    main()

