"""Data download helpers for InsideAirbnb Madrid and Madrid Open Data."""

from __future__ import annotations

import csv
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm

from src.paths import ROOT

HTTP_HEADERS = {"User-Agent": "airbnb-madrid-analysis/1.0 (educational; +https://insideairbnb.com)"}

INSIDEAIRBNB_DATA_PAGE = "https://insideairbnb.com/get-the-data/"
# Ayuntamiento catalog URL (may change; see fallback below)
MADRID_METRO_PRIMARY = (
    "https://datos.madrid.es/egob/catalogo/300440-0-metro_estaciones.json"
)


def _metro_geojson_from_overpass() -> dict:
    """Build a GeoJSON FeatureCollection of Madrid metro stations from OSM (Overpass)."""
    query = """
    [out:json][timeout:120];
    (
      node["station"="subway"](40.25,-3.95,40.55,-3.45);
    );
    out body;
    """
    r = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=query.strip(),
        timeout=180,
        headers=HTTP_HEADERS,
    )
    r.raise_for_status()
    data = r.json()
    feats = []
    for el in data.get("elements", []):
        if el.get("type") != "node":
            continue
        lon, lat = el.get("lon"), el.get("lat")
        if lon is None or lat is None:
            continue
        feats.append(
            {
                "type": "Feature",
                "properties": {"name": el.get("tags", {}).get("name", "")},
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def _find_latest_madrid_listings_url() -> str:
    """Scrape InsideAirbnb data page for the most recent Madrid listings.csv.gz URL."""
    try:
        resp = requests.get(INSIDEAIRBNB_DATA_PAGE, timeout=120, headers=HTTP_HEADERS)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to fetch {INSIDEAIRBNB_DATA_PAGE}: {exc}"
        ) from exc

    html = resp.text
    pattern = re.compile(
        r'href=["\']([^"\']*madrid[^"\']*listings\.csv\.gz)["\']',
        re.IGNORECASE,
    )
    candidates: list[tuple[str, str]] = []
    for m in pattern.finditer(html):
        url = m.group(1)
        if url.startswith("//"):
            url = "https:" + url
        elif url.startswith("/"):
            url = "https://insideairbnb.com" + url
        path = urlparse(url).path
        date_m = re.search(r"/madrid/(\d{4}-\d{2}-\d{2})/", path)
        date_key = date_m.group(1) if date_m else ""
        candidates.append((date_key, url))

    if not candidates:
        raise RuntimeError(
            "No Madrid listings.csv.gz link found on InsideAirbnb data page."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def download_listings(save_dir: str | Path | None = None) -> Path:
    """
    Download the latest Madrid listings CSV from InsideAirbnb (scraped URL),
    decompress to listings.csv, write metadata JSON, return CSV path.
    Idempotent if listings.csv exists and is non-empty.
    """
    save_dir = Path(save_dir) if save_dir is not None else ROOT / "data/raw"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_csv = save_dir / "listings.csv"
    meta_path = save_dir / "download_meta.json"

    if out_csv.exists() and out_csv.stat().st_size > 0:
        print(f"Listings CSV already present at {out_csv}, skipping download.")
        return out_csv.resolve()

    url = _find_latest_madrid_listings_url()
    gz_path = save_dir / "listings.csv.gz"

    try:
        with requests.get(url, stream=True, timeout=300, headers=HTTP_HEADERS) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0) or 0)
            chunk_size = 8192
            with open(gz_path, "wb") as f, tqdm(
                desc="listings.csv.gz",
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    except requests.RequestException as exc:
        if gz_path.exists():
            gz_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc

    file_size = gz_path.stat().st_size
    with gzip.open(gz_path, "rb") as gz_in, open(out_csv, "wb") as raw_out:
        raw_out.write(gz_in.read())

    ncols = len(pd.read_csv(out_csv, nrows=0, low_memory=False).columns)
    with open(out_csv, encoding="utf-8", errors="replace", newline="") as f:
        nrow = sum(1 for _ in csv.reader(f)) - 1

    meta = {
        "source_url": url,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": file_size,
        "n_columns": ncols,
        "n_rows": nrow,
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    print(
        f"Downloaded Madrid listings: {nrow:,} rows, {ncols} columns → {out_csv}"
    )
    return out_csv.resolve()


def download_metro_stations(save_dir: str | Path | None = None) -> Path:
    """
    Download Madrid metro station GeoJSON from Madrid Open Data.
    Idempotent if metro_stations.json exists and is non-empty.
    """
    save_dir = Path(save_dir) if save_dir is not None else ROOT / "data/raw"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "metro_stations.json"

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Metro stations file already present at {out_path}, skipping download.")
        return out_path.resolve()

    try:
        resp = requests.get(MADRID_METRO_PRIMARY, timeout=120, headers=HTTP_HEADERS)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        print(f"Saved metro stations GeoJSON to {out_path} (Ayuntamiento portal).")
        return out_path.resolve()
    except requests.RequestException as exc:
        print(
            f"Primary metro URL failed ({MADRID_METRO_PRIMARY}): {exc}. "
            "Using OpenStreetMap (Overpass) fallback for station coordinates."
        )
        try:
            gj = _metro_geojson_from_overpass()
        except requests.RequestException as exc2:
            raise RuntimeError(
                "Could not download metro stations from Madrid Open Data or OSM Overpass."
            ) from exc2
        if not gj.get("features"):
            raise RuntimeError("Overpass returned no metro station features.")
        out_path.write_text(json.dumps(gj), encoding="utf-8")
        print(
            f"Saved {len(gj['features'])} metro station points to {out_path} (OSM fallback)."
        )
        return out_path.resolve()
