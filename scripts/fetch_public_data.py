"""Download public reanalysis subsets and Natural Earth coastline geometry."""

import argparse
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import shapefile


NC_SOURCES = {
    "ncep_pres_sfc_asia.nc": (
        "https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis.derived/surface/"
        "pres.sfc.mon.mean.nc?var=pres&north=65&west=65&east=170&south=5"
        "&time_start=1961-01-01T00:00:00Z&time_end=2025-12-31T23:59:59Z&accept=netcdf4"
    ),
    "ncep2_hgt500_asia.nc": (
        "https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis2.derived/pressure/"
        "hgt.mon.mean.nc?var=hgt&north=65&west=65&east=170&south=5&vertCoord=500"
        "&time_start=1979-01-01T00:00:00Z&time_end=2025-12-31T23:59:59Z&accept=netcdf4"
    ),
    "ncep2_hgt200_asia.nc": (
        "https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis2.derived/pressure/"
        "hgt.mon.mean.nc?var=hgt&north=65&west=65&east=170&south=5&vertCoord=200"
        "&time_start=1979-01-01T00:00:00Z&time_end=2025-12-31T23:59:59Z&accept=netcdf4"
    ),
}
TEXT_SOURCES = {
    "cpc_oni.ascii.txt": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "psl_soi.data": "https://psl.noaa.gov/data/correlation/soi.data",
    "psl_nao.data": "https://psl.noaa.gov/data/correlation/nao.data",
}
COASTLINE_ZIP = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip"


def download_all(raw_dir):
    import time

    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in {**NC_SOURCES, **TEXT_SOURCES}.items():
        target = out_dir / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"exists: {target.name}")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            target.write_bytes(resp.read())
        print(f"fetched: {target.name} ({target.stat().st_size} bytes)")
        time.sleep(1.5)

    coast_out = out_dir / "ne_50m_coastline.geojson"
    if not coast_out.exists() or coast_out.stat().st_size == 0:
        req = urllib.request.Request(COASTLINE_ZIP, headers={"User-Agent": "Mozilla/5.0"})
        zbytes = urllib.request.urlopen(req, timeout=60).read()
        zf = zipfile.ZipFile(io.BytesIO(zbytes))
        shp_name = next(n for n in zf.namelist() if n.endswith(".shp"))
        dbf_name = next(n for n in zf.namelist() if n.endswith(".dbf"))
        sf = shapefile.Reader(shp=io.BytesIO(zf.read(shp_name)), dbf=io.BytesIO(zf.read(dbf_name)))
        feats = [
            {"type": "Feature", "geometry": s.__geo_interface__, "properties": {}}
            for s in sf.shapes()
        ]
        coast_out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
        print(f"fetched: {coast_out.name} ({len(feats)} features)")


def main():
    parser = argparse.ArgumentParser(description="Fetch public subsets for Yangtze transition diagnostics.")
    parser.add_argument("--raw-dir", required=True, help="Directory to store downloaded raw files.")
    args = parser.parse_args()
    download_all(args.raw_dir)


if __name__ == "__main__":
    main()
