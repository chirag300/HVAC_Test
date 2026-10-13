"""
download.py -- fetch the public FDD datasets into data/raw/<name>/ and unpack them.

    python download.py --list
    python download.py sdahu rtu fcu nist        # any subset
    python download.py all

Sources
  sdahu  LBNL single-duct AHU, 5 faults x severities + fault-free, 1-min, 1 year (simulated)
  rtu    LBNL rooftop unit (simulated, 5-ton), 10 faults x severities + 7 fault-free cases
  fcu    LBNL fan-coil unit, 17 faults x severities + fault-free (OEDI submission 5763)
  nist   NIST residential heat pump, 14/16 SEER cooling-mode fault tests (xlsx)

LBNL asks visitors to fill in a short form before downloading. If the direct link returns a
web page instead of a zip, this script says so and prints where to put the file you download
in a browser (then re-run: it will just unpack it). Downloads resume if interrupted.
"""
import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

LBNL = "https://fdddata.lbl.gov/data"
DATASETS = {
    "sdahu": {
        "page": "https://faultdetection.lbl.gov/dataset/simulated-sd-ahu/",
        "files": {
            "LBNL_FDD_Data_Sets_SDAHU.zip": f"{LBNL}/Simulated_LBNL_FDD_Data_Sets_SDAHU/LBNL_FDD_Data_Sets_SDAHU.zip",
            "LBNL_FDD_Data_Sets_SDAHU.pdf": f"{LBNL}/Simulated_LBNL_FDD_Data_Sets_SDAHU/LBNL_FDD_Data_Sets_SDAHU.pdf",
        },
    },
    "rtu": {
        "page": "https://faultdetection.lbl.gov/dataset/rtu/",
        "files": {
            "LBNL_FDD_Data_Sets_RTU.zip": f"{LBNL}/Simulated_LBNL_FDD_Data_Sets_RTU/LBNL_FDD_Data_Sets_RTU.zip",
            "LBNL_FDD_Data_Sets_RTU.pdf": f"{LBNL}/Simulated_LBNL_FDD_Data_Sets_RTU/LBNL_FDD_Data_Sets_RTU.pdf",
        },
    },
    "fcu": {
        "page": "https://data.openei.org/submissions/5763",
        "files": {
            "LBNL_FDD_Data_Sets_FCU.zip": f"{LBNL}/Simulated_LBNL_FDD_Data_Sets_FCU/LBNL_FDD_Data_Sets_FCU.zip",
            "Inventory_of_LBNL_FDD_Data_Sets_FCU.pdf":
                "https://data.openei.org/files/5763/Inventory%20of%20LBNL%20FDD%20Data%20Sets_FCU.pdf",
        },
    },
    "nist": {
        "page": "https://www.nist.gov/el/beed/hvacr-equipment-performance-group/fault-detection-and-diagnosis-research-data",
        "files": {
            "fdd_heat_pump_cooling_1416seer_longshort_alldata_2019.xlsx":
                "https://www.nist.gov/document/fddheatpumpcooling1416seerlongshortalldata2019xlsx",
            "description_of_data_fdd_1416seer_hp_cooling.xlsx":
                "https://www.nist.gov/document/descriptionofdatafdd1416seerhpcoolingxlsx",
            "nistir_7350_data_in_appendix_d.xls":
                "https://www.nist.gov/document/nistir7350datainappendixdxls",
        },
    },
}

MAGIC = {".zip": b"PK", ".xlsx": b"PK", ".xls": b"\xd0\xcf\x11\xe0", ".pdf": b"%PDF"}
UA = {"User-Agent": "Mozilla/5.0 (research download; fdd-shortcut-study)"}


def _looks_ok(path):
    ext = path.suffix.lower()
    if ext not in MAGIC or not path.exists():
        return path.exists()
    with open(path, "rb") as f:
        return f.read(4).startswith(MAGIC[ext])


def fetch(url, dest: Path):
    if dest.exists() and _looks_ok(dest):
        print(f"  have {dest.name}")
        return True
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            ctype = r.headers.get("Content-Type", "")
            if "text/html" in ctype:
                print(f"  ! {url} returned a web page (registration form?), not the file.")
                return False
            mode = "ab" if (have and r.status == 206) else "wb"
            total = int(r.headers.get("Content-Length", 0)) + (have if mode == "ab" else 0)
            done = have if mode == "ab" else 0
            with open(part, mode) as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {dest.name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="")
            print()
    except Exception as e:  # network errors, 403s, timeouts
        print(f"  ! failed {url}: {e}")
        return False
    part.rename(dest)
    if not _looks_ok(dest):
        print(f"  ! {dest.name} does not look like a {dest.suffix} file (got HTML?)")
        dest.rename(dest.with_suffix(dest.suffix + ".bad"))
        return False
    return True


def safe_extract(zpath: Path, out: Path, depth=0):
    """Extract with path-traversal protection; recurse into nested zips."""
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = (out / m.filename).resolve()
            if not str(target).startswith(str(out.resolve())):
                print(f"  ! skipping unsafe path {m.filename}")
                continue
            if m.filename.startswith("__MACOSX") or Path(m.filename).name.startswith("._"):
                continue
            if m.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == m.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if target.suffix.lower() == ".zip" and depth < 2:
                safe_extract(target, target.with_suffix(""), depth + 1)


def get(name, root: Path):
    spec = DATASETS[name]
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] -> {d}")
    missing = []
    for fname, url in spec["files"].items():
        if not fetch(url, d / fname):
            missing.append((fname, url))
    for z in d.glob("*.zip"):
        if _looks_ok(z):
            print(f"  unpacking {z.name}")
            safe_extract(z, d / "extracted")
    if missing:
        print(f"\n  Manual step for [{name}]: open {spec['page']}, fill in the download form,")
        print(f"  and save these files into {d.resolve()}/ with exactly these names:")
        for fname, url in missing:
            print(f"     {fname}    (link: {url})")
        print(f"  Then re-run:  python download.py {name}\n")
    n_csv = len(list((d / "extracted").rglob("*.csv"))) if (d / "extracted").exists() else 0
    print(f"  {n_csv} CSV files available under {d / 'extracted'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.names:
        for n, s in DATASETS.items():
            print(f"{n:6s} {s['page']}")
        return
    names = list(DATASETS) if a.names == ["all"] else a.names
    for n in names:
        if n not in DATASETS:
            sys.exit(f"unknown dataset {n}; choose from {list(DATASETS)}")
        get(n, Path(a.root))


if __name__ == "__main__":
    main()
