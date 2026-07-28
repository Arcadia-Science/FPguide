#!/usr/bin/env python3
"""Fetch the Atkinson Hyperlegible fonts the Arcadia style guide specifies, as static faces.

The 2026 Arcadia style guide asks for specific weights -- ExtraLight for annotations, Regular
for body text and axis labels, Medium for axis titles, SemiBold for key titles -- plus
Atkinson Hyperlegible Mono for numerals. Google Fonts ships both families only as *variable*
fonts, and matplotlib cannot select a position on a variable font's weight axis: it sees one
face at the default weight. So each required weight is cut into its own static instance here.

Everything is written into ``fonts/`` next to this script and loaded from there via
``apc.mpl.setup(font_dirpath=...)``, so nothing is installed into the system font directories.
The directory is gitignored; rerun this script rather than committing the binaries.

Fonts are SIL Open Font License 1.1; the license is downloaded alongside them.

Usage:
    python fetch_arcadia_fonts.py           # skip if the faces are already present
    python fetch_arcadia_fonts.py --force   # re-download and rebuild
"""
from __future__ import annotations

import argparse
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "fonts")

REPO = "https://github.com/google/fonts/raw/main/ofl"
FAMILIES = {
    "Atkinson Hyperlegible Next": f"{REPO}/atkinsonhyperlegiblenext/AtkinsonHyperlegibleNext%5Bwght%5D.ttf",
    "Atkinson Hyperlegible Mono": f"{REPO}/atkinsonhyperlegiblemono/AtkinsonHyperlegibleMono%5Bwght%5D.ttf",
}
LICENSE_URL = f"{REPO}/atkinsonhyperlegiblenext/OFL.txt"

# weight axis position -> style name. 200/400/500/600 are the ones the guide uses;
# 300 and 700 are cheap to include and stop matplotlib substituting when asked for them.
WEIGHTS = {200: "ExtraLight", 300: "Light", 400: "Regular",
           500: "Medium", 600: "SemiBold", 700: "Bold"}


def expected_faces():
    return [os.path.join(DEST, f"{fam.replace(' ', '')}{style}.ttf")
            for fam in FAMILIES for style in WEIGHTS.values()]


def build():
    from fontTools import ttLib
    from fontTools.varLib import instancer

    os.makedirs(DEST, exist_ok=True)
    for family, url in FAMILIES.items():
        variable_path = os.path.join(DEST, f"_{family.replace(' ', '')}-variable.ttf")
        print(f"downloading {family} ...")
        urllib.request.urlretrieve(url, variable_path)

        for wght, style in WEIGHTS.items():
            font = ttLib.TTFont(variable_path)
            instancer.instantiateVariableFont(font, {"wght": wght}, inplace=True,
                                              updateFontNames=False)
            full = f"{family} {style}"
            postscript = full.replace(" ", "")
            # rewrite the name table so matplotlib reads one family with distinct styles
            names = font["name"]
            for name_id, value in ((1, family), (2, style), (3, f"{postscript};static"),
                                   (4, full), (6, postscript), (16, family), (17, style)):
                names.setName(value, name_id, 3, 1, 0x409)   # Windows / Unicode BMP / en-US
                names.setName(value, name_id, 1, 0, 0)       # Macintosh / Roman / English
            font["OS/2"].usWeightClass = wght
            font.save(os.path.join(DEST, f"{postscript}.ttf"))
            print(f"  {full}")

        os.remove(variable_path)

    urllib.request.urlretrieve(LICENSE_URL, os.path.join(DEST, "OFL.txt"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download and rebuild")
    a = ap.parse_args()

    if all(os.path.exists(p) for p in expected_faces()) and not a.force:
        print(f"{len(expected_faces())} static faces already in {os.path.relpath(DEST, HERE)}/ "
              f"(use --force to rebuild)")
        return

    build()
    print(f"\nwrote {len(expected_faces())} static faces to {os.path.relpath(DEST, HERE)}/")


if __name__ == "__main__":
    main()
