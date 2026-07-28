#!/usr/bin/env python
"""Shared xlsx writer for the campaign's shortlists and batch sheets.

`write_xlsx` is `DataFrame.to_excel` plus column widths fitted to the content, a frozen header row
and an autofilter, so the files open readable instead of needing every column dragged out by hand.

Widths are clamped at both ends. The floor keeps short numeric columns from collapsing to something
narrower than their own header; the ceiling matters because `aa_sequence` and `dna_sequence` are 239
and 720 characters -- fitting those literally would produce a sheet nobody can scroll. They are
capped instead, which shows enough of the sequence to identify a row while leaving the rest one
click away in the formula bar.
"""
from openpyxl.utils import get_column_letter

MIN_WIDTH, MAX_WIDTH, PAD = 7, 44, 2


def fit_widths(worksheet, df, min_width=MIN_WIDTH, max_width=MAX_WIDTH, pad=PAD):
    """Set each column's width from the longest string it has to display, header included."""
    for i, col in enumerate(df.columns, start=1):
        longest = int(df[col].astype(str).str.len().max()) if len(df) else 0
        width = max(len(str(col)), longest) + pad
        worksheet.column_dimensions[get_column_letter(i)].width = min(max(width, min_width), max_width)


def write_xlsx(df, path, sheet_name="Sheet1", freeze_header=True, autofilter=True):
    """Write `df` to `path` as a single formatted sheet."""
    import pandas as pd

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        df.to_excel(xl, index=False, sheet_name=sheet_name)
        ws = xl.sheets[sheet_name]
        fit_widths(ws, df)
        if freeze_header:
            ws.freeze_panes = "A2"
        if autofilter and len(df):
            ws.auto_filter.ref = ws.dimensions
    return path
