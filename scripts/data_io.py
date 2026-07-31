#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yu Peng
"""
Data integration for the BO-electrocatalyst skill.

Loads historical experiment data from multiple sources and unifies them into
the CSV schema consumed by bo_pipeline.py. Sources (Garnett on "data" D=(x,y)):

  1. Local CSV  (assets/experiment_template.csv)
  2. WorkBuddy task history / workspace memory logs  (.workbuddy/memory/*.md)
  3. Notion experiment logs   (via MCP notion-search / notion-fetch at runtime)
  4. ima knowledge base       (catalyst literature / material databases)

The Notion / ima connectors are MCP tools handled by the WorkBuddy agent at
runtime; this script provides the parsing helpers so those retrieved texts can
be normalized into the standard schema below.

Standard schema columns (extend as needed):
  sample_id, T_C, t_h, pH, E_V, m_cat, x_M1, x_M2, eta_10mA, tafel,
  mass_activity, FE, ECSA, stability_100h_loss, cost, block
"""
from __future__ import annotations
import re, json, csv
from pathlib import Path
from typing import Iterable
import pandas as pd


SCHEMA = [
    "sample_id", "T_C", "t_h", "pH", "E_V", "m_cat",
    "x_M1", "x_M2", "eta_10mA", "tafel",
    "mass_activity", "FE", "ECSA", "stability_100h_loss", "cost", "block",
]


def from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def from_workbuddy_memory(memory_dir: str) -> pd.DataFrame:
    """Parse rows embedded in .workbuddy/memory/*.md daily logs.

    Convention: logs may contain fenced ```bo-table blocks with TSV rows.
    """
    rows = []
    for md in Path(memory_dir).glob("*.md"):
        text = md.read_text(encoding="utf-8")
        for block in re.findall(r"```bo-table\n(.*?)```", text, re.S):
            for line in block.strip().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                rows.append(line.split("\t"))
    if not rows:
        return pd.DataFrame(columns=SCHEMA)
    return pd.DataFrame(rows, columns=SCHEMA[: len(rows[0])])


def from_notion_text(markdown_pages: Iterable[str]) -> pd.DataFrame:
    """Normalize Notion page text (retrieved via MCP notion-fetch) into rows.

    Expects each page to contain a markdown table whose first column is
    sample_id. Returns a stacked DataFrame.
    """
    frames = []
    for txt in markdown_pages:
        # crude markdown table parser
        lines = [l for l in txt.splitlines() if l.strip().startswith("|")]
        if len(lines) < 2:
            continue
        header = [c.strip() for c in lines[0].strip("|").split("|")]
        data = [[c.strip() for c in l.strip("|").split("|")] for l in lines[2:]]
        frames.append(pd.DataFrame(data, columns=header))
    if not frames:
        return pd.DataFrame(columns=SCHEMA)
    return pd.concat(frames, ignore_index=True)


def merge_sources(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Concatenate and de-duplicate by sample_id, keeping the most complete row."""
    df = pd.concat(dfs, ignore_index=True)
    if "sample_id" in df.columns:
        df = df.drop_duplicates("sample_id", keep="first")
    return df.reset_index(drop=True)


def to_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "merged_experiments.csv"
    here = Path(__file__).resolve().parent.parent
    csv_df = from_csv(here / "assets" / "experiment_template.csv")
    merged = merge_sources(csv_df)
    to_csv(merged, out)
    print(f"merged {len(merged)} rows -> {out}")
