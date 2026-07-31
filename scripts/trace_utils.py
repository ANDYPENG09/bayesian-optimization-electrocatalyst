#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yu Peng
"""
TRACE — auditable call-chain tracing for the BO-electrocatalyst skill.

Implements a lightweight span recorder so every pipeline stage is auditable and
the optimization chain is fully reconstructable (reproducibility / AI-grading):

  T  race id ...... run-level UUID linking all spans of one BO round
  R  ecord ........ every stage emits a span (load -> preprocess -> surrogate
                    -> acquisition -> recommend -> assess -> persist)
  A  ttributes .... per span: parent linkage, duration_ms, input/output
                    summaries, status
  C  hain ......... spans form a linked list via parent_id (reconstruct order)
  E  mit .......... written to trace.json (machine-readable) alongside result

Usage (see bo_pipeline.py::run):
    tr = TraceRecorder()
    with tr.span("fit_gp", input_summary="Matern5/2 + ARD"):
        opt = fit_gp(space, X, y)
    tr.save("trace.json")
"""
from __future__ import annotations

import time
import uuid
import json
from contextlib import contextmanager
from dataclasses import dataclass, asdict


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    step: str
    status: str
    started_at: float
    duration_ms: float
    input_summary: str
    output_summary: str
    notes: str


class TraceRecorder:
    """Record sibling/nested spans for one BO run and emit trace.json.

    The span is appended to ``self.spans`` immediately on entry so that
    ``record_output`` / ``set_status`` called *inside* the ``with`` block write
    to the *current* span (index-based), not the previously finished one.
    """

    def __init__(self, trace_id: str | None = None, source: str = "bo_pipeline"):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.source = source
        self.spans: list[Span] = []
        self._active: list[int] = []  # span indices currently open (nesting)

    @contextmanager
    def span(self, step: str, input_summary: str = "", notes: str = "",
             parent: str | None = None):
        sid = uuid.uuid4().hex[:12]
        # default linkage: nest under the currently-open span (proper call
        # tree, run = root); otherwise chain to the last closed span.
        if parent is None:
            parent = self._active[-1] and self.spans[self._active[-1]].span_id \
                if self._active else (
                    self.spans[-1].span_id if self.spans else None)
        t0 = time.time()
        idx = len(self.spans)
        self.spans.append(Span(sid, parent, step, "ok", t0, 0.0,
                               input_summary, "", notes))
        self._active.append(idx)
        try:
            yield
        finally:
            self._active.pop()
            self.spans[idx].duration_ms = (time.time() - t0) * 1000.0

    def record_output(self, output_summary: str) -> None:
        """Attach a compact output summary to the most recent open span."""
        if self._active:
            self.spans[self._active[-1]].output_summary = output_summary
        elif self.spans:
            self.spans[-1].output_summary = output_summary

    def set_status(self, status: str, notes: str = "") -> None:
        """Override the status/notes of the most recent open span (e.g. error)."""
        if self._active:
            s = self.spans[self._active[-1]]
            s.status = status
            if notes:
                s.notes = notes
        elif self.spans:
            self.spans[-1].status = status
            if notes:
                self.spans[-1].notes = notes

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "source": self.source,
            "n_spans": len(self.spans),
            "spans": [asdict(s) for s in self.spans],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
