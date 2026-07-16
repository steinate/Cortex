#!/usr/bin/env python3
"""Small web server for manually annotating video subtasks."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2


DEFAULT_OUTPUT_DIR = "annotations/manual"

EPISODE_RE = re.compile(r"episode_(\d+)\.mp4$")
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Subtask Annotator</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #111827;
      --muted: #5b6472;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --warn: #b45309;
      --danger: #b91c1c;
      --shadow: 0 1px 2px rgba(17, 24, 39, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.35;
    }
    button, input, textarea, select {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
      min-height: 34px;
    }
    button:hover { border-color: #aab4c3; }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button.primary:hover { background: var(--accent-strong); }
    button.danger { color: var(--danger); }
    input, textarea, select {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 8px;
      background: #fff;
      color: var(--text);
      min-width: 0;
    }
    input[readonly] {
      background: #f3f6fa;
      color: var(--muted);
    }
    textarea {
      resize: vertical;
      min-height: 72px;
    }
    .app {
      display: grid;
      grid-template-columns: 260px minmax(420px, 1fr) 430px;
      gap: 12px;
      min-height: 100vh;
      padding: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      min-height: 44px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-weight: 650;
    }
    .panel-body { padding: 12px; }
    .sidebar-list {
      max-height: calc(100vh - 88px);
      overflow: auto;
    }
    .episode {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      width: 100%;
      padding: 8px 10px;
      border: 0;
      border-bottom: 1px solid #edf0f5;
      border-radius: 0;
      text-align: left;
      background: #fff;
    }
    .episode.active {
      background: #e6f5f3;
      color: #073b36;
      font-weight: 650;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 22px;
      height: 22px;
      border-radius: 999px;
      padding: 0 7px;
      background: #edf0f5;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    .badge.saved {
      background: #dcfce7;
      color: #166534;
    }
    .video-wrap {
      background: #0b1220;
      border-radius: 8px;
      overflow: hidden;
      aspect-ratio: 16 / 9;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    video {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: contain;
      background: #0b1220;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .meta {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfe;
    }
    .meta span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 2px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
      align-items: center;
    }
    .controls input[type="number"] { width: 110px; }
    .controls select { width: 92px; }
    .timeline {
      position: relative;
      height: 36px;
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: linear-gradient(180deg, #f8fafc, #eef2f7);
      overflow: hidden;
      cursor: pointer;
      touch-action: none;
    }
    .timeline.dragging {
      cursor: ew-resize;
      user-select: none;
    }
    .timeline-segment {
      position: absolute;
      top: 6px;
      bottom: 6px;
      min-width: 2px;
      border-radius: 4px;
      background: rgba(15, 118, 110, 0.28);
      border: 1px solid rgba(15, 118, 110, 0.55);
    }
    .timeline-cursor {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      background: #111827;
      transform: translateX(-1px);
      pointer-events: none;
    }
    .timeline-boundary {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 34px;
      transform: translateX(-17px);
      cursor: ew-resize;
      z-index: 3;
      touch-action: none;
      pointer-events: auto;
    }
    .timeline-boundary::before {
      content: "";
      position: absolute;
      left: 16px;
      top: 0;
      bottom: 0;
      width: 2px;
      background: #b91c1c;
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.9);
    }
    .timeline-boundary::after {
      content: "";
      position: absolute;
      left: 10px;
      top: 50%;
      width: 14px;
      height: 18px;
      border-radius: 5px;
      transform: translateY(-50%);
      background: #ffffff;
      border: 1px solid #b91c1c;
      box-shadow: 0 1px 3px rgba(17, 24, 39, 0.18);
    }
    .timeline-boundary:hover::after,
    .timeline-boundary.active::after {
      background: #fee2e2;
      border-color: #991b1b;
    }
    .timeline-start {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      background: var(--warn);
      transform: translateX(-1px);
      pointer-events: none;
    }
    .form-grid {
      display: grid;
      gap: 10px;
    }
    .field label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    .segments {
      display: grid;
      gap: 8px;
      margin-top: 10px;
      max-height: calc(100vh - 365px);
      overflow: auto;
      padding-right: 2px;
    }
    .segment {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      display: grid;
      gap: 8px;
      background: #fff;
    }
    .segment-row {
      display: grid;
      grid-template-columns: 82px 90px minmax(100px, 1fr) auto;
      gap: 8px;
      align-items: center;
    }
    .segment-row input[type="number"] { width: 100%; }
    .segment-text {
      display: grid;
      grid-template-columns: 120px minmax(0, 1fr);
      gap: 8px;
    }
    .segment-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .segment-actions button {
      min-height: 30px;
      padding: 5px 8px;
    }
    .segment-slider-row {
      display: grid;
      grid-template-columns: 1fr;
    }
    .segment-slider-row input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .preview {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0b1220;
      color: #dbeafe;
      padding: 10px;
      overflow: auto;
      max-height: 240px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      white-space: pre;
    }
    .viz-image {
      display: block;
      width: 100%;
      max-height: 360px;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
    }
    .status {
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }
    .status.error { color: var(--danger); }
    .status.ok { color: #166534; }
    @media (max-width: 1180px) {
      .app { grid-template-columns: 220px minmax(360px, 1fr); }
      .right { grid-column: 1 / -1; }
      .segments { max-height: none; }
    }
    @media (max-width: 760px) {
      .app { grid-template-columns: 1fr; }
      .sidebar-list { max-height: 220px; }
      .meta-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .segment-row { grid-template-columns: 1fr 1fr; }
      .segment-row > input[type="text"] { grid-column: 1 / -1; }
      .segment-text { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="panel">
      <div class="panel-header">
        <span>Episodes</span>
        <span id="videoCount" class="status"></span>
      </div>
      <div id="episodes" class="sidebar-list"></div>
    </section>

    <section class="panel">
      <div class="panel-header">
        <span id="currentTitle">Subtask Annotator</span>
        <span id="saveState" class="status"></span>
      </div>
      <div class="panel-body">
        <div class="video-wrap">
          <video id="video" preload="metadata" controls></video>
        </div>
        <div class="meta-grid">
          <div class="meta"><span>Frame</span><strong id="frameNow">0</strong></div>
          <div class="meta"><span>Length</span><strong id="frameLength">0</strong></div>
          <div class="meta"><span>FPS</span><strong id="fpsValue">0</strong></div>
          <div class="meta"><span>Resolution</span><strong id="resolution">-</strong></div>
        </div>
        <div class="controls">
          <button id="playPause" type="button">Play</button>
          <button data-step="-100" type="button">-100</button>
          <button data-step="-10" type="button">-10</button>
          <button data-step="-1" type="button">-1</button>
          <button data-step="1" type="button">+1</button>
          <button data-step="10" type="button">+10</button>
          <button data-step="100" type="button">+100</button>
          <input id="frameInput" type="number" min="0" step="1" value="0">
          <button id="seekFrame" type="button">Go</button>
          <select id="rate">
            <option value="0.25">0.25x</option>
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="2">2x</option>
            <option value="4">4x</option>
          </select>
        </div>
        <div id="timeline" class="timeline"></div>
        <div class="controls">
          <button id="markEnd" class="primary" type="button">Mark subtask end</button>
          <button id="finishToEnd" type="button">Finish to video end</button>
          <span id="pendingStart" class="status">Next start: 0</span>
        </div>
        <div class="controls">
          <span id="selectedBoundary" class="status">Split: none</span>
          <button data-selected-step="-100" type="button">-100</button>
          <button data-selected-step="-10" type="button">-10</button>
          <button data-selected-step="-1" type="button">-1</button>
          <button data-selected-step="1" type="button">+1</button>
          <button data-selected-step="10" type="button">+10</button>
          <button data-selected-step="100" type="button">+100</button>
          <button id="selectedUseCurrent" onclick="useCurrentFrameForSelectedBoundary()" type="button">Use current frame</button>
        </div>
      </div>
    </section>

    <section class="panel right">
      <div class="panel-header">
        <span>Annotation</span>
        <button id="save" class="primary" type="button">Save JSON</button>
      </div>
      <div class="panel-body">
        <div class="form-grid">
          <div class="field">
            <label for="tasks">Tasks, one per line</label>
            <textarea id="tasks"></textarea>
          </div>
        </div>
        <div id="segments" class="segments"></div>
        <div class="field" style="margin-top: 10px;">
          <label>Visualization</label>
          <img id="visualization" class="viz-image" style="display: none;" alt="">
          <div id="vizMissing" class="status">No visualization.</div>
        </div>
        <pre id="preview" class="preview"></pre>
      </div>
    </section>
  </main>

  <datalist id="skillOptions">
    <option value="Pick"></option>
    <option value="Place"></option>
    <option value="Move"></option>
    <option value="Open"></option>
    <option value="Close"></option>
    <option value="Pour"></option>
    <option value="Wipe"></option>
    <option value="Wash"></option>
  </datalist>

  <script>
    const $ = (id) => document.getElementById(id);
    const video = $("video");
    const state = {
      videos: [],
      current: null,
      meta: { length: 0, fps: 0, width: 0, height: 0, episode_index: 0 },
      pendingStart: 0,
      dragBoundary: null,
      selectedBoundary: null,
      suppressTimelineClick: false,
      segments: []
    };

    async function fetchJson(url, options) {
      const response = await fetch(url, options);
      const text = await response.text();
      let data = {};
      if (text) {
        try {
          data = JSON.parse(text);
        } catch (error) {
          throw new Error(text.slice(0, 300));
        }
      }
      if (!response.ok) {
        throw new Error(data.error || response.statusText);
      }
      return data;
    }

    function frameNow() {
      const fps = state.meta.fps || 0;
      if (!fps) return 0;
      const raw = Math.round(video.currentTime * fps);
      return clampFrame(raw);
    }

    function clampFrame(frame) {
      const maxFrame = Math.max(0, (state.meta.length || 1) - 1);
      return Math.min(maxFrame, Math.max(0, Math.round(Number(frame) || 0)));
    }

    function clampBoundary(frame) {
      const maxFrame = Math.max(0, state.meta.length || 0);
      return Math.min(maxFrame, Math.max(0, Math.round(Number(frame) || 0)));
    }

    function contiguousSegments() {
      let start = 0;
      return state.segments.map((segment) => {
        const item = {
          start_frame: start,
          end_frame: clampBoundary(segment.end_frame),
          action_text: segment.action_text || "",
          skill: segment.skill || ""
        };
        start = item.end_frame;
        return item;
      });
    }

    function reflowSegments() {
      state.segments = contiguousSegments();
      state.pendingStart = state.segments.length ? state.segments[state.segments.length - 1].end_frame : 0;
    }

    function updateSegmentInputs() {
      document.querySelectorAll("[data-start-index]").forEach((element) => {
        const index = Number(element.getAttribute("data-start-index"));
        if (Number.isInteger(index) && state.segments[index]) {
          element.value = state.segments[index].start_frame;
        }
      });
      document.querySelectorAll("[data-field='end_frame']").forEach((element) => {
        const index = Number(element.getAttribute("data-index"));
        if (Number.isInteger(index) && state.segments[index]) {
          element.value = state.segments[index].end_frame;
          const minFrame = Number(state.segments[index].start_frame) + 1;
          const maxFrame = state.segments[index + 1]
            ? Number(state.segments[index + 1].end_frame) - 1
            : Number(state.meta.length || state.segments[index].end_frame);
          element.min = String(Math.max(0, minFrame));
          element.max = String(Math.max(minFrame, maxFrame));
        }
      });
    }

    function frameFromTimelineClientX(clientX) {
      const rect = $("timeline").getBoundingClientRect();
      const ratio = (clientX - rect.left) / Math.max(1, rect.width);
      return clampBoundary(Math.round(ratio * (state.meta.length || 0)));
    }

    function frameFromTimelineEvent(event) {
      return frameFromTimelineClientX(event.clientX);
    }

    function nearestBoundaryIndex(clientX, maxDistancePx = 48) {
      if (state.segments.length < 2) return null;
      const rect = $("timeline").getBoundingClientRect();
      const length = Math.max(1, state.meta.length || 1);
      let bestIndex = null;
      let bestDistance = Infinity;
      state.segments.slice(0, -1).forEach((segment, index) => {
        const x = rect.left + (clampBoundary(segment.end_frame) / length) * rect.width;
        const distance = Math.abs(clientX - x);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIndex = index;
        }
      });
      return bestDistance <= maxDistancePx ? bestIndex : null;
    }

    function clampSplitFrame(index, frame) {
      if (!state.segments[index] || !state.segments[index + 1]) {
        return clampBoundary(frame);
      }
      const minFrame = Number(state.segments[index].start_frame) + 1;
      const maxFrame = Number(state.segments[index + 1].end_frame) - 1;
      if (maxFrame < minFrame) {
        return Number(state.segments[index].end_frame);
      }
      return Math.min(maxFrame, Math.max(minFrame, clampBoundary(frame)));
    }

    function updateTimelineElements() {
      const root = $("timeline");
      const length = Math.max(1, state.meta.length || 1);
      state.segments.forEach((segment, index) => {
        const start = clampBoundary(segment.start_frame);
        const end = clampBoundary(segment.end_frame);
        const segmentEl = root.querySelector(`[data-segment-index="${index}"]`);
        if (segmentEl) {
          segmentEl.style.left = `${(start / length) * 100}%`;
          segmentEl.style.width = `${Math.max(0.15, ((end - start) / length) * 100)}%`;
          segmentEl.title = `${index + 1}: ${start}-${end} ${segment.skill || ""}`;
        }
      });
      root.querySelectorAll("[data-boundary-index]").forEach((handle) => {
        const index = Number(handle.getAttribute("data-boundary-index"));
        if (Number.isInteger(index) && state.segments[index]) {
          const boundary = clampBoundary(state.segments[index].end_frame);
          handle.style.left = `${(boundary / length) * 100}%`;
          handle.title = `Drag split ${index + 1}: frame ${boundary}`;
        }
      });
      const startLine = $("timelineStartLine");
      if (startLine) {
        startLine.style.left = `${(clampBoundary(state.pendingStart) / length) * 100}%`;
      }
      const cursor = $("timelineCursor");
      if (cursor) {
        cursor.style.left = `${(frameNow() / length) * 100}%`;
      }
    }

    function setSplitFrame(index, frame, redrawTimeline = true) {
      if (!state.segments[index] || !state.segments[index + 1]) return;
      state.selectedBoundary = index;
      state.segments[index].end_frame = clampSplitFrame(index, frame);
      reflowSegments();
      updateSegmentInputs();
      renderMeta();
      updateSelectedBoundaryLabel();
      if (redrawTimeline) {
        renderTimeline();
      } else {
        updateTimelineElements();
      }
      $("preview").textContent = JSON.stringify(annotationPayload(), null, 2);
    }

    function startBoundaryDrag(index, pointerId, event) {
      if (!state.segments[index] || !state.segments[index + 1]) return;
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      state.selectedBoundary = index;
      updateSelectedBoundaryLabel();
      state.dragBoundary = { index, pointerId, moved: false };
      $("timeline").classList.add("dragging");
    }

    function startNearestBoundaryDrag(clientX, pointerId, event) {
      const index = nearestBoundaryIndex(clientX);
      if (index === null) return false;
      startBoundaryDrag(index, pointerId, event);
      return true;
    }

    function updateBoundaryDrag(clientX) {
      if (!state.dragBoundary) return;
      state.dragBoundary.moved = true;
      setSplitFrame(state.dragBoundary.index, frameFromTimelineClientX(clientX), false);
      setStatus(`Split ${state.dragBoundary.index + 1}: frame ${state.segments[state.dragBoundary.index].end_frame}`);
    }

    function endBoundaryDrag() {
      if (!state.dragBoundary) return;
      state.suppressTimelineClick = Boolean(state.dragBoundary.moved);
      setStatus(`Split ${state.dragBoundary.index + 1} moved to frame ${state.segments[state.dragBoundary.index].end_frame}.`);
      state.dragBoundary = null;
      $("timeline").classList.remove("dragging");
      renderTimeline();
    }

    function moveBoundary(index, delta) {
      if (!state.segments[index] || !state.segments[index + 1]) return;
      setSplitFrame(index, Number(state.segments[index].end_frame) + Number(delta || 0));
      setStatus(`Split ${index + 1}: frame ${state.segments[index].end_frame}`);
    }

    function applyEndFrameInput(index) {
      const input = document.querySelector(`input[type="number"][data-field="end_frame"][data-index="${index}"]`);
      if (!input || !state.segments[index]) return;
      if (index < state.segments.length - 1) {
        setSplitFrame(index, input.value);
        setStatus(`Split ${index + 1}: frame ${state.segments[index].end_frame}`);
      } else {
        state.segments[index].end_frame = clampBoundary(input.value);
        refreshDerivedViews();
        setStatus(`Final end frame: ${state.segments[index].end_frame}`);
      }
    }

    function setBoundaryToCurrentFrame(index) {
      if (!state.segments[index]) return;
      if (index < state.segments.length - 1) {
        setSplitFrame(index, frameNow());
        setStatus(`Split ${index + 1}: frame ${state.segments[index].end_frame}`);
      } else {
        state.segments[index].end_frame = clampBoundary(frameNow());
        refreshDerivedViews();
        setStatus(`Final end frame: ${state.segments[index].end_frame}`);
      }
    }

    function insertSplitAtFrame(frame) {
      const splitFrame = clampBoundary(frame);
      reflowSegments();
      if (!state.segments.length) {
        if (splitFrame <= 0 || splitFrame >= state.meta.length) {
          setStatus("Split frame must be inside the video.", "error");
          return;
        }
        state.segments = [
          { start_frame: 0, end_frame: splitFrame, action_text: "", skill: "" },
          { start_frame: splitFrame, end_frame: state.meta.length, action_text: "", skill: "" }
        ];
        state.selectedBoundary = 0;
        renderSegments();
        renderMeta();
        updateSelectedBoundaryLabel();
        setStatus(`Inserted split at frame ${splitFrame}.`);
        return;
      }

      const index = state.segments.findIndex((segment) => {
        return Number(segment.start_frame) < splitFrame && splitFrame < Number(segment.end_frame);
      });
      if (index < 0) {
        const existing = state.segments.findIndex((segment, segmentIndex) => {
          return segmentIndex < state.segments.length - 1 && Number(segment.end_frame) === splitFrame;
        });
        if (existing >= 0) {
          selectBoundary(existing);
          setStatus(`Selected existing split ${existing + 1}.`);
        } else {
          setStatus("Split frame must be inside an existing segment.", "error");
        }
        return;
      }

      const oldEnd = Number(state.segments[index].end_frame);
      state.segments[index].end_frame = splitFrame;
      state.segments.splice(index + 1, 0, {
        start_frame: splitFrame,
        end_frame: oldEnd,
        action_text: "",
        skill: ""
      });
      state.selectedBoundary = index;
      reflowSegments();
      renderSegments();
      renderMeta();
      updateSelectedBoundaryLabel();
      setStatus(`Inserted split ${index + 1} at frame ${splitFrame}.`);
    }

    function moveSelectedBoundary(delta) {
      if (state.selectedBoundary === null) {
        setStatus("Select a split first.", "error");
        return;
      }
      moveBoundary(state.selectedBoundary, delta);
    }

    function useCurrentFrameForSelectedBoundary() {
      if (state.selectedBoundary === null) {
        setStatus("Select a split first.", "error");
        return;
      }
      setSplitFrame(state.selectedBoundary, frameNow());
      setStatus(`Split ${state.selectedBoundary + 1}: frame ${state.segments[state.selectedBoundary].end_frame}`);
    }

    window.applyEndFrameInput = applyEndFrameInput;
    window.setBoundaryToCurrentFrame = setBoundaryToCurrentFrame;
    window.moveBoundary = moveBoundary;
    window.moveSelectedBoundary = moveSelectedBoundary;
    window.selectBoundary = selectBoundary;
    window.setSplitFrame = setSplitFrame;
    window.setStatus = setStatus;
    window.useCurrentFrameForSelectedBoundary = useCurrentFrameForSelectedBoundary;

    function seekToFrame(frame) {
      if (!state.meta.fps) return;
      const nextFrame = clampFrame(frame);
      video.currentTime = nextFrame / state.meta.fps;
      updateCurrentFrame(nextFrame);
    }

    function setStatus(message, kind = "") {
      const el = $("saveState");
      el.textContent = message;
      el.className = `status ${kind}`;
    }

    function selectedBoundaryText() {
      if (state.selectedBoundary === null || !state.segments[state.selectedBoundary]) {
        return "Split: none";
      }
      return `Split ${state.selectedBoundary + 1}: frame ${state.segments[state.selectedBoundary].end_frame}`;
    }

    function updateSelectedBoundaryLabel() {
      const el = $("selectedBoundary");
      if (el) el.textContent = selectedBoundaryText();
    }

    function selectBoundary(index) {
      if (!Number.isInteger(index) || !state.segments[index] || !state.segments[index + 1]) {
        state.selectedBoundary = null;
      } else {
        state.selectedBoundary = index;
      }
      updateSelectedBoundaryLabel();
      renderTimeline();
    }

    function annotationPayload() {
      const tasks = $("tasks").value
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
      return {
        episode_index: state.meta.episode_index,
        tasks,
        length: state.meta.length,
        action_config: contiguousSegments()
      };
    }

    function renderEpisodes() {
      $("videoCount").textContent = `${state.videos.length}`;
      const root = $("episodes");
      root.innerHTML = "";
      state.videos.forEach((item) => {
        const button = document.createElement("button");
        button.className = `episode ${state.current === item.filename ? "active" : ""}`;
        button.type = "button";
        button.innerHTML = `<span>${item.filename}</span><span class="badge ${item.annotated ? "saved" : ""}">${item.annotated ? "OK" : ""}</span>`;
        button.addEventListener("click", () => loadEpisode(item.filename));
        root.appendChild(button);
      });
    }

    function renderMeta() {
      $("currentTitle").textContent = state.current || "Subtask Annotator";
      $("frameLength").textContent = state.meta.length || 0;
      $("fpsValue").textContent = state.meta.fps ? state.meta.fps.toFixed(3) : "0";
      $("resolution").textContent = state.meta.width ? `${state.meta.width}x${state.meta.height}` : "-";
      $("pendingStart").textContent = `Next start: ${state.pendingStart} / ${state.meta.length || 0}`;
      updateCurrentFrame(frameNow());
    }

    function renderTimeline() {
      const root = $("timeline");
      root.innerHTML = "";
      const length = Math.max(1, state.meta.length || 1);
      state.segments.forEach((segment, index) => {
        const start = clampBoundary(segment.start_frame);
        const end = clampBoundary(segment.end_frame);
        const left = (start / length) * 100;
        const width = Math.max(0.15, ((end - start) / length) * 100);
        const div = document.createElement("div");
        div.className = "timeline-segment";
        div.style.left = `${left}%`;
        div.style.width = `${width}%`;
        div.dataset.segmentIndex = String(index);
        div.title = `${index + 1}: ${start}-${end} ${segment.skill || ""}`;
        root.appendChild(div);
      });
      state.segments.slice(0, -1).forEach((segment, index) => {
        const boundary = clampBoundary(segment.end_frame);
        const handle = document.createElement("div");
        const active = (state.dragBoundary && state.dragBoundary.index === index) || state.selectedBoundary === index;
        handle.className = `timeline-boundary ${active ? "active" : ""}`;
        handle.style.left = `${(boundary / length) * 100}%`;
        handle.dataset.boundaryIndex = String(index);
        handle.title = `Drag split ${index + 1}: frame ${boundary}`;
        root.appendChild(handle);
      });
      const startLine = document.createElement("div");
      startLine.className = "timeline-start";
      startLine.id = "timelineStartLine";
      startLine.style.left = `${(clampBoundary(state.pendingStart) / length) * 100}%`;
      root.appendChild(startLine);
      const cursor = document.createElement("div");
      cursor.className = "timeline-cursor";
      cursor.id = "timelineCursor";
      cursor.style.left = `${(frameNow() / length) * 100}%`;
      root.appendChild(cursor);
    }

    function updateCurrentFrame(value) {
      const frame = clampFrame(value);
      $("frameNow").textContent = frame;
      $("frameInput").value = frame;
      const cursor = $("timelineCursor");
      if (cursor) {
        const length = Math.max(1, state.meta.length || 1);
        cursor.style.left = `${(frame / length) * 100}%`;
      }
    }

    function renderSegments() {
      reflowSegments();
      const root = $("segments");
      root.innerHTML = "";
      if (!state.segments.length) {
        const empty = document.createElement("div");
        empty.className = "status";
        empty.textContent = "No segments yet.";
        root.appendChild(empty);
      }
      state.segments.forEach((segment, index) => {
        const sliderMin = Number(segment.start_frame) + 1;
        const sliderMax = state.segments[index + 1]
          ? Number(state.segments[index + 1].end_frame) - 1
          : Number(state.meta.length || segment.end_frame);
        const canSlide = index < state.segments.length - 1 && sliderMax >= sliderMin;
        const row = document.createElement("div");
        row.className = "segment";
        row.innerHTML = `
          <div class="segment-row">
            <input data-start-index="${index}" type="number" readonly value="${segment.start_frame}" title="Auto start">
            <input data-field="end_frame" data-index="${index}" type="number" min="0" max="${state.meta.length || 0}" step="1" value="${segment.end_frame}" onchange="applyEndFrameInput(${index})">
            <input data-field="skill" data-index="${index}" type="text" list="skillOptions" placeholder="Skill" value="${escapeHtml(segment.skill || "")}">
            <button class="danger" data-delete="${index}" type="button">Delete</button>
          </div>
          <div class="segment-text">
            <input data-field="action_text" data-index="${index}" type="text" placeholder="Action text" value="${escapeHtml(segment.action_text || "")}">
            <div class="segment-actions">
              <button data-seek-start="${index}" type="button">Seek start</button>
              <button data-seek-end="${index}" type="button">Seek end</button>
              <button data-use-end="${index}" type="button">Use frame as end</button>
              ${index < state.segments.length - 1 ? `
              <button data-select-boundary="${index}" type="button">Select split</button>
              <button data-boundary-index="${index}" data-boundary-delta="-10" type="button">-10</button>
              <button data-boundary-index="${index}" data-boundary-delta="-1" type="button">-1</button>
              <button data-boundary-index="${index}" data-boundary-delta="1" type="button">+1</button>
              <button data-boundary-index="${index}" data-boundary-delta="10" type="button">+10</button>
              <button onclick="applyEndFrameInput(${index})" type="button">Set</button>` : ""}
            </div>
          </div>
          ${canSlide ? `
          <div class="segment-slider-row">
            <input data-field="end_frame" data-index="${index}" type="range" min="${sliderMin}" max="${sliderMax}" step="1" value="${segment.end_frame}" title="Drag subtask split" oninput="setSplitFrame(${index}, this.value, false)" onchange="setSplitFrame(${index}, this.value, true)">
          </div>` : ""}`;
        root.appendChild(row);
      });
      $("preview").textContent = JSON.stringify(annotationPayload(), null, 2);
      renderTimeline();
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    async function loadVideos() {
      const data = await fetchJson("/api/videos");
      state.videos = data.videos;
      renderEpisodes();
      if (state.videos.length) {
        await loadEpisode(state.videos[0].filename);
      }
    }

    async function loadEpisode(filename) {
      state.current = filename;
      renderEpisodes();
      setStatus("Loading...");
      const encoded = encodeURIComponent(filename);
      const [meta, annotation] = await Promise.all([
        fetchJson(`/api/video/${encoded}/meta`),
        fetchJson(`/api/annotation/${encoded}`)
      ]);
      state.meta = meta;
      state.segments = (annotation.action_config || []).map((segment) => ({ ...segment }));
      reflowSegments();
      $("tasks").value = (annotation.tasks || []).join("\n");
      video.src = `/video/${encoded}`;
      video.load();
      if (meta.visualization_exists) {
        $("visualization").src = `${meta.visualization_url}?t=${Date.now()}`;
        $("visualization").style.display = "block";
        $("vizMissing").style.display = "none";
      } else {
        $("visualization").removeAttribute("src");
        $("visualization").style.display = "none";
        $("vizMissing").style.display = "block";
      }
      state.selectedBoundary = null;
      renderMeta();
      updateSelectedBoundaryLabel();
      renderSegments();
      setStatus(annotation.exists ? "Loaded saved JSON" : "New JSON", annotation.exists ? "ok" : "");
    }

    function refreshDerivedViews() {
      reflowSegments();
      updateSegmentInputs();
      renderMeta();
      renderTimeline();
      $("preview").textContent = JSON.stringify(annotationPayload(), null, 2);
    }

    function markSubtaskEnd() {
      const start = state.pendingStart;
      const end = frameNow();
      if (end <= start) {
        setStatus("End point must be greater than the previous boundary.", "error");
        return;
      }
      state.segments.push({
        start_frame: start,
        end_frame: end,
        action_text: "",
        skill: ""
      });
      renderSegments();
      renderMeta();
      setStatus("Subtask endpoint added.");
    }

    function finishToVideoEnd() {
      const start = state.pendingStart;
      const end = state.meta.length || 0;
      if (end <= start) {
        setStatus("The annotation already reaches the video end.", "ok");
        return;
      }
      state.segments.push({
        start_frame: start,
        end_frame: end,
        action_text: "",
        skill: ""
      });
      renderSegments();
      renderMeta();
      setStatus("Final subtask reaches video end.");
    }

    async function saveAnnotation() {
      if (!state.current) return;
      refreshDerivedViews();
      if (!state.segments.length || state.pendingStart !== state.meta.length) {
        setStatus(`Annotation must cover frames 0-${state.meta.length}. Use Finish to video end for the final subtask.`, "error");
        return;
      }
      setStatus("Saving...");
      const encoded = encodeURIComponent(state.current);
      try {
        const data = await fetchJson(`/api/annotation/${encoded}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(annotationPayload())
        });
        const item = state.videos.find((videoItem) => videoItem.filename === state.current);
        if (item) item.annotated = true;
        renderEpisodes();
        setStatus(`Saved ${data.path}`, "ok");
      } catch (error) {
        setStatus(error.message, "error");
      }
    }

    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const step = target.getAttribute("data-step");
      if (step !== null) seekToFrame(frameNow() + Number(step));
      const selectedStep = target.getAttribute("data-selected-step");
      if (selectedStep !== null) moveSelectedBoundary(Number(selectedStep));

      const indexForDelete = target.getAttribute("data-delete");
      if (indexForDelete !== null) {
        state.segments.splice(Number(indexForDelete), 1);
        state.selectedBoundary = null;
        renderSegments();
        renderMeta();
        updateSelectedBoundaryLabel();
      }
      const selectBoundaryIndex = target.getAttribute("data-select-boundary");
      if (selectBoundaryIndex !== null) {
        selectBoundary(Number(selectBoundaryIndex));
        setStatus(`Selected split ${Number(selectBoundaryIndex) + 1}.`);
      }
      const boundaryIndex = target.getAttribute("data-boundary-index");
      const boundaryDelta = target.getAttribute("data-boundary-delta");
      if (boundaryIndex !== null && boundaryDelta !== null) {
        moveBoundary(Number(boundaryIndex), Number(boundaryDelta));
      }
      const seekStart = target.getAttribute("data-seek-start");
      if (seekStart !== null) seekToFrame(state.segments[Number(seekStart)].start_frame);
      const seekEnd = target.getAttribute("data-seek-end");
      if (seekEnd !== null) seekToFrame(Math.min(state.segments[Number(seekEnd)].end_frame, Math.max(0, (state.meta.length || 1) - 1)));
      const useEnd = target.getAttribute("data-use-end");
      if (useEnd !== null) {
        const index = Number(useEnd);
        if (index < state.segments.length - 1) {
          setSplitFrame(index, frameNow());
        } else {
          state.segments[index].end_frame = clampBoundary(frameNow());
          refreshDerivedViews();
        }
      }
    });

    $("segments").addEventListener("input", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLInputElement)) return;
      const index = Number(target.dataset.index);
      const field = target.dataset.field;
      if (!Number.isInteger(index) || !field) return;
      if (field === "end_frame") {
        if (index < state.segments.length - 1) {
          setSplitFrame(index, target.value);
        } else {
          state.segments[index][field] = clampBoundary(target.value);
          refreshDerivedViews();
        }
      } else {
        state.segments[index][field] = target.value;
        $("preview").textContent = JSON.stringify(annotationPayload(), null, 2);
        renderTimeline();
      }
    });

    $("tasks").addEventListener("input", () => {
      $("preview").textContent = JSON.stringify(annotationPayload(), null, 2);
    });
    $("playPause").addEventListener("click", async () => {
      if (video.paused) {
        await video.play();
      } else {
        video.pause();
      }
    });
    video.addEventListener("play", () => { $("playPause").textContent = "Pause"; });
    video.addEventListener("pause", () => { $("playPause").textContent = "Play"; });
    video.addEventListener("timeupdate", () => updateCurrentFrame(frameNow()));
    video.addEventListener("seeked", () => updateCurrentFrame(frameNow()));
    $("seekFrame").addEventListener("click", () => seekToFrame($("frameInput").value));
    $("frameInput").addEventListener("keydown", (event) => {
      if (event.key === "Enter") seekToFrame($("frameInput").value);
    });
    $("rate").addEventListener("change", () => { video.playbackRate = Number($("rate").value); });
    $("markEnd").addEventListener("click", markSubtaskEnd);
    $("finishToEnd").addEventListener("click", finishToVideoEnd);
    $("save").addEventListener("click", saveAnnotation);
    $("selectedUseCurrent").addEventListener("click", useCurrentFrameForSelectedBoundary);
    $("timeline").addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      if (startNearestBoundaryDrag(event.clientX, event.pointerId, event)) {
        $("timeline").setPointerCapture(event.pointerId);
      }
    });
    $("timeline").addEventListener("click", (event) => {
      if (state.suppressTimelineClick) {
        state.suppressTimelineClick = false;
        return;
      }
      const nearest = nearestBoundaryIndex(event.clientX, 48);
      if (nearest !== null) {
        selectBoundary(nearest);
        setStatus(`Selected split ${nearest + 1}.`);
        return;
      }
      const frame = frameFromTimelineEvent(event);
      seekToFrame(frame);
      insertSplitAtFrame(frame);
    });
    $("timeline").addEventListener("pointermove", (event) => {
      if (!state.dragBoundary) return;
      if (event.pointerId !== state.dragBoundary.pointerId) return;
      event.preventDefault();
      updateBoundaryDrag(event.clientX);
    });
    $("timeline").addEventListener("pointerup", (event) => {
      if (!state.dragBoundary || event.pointerId !== state.dragBoundary.pointerId) return;
      if ($("timeline").hasPointerCapture(event.pointerId)) {
        $("timeline").releasePointerCapture(event.pointerId);
      }
      endBoundaryDrag();
    });
    $("timeline").addEventListener("pointercancel", (event) => {
      if (!state.dragBoundary || event.pointerId !== state.dragBoundary.pointerId) return;
      endBoundaryDrag();
    });
    document.addEventListener("keydown", (event) => {
      const activeTag = document.activeElement ? document.activeElement.tagName : "";
      if (["INPUT", "TEXTAREA", "SELECT"].includes(activeTag)) return;
      if (event.code === "Space") {
        event.preventDefault();
        $("playPause").click();
      } else if (event.key === "ArrowLeft") {
        seekToFrame(frameNow() - (event.shiftKey ? 10 : 1));
      } else if (event.key === "ArrowRight") {
        seekToFrame(frameNow() + (event.shiftKey ? 10 : 1));
      } else if (event.key === "]") {
        $("markEnd").click();
      } else if (event.key === "\\") {
        $("finishToEnd").click();
      }
    });

    loadVideos().catch((error) => setStatus(error.message, "error"));
  </script>
</body>
</html>
"""


class AnnotationError(ValueError):
    pass


def episode_index(filename: str) -> int:
    match = EPISODE_RE.match(filename)
    if not match:
        raise AnnotationError(f"invalid episode filename: {filename}")
    return int(match.group(1))


def video_metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AnnotationError(f"failed to open video: {path}")
    try:
        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    return {
        "filename": path.name,
        "episode_index": episode_index(path.name),
        "length": length,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": length / fps if fps > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True, help="Directory containing episode_*.mp4 videos.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where episode_*.json files are saved.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP bind port.")
    return parser.parse_args()


class AnnotatorHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], video_dir: Path, output_dir: Path):
        super().__init__(server_address, handler_class)
        self.video_dir = video_dir
        self.output_dir = output_dir


class Handler(BaseHTTPRequestHandler):
    server: AnnotatorHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/videos":
                self.api_videos()
            elif path.startswith("/api/video/") and path.endswith("/meta"):
                filename = unquote(path[len("/api/video/") : -len("/meta")])
                self.api_video_meta(filename)
            elif path.startswith("/api/annotation/"):
                filename = unquote(path[len("/api/annotation/") :])
                self.api_get_annotation(filename)
            elif path.startswith("/visualization/"):
                filename = unquote(path[len("/visualization/") :])
                self.serve_visualization(filename)
            elif path.startswith("/video/"):
                filename = unquote(path[len("/video/") :])
                self.serve_video(filename)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
        except AnnotationError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except BrokenPipeError:
            return
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path.startswith("/api/annotation/"):
                filename = unquote(path[len("/api/annotation/") :])
                self.api_save_annotation(filename)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
        except AnnotationError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def safe_video_path(self, filename: str) -> Path:
        if not filename or "/" in filename or "\\" in filename:
            raise AnnotationError("invalid filename")
        path = self.server.video_dir / filename
        if not path.exists() or not path.is_file() or path.suffix.lower() != ".mp4":
            raise AnnotationError(f"video not found: {filename}")
        episode_index(filename)
        return path

    def annotation_path(self, filename: str) -> Path:
        self.safe_video_path(filename)
        return self.server.output_dir / f"{Path(filename).stem}.json"

    def visualization_path(self, filename: str) -> Path:
        self.safe_video_path(filename)
        return self.server.output_dir.parent / "subtask_annotation_visualizations" / f"{Path(filename).stem}.jpg"

    def api_videos(self) -> None:
        videos = []
        for path in sorted(self.server.video_dir.glob("episode_*.mp4")):
            try:
                index = episode_index(path.name)
            except AnnotationError:
                continue
            annotation_path = self.server.output_dir / f"{path.stem}.json"
            videos.append(
                {
                    "filename": path.name,
                    "episode_index": index,
                    "annotated": annotation_path.exists(),
                }
            )
        self.send_json({"videos": videos, "video_dir": str(self.server.video_dir), "output_dir": str(self.server.output_dir)})

    def api_video_meta(self, filename: str) -> None:
        path = self.safe_video_path(filename)
        meta = video_metadata(path)
        meta["annotation_path"] = str(self.annotation_path(filename))
        meta["annotation_exists"] = self.annotation_path(filename).exists()
        viz_path = self.visualization_path(filename)
        meta["visualization_path"] = str(viz_path)
        meta["visualization_exists"] = viz_path.exists()
        meta["visualization_url"] = f"/visualization/{Path(filename).stem}.jpg"
        self.send_json(meta)

    def api_get_annotation(self, filename: str) -> None:
        path = self.annotation_path(filename)
        if not path.exists():
            meta = video_metadata(self.safe_video_path(filename))
            self.send_json(
                {
                    "exists": False,
                    "episode_index": meta["episode_index"],
                    "tasks": [],
                    "length": meta["length"],
                    "action_config": [],
                }
            )
            return
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["exists"] = True
        self.send_json(data)

    def api_save_annotation(self, filename: str) -> None:
        length_header = self.headers.get("Content-Length")
        if not length_header:
            raise AnnotationError("missing request body")
        try:
            content_length = int(length_header)
        except ValueError as exc:
            raise AnnotationError("invalid Content-Length") from exc
        if content_length <= 0:
            raise AnnotationError("missing request body")
        if content_length > 10 * 1024 * 1024:
            raise AnnotationError("request body exceeds 10 MiB")
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AnnotationError("request body must be valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise AnnotationError("request body must be a JSON object")
        annotation = self.normalize_annotation(filename, payload)
        path = self.annotation_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(annotation, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, path)
        self.send_json({"ok": True, "path": str(path), "annotation": annotation})

    def normalize_annotation(self, filename: str, payload: dict) -> dict:
        meta = video_metadata(self.safe_video_path(filename))
        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list):
            raise AnnotationError("tasks must be a list")
        normalized_tasks = [str(task).strip() for task in tasks if str(task).strip()]

        configs = payload.get("action_config", [])
        if not isinstance(configs, list):
            raise AnnotationError("action_config must be a list")
        normalized_configs = []
        expected_start = 0
        for index, item in enumerate(configs):
            if not isinstance(item, dict):
                raise AnnotationError(f"action_config[{index}] must be an object")
            try:
                end_frame = int(item.get("end_frame"))
            except (TypeError, ValueError) as exc:
                raise AnnotationError(f"invalid frame in action_config[{index}]") from exc
            if end_frame < 0:
                raise AnnotationError(f"negative frame in action_config[{index}]")
            if end_frame <= expected_start:
                raise AnnotationError(f"end_frame must be greater than start_frame in action_config[{index}]")
            if meta["length"] and end_frame > meta["length"]:
                raise AnnotationError(f"end_frame exceeds video length in action_config[{index}]")
            normalized_configs.append(
                {
                    "seg_id": index,
                    "start_frame": expected_start,
                    "end_frame": end_frame,
                    "action_text": str(item.get("action_text", "")).strip(),
                    "skill": str(item.get("skill", "")).strip(),
                }
            )
            expected_start = end_frame
        if meta["length"]:
            if not normalized_configs:
                raise AnnotationError("action_config must cover the whole video")
            if expected_start != meta["length"]:
                raise AnnotationError(f"last end_frame must equal video length ({meta['length']})")
        return {
            "episode_index": meta["episode_index"],
            "tasks": normalized_tasks,
            "length": meta["length"],
            "fps": meta["fps"],
            "action_config": normalized_configs,
        }

    def serve_video(self, filename: str) -> None:
        path = self.safe_video_path(filename)
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        if not range_header:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.copy_file_range(path, 0, file_size)
            return

        match = RANGE_RE.match(range_header.strip())
        if not match:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        start_raw, end_raw = match.groups()
        if start_raw == "" and end_raw == "":
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        if start_raw == "":
            suffix_length = int(end_raw)
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        else:
            start = int(start_raw)
            end = int(end_raw) if end_raw else file_size - 1
        if start >= file_size or end < start:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        end = min(end, file_size - 1)
        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.copy_file_range(path, start, length)

    def serve_visualization(self, filename: str) -> None:
        if not filename.endswith(".jpg"):
            raise AnnotationError("invalid visualization filename")
        stem = Path(filename).stem
        video_filename = f"{stem}.mp4"
        path = self.visualization_path(video_filename)
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "visualization not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        self.copy_file_range(path, 0, path.stat().st_size)

    def copy_file_range(self, path: Path, start: int, length: int) -> None:
        remaining = length
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_bytes(data, "application/json; charset=utf-8", status)

    def send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not video_dir.exists():
        raise SystemExit(f"video directory does not exist: {video_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    server = AnnotatorHTTPServer((args.host, args.port), Handler, video_dir, output_dir)
    print(f"Serving subtask annotator at http://{args.host}:{args.port}", flush=True)
    print(f"Video directory: {video_dir}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
