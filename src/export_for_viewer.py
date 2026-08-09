"""
export_for_viewer.py

Step 8a: Export recorded sign samples into a JSON file for 3D playback
in a web-based viewer (sign_viewer.html).

For each vocabulary class in data/raw/, picks ONE representative sample
and converts its (30, 126) landmark array into a more explicit
per-frame, per-hand, per-joint structure that's easy for JavaScript to
animate.

Sample selection:
    By default (--strategy best) every sample file for a class is
    scored by how many of its 30 frames have at least one hand
    detected, and the highest-scoring sample is exported. This avoids
    exporting a mostly-empty recording just because it happened to be
    first alphabetically. Pass --strategy first to restore the old
    "first file found" behavior.

Coordinate note: our normalized coordinates have y increasing DOWNWARD
(image convention), but 3D viewers typically expect y increasing UPWARD.
We flip the y-axis here so the hand appears right-side-up in the viewer.

Viewer template:
    The HTML below is embedded as a single string with one placeholder,
    VOCAB_PLACEHOLDER, that gets swapped out for the real vocabulary
    JSON via a plain str.replace(). Earlier versions used str.format(),
    which meant every literal "{" and "}" in the CSS/JS had to be
    doubled - fragile to edit. Plain substitution avoids that entirely,
    so the HTML/CSS/JS below reads and edits like a normal template.

Output:
    docs/vocabulary_export.json
    docs/sign_viewer.html   (unless --json-only is passed; uses the licensed OBJ)

Usage:
    python src/export_for_viewer.py
    python src/export_for_viewer.py --strategy first
    python src/export_for_viewer.py --json-only
"""

import argparse
import json
import os
import shutil

import numpy as np

from dataset_io import DATA_DIR, SEQUENCE_LENGTH
from landmark_utils import FEATURES_PER_HAND, NUM_LANDMARKS, COORDS_PER_LANDMARK

OUTPUT_JSON_PATH = "docs/vocabulary_export.json"
OUTPUT_HTML_PATH = "docs/sign_viewer.html"
VOCAB_PLACEHOLDER = "__VOCAB_JSON__"
MODEL_URL_PLACEHOLDER = "__MODEL_URL__"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
HAND_TASK_SOURCE = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
LABEL_MAP_SOURCE = os.path.join(PROJECT_ROOT, "models", "label_map.json")

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SignSpeak - Sign Viewer</title>
<link rel="manifest" href="manifest.webmanifest">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<style>
  :root {
    --bg: #0a0c0f;
    --bg-grain: radial-gradient(circle at 18% 12%, rgba(255,157,92,0.05), transparent 42%),
                radial-gradient(circle at 82% 88%, rgba(111,183,255,0.045), transparent 46%);
    --panel: #14171b;
    --panel-alt: #191d22;
    --panel-raised: #1d2126;
    --border: #262b31;
    --border-soft: rgba(237,239,242,0.07);
    --text: #edeff2;
    --text-dim: #8b929c;
    --text-faint: #5c636d;
    --accent: #ff9d5c;
    --accent-dim: #c97a44;
    --accent-soft: rgba(255,157,92,0.12);
    --data: #6fb7ff;
    --data-dim: #3f7bb8;
    --danger: #ef8080;
    --radius: 10px;
    --radius-sm: 7px;
    --shadow-panel: 0 1px 0 rgba(255,255,255,0.02) inset, 0 10px 30px -18px rgba(0,0,0,0.65);
  }
  * { box-sizing: border-box; }
  ::selection { background: var(--accent-soft); color: var(--accent); }
  html { color-scheme: dark; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg) var(--bg-grain);
    background-attachment: fixed;
    color: var(--text);
    margin: 0;
    padding: 28px 24px 40px;
    -webkit-font-smoothing: antialiased;
  }
  :focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 4px;
  }

  #topbar {
    max-width: 1080px;
    margin: 0 auto 20px;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
  }
  h1 {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 19px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text);
    margin: 0;
    display: flex;
    align-items: baseline;
    gap: 9px;
  }
  h1 .divider { color: var(--text-faint); font-weight: 400; }
  h1 .sub {
    color: var(--accent);
    font-weight: 500;
  }
  #buildTag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.03em;
    color: var(--text-faint);
    padding-top: 2px;
  }
  #buildTag b { color: var(--data); font-weight: 500; }

  #layout { display: flex; gap: 18px; max-width: 1080px; margin: 0 auto; align-items: flex-start; }
  #left { flex: 1; min-width: 240px; display: flex; flex-direction: column; gap: 12px; }
  #right { flex: 2; min-width: 320px; }

  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-panel);
  }

  #searchWrap { position: relative; }
  input[type=text] {
    width: 100%;
    padding: 11px 13px;
    font-size: 15px;
    font-family: inherit;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: var(--radius-sm);
    box-sizing: border-box;
    outline: none;
    transition: border-color .15s ease;
  }
  input[type=text]::placeholder { color: var(--text-faint); }
  input[type=text]:focus { border-color: var(--accent-dim); }
  #suggestions {
    display: none;
    position: absolute;
    top: 46px;
    left: 0;
    right: 0;
    background: var(--panel-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    overflow: hidden;
    z-index: 5;
    box-shadow: 0 14px 34px -14px rgba(0,0,0,0.7);
  }
  .suggestionItem { padding: 9px 13px; font-size: 13px; cursor: pointer; }
  .suggestionItem:hover { background: var(--accent-soft); color: var(--accent); }

  #status { min-height: 18px; font-size: 13px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .handBadge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.04em;
    padding: 2px 7px;
    border-radius: 4px;
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid rgba(255,157,92,0.28);
  }

  .sectionLabel {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11.5px;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: var(--text-dim);
    padding: 12px 13px 0 13px;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
  }
  .sectionLabel b { color: var(--accent); font-family: 'JetBrains Mono', monospace; font-weight: 500; }

  #vocabList {
    max-height: 300px;
    overflow-y: auto;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 0 11px 11px;
  }
  #vocabList button {
    font-size: 12px;
    padding: 5px 10px;
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 999px;
    cursor: pointer;
    font-family: inherit;
    transition: border-color .12s ease, background .12s ease, color .12s ease;
  }
  #vocabList button:hover { border-color: var(--accent-dim); color: var(--accent); }
  #vocabList button.active { background: var(--accent); color: #1a0f07; border-color: var(--accent); font-weight: 600; }

  #sentencePanel { padding: 12px 13px; }
  #sentenceItems { min-height: 26px; display: flex; flex-wrap: wrap; gap: 5px; margin: 8px 0; }
  .sentenceChip { padding: 4px 8px; border-radius: 5px; background: var(--accent-soft); color: var(--accent); font-size: 11px; cursor: pointer; border: 1px solid rgba(255,157,92,0.25); }
  .sentenceEmpty { color: var(--text-faint); font-size: 11px; }
  #sentencePanel button {
    font: 11px inherit; padding: 6px 10px; border: 1px solid var(--border); border-radius: var(--radius-sm);
    background: var(--panel-alt); color: var(--text); cursor: pointer; transition: border-color .12s ease;
  }
  #sentencePanel button:hover { border-color: var(--accent-dim); }
  #sentenceActions { display: flex; gap: 6px; align-items: center; }
  #blendRange { flex: 1; accent-color: var(--accent); }

  #skinRow { padding: 12px 13px; font-size: 12px; color: var(--text-dim); }
  #skinRow .rowTop { display: flex; align-items: center; gap: 8px; }
  #skinRow input[type=color] {
    width: 26px; height: 26px; padding: 0; border: 1px solid var(--border); border-radius: 50%; background: none; cursor: pointer;
  }
  #skinSwatches { display: flex; gap: 6px; margin-left: auto; }
  .swatch {
    width: 19px; height: 19px; border-radius: 50%; border: 1px solid var(--border);
    cursor: pointer; padding: 0; transition: transform .1s ease, border-color .1s ease;
  }
  .swatch:hover { border-color: var(--accent); transform: scale(1.12); }

  #optionsRow { padding: 12px 13px; font-size: 12px; color: var(--text-dim); display: flex; flex-direction: column; gap: 9px; }
  #optionsRow .optLine { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  #optionsRow label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
  #optionsRow select {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; background: var(--panel-alt);
    color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 5px 7px;
  }
  #optionsRow input[type=checkbox] { accent-color: var(--accent); width: 14px; height: 14px; }
  body.highContrast { --bg:#000; --panel:#080808; --panel-alt:#111; --border:#fff; --text:#fff; --text-dim:#fff; --accent:#00ffff; --accent-dim:#00a0a0; }

  #learningPanel, #webcamPanel { padding: 12px 13px; font-size: 12px; }
  #learningPanel button, #webcamPanel button {
    padding: 6px 9px; border: 1px solid var(--border); border-radius: var(--radius-sm);
    background: var(--panel-alt); color: var(--text); cursor: pointer; transition: border-color .12s ease;
  }
  #learningPanel button:hover, #webcamPanel button:hover { border-color: var(--accent-dim); }
  #webcamVideo { display:none; width:100%; margin-top:8px; border-radius: var(--radius-sm); transform:scaleX(-1); }
  #quizPrompt { color: var(--accent); min-height:18px; margin:6px 0; }

  #right.panel { position: relative; overflow: hidden; }
  #canvasWrap { width: 100%; height: 480px; border-radius: var(--radius) var(--radius) 0 0; overflow: hidden; position: relative; background: radial-gradient(ellipse at 50% 38%, #171a1e 0%, #0e1013 78%); }
  .reticle { position: absolute; width: 22px; height: 22px; pointer-events: none; opacity: .55; z-index: 2; }
  .reticle::before, .reticle::after { content: ""; position: absolute; background: var(--accent); }
  .reticle::before { width: 100%; height: 1.5px; top: 0; }
  .reticle::after { width: 1.5px; height: 100%; left: 0; }
  .reticle.tl { top: 10px; left: 10px; }
  .reticle.tr { top: 10px; right: 10px; transform: scaleX(-1); }
  .reticle.bl { bottom: 10px; left: 10px; transform: scaleY(-1); }
  .reticle.br { bottom: 10px; right: 10px; transform: scale(-1); }
  #captureLabel {
    position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
    font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.08em;
    color: var(--text-faint); pointer-events: none; z-index: 2; text-transform: uppercase;
  }
  #hint {
    text-align: center; font-size: 11px; color: var(--text-faint); padding: 8px 0;
    letter-spacing: 0.02em; border-top: 1px solid var(--border-soft);
  }
  #playbackBar {
    display: flex; align-items: center; gap: 8px; padding: 10px 14px;
    border-top: 1px solid var(--border); flex-wrap: wrap;
  }
  #playBtn, #randomBtn, #saveImgBtn {
    width: 30px; height: 30px; border-radius: var(--radius-sm); border: 1px solid var(--border);
    background: var(--panel-alt); color: var(--text); cursor: pointer; font-size: 13px;
    display: flex; align-items: center; justify-content: center; flex: none;
    transition: border-color .12s ease, color .12s ease;
  }
  #playBtn:hover, #randomBtn:hover, #saveImgBtn:hover { border-color: var(--accent-dim); color: var(--accent); }
  #frameSlider { flex: 1; min-width: 80px; accent-color: var(--accent); }
  #frameLabel { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-dim); min-width: 52px; text-align: right; }
  #speedSelect, #loopSelect {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; background: var(--panel-alt);
    color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 5px 7px;
  }
  @media (max-width: 720px) {
    #layout { flex-direction: column; }
    #canvasWrap { height: 360px; }
  }

  #modelPanel { padding: 12px 13px; font-size: 12px; color: var(--text-dim); }
  #modelPanel .modelTitle { font-family: 'Space Grotesk', sans-serif; font-weight: 500; display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
  #modelState { font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--text-dim); }
  #modelState.ready { color: var(--accent); }
  #modelState.warn { color:#e0b56f; }
  .rangeRow { display:grid; grid-template-columns: 1fr auto; gap:8px; align-items:center; margin-top:8px; }
  .rangeRow input { width:100%; accent-color: var(--accent); }
  .rangeValue { font-family:'JetBrains Mono',monospace; min-width:42px; text-align:right; color: var(--text-dim); }
  .modelBtns { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
  .modelBtns button {
    flex:1; min-width:92px; padding:6px 8px; border:1px solid var(--border); border-radius: var(--radius-sm);
    background: var(--panel-alt); color: var(--text); cursor:pointer; font:11px 'Inter',sans-serif;
    transition: border-color .12s ease, color .12s ease;
  }
  .modelBtns button:hover { border-color: var(--accent-dim); color: var(--accent); }
  .diag { margin-top:8px; padding:8px 9px; border:1px solid var(--border); border-radius: var(--radius-sm); background:#101215; font:10px/1.45 'JetBrains Mono',monospace; white-space:pre-wrap; max-height:110px; overflow:auto; color: var(--text-dim); }
  .toggle { display:flex; align-items:center; gap:6px; cursor:pointer; }
  #performanceBadge { font-family:'JetBrains Mono',monospace; font-size:10px; color: var(--text-faint); }
  .requiredAsset {
    display:flex; align-items:center; gap:9px; padding:9px 10px; border:1px solid rgba(255,157,92,.22);
    border-radius: var(--radius-sm); background: var(--accent-soft); color: var(--text-dim); margin-bottom:9px;
  }
  .requiredAsset b { color: var(--text); font-size:11px; }
  .requiredAsset code { font:10px 'JetBrains Mono',monospace; color: var(--accent); }
  .assetDot { width:7px; height:7px; border-radius:50%; background: var(--accent); box-shadow:0 0 10px rgba(255,157,92,.6); flex:none; }
  .assetLock { margin-left:auto; font:9px 'JetBrains Mono',monospace; color: var(--accent); border:1px solid var(--accent-dim); border-radius:4px; padding:2px 5px; letter-spacing: 0.04em; }

  #vocabList::-webkit-scrollbar, .diag::-webkit-scrollbar { width: 7px; }
  #vocabList::-webkit-scrollbar-thumb, .diag::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  /* ---- Tabs ---- */
  #tabBar { display: flex; gap: 4px; background: var(--panel-alt); border: 1px solid var(--border); border-radius: 999px; padding: 3px; }
  #tabBar button {
    font-family: 'Space Grotesk', sans-serif; font-size: 12.5px; font-weight: 500;
    padding: 7px 16px; border-radius: 999px; border: none; background: transparent;
    color: var(--text-dim); cursor: pointer; transition: background .15s ease, color .15s ease;
  }
  #tabBar button.active { background: var(--accent); color: #1a0f07; }
  #tabBar button:hover:not(.active) { color: var(--text); }

  /* ---- Recognize layout (mirrors #layout / #left / #right) ---- */
  #recLayout { display: flex; gap: 18px; max-width: 1080px; margin: 0 auto; align-items: flex-start; }
  #recLeft { flex: 1; min-width: 240px; display: flex; flex-direction: column; gap: 12px; }
  #recRight { flex: 2; min-width: 320px; }
  @media (max-width: 720px) { #recLayout { flex-direction: column; } }

  #recModelPanel, #recSettingsPanel, #recSentencePanel, #recSessionPanel { padding: 12px 13px; font-size: 12px; color: var(--text-dim); }
  #recModelPanel .modelTitle { font-family: 'Space Grotesk', sans-serif; font-weight: 500; color: var(--text); display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
  #recSessionPanel .modelTitle { font-family: 'Space Grotesk', sans-serif; font-weight: 500; color: var(--text); display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
  #recSessionState { font: 10px 'JetBrains Mono', monospace; color: var(--text-faint); }
  #recSessionState.recording { color: var(--accent); }
  #recSessionState.processing { color: #e0b56f; }
  #recSessionState.completed { color: var(--data); }
  #recSessionSigns { min-height: 34px; padding: 8px 9px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--panel-alt); line-height: 1.45; color: var(--text); word-break: break-word; }
  #recSessionActions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 9px; }
  #recSessionActions button { font: 11px 'Inter', sans-serif; padding: 6px 9px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--panel-alt); color: var(--text); cursor: pointer; }
  #recSessionActions button:first-child { border-color: var(--accent-dim); color: var(--accent); }
  #recSessionActions button:hover:not(:disabled) { border-color: var(--accent-dim); }
  #recSessionActions button:disabled { opacity: .45; cursor: default; }
  #recSessionResult { margin-top: 8px; color: var(--accent); font-size: 13px; min-height: 18px; }
  #preloader {
    position: fixed; inset: 0; z-index: 50; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 18px; background: #05070d;
    transition: opacity .35s ease, visibility .35s ease;
  }
  #preloader.hidden { opacity: 0; visibility: hidden; pointer-events: none; }
  #preloader img { width: min(430px, 72vw); height: auto; display: block; }
  #preloaderBar { width: min(260px, 60vw); height: 3px; overflow: hidden; border-radius: 4px; background: #17213b; }
  #preloaderBar::after { content: ""; display: block; width: 42%; height: 100%; background: #2866ff; border-radius: inherit; animation: preload 1.1s ease-in-out infinite; }
  @keyframes preload { 0% { transform: translateX(-140%); } 100% { transform: translateX(340%); } }
  #recModelState { font-family:'JetBrains Mono',monospace; font-size:10px; }
  #recModelState.ready { color: var(--accent); }
  #recModelState.warn { color:#e0b56f; }
  #recModelState.err { color: var(--danger); }
  .pathRow { display: flex; flex-direction: column; gap: 3px; margin-top: 8px; }
  .pathRow label { font-size: 10.5px; color: var(--text-faint); font-family: 'JetBrains Mono', monospace; letter-spacing: .02em; }
  .pathRow input[type=text] { font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 7px 9px; }
  #recLoadBtn {
    margin-top: 10px; width: 100%; padding: 8px; border-radius: var(--radius-sm); border: 1px solid var(--accent-dim);
    background: var(--accent-soft); color: var(--accent); font: 12px 'Space Grotesk', sans-serif; font-weight: 500; cursor: pointer;
  }
  #recLoadBtn:hover { background: var(--accent); color: #1a0f07; }
  #recLoadBtn:disabled { opacity: .5; cursor: default; background: var(--panel-alt); color: var(--text-faint); border-color: var(--border); }
  .setupNote { margin-top: 9px; font-size: 10.5px; line-height: 1.5; color: var(--text-faint); }
  .setupNote code { font: 10px 'JetBrains Mono', monospace; color: var(--data); background: rgba(111,183,255,0.08); padding: 1px 4px; border-radius: 4px; }

  #recSettingsPanel .optLine { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 9px; }
  #recSettingsPanel .optLine:first-child { margin-top: 0; }
  #recSettingsPanel label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
  #recSettingsPanel input[type=range] { width: 100%; accent-color: var(--accent); }
  #confThreshValue { font-family: 'JetBrains Mono', monospace; color: var(--text-dim); min-width: 34px; text-align: right; }

  #recSentenceBox {
    min-height: 52px; font-size: 15px; line-height: 1.5; padding: 9px 10px; margin: 8px 0;
    background: var(--panel-alt); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text);
    word-break: break-word;
  }
  #recSentenceBox:empty::before { content: "Recognized signs will build a sentence here."; color: var(--text-faint); }
  #recSentenceActions { display: flex; gap: 6px; }
  #recSentenceActions button {
    font: 11px 'Inter', sans-serif; padding: 6px 10px; border: 1px solid var(--border); border-radius: var(--radius-sm);
    background: var(--panel-alt); color: var(--text); cursor: pointer; transition: border-color .12s ease;
  }
  #recSentenceActions button:hover { border-color: var(--accent-dim); }

  #recRight.panel { position: relative; overflow: hidden; }
  #camWrap {
    width: 100%; height: 480px; border-radius: var(--radius) var(--radius) 0 0; overflow: hidden; position: relative;
    background: radial-gradient(ellipse at 50% 42%, #171a1e 0%, #0e1013 78%);
    display: flex; align-items: center; justify-content: center;
  }
  #recCanvas { width: 100%; height: 100%; object-fit: cover; display: block; }
  #camIdle {
    position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 12px; color: var(--text-faint); font-size: 13px; text-align: center; padding: 0 24px;
  }
  #camIdle button {
    font: 13px 'Space Grotesk', sans-serif; font-weight: 500; padding: 10px 20px; border-radius: 999px;
    border: 1px solid var(--accent-dim); background: var(--accent-soft); color: var(--accent); cursor: pointer;
  }
  #camIdle button:hover { background: var(--accent); color: #1a0f07; }
  #predictionBadge {
    position: absolute; top: 12px; left: 12px; z-index: 3;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; padding: 6px 11px; border-radius: 999px;
    background: rgba(10,12,15,0.72); border: 1px solid var(--border); color: var(--text-dim); backdrop-filter: blur(6px);
    display: flex; align-items: center; gap: 7px;
  }
  #predictionBadge .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text-faint); flex: none; }
  #predictionBadge.locked { color: var(--accent); border-color: rgba(255,157,92,.35); }
  #predictionBadge.locked .dot { background: var(--accent); box-shadow: 0 0 8px rgba(255,157,92,.7); }
  #bufferBar {
    position: absolute; bottom: 0; left: 0; height: 3px; background: var(--accent); z-index: 3;
    transition: width .08s linear; width: 0%;
  }
  #ttsBadge {
    position: absolute; top: 12px; right: 12px; z-index: 3;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px; padding: 5px 9px; border-radius: 999px;
    background: rgba(10,12,15,0.72); border: 1px solid var(--border); color: var(--text-faint); cursor: pointer;
    backdrop-filter: blur(6px);
  }
  #ttsBadge.on { color: var(--data); border-color: rgba(111,183,255,.35); }
  #recControlsBar {
    display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-top: 1px solid var(--border); flex-wrap: wrap;
  }
  #camToggleBtn {
    padding: 7px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border); background: var(--panel-alt);
    color: var(--text); cursor: pointer; font: 12px 'Inter', sans-serif; transition: border-color .12s ease;
  }
  #camToggleBtn:hover { border-color: var(--accent-dim); }
  #camToggleBtn.live { border-color: var(--danger); color: var(--danger); }
  #recFpsBadge { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--text-faint); margin-left: auto; }

</style>
</head>
<body>
<div id="preloader" role="status" aria-label="Loading SignSpeak">
  <img src="SignSpeak.png" alt="SignSpeak" />
  <div id="preloaderBar"></div>
</div>
<div id="topbar">
  <h1>SignSpeak <span class="divider">/</span> <span class="sub" id="tabHeading">Recognize</span></h1>
  <div style="display:flex; align-items:center; gap:14px;">
    <div id="tabBar">
      <button id="tabBtnRecognize" class="active" onclick="switchTab('recognize')">Recognize</button>
      <button id="tabBtnViewer" onclick="switchTab('viewer')">Viewer</button>
    </div>
    <div id="buildTag">landmark capture &middot; <b>21pt</b> hand rig</div>
  </div>
</div>
<div id="recognizeView">
<div id="recLayout">
  <div id="recLeft">
    <div class="panel" id="recModelPanel">
      <div class="modelTitle"><span>Classifier</span><span id="recModelState">not loaded</span></div>
      <div class="pathRow">
        <label for="recModelPath">Model (converted to TFJS)</label>
        <input id="recModelPath" type="text" value="models_web/model.json" />
      </div>
      <div class="pathRow">
        <label for="recLabelPath">Label map JSON</label>
        <input id="recLabelPath" type="text" value="models_web/label_map.json" />
      </div>
      <button id="recLoadBtn">Load model</button>
      <div class="setupNote">
        Convert your Keras model once with <code>pip install tensorflowjs</code> then
        <code>tensorflowjs_converter --input_format=keras models/sign_classifier.keras models_web/</code>,
        and copy <code>models/label_map.json</code> next to it. Serve this folder over HTTP so the
        browser can fetch these paths.
      </div>
    </div>

    <div class="panel" id="recSentencePanel">
      <div class="sectionLabel">Transcript</div>
      <div id="recSentenceBox"></div>
      <div id="recSentenceActions">
        <button id="recClearBtn">Clear</button>
        <button id="recSpeakBtn">Speak last sign</button>
        <button id="recListenBtn">Listen to final sentence</button>
        <button id="recCopyBtn">Copy</button>
      </div>
    </div>

    <div class="panel" id="recSessionPanel">
      <div class="modelTitle"><span>Sentence recognition</span><span id="recSessionState">Idle</span></div>
      <div id="recSessionSigns">Press Start Recognition, then sign naturally. Detected signs will appear here.</div>
      <div id="recSessionActions">
        <button id="recSessionStartBtn" disabled>Start Recognition</button>
        <button id="recSessionStopBtn" disabled>Stop Recognition</button>
        <button id="recSessionResetBtn">New / Reset</button>
      </div>
      <div class="setupNote">The final sentence is created only after you press Stop Recognition.</div>
      <div id="recSessionResult" aria-live="polite"></div>
    </div>

    <div class="panel" id="recSettingsPanel">
      <div class="sectionLabel" style="padding:0; margin-bottom:2px;">Recognition settings</div>
      <div class="optLine">
        <label for="confThreshRange">Confidence threshold</label>
        <span id="confThreshValue">60%</span>
      </div>
      <input id="confThreshRange" type="range" min="30" max="95" value="60" step="1" />
      <div class="optLine">
        <label><input id="recSpeakToggle" type="checkbox" checked /> Speak recognized signs</label>
      </div>
      <div class="optLine">
        <label><input id="recDrawToggle" type="checkbox" checked /> Draw hand skeleton</label>
      </div>
    </div>
  </div>

  <div id="recRight" class="panel">
    <div id="camWrap">
      <video id="recVideoHidden" style="display:none;" playsinline muted></video>
      <canvas id="recCanvas"></canvas>
      <span class="reticle tl"></span><span class="reticle tr"></span>
      <span class="reticle bl"></span><span class="reticle br"></span>
      <div id="predictionBadge"><span class="dot"></span><span id="predictionText">Camera off</span></div>
      <div id="ttsBadge" class="on" title="Toggle speech output">&#128264; Speech on</div>
      <div id="bufferBar"></div>
      <div id="camIdle">
        <div>Turn on your camera to start recognizing signs.<br>Nothing is recorded or sent anywhere — everything runs locally in this tab.</div>
        <button id="camStartBtn" disabled>Start camera</button>
      </div>
    </div>
    <div id="recControlsBar">
      <button id="camToggleBtn" disabled>Start camera</button>
      <span style="font-size:11px; color:var(--text-faint);">c &middot; clear transcript</span>
      <span id="recFpsBadge">-- FPS</span>
    </div>
  </div>
</div>
</div>

<div id="viewerView" style="display:none;">
<div id="layout">
  <div id="left">
    <div id="searchWrap">
      <input id="wordInput" type="text" placeholder="Type a word or letter, press Enter" autocomplete="off" />
      <div id="suggestions"></div>
    </div>
    <div id="status"></div>
    <div class="panel">
      <div class="sectionLabel" style="padding: 10px 12px 0 12px;">Available signs <b id="vocabCount">0</b></div>
      <div id="vocabList"></div>
    </div>
    <div class="panel" id="sentencePanel">
      <div class="sectionLabel">Sentence builder <b id="sentenceCount">0</b></div>
      <div id="sentenceItems"><span class="sentenceEmpty">Click signs to add them here</span></div>
      <div id="sentenceActions">
        <button id="playSentenceBtn">Play sentence</button>
        <button id="clearSentenceBtn">Clear</button>
      </div>
      <div class="rangeRow" style="margin-top:8px;"><label for="blendRange">Transition blend</label><span class="rangeValue" id="blendValue">6 frames</span></div>
      <input id="blendRange" type="range" min="0" max="15" value="6" step="1" />
    </div>
    <div class="panel" id="skinRow">
      <div class="rowTop">
        <label for="skinPicker">Skin tone</label>
        <input id="skinPicker" type="color" value="#e0ac85" />
        <div id="skinSwatches"></div>
      </div>
    </div>

    <div class="panel" id="modelPanel">
      <div class="modelTitle">
        <span>3D model</span><span id="modelState">loading articulated hand…</span>
      </div>
      <div class="requiredAsset"><span class="assetDot"></span><span><b>Articulated landmark hand</b><br><code>Independent thumb and finger joints</code></span><span class="assetLock">READY</span></div>
      <div class="rangeRow">
        <label for="smoothRange">Pose smoothing</label>
        <span class="rangeValue" id="smoothValue">45%</span>
      </div>
      <input id="smoothRange" type="range" min="0" max="90" value="45" step="5" />
      <div class="rangeRow">
        <label for="modelScaleRange">Model scale</label>
        <span class="rangeValue" id="modelScaleValue">100%</span>
      </div>
      <input id="modelScaleRange" type="range" min="60" max="150" value="100" step="1" />
      <div class="modelBtns">
        <button id="resetViewBtn">Reset view</button>
        <button id="resetPoseBtn">Reset pose</button>
        <button id="frontViewBtn">Front</button>
        <button id="sideViewBtn">Side</button>
        <button id="fitViewBtn">Fit hand</button>
        <button id="poseJsonBtn">Pose JSON</button>
        <button id="glbBtn">Export GLB</button>
        <button id="diagnosticBtn">Diagnostics</button>
      </div>
      <div id="modelDiag" class="diag" hidden></div>
      <div id="performanceBadge">Articulated FBX · independent joint driver</div>
    </div>

    <div class="panel" id="optionsRow">
      <div class="optLine">
        <span>Lighting</span>
        <select id="lightSelect">
          <option value="studio" selected>Studio</option>
          <option value="soft">Soft</option>
          <option value="warm">Warm</option>
        </select>
      </div>
      <div class="optLine">
        <label><input id="mirrorToggle" type="checkbox" /> Mirror view</label>
        <label><input id="autoRotateToggle" type="checkbox" checked /> Auto-rotate</label>
      </div>
      <div class="optLine">
        <label><input id="debugToggle" type="checkbox" /> Show landmark skeleton</label>
      </div>
      <div class="optLine">
        <button id="fullscreenBtn">Fullscreen</button>
        <label><input id="contrastToggle" type="checkbox" /> High contrast</label>
      </div>
    </div>
    <div class="panel" id="learningPanel">
      <div class="sectionLabel">Practice mode <b id="scoreLabel">0%</b></div>
      <div id="quizPrompt">Choose a sign to practice.</div>
      <button id="practiceBtn">Practice random sign</button>
      <button id="quizBtn">Start quiz</button>
    </div>
    <div class="panel" id="webcamPanel">
      <div class="sectionLabel">Webcam practice <b id="webcamState">OFF</b></div>
      <button id="webcamBtn">Enable webcam</button>
      <video id="webcamVideo" autoplay muted playsinline></video>
      <div id="confidenceLabel">Camera landmarks: unavailable (use exported pose playback)</div>
    </div>
  </div>
  <div id="right" class="panel">
    <div id="canvasWrap">
      <div id="captureLabel">live capture</div>
      <span class="reticle tl"></span><span class="reticle tr"></span>
      <span class="reticle bl"></span><span class="reticle br"></span>
    </div>
    <div id="hint">drag to rotate &middot; scroll to zoom &middot; space to play/pause &middot; &#8592;/&#8594; to step &middot; R for random</div>
    <div id="playbackBar">
      <button id="playBtn" title="Play / pause">&#10074;&#10074;</button>
      <input id="frameSlider" type="range" min="0" max="0" value="0" step="1" />
      <span id="frameLabel">0 / 0</span>
      <select id="speedSelect" title="Playback speed">
        <option value="160">0.5x</option>
        <option value="80" selected>1x</option>
        <option value="40">2x</option>
      </select>
      <select id="loopSelect" title="Loop mode">
        <option value="loop" selected>Loop</option>
        <option value="pingpong">Ping-pong</option>
        <option value="once">Once</option>
      </select>
      <button id="randomBtn" title="Play a random sign">&#127922;</button>
      <button id="saveImgBtn" title="Save current frame as PNG">&#128247;</button>
    </div>
  </div>
</div>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/fflate@0.7.4/umd/index.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/FBXLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/utils/SkeletonUtils.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/exporters/GLTFExporter.js"></script>
<script>
// Embedded directly by export_for_viewer.py - regenerate this file by
// re-running that script whenever your dataset changes.
var vocab = __VOCAB_JSON__;

// ---- hand geometry constants (MediaPipe 21-point layout) ----
var palmLoop = [0, 1, 5, 9, 13, 17]; // wrist, thumb CMC, index/middle/ring/pinky MCP
var radiusByLandmark = [0.068,0.038,0.032,0.027,0.022,0.038,0.033,0.027,0.021,0.040,0.035,0.029,0.022,0.038,0.033,0.027,0.021,0.035,0.030,0.025,0.019];
var fingerChains = [
  [1, 2, 3, 4],     // thumb
  [5, 6, 7, 8],     // index
  [9, 10, 11, 12],  // middle
  [13, 14, 15, 16], // ring
  [17, 18, 19, 20]  // pinky
];
var fingertips = [
  { tip: 4, prev: 3 },
  { tip: 8, prev: 7 },
  { tip: 12, prev: 11 },
  { tip: 16, prev: 15 },
  { tip: 20, prev: 19 }
];
var skeletonConnections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

// ---- tunable shape parameters ----
var SEGMENTS_PER_SPAN = 10;    // tube subdivisions between two consecutive joints
var FINGER_RADIAL_SEGMENTS = 14;
var KNUCKLE_BUMP = 0.16;      // subtle joint fullness without separate ball-like joints
var FLATTEN_NORMAL = 0.78;    // finger cross-section squash along the palm-normal axis
var FLATTEN_SIDE = 1.02;      // finger cross-section stretch along the side axis
var PALM_RIM_SEGMENTS = 32;
var PALM_RIM_HALF = 0.016;
var PALM_FRONT_HEIGHT = 0.19;
var PALM_BACK_HEIGHT = 0.13;

var scene, camera, renderer, controls, group, debugGroup;
var skinMat, nailMat;
var currentFrames = null;
var frameIdx = 0;
var playDir = 1;
var isPlaying = true;
var frameIntervalMs = 80;
var lastTick = 0;
var idleTimer = null;
var activeVocabBtn = null;
var lastWord = "";
var mirrorOn = false;
var showSkeleton = false;
var loopMode = "loop"; // loop | pingpong | once
var wrap = document.getElementById("canvasWrap");

var SKIN_PRESETS = ["#f6d2b0", "#e0ac85", "#c68642", "#8d5524", "#5c3a21"];

var LIGHT_PRESETS = {
  studio: { bg: 0x0e1013, hemi: [0xfff4e0, 0x1a1410, 0.55], key: [0xfff2e0, 1.0], fill: [0xcfe8ff, 0.35], rim: [0x6fb7ff, 0.4] },
  soft:   { bg: 0x1c1c22, hemi: [0xffffff, 0x2a2a35, 0.7],  key: [0xffffff, 0.7],  fill: [0xffffff, 0.45], rim: [0x8888ff, 0.2] },
  warm:   { bg: 0x1a1410, hemi: [0xffe0b0, 0x1a0f08, 0.55], key: [0xffcf9a, 1.05], fill: [0xff9a5a, 0.25], rim: [0xffb56b, 0.35] }
};


// ---- FBX human-hand model / pose driver ----
var MODEL_URL = "__MODEL_URL__";
var FALLBACK_MODEL_URL = "handct_hand.obj";
var objLoader = null;
var objSource = null;
var fbxLoader = null;
var fbxSource = null;
var fbxReady = false;
var fbxHasRig = false;
var modelInstances = { left: null, right: null };
var modelGroup = null;
var modelScaleUser = 1.0;
var poseSmoothing = 0.45;
var previousPose = { left: null, right: null };
var sentence = [];
var sentenceBlendFrames = 6;
var isSentencePlayback = false;
var manualPose = { left: {}, right: {} };
var dragState = null;
var quizActive = false;
var quizTarget = "";
var quizScore = 0;
var quizTotal = 0;
var webcamStream = null;
var rigProfiles = new WeakMap();
var modelBaseScale = 1.0;
var modelDiagnostics = "";
var objPoseUniforms = [];
var objRestLandmarks = [
  [[0,-0.85,0],[0.02,-0.25,0.02],[0.25,0.08,0.01],[0.40,0.38,0.00],[0.34,0.68,-0.01]],
  [[0,-0.85,0],[0.30,-0.05,0.01],[0.34,0.30,0.01],[0.32,0.64,0.00],[0.30,0.96,0.00]],
  [[0,-0.85,0],[0.06,-0.04,0.02],[0.08,0.32,0.02],[0.08,0.68,0.01],[0.08,1.04,0.00]],
  [[0,-0.85,0],[-0.17,-0.06,0.01],[-0.19,0.28,0.01],[-0.20,0.60,0.00],[-0.20,0.91,0.00]],
  [[0,-0.85,0],[-0.33,-0.11,0.00],[-0.36,0.19,0.00],[-0.36,0.46,0.00],[-0.35,0.72,0.00]]
];
var objDeformChains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]];
var fpsLastTime=performance.now(), fpsFrames=0, currentFPS=0;

function setModelState(text, cls) {
  var el = document.getElementById("modelState");
  if (!el) return;
  el.textContent = text;
  el.className = cls || "";
}

function safeDisposeObject(obj) {
  if (!obj) return;
  obj.traverse(function (o) {
    if (o.geometry && o.geometry.dispose) o.geometry.dispose();
    if (o.material) {
      var mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach(function (m) { if (m && m.dispose) m.dispose(); });
    }
  });
}

function cloneFBX(src) {
  if (THREE.SkeletonUtils && THREE.SkeletonUtils.clone) return THREE.SkeletonUtils.clone(src);
  return src.clone(true);
}

function allBones(root) {
  var out = [];
  root.traverse(function (o) { if (o.isBone) out.push(o); });
  return out;
}

function childBones(b) { return b.children.filter(function (c) { return c.isBone; }); }

function descendantsCount(b) {
  var n = 0;
  childBones(b).forEach(function (c) { n += 1 + descendantsCount(c); });
  return n;
}

function worldPointOf(obj) {
  var p = new THREE.Vector3();
  obj.getWorldPosition(p);
  return p;
}

function boneByNameMap(bones){
  var map={};
  bones.forEach(function(b){
    if(!map[b.name]) map[b.name]=[];
    map[b.name].push(b);
  });
  return map;
}

function findConnectedNamedChain(nameMap,names){
  var candidates=names.map(function(name){return nameMap[name]||[];});
  if(candidates.some(function(a){return !a.length;})) return null;
  function walk(level,previous){
    if(level===candidates.length) return [];
    for(var i=0;i<candidates[level].length;i++){
      var b=candidates[level][i];
      if(level>0 && b.parent!==previous) continue;
      var tail=walk(level+1,b);
      if(tail!==null) return [b].concat(tail);
    }
    return null;
  }
  return walk(0,null);
}

function commonBoneAncestor(bones){
  if(!bones.length) return null;
  var paths=[];
  bones.forEach(function(b){
    var p=[]; var x=b;
    while(x && x.isBone){p.push(x);x=x.parent;}
    paths.push(p);
  });
  for(var i=0;i<paths[0].length;i++){
    var candidate=paths[0][i];
    if(paths.every(function(path){return path.indexOf(candidate)!==-1;})) return candidate;
  }
  return null;
}

function saveMappingRestState(mappings){
  mappings.forEach(function(m){
    m.restQuat=m.bone.quaternion.clone();
    m.restPos=m.bone.position.clone();
    var child=childBones(m.bone)[0];
    var restDirWorld;
    if(child){
      restDirWorld=worldPointOf(child).sub(worldPointOf(m.bone)).normalize();
    }else{
      restDirWorld=new THREE.Vector3(0,1,0).applyQuaternion(m.bone.getWorldQuaternion(new THREE.Quaternion())).normalize();
    }
    var parent=m.bone.parent;
    var parentInv=parent?parent.getWorldQuaternion(new THREE.Quaternion()).invert():new THREE.Quaternion();
    m.restDirParent=restDirWorld.clone().applyQuaternion(parentInv).normalize();
  });
}

function buildExplicitRigProfile(bones){
  // Robust mapper for the supplied FBX. FBXLoader may normalize/rename node
  // labels differently from the raw FBX strings, so compare normalized names
  // and also accept a Bone + numeric suffix representation.
  function norm(s){
    return String(s||"").toLowerCase().replace(/[^a-z0-9]/g,"");
  }

  function boneNumber(name){
    var m=String(name||"").match(/bone[^0-9]*([0-9]+)/i);
    return m ? parseInt(m[1],10) : null;
  }

  var byNorm={};
  var byNumber={};
  bones.forEach(function(b){
    var n=norm(b.name);
    if(!byNorm[n]) byNorm[n]=[];
    byNorm[n].push(b);
    var num=boneNumber(b.name);
    if(num!==null){
      if(!byNumber[num]) byNumber[num]=[];
      byNumber[num].push(b);
    }
  });

  function resolve(expected){
    var n=norm(expected);
    if(byNorm[n] && byNorm[n].length) return byNorm[n][0];

    var num=boneNumber(expected);
    if(num!==null && byNumber[num] && byNumber[num].length){
      return byNumber[num][0];
    }

    return null;
  }

  var chainNames=[
    ["Bone.001","Bone.011","Bone.012","Bone.013"],
    ["Bone.002","Bone.008","Bone.009","Bone.010"],
    ["Bone.003","Bone.005","Bone.006","Bone.007"],
    ["Bone.004","Bone.014","Bone.015","Bone.016"],
    ["Bone.017","Bone.018","Bone.019"]
  ];

  var landmarkChains=[
    [0,1,2,3,4],
    [0,5,6,7,8],
    [0,9,10,11,12],
    [0,13,14,15,16],
    [0,17,18,19,20]
  ];

  var chains=[];
  var mappings=[];
  var missing=[];

  chainNames.forEach(function(names,finger){
    var chain=names.map(resolve);
    if(chain.some(function(b){return !b;})){
      names.forEach(function(n){
        if(!resolve(n) && missing.indexOf(n)<0) missing.push(n);
      });
      return;
    }

    chains.push(chain);
    var target=landmarkChains[finger];
    for(var i=0;i<chain.length;i++){
      mappings.push({
        bone:chain[i],
        a:target[i],
        b:target[i+1],
        finger:finger,
        depth:i
      });
    }
  });

  if(missing.length){
    var sample=bones.slice(0,40).map(function(b){return b.name;}).join(", ");
    return {
      valid:false,
      reason:"Could not resolve supplied FBX deform bones: "+missing.join(", ") +
             ". Runtime bone names begin with: "+sample
    };
  }

  var root=commonBoneAncestor(chains.map(function(c){return c[0];}));
  if(!root){
    root=bones.find(function(b){return /^bone$/i.test(String(b.name||""))}) ||
         bones.find(function(b){return boneNumber(b.name)===0}) ||
         chains[0][0];
  }

  saveMappingRestState(mappings);

  return {
    valid:true,
    mode:"explicit-name-normalized",
    root:root,
    bones:bones,
    mappings:mappings,
    branches:chains.map(function(c){return c[0];}),
    rootPos:worldPointOf(root),
    chainNames:chainNames
  };
}

function buildTopologyRigProfile(bones){
  // Conservative fallback for a differently exported copy of the same asset.
  // It looks for five non-overlapping long chains, then orders them spatially.
  var boneSet=new Set(bones);
  var roots=bones.filter(function(b){return !b.parent || !boneSet.has(b.parent);});
  var root=roots.sort(function(a,b){return descendantsCount(b)-descendantsCount(a);})[0];
  if(!root) return null;

  function collectChain(start){
    var chain=[];var b=start;
    while(b&&b.isBone&&chain.length<4){
      chain.push(b);
      var cs=childBones(b);
      if(!cs.length) break;
      cs.sort(function(a,c){return descendantsCount(c)-descendantsCount(a);});
      b=cs[0];
    }
    return chain;
  }

  var candidates=[];
  bones.forEach(function(b){
    var chain=collectChain(b);
    if(chain.length>=3){
      var end=chain[chain.length-1];
      var d=worldPointOf(end).sub(worldPointOf(b));
      if(d.lengthSq()>1e-8) candidates.push({root:b,chain:chain,dir:d.normalize(),score:chain.length});
    }
  });
  candidates.sort(function(a,b){return b.score-a.score;});
  var chosen=[];
  candidates.forEach(function(c){
    if(chosen.length>=5) return;
    if(chosen.some(function(x){return isAncestorOrSame(c.root,x.root)||isAncestorOrSame(x.root,c.root);})) return;
    chosen.push(c);
  });
  if(chosen.length!==5) return null;

  var center=new THREE.Vector3();chosen.forEach(function(c){center.add(c.dir);});center.normalize();
  var thumbIndex=0,thumbScore=-1;
  chosen.forEach(function(c,i){var sc=1-Math.abs(c.dir.dot(center));if(sc>thumbScore){thumbScore=sc;thumbIndex=i;}});
  var thumb=chosen.splice(thumbIndex,1)[0];
  chosen.sort(function(a,b){return worldPointOf(a.root).x-worldPointOf(b.root).x;});
  chosen.unshift(thumb);

  var landmarkChains=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]];
  var mappings=[];
  chosen.forEach(function(c,finger){
    var target=landmarkChains[finger];
    for(var i=0;i<c.chain.length&&i<4;i++) mappings.push({bone:c.chain[i],a:target[i],b:target[i+1],finger:finger,depth:i});
  });
  if(mappings.length<15) return null;
  saveMappingRestState(mappings);
  return {valid:true,mode:"topology",root:root,bones:bones,mappings:mappings,branches:chosen.map(function(c){return c.root;}),rootPos:worldPointOf(root)};
}

function isAncestorOrSame(a,b){
  var p=a;
  while(p&&p.isBone){if(p===b)return true;p=p.parent;}
  return false;
}

function makeRigProfile(instance) {
  var bones=allBones(instance);
  if(!bones.length) return {valid:false,reason:"No skeleton bones found in FBX."};

  var explicit=buildExplicitRigProfile(bones);
  if(explicit) return explicit;

  var fallback=buildTopologyRigProfile(bones);
  if(fallback) return fallback;

  return {valid:false,reason:"FBX loaded, but no confident five-finger deform hierarchy was found. The exact supplied FBX remains loaded; automatic landmark articulation is disabled for safety."};
}
function prepareFBXMaterials(root) {
  root.traverse(function(o){
    if(!o.isMesh) return;
    o.castShadow=false; o.receiveShadow=false;
    var mats=Array.isArray(o.material)?o.material:[o.material];
    mats.forEach(function(m){
      if(!m) return;
      var isNail=/nail|finger.?tip|fingernail/i.test((m.name||"")+" "+(o.name||""));
      if(m.color) m.color.set(isNail ? 0xf1ddd2 : 0xe0ac85);
      if("skinning" in m) { m.skinning=true; m.needsUpdate=true; }
    });
  });
}

function normalizeModel(root){
  var box=new THREE.Box3().setFromObject(root);
  var size=box.getSize(new THREE.Vector3());
  var maxDim=Math.max(size.x,size.y,size.z)||1;
  modelBaseScale=1.9/maxDim;
  root.scale.setScalar(modelBaseScale*modelScaleUser);
  box.setFromObject(root);
  var center=box.getCenter(new THREE.Vector3());
  root.position.sub(center.multiplyScalar(1));
  root.position.y += 0.1;
}

function prepareOBJDeformation(root) {
  objPoseUniforms = null;
}

function deformOBJForPose(points) {
  if (!objSource || !points) return;
  var wrist = vec(points[0]);
  var fingertips = [4, 8, 12, 16, 20].map(function(i) { return vec(points[i]).sub(wrist); });
  var average = fingertips.reduce(function(sum, p) { return sum.add(p); }, new THREE.Vector3()).multiplyScalar(1 / fingertips.length);
  var spread = Math.max.apply(null, fingertips.map(function(p) { return p.x; })) -
    Math.min.apply(null, fingertips.map(function(p) { return p.x; }));
  var curl = fingertips.reduce(function(sum, p) { return sum + p.y; }, 0) / fingertips.length;
  objSource.rotation.set(
    THREE.MathUtils.clamp(-average.z * 0.55, -0.42, 0.42),
    THREE.MathUtils.clamp(average.x * 0.5, -0.38, 0.38),
    THREE.MathUtils.clamp((spread - 0.55) * 0.35, -0.22, 0.22)
  );
  var gestureScale = THREE.MathUtils.clamp(0.94 + curl * 0.045, 0.88, 1.08);
  objSource.scale.setScalar(modelBaseScale * modelScaleUser * gestureScale);
}

function loadFBXModel(){
  if(!window.THREE || !THREE.FBXLoader){
    setModelState("FBX loader unavailable","warn");
    return;
  }
  if(window.location.protocol === "file:"){
    setModelState("Run with local server","warn");
    modelDiagnostics="Browsers block OBJ loading from file:// pages. Use: python -m http.server 8000 --directory docs";
    return;
  }
  fbxLoader=new THREE.FBXLoader();
  setModelState("loading articulated FBX hand…","");
  fbxLoader.load(MODEL_URL,function(obj){
    fbxSource=obj;
    prepareFBXMaterials(obj);
    normalizeModel(obj);
    createModelInstances();
    fbxReady=true;
    setModelState("Articulated FBX hand ready","ready");
    modelDiagnostics="Supplied FBX hand loaded\nIndependent bone chains: thumb, index, middle, ring, pinky\nEach mapped bone is rotated from its own landmark segment.\nThe HandCT OBJ remains available as a licensed anatomical fallback.";
    if(currentFrames) renderFrame(lastWord);
  },function(xhr){
    if(xhr && xhr.total) setModelState("loading "+Math.round(xhr.loaded/xhr.total*100)+"%","");
  },function(err){
    console.error("Anatomical hand load failed",err);
    setModelState("FBX failed — loading anatomical fallback","warn");
    loadOBJFallback();
  });
}

function loadOBJFallback(){
  if(!THREE.OBJLoader) {
    modelDiagnostics="Neither the supplied FBX nor the OBJ fallback could be loaded.";
    return;
  }
  objLoader=new THREE.OBJLoader();
  objLoader.load(FALLBACK_MODEL_URL,function(obj){
    objSource=obj;
    obj.traverse(function(node){
      if(!node.isMesh) return;
      node.material=new THREE.MeshStandardMaterial({color:0xe0ac85,roughness:0.42});
    });
    normalizeModel(obj);
    modelGroup.add(obj);
    setModelState("Anatomical fallback ready","ready");
    modelDiagnostics="The supplied FBX could not be loaded, so the licensed HandCT OBJ is displayed.";
    if(currentFrames) renderFrame(lastWord);
  },null,function(){
    setModelState("3D model unavailable","warn");
    modelDiagnostics="WebGL or model loading failed. Close other WebGL tabs and reload this page.";
  });
}

function createModelInstances(){
  if(!fbxSource || !modelGroup) return;
  ["right"].forEach(function(k){
    if(modelInstances[k]) { modelGroup.remove(modelInstances[k]); safeDisposeObject(modelInstances[k]); }
    modelInstances[k]=cloneFBX(fbxSource);
    modelInstances[k].visible=false;
    modelGroup.add(modelInstances[k]);
    normalizeModel(modelInstances[k]);
    rigProfiles.set(modelInstances[k],makeRigProfile(modelInstances[k]));
  });
  modelInstances.left=null;
}

function lerpPoints(prev, next, amount){
  if(!next) return null;
  if(!prev || amount<=0) return next.map(function(p){return p.slice();});
  return next.map(function(p,i){
    var q=prev[i]||p;
    return [q[0]+(p[0]-q[0])*amount,q[1]+(p[1]-q[1])*amount,q[2]+(p[2]-q[2])*amount];
  });
}

function smoothFrame(frame){
  var out={left:null,right:null};
  ["left","right"].forEach(function(side){
    out[side]=lerpPoints(previousPose[side],frame[side],poseSmoothing);
    if(frame[side]) previousPose[side]=out[side];
    else previousPose[side]=null;
  });
  return out;
}

function setMaterialSkin(root,hex){
  root.traverse(function(o){
    if(!o.isMesh) return;
    var mats=Array.isArray(o.material)?o.material:[o.material];
    mats.forEach(function(m){
      if(!m || !m.color) return;
      var nail=/nail|finger.?tip|fingernail/i.test((m.name||"")+" "+(o.name||""));
      m.color.set(nail ? hex : hex);
      if(nail) m.color.lerp(new THREE.Color(0xffffff),0.58);
    });
  });
}

function clampAngle(a, maxAbs){
  return Math.max(-maxAbs, Math.min(maxAbs, a));
}

function smoothBoneRotation(bone, target, alpha){
  // Quaternion slerp prevents abrupt Euler-angle jumps and avoids gimbal lock.
  bone.quaternion.slerp(target, alpha);
}

function poseFBXInstance(instance, points){
  if(!instance || !points) return false;
  var profile=rigProfiles.get(instance);
  if(!profile || !profile.valid) return false;

  instance.visible=true;
  var alpha = Math.max(0.08, 1.0 - poseSmoothing * 0.82);

  // First solve all finger directions in world space. The desired direction is
  // then converted into the bone parent's local space, preserving the exact
  // rest orientation of the supplied FBX rig.
  profile.mappings.forEach(function(m){
    var a=vec(points[m.a]), b=vec(points[m.b]);
    var desired=b.sub(a);
    if(desired.lengthSq()<1e-8) return;
    desired.normalize();

    var parent=m.bone.parent;
    var parentQ=parent ? parent.getWorldQuaternion(new THREE.Quaternion()) : new THREE.Quaternion();
    var desiredLocal=desired.clone().applyQuaternion(parentQ.clone().invert()).normalize();
    var deltaQ=new THREE.Quaternion().setFromUnitVectors(m.restDirParent,desiredLocal);
    var targetQ=deltaQ.multiply(m.restQuat.clone());

    // Limit extreme single-frame rotations. This keeps noisy landmark frames
    // from folding a finger through the palm while retaining real sign motion.
    var dot=Math.abs(m.bone.quaternion.dot(targetQ));
    if(dot < 0.15) targetQ.slerp(m.restQuat, 0.12);
    smoothBoneRotation(m.bone,targetQ,alpha);
  });

  // Re-anchor after bone rotations so the wrist stays exactly on landmark 0.
  instance.updateMatrixWorld(true);
  var rootWorld=worldPointOf(profile.root);
  var desiredWrist=vec(points[0]);
  instance.position.add(desiredWrist.sub(rootWorld));
  instance.updateMatrixWorld(true);

  return true;
}

function hideModels(){
  Object.keys(modelInstances).forEach(function(k){ if(modelInstances[k]) modelInstances[k].visible=false; });
}

function renderModelsForFrame(frame){
  hideModels();
  var ok=false;
  if(frame.right && modelInstances.right) ok=poseFBXInstance(modelInstances.right,frame.right)||ok;
  else if(frame.left && modelInstances.right) ok=poseFBXInstance(modelInstances.right,frame.left)||ok;
  // Put left/right models apart only when both are present and the FBX rig is
  // unable to preserve their captured relative positions.
  return ok;
}

function resetCamera(){
  if(!camera||!controls) return;
  camera.position.set(0,0.15,3.4);
  controls.target.set(0,0,0);
  if(group) group.scale.setScalar(1);
  if(objSource){ objSource.rotation.set(0,0,0); objSource.scale.setScalar(modelBaseScale*modelScaleUser); }
  Object.keys(modelInstances).forEach(function(k){
    if(modelInstances[k]) modelInstances[k].scale.setScalar(modelBaseScale*modelScaleUser);
  });
  if (controls) controls.update();
}
function setCameraPreset(mode){
  if(!camera||!controls) return;
  if(mode==='front') camera.position.set(0,0.12,3.4);
  if(mode==='side') camera.position.set(3.4,0.12,0);
  controls.target.set(0,0,0); controls.update();
}
function fitHandToView(){
  if(!group||!camera||!controls||!group.children.length) return;
  var box=new THREE.Box3().setFromObject(group);
  var center=box.getCenter(new THREE.Vector3());
  var size=box.getSize(new THREE.Vector3());
  var radius=Math.max(size.x,size.y,size.z)*0.5;
  var distance=Math.max(radius/Math.tan(THREE.MathUtils.degToRad(camera.fov*0.5))*1.35,1.7);
  camera.position.copy(center).add(new THREE.Vector3(0,0,distance));
  controls.target.copy(center);
  controls.update();
}
function downloadPoseJson(){
  if(!currentFrames||!lastWord) return;
  var payload={sign:lastWord,frame:frameIdx+1,totalFrames:currentFrames.length,pose:currentFrames[frameIdx]};
  var blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});
  var link=document.createElement("a");
  link.download=lastWord+"_frame_"+(frameIdx+1)+".json";
  link.href=URL.createObjectURL(blob);
  link.click();
  setTimeout(function(){URL.revokeObjectURL(link.href);},1000);
}
function showDiagnostics(){
  var d=document.getElementById('modelDiag'); if(!d) return;
  d.hidden=!d.hidden;
  if(!d.hidden){
    var leftP=modelInstances.right?rigProfiles.get(modelInstances.right):null;
    if(leftP&&leftP.valid){
      d.textContent=modelDiagnostics+"\nRig mapping: "+leftP.mode+"\nMapped finger bones: "+leftP.mappings.length+"\nThumb: "+leftP.mappings.filter(function(m){return m.finger===0;}).length+" bones\nIndex: "+leftP.mappings.filter(function(m){return m.finger===1;}).length+" bones\nMiddle: "+leftP.mappings.filter(function(m){return m.finger===2;}).length+" bones\nRing: "+leftP.mappings.filter(function(m){return m.finger===3;}).length+" bones\nPinky: "+leftP.mappings.filter(function(m){return m.finger===4;}).length+" bones";
    }else{
      d.textContent=modelDiagnostics+"\nRig mapping: unavailable"+(leftP&&leftP.reason?"\nReason: "+leftP.reason:"");
    }
  }
}


// ---- math / geometry helpers ----
function vec(p) { return new THREE.Vector3(p[0], p[1], p[2]); }

function palmNormal(points) {
  var a = vec(points[0]), b = vec(points[5]), c = vec(points[17]);
  var v1 = new THREE.Vector3().subVectors(b, a);
  var v2 = new THREE.Vector3().subVectors(c, a);
  return new THREE.Vector3().crossVectors(v1, v2).normalize();
}

// Base radius at sample j (0..totalSegments) interpolated between control-point
// radii, plus a cosine "knuckle" bump centered on every interior joint so
// fingers taper smoothly but still swell a little at each joint like real
// knuckles instead of reading as a uniform tapered cone.
function radiusAt(j, ctrlRadii, n, totalSegments) {
  var t = j / totalSegments;
  var f = t * (n - 1);
  var i0 = Math.min(n - 2, Math.floor(f));
  var frac = f - i0;
  var base = ctrlRadii[i0] + (ctrlRadii[i0 + 1] - ctrlRadii[i0]) * frac;
  var bump = 0;
  for (var k = 1; k < n - 1; k++) {
    var jointJ = k * SEGMENTS_PER_SPAN;
    var dist = Math.abs(j - jointJ);
    var width = SEGMENTS_PER_SPAN * 0.9;
    if (dist < width) {
      var w = Math.cos((dist / width) * (Math.PI / 2));
      if (w > bump) bump = w;
    }
  }
  return base * (1 + bump * KNUCKLE_BUMP);
}

// Builds one finger (or thumb) as a smooth lofted tube along a Catmull-Rom
// spline through its joints, with an elliptical cross-section (flattened
// front-to-back like a real finger) and a rounded fingertip cap.
function buildFingerTube(points, chain, flattenAxisRef) {
  var ctrlPts = chain.map(function (i) { return vec(points[i]); });
  var ctrlRadii = chain.map(function (i) { return radiusByLandmark[i]; });
  var n = ctrlPts.length;
  var curve = new THREE.CatmullRomCurve3(ctrlPts);
  var totalSegments = (n - 1) * SEGMENTS_PER_SPAN;
  var RS = FINGER_RADIAL_SEGMENTS;

  var positions = [];
  var uvs = [];
  var ringStart = [];

  for (var j = 0; j <= totalSegments; j++) {
    var t = j / totalSegments;
    var center = curve.getPoint(t);
    var tangent = curve.getTangent(t);
    if (tangent.lengthSq() < 1e-8) tangent.set(0, 1, 0);
    var sideAxis = new THREE.Vector3().crossVectors(tangent, flattenAxisRef);
    if (sideAxis.lengthSq() < 1e-6) sideAxis.set(1, 0, 0); else sideAxis.normalize();
    var normalAxis = new THREE.Vector3().crossVectors(sideAxis, tangent).normalize();
    var r = radiusAt(j, ctrlRadii, n, totalSegments);

    ringStart.push(positions.length / 3);
    for (var a = 0; a < RS; a++) {
      var ang = (a / RS) * Math.PI * 2;
      var cx = Math.cos(ang) * r * FLATTEN_NORMAL;
      var sx = Math.sin(ang) * r * FLATTEN_SIDE;
      positions.push(
        center.x + normalAxis.x * cx + sideAxis.x * sx,
        center.y + normalAxis.y * cx + sideAxis.y * sx,
        center.z + normalAxis.z * cx + sideAxis.z * sx
      );
      uvs.push(a / RS, t);
    }
  }

  var indices = [];
  for (var j2 = 0; j2 < totalSegments; j2++) {
    var r0 = ringStart[j2], r1 = ringStart[j2 + 1];
    for (var a2 = 0; a2 < RS; a2++) {
      var a3 = (a2 + 1) % RS;
      indices.push(r0 + a2, r1 + a2, r1 + a3);
      indices.push(r0 + a2, r1 + a3, r0 + a3);
    }
  }

  // rounded fingertip cap: a single pole projected forward along the tip
  // tangent, fanned from the last ring
  var tipT = 1;
  var tipCenter = curve.getPoint(tipT);
  var tipTangent = curve.getTangent(tipT);
  var tipRadius = radiusAt(totalSegments, ctrlRadii, n, totalSegments);
  var poleIndex = positions.length / 3;
  var pole = tipCenter.clone().addScaledVector(tipTangent, tipRadius * 0.85);
  positions.push(pole.x, pole.y, pole.z);
  uvs.push(0.5, 1);
  var lastRing = ringStart[totalSegments];
  for (var a4 = 0; a4 < RS; a4++) {
    var a5 = (a4 + 1) % RS;
    indices.push(lastRing + a4, poleIndex, lastRing + a5);
  }

  var geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  var fingerMesh = new THREE.Mesh(geo, skinMat);
  fingerMesh.castShadow = true;
  fingerMesh.receiveShadow = true;
  group.add(fingerMesh);
}

// Builds the palm as a smooth "pillow" mass: a closed Catmull-Rom rim
// through the wrist / thumb-base / knuckle landmarks, domed outward on the
// palm-pad side and more subtly on the back-of-hand side, instead of a flat
// prism. This is what previously made the hand read as a thin plate with
// sausage fingers glued on.
function buildPalmMound(points, normal) {
  var loopPts = palmLoop.map(function (i) { return vec(points[i]); });
  var rimCurve = new THREE.CatmullRomCurve3(loopPts, true, "catmullrom", 0.4);
  var rim = rimCurve.getPoints(PALM_RIM_SEGMENTS);
  rim.pop(); // closed curve repeats the first point at the end

  var center = new THREE.Vector3();
  rim.forEach(function (p) { center.add(p); });
  center.multiplyScalar(1 / rim.length);

  var frontApex = center.clone().addScaledVector(normal, PALM_FRONT_HEIGHT);
  var backApex = center.clone().addScaledVector(normal, -PALM_BACK_HEIGHT);

  var positions = [];
  var uvs = [];
  var N = rim.length;
  var frontStart = 0;
  rim.forEach(function (p, i) {
    var fp = p.clone().addScaledVector(normal, PALM_RIM_HALF);
    positions.push(fp.x, fp.y, fp.z);
    uvs.push(i / N, 0.72);
  });
  var backStart = N;
  rim.forEach(function (p, i) {
    var bp = p.clone().addScaledVector(normal, -PALM_RIM_HALF);
    positions.push(bp.x, bp.y, bp.z);
    uvs.push(i / N, 0.28);
  });
  var frontApexIdx = positions.length / 3;
  positions.push(frontApex.x, frontApex.y, frontApex.z);
  uvs.push(0.5, 1);
  var backApexIdx = positions.length / 3;
  positions.push(backApex.x, backApex.y, backApex.z);
  uvs.push(0.5, 0);

  var indices = [];
  for (var i = 0; i < N; i++) {
    var a = i, b = (i + 1) % N;
    indices.push(frontStart + a, frontStart + b, frontApexIdx);
    indices.push(backStart + b, backStart + a, backApexIdx);
    indices.push(frontStart + a, backStart + a, backStart + b);
    indices.push(frontStart + a, backStart + b, frontStart + b);
  }

  var geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  var palmMesh = new THREE.Mesh(geo, skinMat);
  palmMesh.castShadow = true;
  palmMesh.receiveShadow = true;
  group.add(palmMesh);
}

function addForearmStub(points) {
  var wrist = vec(points[0]);
  var midMcp = vec(points[9]);
  var dir = new THREE.Vector3().subVectors(wrist, midMcp).normalize();
  var end = wrist.clone().addScaledVector(dir, 0.55);
  var r0 = radiusByLandmark[0];

  var pa = wrist, pb = end;
  var boneDir = new THREE.Vector3().subVectors(pb, pa);
  var length = boneDir.length() || 0.001;
  var geo = new THREE.CylinderGeometry(r0 * 0.92, r0 * 1.18, length, 20, 2);
  var mesh = new THREE.Mesh(geo, skinMat);
  mesh.position.copy(pa.clone().add(pb).multiplyScalar(0.5));
  var quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), boneDir.clone().normalize());
  mesh.setRotationFromQuaternion(quat);
  group.add(mesh);

  var wristCuff = new THREE.Mesh(new THREE.SphereGeometry(r0 * 1.05, 16, 10), skinMat);
  wristCuff.position.copy(pa);
  wristCuff.scale.set(1.15, 0.9, 1.05);
  group.add(wristCuff);

  var cap = new THREE.Mesh(new THREE.SphereGeometry(r0 * 0.8, 12, 10), skinMat);
  cap.position.copy(end);
  group.add(cap);
}

function addAnatomicalPads(points, normal) {
  // Rounded thenar/hypothenar pads and soft MCP knuckles give the landmark
  // skeleton a human silhouette instead of a collection of tubes.
  var palmCenter = vec(points[0]).lerp(vec(points[9]), 0.48);
  var padSpecs = [
    { p: vec(points[1]).lerp(palmCenter, 0.35), scale: [0.18, 0.12, 0.075] },
    { p: vec(points[17]).lerp(palmCenter, 0.38), scale: [0.15, 0.11, 0.065] }
  ];
  padSpecs.forEach(function (spec) {
    var mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 12), skinMat);
    mesh.position.copy(spec.p).addScaledVector(normal, 0.035);
    mesh.scale.set(spec.scale[0], spec.scale[1], spec.scale[2]);
    var q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
    mesh.quaternion.copy(q);
    group.add(mesh);
  });

  [5, 9, 13, 17].forEach(function (idx) {
    var knuckle = new THREE.Mesh(
      new THREE.SphereGeometry(radiusByLandmark[idx] * 1.12, 14, 10),
      skinMat
    );
    knuckle.position.copy(vec(points[idx])).addScaledVector(normal, 0.018);
    group.add(knuckle);
  });
}

// A slightly domed, elongated nail plate instead of a flat box - built from
// a partial sphere ("cap") so it reads as a convex nail rather than a chip
// of plastic glued to the fingertip.
function addNail(points, normal, tipIdx, prevIdx) {
  var tip = vec(points[tipIdx]);
  var prev = vec(points[prevIdx]);
  var boneDir = new THREE.Vector3().subVectors(tip, prev).normalize();
  var side = new THREE.Vector3().crossVectors(boneDir, normal).normalize();
  var normal2 = new THREE.Vector3().crossVectors(side, boneDir).normalize();
  var r = radiusByLandmark[tipIdx];

  var geo = new THREE.SphereGeometry(r * 0.98, 12, 8, 0, Math.PI * 2, 0, Math.PI * 0.5);
  geo.scale(1.05, 0.4, 1.3); // width, dome thickness, length (local axes before basis rotation)

  var nail = new THREE.Mesh(geo, nailMat);
  var pos = tip.clone().addScaledVector(boneDir, -r * 0.35).addScaledVector(normal2, r * 0.6);
  nail.position.copy(pos);
  var m = new THREE.Matrix4().makeBasis(side, normal2, boneDir);
  nail.quaternion.setFromRotationMatrix(m);
  group.add(nail);
}

// Optional thin skeleton overlay (small dots + lines) for debugging the
// underlying landmark data - handy since this viewer doubles as a sanity
// check for the dataset export pipeline.
function addSkeletonOverlay(points) {
  var dotGeo = new THREE.SphereGeometry(0.012, 8, 8);
  var dotMat = new THREE.MeshBasicMaterial({ color: 0x5fd9c7 });
  points.forEach(function (p) {
    var m = new THREE.Mesh(dotGeo, dotMat);
    m.position.copy(vec(p));
    debugGroup.add(m);
  });
  var lineMat = new THREE.LineBasicMaterial({ color: 0x5fd9c7 });
  var verts = [];
  skeletonConnections.forEach(function (pair) {
    var a = points[pair[0]], b = points[pair[1]];
    verts.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  });
  var lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  debugGroup.add(new THREE.LineSegments(lineGeo, lineMat));
}

function addCapsuleBetween(a, b, r0, r1) {
  var start = vec(a), end = vec(b);
  var axis = new THREE.Vector3().subVectors(end, start);
  var length = Math.max(axis.length(), 0.001);
  var mid = start.clone().add(end).multiplyScalar(0.5);
  var geo = new THREE.CylinderGeometry(r1, r0, length, 16, 2);
  var mesh = new THREE.Mesh(geo, skinMat);
  mesh.position.copy(mid);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.normalize());
  group.add(mesh);
  var joint = new THREE.Mesh(new THREE.SphereGeometry(Math.max(r0, r1) * 1.05, 14, 10), skinMat);
  joint.position.copy(start);
  group.add(joint);
}

function addJointPad(point, radius, normal, direction, scale) {
  var pad = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 12), skinMat);
  pad.position.copy(vec(point)).addScaledVector(normal, radius * 0.16);
  pad.scale.set(radius * scale[0], radius * scale[1], radius * scale[2]);
  var axis = direction.clone().normalize();
  var side = new THREE.Vector3().crossVectors(axis, normal).normalize();
  var face = new THREE.Vector3().crossVectors(side, axis).normalize();
  pad.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(side, axis, face));
  group.add(pad);
}

function addThumbWebbing(points, normal) {
  var thumbBase = vec(points[1]).lerp(vec(points[2]), 0.25);
  var indexBase = vec(points[5]).lerp(vec(points[6]), 0.12);
  var web = new THREE.Mesh(
    new THREE.SphereGeometry(1, 20, 14),
    skinMat
  );
  web.position.copy(thumbBase).lerp(indexBase, 0.5).addScaledVector(normal, 0.01);
  web.scale.set(thumbBase.distanceTo(indexBase) * 0.58, 0.075, 0.055);
  var axis = new THREE.Vector3().subVectors(indexBase, thumbBase).normalize();
  web.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(
    axis, normal, new THREE.Vector3().crossVectors(axis, normal).normalize()
  ));
  group.add(web);
}

function addFingerJointDetails(points, normal) {
  fingerChains.forEach(function(chain, finger) {
    for (var i = 0; i < chain.length; i++) {
      var index = chain[i];
      var next = points[chain[Math.min(i + 1, chain.length - 1)]];
      var previous = points[chain[Math.max(i - 1, 0)]];
      var direction = new THREE.Vector3().subVectors(vec(next), vec(previous));
      var jointScale = finger === 0
        ? [1.18, 0.82, 0.72]
        : (i === chain.length - 1 ? [1.02, 0.78, 0.66] : [1.12, 0.86, 0.72]);
      addJointPad(points[index], radiusByLandmark[index] * 0.82, normal, direction, jointScale);
    }
  });
}

function addPalmBody(points, normal) {
  var wrist = vec(points[0]), middle = vec(points[9]);
  var across = new THREE.Vector3().subVectors(vec(points[5]), vec(points[17]));
  var up = new THREE.Vector3().subVectors(middle, wrist).normalize();
  var side = across.normalize();
  var palmNormalAxis = new THREE.Vector3().crossVectors(side, up).normalize();
  if (palmNormalAxis.dot(normal) < 0) palmNormalAxis.negate();
  var center = wrist.clone().lerp(middle, 0.52);
  // SphereGeometry has a diameter of two units; use half-extents here so the
  // palm stays proportional to the captured fingers.
  var width = Math.max(across.length() * 0.31, 0.09);
  var height = Math.max(wrist.distanceTo(middle) * 0.36, 0.14);
  var depth = Math.max(width * 0.48, 0.06);
  var palm = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), skinMat);
  palm.position.copy(center).addScaledVector(palmNormalAxis, depth * 0.05);
  palm.scale.set(width, height, depth);
  palm.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(side, up, palmNormalAxis));
  group.add(palm);
  // A smaller thenar mound gives the thumb a natural web connection.
  var thenar = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 12), skinMat);
  thenar.position.copy(wrist).lerp(vec(points[5]), 0.42).addScaledVector(palmNormalAxis, depth * 0.5);
  thenar.scale.set(width * 0.42, height * 0.55, depth * 0.55);
  thenar.quaternion.copy(palm.quaternion);
  group.add(thenar);
}

function addFreshPalm(points, normal) {
  var wrist = vec(points[0]);
  var up = new THREE.Vector3().subVectors(vec(points[9]), wrist).normalize();
  var side = new THREE.Vector3().subVectors(vec(points[5]), vec(points[17])).normalize();
  var face = new THREE.Vector3().crossVectors(side, up).normalize();
  if (face.dot(normal) < 0) face.negate();
  var width = Math.max(vec(points[5]).distanceTo(vec(points[17])) * 0.62, 0.18);
  var height = Math.max(wrist.distanceTo(vec(points[9])) * 0.74, 0.22);
  var depth = Math.max(width * 0.30, 0.055);
  var shape = new THREE.Shape();
  shape.moveTo(-width * 0.42, -height * 0.50);
  shape.quadraticCurveTo(-width * 0.56, -height * 0.10, -width * 0.51, height * 0.28);
  shape.quadraticCurveTo(-width * 0.40, height * 0.56, -width * 0.18, height * 0.53);
  shape.quadraticCurveTo(0, height * 0.48, width * 0.18, height * 0.53);
  shape.quadraticCurveTo(width * 0.43, height * 0.55, width * 0.51, height * 0.22);
  shape.quadraticCurveTo(width * 0.56, -height * 0.18, width * 0.40, -height * 0.50);
  shape.quadraticCurveTo(0, -height * 0.62, -width * 0.42, -height * 0.50);
  var geo = new THREE.ExtrudeGeometry(shape, {
    depth: depth, bevelEnabled: true, bevelSegments: 3,
    bevelSize: Math.min(width, height) * 0.08, bevelThickness: depth * 0.35,
    curveSegments: 10
  });
  geo.translate(0, 0, -depth * 0.5);
  var palm = new THREE.Mesh(geo, skinMat);
  palm.position.copy(wrist).addScaledVector(up, height * 0.48);
  palm.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(side, up, face));
  group.add(palm);
  return { side: side, up: up, face: face };
}

function addFreshSegment(a, b, r0, r1, normal) {
  var start = vec(a), end = vec(b);
  var axis = new THREE.Vector3().subVectors(end, start);
  var length = Math.max(axis.length(), 0.002);
  var radius = Math.max(r0, r1);
  var geo = new THREE.CylinderGeometry(r1, r0, length, 12, 3);
  var mesh = new THREE.Mesh(geo, skinMat);
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.normalize());
  group.add(mesh);
  var joint = new THREE.Mesh(new THREE.SphereGeometry(radius * 0.78, 12, 8), skinMat);
  joint.position.copy(start).addScaledVector(normal, radius * 0.08);
  joint.scale.set(1.05, 0.82, 0.88);
  group.add(joint);
}

function addPalmCreases(points, normal) {
  var wrist = vec(points[0]), index = vec(points[5]), pinky = vec(points[17]);
  var center = wrist.clone().lerp(vec(points[9]), 0.48).addScaledVector(normal, 0.055);
  var side = new THREE.Vector3().subVectors(index, pinky).normalize();
  var up = new THREE.Vector3().subVectors(vec(points[9]), wrist).normalize();
  var mat = new THREE.LineBasicMaterial({color: 0x8c5d45, transparent: true, opacity: 0.34});
  [[-0.18, 0.12, 0.22], [-0.15, -0.03, 0.19], [0.05, -0.16, 0.16]].forEach(function(spec) {
    var curve = new THREE.CatmullRomCurve3([
      center.clone().addScaledVector(side, spec[0]).addScaledVector(up, spec[1]),
      center.clone().addScaledVector(side, spec[2]).addScaledVector(up, spec[1] * 0.55),
      center.clone().addScaledVector(side, spec[2] * 0.72).addScaledVector(up, spec[1] * 0.15)
    ]);
    var geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(10));
    group.add(new THREE.Line(geo, mat));
  });
}

function buildFreshHand(points) {
  var normal = palmNormal(points);
  addFreshPalm(points, normal);
  fingerChains.forEach(function(chain) {
    for (var i = 0; i < chain.length - 1; i++) {
      addFreshSegment(points[chain[i]], points[chain[i + 1]],
        radiusByLandmark[chain[i]] * 1.08,
        radiusByLandmark[chain[i + 1]] * 0.92, normal);
    }
    var tip = vec(points[chain[chain.length - 1]]);
    var tipSphere = new THREE.Mesh(
      new THREE.SphereGeometry(radiusByLandmark[chain[chain.length - 1]] * 0.9, 14, 9),
      skinMat
    );
    tipSphere.position.copy(tip);
    group.add(tipSphere);
  });
  addForearmStub(points);
  addPalmCreases(points, normal);
  fingertips.forEach(function (f) { addNail(points, normal, f.tip, f.prev); });
  if (showSkeleton) addSkeletonOverlay(points);
}

function buildHumanHand(points) {
  buildFreshHand(points);
}

function buildHandFull(points) {
  buildHumanHand(points);
}

// ---- scene setup ----
function setSkinColor(hex) {
  var c = new THREE.Color(hex);
  skinMat.color.copy(c);
  nailMat.color.copy(c).lerp(new THREE.Color(0xffffff), 0.55);
  Object.keys(modelInstances).forEach(function(k){ if(modelInstances[k]) setMaterialSkin(modelInstances[k],hex); });
  if (objSource) setMaterialSkin(objSource,hex);
  document.getElementById("skinPicker").value = hex;
}

// Generates a small tileable normal map (fine pore/crease noise) and a
// matching roughness map so the skin reads as soft organic tissue under
// the studio lights instead of a uniformly smooth plastic shell.
function makeSkinTextures() {
  var size = 256;
  var noiseCanvas = document.createElement("canvas");
  noiseCanvas.width = noiseCanvas.height = size;
  var nctx = noiseCanvas.getContext("2d");
  var img = nctx.createImageData(size, size);
  for (var i = 0; i < size * size; i++) {
    // layered value noise: fine pores + slightly larger crease clusters
    var fine = Math.random();
    var cx = (i % size) / size, cy = Math.floor(i / size) / size;
    var coarse = 0.5 + 0.5 * Math.sin(cx * 37 + Math.sin(cy * 29) * 2.1) * Math.sin(cy * 31 + Math.sin(cx * 23) * 1.7);
    var v = Math.max(0, Math.min(1, fine * 0.55 + coarse * 0.45));
    var idx = i * 4;
    img.data[idx] = img.data[idx + 1] = img.data[idx + 2] = Math.floor(v * 255);
    img.data[idx + 3] = 255;
  }
  nctx.putImageData(img, 0, 0);

  // Convert the greyscale height noise into an approximate normal map.
  var normalCanvas = document.createElement("canvas");
  normalCanvas.width = normalCanvas.height = size;
  var nrmCtx = normalCanvas.getContext("2d");
  var src = nctx.getImageData(0, 0, size, size).data;
  var out = nrmCtx.createImageData(size, size);
  var strength = 1.6;
  for (var y = 0; y < size; y++) {
    for (var x = 0; x < size; x++) {
      var xL = src[(y * size + ((x - 1 + size) % size)) * 4] / 255;
      var xR = src[(y * size + ((x + 1) % size)) * 4] / 255;
      var yU = src[(((y - 1 + size) % size) * size + x) * 4] / 255;
      var yD = src[(((y + 1) % size) * size + x) * 4] / 255;
      var nx = (xL - xR) * strength;
      var ny = (yU - yD) * strength;
      var nz = 1.0;
      var len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      var o = (y * size + x) * 4;
      out.data[o] = Math.floor(((nx / len) * 0.5 + 0.5) * 255);
      out.data[o + 1] = Math.floor(((ny / len) * 0.5 + 0.5) * 255);
      out.data[o + 2] = Math.floor(((nz / len) * 0.5 + 0.5) * 255);
      out.data[o + 3] = 255;
    }
  }
  nrmCtx.putImageData(out, 0, 0);

  var normalTex = new THREE.CanvasTexture(normalCanvas);
  normalTex.wrapS = normalTex.wrapT = THREE.RepeatWrapping;
  normalTex.repeat.set(6, 6);

  var roughTex = new THREE.CanvasTexture(noiseCanvas);
  roughTex.wrapS = roughTex.wrapT = THREE.RepeatWrapping;
  roughTex.repeat.set(6, 6);

  return { normal: normalTex, roughness: roughTex };
}

function makeShadowTexture() {
  var size = 128;
  var canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  var ctx = canvas.getContext("2d");
  var grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, "rgba(0,0,0,0.45)");
  grad.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

var keyLight, fillLight, rimLight, hemiLight, shadowMesh;

function applyLightPreset(name) {
  var p = LIGHT_PRESETS[name] || LIGHT_PRESETS.studio;
  scene.background = new THREE.Color(p.bg);
  hemiLight.color.setHex(p.hemi[0]);
  hemiLight.groundColor.setHex(p.hemi[1]);
  hemiLight.intensity = p.hemi[2];
  keyLight.color.setHex(p.key[0]);
  keyLight.intensity = p.key[1];
  fillLight.color.setHex(p.fill[0]);
  fillLight.intensity = p.fill[1];
  rimLight.color.setHex(p.rim[0]);
  rimLight.intensity = p.rim[1];
}

function initThree() {
  scene = new THREE.Scene();
  var w = wrap.clientWidth, h = wrap.clientHeight;
  camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
  camera.position.set(0, 0.15, 3.4);

  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: false, powerPreference: "high-performance" });
  } catch (err) {
    setModelState("WebGL unavailable — close other tabs and reload","warn");
    document.getElementById("performanceBadge").textContent = "3D renderer could not create a WebGL context";
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h);
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  wrap.appendChild(renderer.domElement);
  renderer.domElement.addEventListener("webglcontextlost", function(e) {
    e.preventDefault();
    setModelState("WebGL context lost — reload this page","warn");
  }, false);

  hemiLight = new THREE.HemisphereLight(0xfff4e0, 0x1a1410, 0.55);
  scene.add(hemiLight);
  keyLight = new THREE.DirectionalLight(0xfff2e0, 1.0);
  keyLight.position.set(1.2, 2.4, 2.2);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(1024, 1024);
  keyLight.shadow.camera.near = 0.1;
  keyLight.shadow.camera.far = 8;
  keyLight.shadow.bias = -0.0005;
  scene.add(keyLight);
  fillLight = new THREE.DirectionalLight(0xcfe8ff, 0.35);
  fillLight.position.set(-1.5, -0.3, 1.4);
  scene.add(fillLight);
  rimLight = new THREE.DirectionalLight(0x5fd9c7, 0.4);
  rimLight.position.set(-0.6, 1.2, -2);
  scene.add(rimLight);

  var skinMaps = makeSkinTextures();
  skinMat = new THREE.MeshPhysicalMaterial({
    color: 0xe0ac85, roughness: 0.42, metalness: 0.0,
    clearcoat: 0.05, clearcoatRoughness: 0.75,
    normalMap: skinMaps.normal, normalScale: new THREE.Vector2(0.35, 0.35),
    roughnessMap: skinMaps.roughness,
    emissive: 0x5a1c10, emissiveIntensity: 0.035
  });
  nailMat = new THREE.MeshPhysicalMaterial({ color: 0xf2e4d8, roughness: 0.22, metalness: 0.0, clearcoat: 0.65, clearcoatRoughness: 0.2 });

  group = new THREE.Group();
  scene.add(group);
  modelGroup = new THREE.Group();
  scene.add(modelGroup);
  debugGroup = new THREE.Group();
  scene.add(debugGroup);

  var grid = new THREE.GridHelper(2.2, 12, 0x6fb7ff, 0x2a2f36);
  grid.position.y = -0.92;
  grid.material.transparent = true;
  grid.material.opacity = 0.1;
  scene.add(grid);

  var shadowMat = new THREE.MeshBasicMaterial({ map: makeShadowTexture(), transparent: true, depthWrite: false });
  shadowMesh = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 1.7), shadowMat);
  shadowMesh.rotation.x = -Math.PI / 2;
  shadowMesh.position.y = -0.91;
  scene.add(shadowMesh);

  // Invisible catcher plane that receives the real-time shadow map so the
  // hand grounds itself under the studio key light instead of floating.
  var catcherMat = new THREE.ShadowMaterial({ opacity: 0.32 });
  var shadowCatcher = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), catcherMat);
  shadowCatcher.rotation.x = -Math.PI / 2;
  shadowCatcher.position.y = -0.905;
  shadowCatcher.receiveShadow = true;
  scene.add(shadowCatcher);

  applyLightPreset("studio");

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.minDistance = 1.6;
  controls.maxDistance = 7;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.3;
  controls.addEventListener("start", function () {
    controls.autoRotate = false;
    if (idleTimer) clearTimeout(idleTimer);
  });
  controls.addEventListener("end", function () {
    if (!document.getElementById("autoRotateToggle").checked) return;
    idleTimer = setTimeout(function () { controls.autoRotate = true; }, 4000);
  });

  window.addEventListener("resize", onResize);
  setModelState("Articulated landmark hand ready","ready");
  modelDiagnostics="Runtime articulated hand active\nEach thumb and finger segment is generated from its own landmark joint.\nThe supplied FBX remains available in docs for reference.";
  animate();
}

function onResize() {
  if (!renderer || !camera) return;
  var w = wrap.clientWidth, h = wrap.clientHeight;
  if (!w || !h) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

function clearGroup(g) {
  if (!g) return;
  while (g.children.length) {
    var child = g.children.pop();
    if (child.geometry && child.geometry.dispose) child.geometry.dispose();
    if (child.material) {
      var mats=Array.isArray(child.material)?child.material:[child.material];
      mats.forEach(function(m){ if(m && m.dispose) m.dispose(); });
    }
  }
}

function updateStatus(word, handsPresent) {
  var status = document.getElementById("status");
  status.innerHTML = "";
  var label = document.createElement("span");
  label.textContent = "Showing: " + word;
  label.style.color = "#7fd995";
  status.appendChild(label);
  handsPresent.forEach(function (h) {
    var badge = document.createElement("span");
    badge.className = "handBadge";
    badge.textContent = h === "L" ? "LEFT" : "RIGHT";
    status.appendChild(badge);
  });
}

function syncPlaybackUI() {
  var slider = document.getElementById("frameSlider");
  var label = document.getElementById("frameLabel");
  if (!currentFrames) {
    slider.max = 0; slider.value = 0; label.textContent = "0 / 0";
    return;
  }
  slider.max = currentFrames.length - 1;
  slider.value = frameIdx;
  label.textContent = (frameIdx + 1) + " / " + currentFrames.length;
}

function offsetHandPoints(points, xOffset) {
  return points.map(function (p) {
    return [p[0] + xOffset, p[1], p[2]];
  });
}

function adjustedPoints(points, side) {
  if (!points) return null;
  var out = points.map(function (p) { return p.slice(); });
  var edits = manualPose[side] || {};
  Object.keys(edits).forEach(function (key) {
    var index = parseInt(key, 10);
    var delta = edits[key];
    var chain = fingerChains.find(function (c) { return c.indexOf(index) !== -1; }) || [index];
    var start = chain.indexOf(index);
    for (var i = start; i < chain.length; i++) {
      var joint = chain[i];
      out[joint][0] += delta[0];
      out[joint][1] += delta[1];
      out[joint][2] += delta[2];
    }
  });
  return out;
}

function renderHand(points, xOffset) {
  if (!points) return;
  buildHandFull(offsetHandPoints(points, xOffset));
}

function renderFrame(word) {
  if (!group) return;
  clearGroup(group);
  if (!currentFrames || !currentFrames.length) { hideModels(); syncPlaybackUI(); return; }
  var rawFrame=currentFrames[frameIdx];
  var frame=smoothFrame(rawFrame);
  var leftPoints = adjustedPoints(frame.left, "left");
  var rightPoints = adjustedPoints(frame.right, "right");
  var handsPresent=[];
  if (leftPoints) handsPresent.push("L");
  if (rightPoints) handsPresent.push("R");
  hideModels();
  modelGroup.visible=false;
  var handCount = (leftPoints ? 1 : 0) + (rightPoints ? 1 : 0);
  if (handCount === 2) {
    // Keep two-hand signs readable without changing the captured relative pose.
    renderHand(leftPoints, -0.30);
    renderHand(rightPoints, 0.30);
  } else {
    renderHand(rightPoints || leftPoints, 0);
  }
  group.traverse(function (o) { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  group.visible=handCount > 0;
  if (showSkeleton) {
    clearGroup(debugGroup);
    if (handCount === 2) {
      if (leftPoints) addSkeletonOverlay(offsetHandPoints(leftPoints, -0.30));
      if (rightPoints) addSkeletonOverlay(offsetHandPoints(rightPoints, 0.30));
    } else {
      if (leftPoints) addSkeletonOverlay(leftPoints);
      if (rightPoints) addSkeletonOverlay(rightPoints);
    }
  }
  if (word) updateStatus(word, handsPresent);
  syncPlaybackUI();
}

function stepFrame() {
  if (!currentFrames || !currentFrames.length) return;
  var last = currentFrames.length - 1;
  if (loopMode === "loop") {
    frameIdx = (frameIdx + 1) % currentFrames.length;
  } else if (loopMode === "pingpong") {
    frameIdx += playDir;
    if (frameIdx >= last) { frameIdx = last; playDir = -1; }
    else if (frameIdx <= 0) { frameIdx = 0; playDir = 1; }
  } else { // once
    if (frameIdx >= last) { setPlaying(false); return; }
    frameIdx += 1;
  }
  renderFrame(lastWord);
}

function animate(t) {
  if (typeof t !== "number") t = performance.now();
  requestAnimationFrame(animate);
  var viewerEl = document.getElementById("viewerView");
  if (!viewerEl || viewerEl.style.display === "none") return; // Recognize tab active, skip 3D work
  fpsFrames++;
  if(t-fpsLastTime>=500){
    currentFPS=Math.round((fpsFrames*1000)/(t-fpsLastTime));
    fpsFrames=0; fpsLastTime=t;
    var perf=document.getElementById("performanceBadge");
    if(perf) perf.textContent="Articulated landmark hand · independent joint driver · "+currentFPS+" FPS";
  }
  controls.update();
  if (isPlaying && currentFrames && (!lastTick || t - lastTick > frameIntervalMs)) {
    stepFrame();
    lastTick = t;
  }
  if (renderer) renderer.render(scene, camera);
}

function setPlaying(playing) {
  isPlaying = playing;
  document.getElementById("playBtn").innerHTML = isPlaying ? "&#10074;&#10074;" : "&#9654;";
}

function refreshVocabList() {
  var list = document.getElementById("vocabList");
  var keys = Object.keys(vocab).sort();
  document.getElementById("vocabCount").textContent = keys.length;
  list.innerHTML = "";
  keys.forEach(function (k) {
    var btn = document.createElement("button");
    btn.textContent = k;
    btn.dataset.key = k;
    btn.onclick = function () { addSentenceSign(k); playWord(k); };
    list.appendChild(btn);
  });
}

function renderSentenceItems() {
  var root = document.getElementById("sentenceItems");
  var count = document.getElementById("sentenceCount");
  root.innerHTML = "";
  count.textContent = sentence.length;
  if (!sentence.length) {
    root.innerHTML = '<span class="sentenceEmpty">Click signs to add them here</span>';
    return;
  }
  sentence.forEach(function (key, index) {
    var chip = document.createElement("button");
    chip.className = "sentenceChip";
    chip.textContent = (index + 1) + ". " + key + " ×";
    chip.title = "Remove " + key;
    chip.onclick = function () {
      sentence.splice(index, 1);
      renderSentenceItems();
    };
    root.appendChild(chip);
  });
}

function addSentenceSign(key) {
  if (!vocab[key]) return;
  sentence.push(key);
  renderSentenceItems();
  document.getElementById("wordInput").value = key;
}

function blendBoundary(fromFrame, toFrame, count) {
  var frames = [];
  for (var i = 1; i <= count; i++) {
    var amount = i / (count + 1);
    var blended = { left: null, right: null };
    ["left", "right"].forEach(function (side) {
      if (fromFrame[side] && toFrame[side]) {
        blended[side] = lerpPoints(fromFrame[side], toFrame[side], amount);
      } else {
        blended[side] = amount < 0.5 ? fromFrame[side] : toFrame[side];
      }
    });
    frames.push(blended);
  }
  return frames;
}

function buildSentenceFrames() {
  var frames = [];
  sentence.forEach(function (key, index) {
    var signFrames = vocab[key] || [];
    if (!signFrames.length) return;
    if (index > 0 && frames.length) {
      var previous = frames[frames.length - 1];
      frames.push.apply(frames, blendBoundary(previous, signFrames[0], sentenceBlendFrames));
    }
    signFrames.forEach(function (frame) {
      frames.push({
        left: frame.left ? frame.left.map(function (p) { return p.slice(); }) : null,
        right: frame.right ? frame.right.map(function (p) { return p.slice(); }) : null
      });
    });
  });
  return frames;
}

function playSentence() {
  var frames = buildSentenceFrames();
  if (!frames.length) {
    document.getElementById("status").textContent = "Add at least one sign to the sentence first.";
    return;
  }
  currentFrames = frames;
  frameIdx = 0;
  playDir = 1;
  lastWord = sentence.join(" ");
  previousPose = { left: null, right: null };
  isSentencePlayback = true;
  setPlaying(true);
  renderFrame(lastWord);
}

function highlightVocabButton(key) {
  if (activeVocabBtn) activeVocabBtn.classList.remove("active");
  var btn = document.querySelector('#vocabList button[data-key="' + CSS.escape(key) + '"]');
  if (btn) { btn.classList.add("active"); activeVocabBtn = btn; }
}

function playWord(word) {
  var key = word.trim().toLowerCase().replace(/\s+/g, "_");
  var status = document.getElementById("status");
  if (vocab[key]) {
    isSentencePlayback = false;
    currentFrames = vocab[key];
    frameIdx = 0;
    playDir = 1;
    lastWord = key;
    previousPose={left:null,right:null};
    setPlaying(true);
    renderFrame(key);
    highlightVocabButton(key);
    evaluateQuiz(key);
  } else {
    status.innerHTML = "";
    var span = document.createElement("span");
    span.textContent = '"' + word + '" is not in the vocabulary yet.';
    span.style.color = "#e08585";
    status.appendChild(span);
  }
}

function playRandomWord() {
  var keys = Object.keys(vocab);
  if (!keys.length) return;
  var k = keys[Math.floor(Math.random() * keys.length)];
  document.getElementById("wordInput").value = k;
  playWord(k);
}

function saveFrameAsImage() {
  if (!renderer) return;
  renderer.render(scene, camera);
  var link = document.createElement("a");
  var name = (lastWord || "hand") + "_frame" + (frameIdx + 1) + ".png";
  link.download = name;
  link.href = renderer.domElement.toDataURL("image/png");
  link.click();
}

function downloadText(name, text, type) {
  var blob = new Blob([text], {type: type || "application/json"});
  var link = document.createElement("a");
  link.download = name;
  link.href = URL.createObjectURL(blob);
  link.click();
  setTimeout(function () { URL.revokeObjectURL(link.href); }, 1000);
}

function exportGLB() {
  if (!scene || !window.GLTFExporter) return;
  var exporter = new THREE.GLTFExporter();
  exporter.parse(group, function (result) {
    var data = result instanceof ArrayBuffer ? result : JSON.stringify(result);
    downloadText((lastWord || "hand") + ".glb", data, result instanceof ArrayBuffer ? "model/gltf-binary" : "application/json");
  }, {binary: true});
}

function startPractice() {
  var keys = Object.keys(vocab);
  if (!keys.length) return;
  var key = keys[Math.floor(Math.random() * keys.length)];
  document.getElementById("quizPrompt").textContent = "Practice: " + key.replace(/_/g, " ");
  playWord(key);
}

function startQuiz() {
  var keys = Object.keys(vocab);
  if (!keys.length) return;
  quizActive = true;
  quizTarget = keys[Math.floor(Math.random() * keys.length)];
  quizTotal++;
  document.getElementById("quizPrompt").textContent = "Quiz: select \"" + quizTarget.replace(/_/g, " ") + "\"";
  document.getElementById("wordInput").focus();
}

function evaluateQuiz(key) {
  if (!quizActive) return;
  if (key === quizTarget) quizScore++;
  quizActive = false;
  document.getElementById("scoreLabel").textContent = Math.round((quizScore / Math.max(quizTotal, 1)) * 100) + "%";
  document.getElementById("quizPrompt").textContent = key === quizTarget ? "Correct!" : "Target was " + quizTarget;
}

function toggleWebcam() {
  var video = document.getElementById("webcamVideo");
  if (webcamStream) {
    webcamStream.getTracks().forEach(function (track) { track.stop(); });
    webcamStream = null;
    video.srcObject = null;
    video.style.display = "none";
    document.getElementById("webcamState").textContent = "OFF";
    document.getElementById("webcamBtn").textContent = "Enable webcam";
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    document.getElementById("confidenceLabel").textContent = "Webcam is unavailable in this browser.";
    return;
  }
  navigator.mediaDevices.getUserMedia({video: {facingMode: "user"}, audio: false}).then(function (stream) {
    webcamStream = stream;
    video.srcObject = stream;
    video.style.display = "block";
    document.getElementById("webcamState").textContent = "ON";
    document.getElementById("webcamBtn").textContent = "Disable webcam";
    document.getElementById("confidenceLabel").textContent = "Camera active. Landmark recognition requires a compatible CV model.";
  }).catch(function (error) {
    document.getElementById("confidenceLabel").textContent = "Camera permission failed: " + error.name;
  });
}

// ---- search + autocomplete ----
var wordInput = document.getElementById("wordInput");
var suggestBox = document.getElementById("suggestions");

function matchesFor(q) {
  var norm = q.toLowerCase().replace(/\s+/g, "_");
  return Object.keys(vocab).filter(function (k) {
    return k.indexOf(norm) === 0 || k.indexOf(norm) !== -1;
  }).sort().slice(0, 8);
}

wordInput.addEventListener("input", function () {
  var q = this.value.trim();
  suggestBox.innerHTML = "";
  if (!q) { suggestBox.style.display = "none"; return; }
  var matches = matchesFor(q);
  if (!matches.length) { suggestBox.style.display = "none"; return; }
  matches.forEach(function (k) {
    var item = document.createElement("div");
    item.className = "suggestionItem";
    item.textContent = k;
    item.onclick = function () {
      wordInput.value = k;
      suggestBox.style.display = "none";
      playWord(k);
    };
    suggestBox.appendChild(item);
  });
  suggestBox.style.display = "block";
});

wordInput.addEventListener("keydown", function (e) {
  if (e.key === "Enter") {
    suggestBox.style.display = "none";
    var q = this.value.trim();
    var norm = q.toLowerCase().replace(/\s+/g, "_");
    if (vocab[norm]) { playWord(q); return; }
    var matches = matchesFor(q);
    if (matches.length) playWord(matches[0]);
    else playWord(q);
  } else if (e.key === "Escape") {
    suggestBox.style.display = "none";
  }
});

document.addEventListener("click", function (e) {
  if (e.target !== wordInput) suggestBox.style.display = "none";
});

// ---- playback bar ----
document.getElementById("playBtn").addEventListener("click", function () {
  setPlaying(!isPlaying);
});
document.getElementById("frameSlider").addEventListener("input", function () {
  if (!currentFrames) return;
  setPlaying(false);
  frameIdx = parseInt(this.value, 10);
  renderFrame(lastWord);
});
document.getElementById("speedSelect").addEventListener("change", function () {
  frameIntervalMs = parseInt(this.value, 10);
});
document.getElementById("loopSelect").addEventListener("change", function () {
  loopMode = this.value;
  playDir = 1;
});
document.getElementById("playSentenceBtn").addEventListener("click", playSentence);
document.getElementById("clearSentenceBtn").addEventListener("click", function () {
  sentence = [];
  isSentencePlayback = false;
  renderSentenceItems();
});
document.getElementById("blendRange").addEventListener("input", function () {
  sentenceBlendFrames = parseInt(this.value, 10);
  document.getElementById("blendValue").textContent = sentenceBlendFrames + " frames";
});
document.getElementById("randomBtn").addEventListener("click", playRandomWord);
document.getElementById("saveImgBtn").addEventListener("click", saveFrameAsImage);
document.getElementById("practiceBtn").addEventListener("click", startPractice);
document.getElementById("quizBtn").addEventListener("click", startQuiz);
document.getElementById("webcamBtn").addEventListener("click", toggleWebcam);
document.getElementById("contrastToggle").addEventListener("change", function () {
  document.body.classList.toggle("highContrast", this.checked);
});
document.getElementById("fullscreenBtn").addEventListener("click", function () {
  var target = document.getElementById("right");
  if (!document.fullscreenElement) target.requestFullscreen().catch(function () {});
  else document.exitFullscreen();
});

// ---- skin tone ----
document.getElementById("skinPicker").addEventListener("input", function () {
  setSkinColor(this.value);
});
(function buildSkinSwatches() {
  var row = document.getElementById("skinSwatches");
  SKIN_PRESETS.forEach(function (hex) {
    var sw = document.createElement("button");
    sw.className = "swatch";
    sw.style.background = hex;
    sw.title = hex;
    sw.onclick = function () { setSkinColor(hex); };
    row.appendChild(sw);
  });
})();

// ---- view options ----
document.getElementById("mirrorToggle").addEventListener("change", function () {
  mirrorOn = this.checked;
  group.scale.x = mirrorOn ? -1 : 1;
  modelGroup.scale.x = mirrorOn ? -1 : 1;
  debugGroup.scale.x = mirrorOn ? -1 : 1;
});
document.getElementById("autoRotateToggle").addEventListener("change", function () {
  if (!this.checked) {
    controls.autoRotate = false;
    if (idleTimer) clearTimeout(idleTimer);
  } else {
    controls.autoRotate = true;
  }
});
document.getElementById("debugToggle").addEventListener("change", function () {
  showSkeleton = this.checked;
  clearGroup(debugGroup);
  if (showSkeleton && currentFrames) renderFrame(lastWord);
});
document.getElementById("lightSelect").addEventListener("change", function () {
  applyLightPreset(this.value);
});


// ---- FBX model controls ----
document.getElementById("smoothRange").addEventListener("input", function(){
  poseSmoothing=parseInt(this.value,10)/100;
  document.getElementById("smoothValue").textContent=this.value+"%";
});
document.getElementById("modelScaleRange").addEventListener("input", function(){
  modelScaleUser=parseInt(this.value,10)/100;
  document.getElementById("modelScaleValue").textContent=this.value+"%";
  group.scale.setScalar(modelScaleUser);
  Object.keys(modelInstances).forEach(function(k){
    if(modelInstances[k]) modelInstances[k].scale.setScalar(modelBaseScale*modelScaleUser);
  });
  if(objSource) objSource.scale.setScalar(modelBaseScale*modelScaleUser);
  if(currentFrames) renderFrame(lastWord);
});
document.getElementById("resetViewBtn").addEventListener("click", resetCamera);
document.getElementById("frontViewBtn").addEventListener("click", function(){setCameraPreset('front');});
document.getElementById("sideViewBtn").addEventListener("click", function(){setCameraPreset('side');});
document.getElementById("fitViewBtn").addEventListener("click", fitHandToView);
document.getElementById("poseJsonBtn").addEventListener("click", downloadPoseJson);
document.getElementById("glbBtn").addEventListener("click", exportGLB);
document.getElementById("diagnosticBtn").addEventListener("click", showDiagnostics);
document.getElementById("resetPoseBtn").addEventListener("click", function () {
  manualPose = { left: {}, right: {} };
  if (currentFrames) renderFrame(lastWord);
});

function nearestEditableJoint(event) {
  if (!renderer || !currentFrames || !currentFrames.length) return null;
  var frame = smoothFrame(currentFrames[frameIdx]);
  var rect = renderer.domElement.getBoundingClientRect();
  var pointer = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1
  );
  var best = null;
  ["left", "right"].forEach(function (side) {
    var points = adjustedPoints(frame[side], side);
    if (!points) return;
    var xOffset = frame.left && frame.right ? (side === "left" ? -0.30 : 0.30) : 0;
    points.forEach(function (p, index) {
      var projected = new THREE.Vector3(p[0] + xOffset, p[1], p[2]).project(camera);
      var px = (projected.x + 1) * 0.5 * rect.width;
      var py = (1 - projected.y) * 0.5 * rect.height;
      var distance = Math.hypot(px - (event.clientX - rect.left), py - (event.clientY - rect.top));
      if (distance < 42 && (!best || distance < best.distance)) {
        best = { side: side, index: index, distance: distance, xOffset: xOffset };
      }
    });
  });
  return best;
}

function installPointerControls() {
  if (!renderer) return;
  renderer.domElement.addEventListener("pointerdown", function (event) {
  var picked = nearestEditableJoint(event);
  if (!picked) return;
  dragState = { picked: picked, x: event.clientX, y: event.clientY };
  renderer.domElement.setPointerCapture(event.pointerId);
  });
  renderer.domElement.addEventListener("pointermove", function (event) {
  if (!dragState || !camera) return;
  var dx = event.clientX - dragState.x;
  var dy = event.clientY - dragState.y;
  dragState.x = event.clientX;
  dragState.y = event.clientY;
  var scale = Math.max(camera.position.distanceTo(controls.target), 1) / Math.max(renderer.domElement.clientHeight, 1) * 1.8;
  var right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  var up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
  var delta = right.multiplyScalar(dx * scale).add(up.multiplyScalar(-dy * scale));
  var sideEdits = manualPose[dragState.picked.side];
  var index = dragState.picked.index;
  var current = sideEdits[index] || [0, 0, 0];
  sideEdits[index] = [
    THREE.MathUtils.clamp(current[0] + delta.x, -0.22, 0.22),
    THREE.MathUtils.clamp(current[1] + delta.y, -0.22, 0.22),
    THREE.MathUtils.clamp(current[2] + delta.z, -0.12, 0.12)
  ];
  renderFrame(lastWord);
  });
  renderer.domElement.addEventListener("pointerup", function (event) {
  dragState = null;
  renderer.domElement.releasePointerCapture(event.pointerId);
  });
}

// ---- keyboard shortcuts ----
document.addEventListener("keydown", function (e) {
  if (document.activeElement === wordInput) return;
  if (e.code === "Space") { e.preventDefault(); setPlaying(!isPlaying); }
  else if (e.code === "ArrowRight") { setPlaying(false); stepFrame(); }
  else if (e.code === "ArrowLeft") {
    if (!currentFrames || !currentFrames.length) return;
    setPlaying(false);
    frameIdx = (frameIdx - 1 + currentFrames.length) % currentFrames.length;
    renderFrame(lastWord);
  } else if (e.key === "r" || e.key === "R") { playRandomWord(); }
});

refreshVocabList();
renderSentenceItems();
initThree();
installPointerControls();
if ("serviceWorker" in navigator) navigator.serviceWorker.register("service-worker.js").catch(function () {});
if (renderer) setSkinColor("#e0ac85");
document.getElementById("status").textContent = "Loaded " + Object.keys(vocab).length + " signs. Articulated landmark hand is ready.";

</script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js"></script>
<script>
// ---- Tab switching (shared by both views) ----
function switchTab(name) {
  var recognizeEl = document.getElementById("recognizeView");
  var viewerEl = document.getElementById("viewerView");
  var btnR = document.getElementById("tabBtnRecognize");
  var btnV = document.getElementById("tabBtnViewer");
  var heading = document.getElementById("tabHeading");
  var toViewer = name === "viewer";
  recognizeEl.style.display = toViewer ? "none" : "";
  viewerEl.style.display = toViewer ? "" : "none";
  btnR.classList.toggle("active", !toViewer);
  btnV.classList.toggle("active", toViewer);
  heading.textContent = toViewer ? "Sign Viewer" : "Recognize";
  if (toViewer) {
    onResize();
  } // three.js canvas may have been 0-sized while hidden
}
</script>
<script type="module">
import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// =====================================================================
// Live sign recognition, ported from recognize_live.py.
//
// Feature normalization note: landmark_utils.extract_feature_vector()
// wasn't available when this was written, so the normalization below
// (wrist-relative, unit-scaled to the wrist -> middle-fingertip
// distance) was reverse-engineered from the exported sign data in the
// Viewer tab, where it matched to ~5 decimal places. If live
// predictions look off, compare this against the real
// extract_feature_vector and adjust normalizeHandLandmarks() below.
// =====================================================================

const HAND_LANDMARKER_URL =
  "models_web/hand_landmarker.task";
const HAND_LANDMARKER_REMOTE_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const SEQUENCE_LENGTH = 30;
const NUM_LANDMARKS = 21;
const COORDS_PER_LANDMARK = 3;
const FEATURES_PER_HAND = NUM_LANDMARKS * COORDS_PER_LANDMARK; // 63
const VOTE_WINDOW = 3;

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

var confidenceThreshold = 0.6;
var ttsEnabled = true;
var drawSkeleton = true;

var handLandmarker = null;
var classifierModel = null;
var labelNames = null;
var fallbackPrototypes = [];

var stream = null;
var running = false;
var rafId = null;

var frameBuffer = []; // ring buffer of Float32Array(126), max SEQUENCE_LENGTH
var recentLabels = []; // ring buffer of label|null, max VOTE_WINDOW
var lastSpokenLabel = null;
var sentence = "";
var sessionState = "idle";
var sessionLabels = [];
var sessionLastLabel = null;
var sessionCandidateLabel = null;
var sessionCandidateCount = 0;
var SESSION_STABILITY_FRAMES = 5;

var fpsFrames = 0, fpsLastTime = performance.now();

var videoEl = document.getElementById("recVideoHidden");
var canvasEl = document.getElementById("recCanvas");
var ctx = canvasEl.getContext("2d");

// ---- label formatting (mirrors label_to_display / label_to_sentence_fragment) ----
function labelToDisplay(label) {
  if (label.startsWith("letter_")) return label.replace("letter_", "").toUpperCase();
  return label.replace(/_/g, " ");
}
function labelToSentenceFragment(label) {
  if (label.startsWith("letter_")) return label.replace("letter_", "").toUpperCase();
  return " " + label.replace(/_/g, " ");
}

// ---- majority_vote() port: winner needs >=2 of the last VOTE_WINDOW votes ----
function majorityVote(recent) {
  if (recent.length < VOTE_WINDOW) return null;
  var counts = new Map();
  recent.forEach(function (l) { counts.set(l, (counts.get(l) || 0) + 1); });
  var bestLabel = null, bestCount = -1;
  counts.forEach(function (count, label) {
    if (count > bestCount) { bestCount = count; bestLabel = label; }
  });
  if (bestLabel === null || bestCount < 2) return null;
  return bestLabel;
}

// ---- feature extraction: wrist-relative, unit-scaled to wrist->middle-fingertip ----
function normalizeHandLandmarks(landmarks) {
  var wrist = landmarks[0];
  var mid = landmarks[12];
  var relX12 = mid.x - wrist.x, relY12 = mid.y - wrist.y, relZ12 = mid.z - wrist.z;
  var scale = Math.sqrt(relX12 * relX12 + relY12 * relY12 + relZ12 * relZ12);
  if (!scale || scale < 1e-6) scale = 1;
  var out = new Float32Array(FEATURES_PER_HAND);
  for (var i = 0; i < NUM_LANDMARKS; i++) {
    var p = landmarks[i];
    out[i * 3] = (p.x - wrist.x) / scale;
    out[i * 3 + 1] = (p.y - wrist.y) / scale;
    out[i * 3 + 2] = (p.z - wrist.z) / scale;
  }
  return out;
}

function extractFeatureVector(result) {
  var vec = new Float32Array(FEATURES_PER_HAND * 2); // [left(63), right(63)]
  if (!result || !result.landmarks) return vec;
  for (var i = 0; i < result.landmarks.length; i++) {
    var handedness = result.handednesses && result.handednesses[i] && result.handednesses[i][0]
      ? result.handednesses[i][0].categoryName : null;
    if (handedness !== "Left" && handedness !== "Right") continue;
    var offset = handedness === "Left" ? 0 : FEATURES_PER_HAND;
    vec.set(normalizeHandLandmarks(result.landmarks[i]), offset);
  }

  return vec;
}

function frameFeature(frame) {
  var points = frame && (frame.right || frame.left);
  if (!points) return null;
  var out = new Float32Array(FEATURES_PER_HAND);
  points.forEach(function (p, i) {
    out[i * 3] = p[0];
    out[i * 3 + 1] = -p[1];
    out[i * 3 + 2] = p[2];
  });
  return out;
}

function buildFallbackPrototypes() {
  fallbackPrototypes = [];
  Object.keys(vocab).forEach(function (label) {
    var sum = new Float32Array(FEATURES_PER_HAND), count = 0;
    var samples = [];
    vocab[label].forEach(function (frame) {
      var feature = frameFeature(frame);
      if (!feature) return;
      samples.push(feature);
      for (var i = 0; i < feature.length; i++) sum[i] += feature[i];
      count++;
    });
    if (count) {
      for (var j = 0; j < sum.length; j++) sum[j] /= count;
      fallbackPrototypes.push({ label: label, feature: sum, samples: samples });
    }
  });
}

function fallbackRecognize(featureVector) {
  if (!fallbackPrototypes.length) return { label: null, confidence: 0 };
  var best = null, bestDistance = Infinity;
  fallbackPrototypes.forEach(function (prototype) {
    [0, FEATURES_PER_HAND].forEach(function (offset) {
      var candidates = [prototype.feature].concat(prototype.samples);
      candidates.forEach(function (candidate) {
        var distance = 0, active = 0;
        for (var i = 0; i < FEATURES_PER_HAND; i++) {
          if (featureVector[offset + i] === 0) continue;
          var delta = featureVector[offset + i] - candidate[i];
          distance += delta * delta;
          active++;
        }
        if (!active) return;
        distance = Math.sqrt(distance / active);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = prototype;
        }
      });
    });
  });
  if (!best) return { label: null, confidence: 0 };
  return { label: best.label, confidence: THREE.MathUtils.clamp(1 - bestDistance / 0.85, 0, 1) };
}

function recognizeFallbackFrame(featureVector) {
  // The browser fallback has no temporal classifier, so classify each
  // detected frame and let the same 3-frame vote used by recognize_live.py
  // remove transient mismatches.
  var fallback = fallbackRecognize(featureVector);
  if (fallback.confidence >= 0.35) return fallback;
  return { label: null, confidence: fallback.confidence };
}

var speechVoice = null;
function chooseSpeechVoice() {
  if (!("speechSynthesis" in window)) return;
  var voices = window.speechSynthesis.getVoices();
  speechVoice = voices.find(function (voice) {
    return /^en(-|_)/i.test(voice.lang) && /google|microsoft|natural|samantha/i.test(voice.name);
  }) || voices.find(function (voice) { return /^en(-|_)/i.test(voice.lang); }) || voices[0] || null;
}
if ("speechSynthesis" in window) {
  chooseSpeechVoice();
  window.speechSynthesis.addEventListener("voiceschanged", chooseSpeechVoice);
}

// Cancel stale queued utterances so rapid predictions do not produce delayed,
// overlapping speech. A user can always replay the current transcript.
function speak(text) {
  if (!ttsEnabled || !text || !("speechSynthesis" in window)) return;
  chooseSpeechVoice();
  window.speechSynthesis.cancel();
  window.speechSynthesis.resume();
  var u = new SpeechSynthesisUtterance(text);
  if (speechVoice) u.voice = speechVoice;
  u.lang = speechVoice ? speechVoice.lang : "en-US";
  u.rate = 1.05;
  u.pitch = 1;
  u.volume = 1;
  u.onerror = function (event) {
    document.getElementById("confidenceLabel").textContent = "Speech error: " + (event.error || "browser voice unavailable");
  };
  window.speechSynthesis.speak(u);
}

// ---- drawing ----
function drawLandmarks(result, w, h) {
  if (!drawSkeleton || !result || !result.landmarks) return;
  result.landmarks.forEach(function (hand) {
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 2;
    HAND_CONNECTIONS.forEach(function (pair) {
      var a = hand[pair[0]], b = hand[pair[1]];
      ctx.beginPath();
      ctx.moveTo(a.x * w, a.y * h);
      ctx.lineTo(b.x * w, b.y * h);
      ctx.stroke();
    });
    ctx.fillStyle = "#ff9d5c";
    hand.forEach(function (p) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}

function setPredictionBadge(text, locked) {
  var badge = document.getElementById("predictionBadge");
  document.getElementById("predictionText").textContent = text;
  badge.classList.toggle("locked", !!locked);
}

function setModelState(text, cls) {
  var el = document.getElementById("recModelState");
  el.textContent = text;
  el.className = cls || "";
}

function setSessionState(state, text) {
  sessionState = state;
  var stateEl = document.getElementById("recSessionState");
  stateEl.textContent = text || state.charAt(0).toUpperCase() + state.slice(1);
  stateEl.className = state;
  document.getElementById("recSessionStartBtn").disabled = !handLandmarker || state === "recording" || state === "processing";
  document.getElementById("recSessionStopBtn").disabled = state !== "recording";
}

function renderSessionSigns() {
  var target = document.getElementById("recSessionSigns");
  target.textContent = sessionLabels.length
    ? sessionLabels.map(labelToDisplay).join("  ·  ")
    : "No signs collected yet.";
}

function collectSessionLabel(label) {
  if (label === sessionCandidateLabel) sessionCandidateCount++;
  else {
    sessionCandidateLabel = label;
    sessionCandidateCount = 1;
  }
  if (sessionCandidateCount < SESSION_STABILITY_FRAMES || label === sessionLastLabel) return;
  sessionLabels.push(label);
  sessionLastLabel = label;
  renderSessionSigns();
}

var COMMON_SESSION_WORDS = new Set([
  "a", "am", "and", "are", "bye", "can", "cow", "eat", "hello", "help",
  "how", "i", "is", "it", "me", "my", "no", "please", "sorry", "thank",
  "the", "what", "when", "where", "why", "yes", "you"
]);

function cleanSessionTokens(tokens) {
  var knownWords = new Set(COMMON_SESSION_WORDS);
  Object.keys(vocab).forEach(function (label) {
    if (!label.startsWith("letter_")) knownWords.add(label.replace(/_/g, ""));
  });
  var cleaned = [];
  tokens.forEach(function (token) {
    var value = token.toLowerCase().replace(/[^a-z ]/g, "").replace(/\s+/g, " ").trim();
    var compactValue = value.replace(/ /g, "");
    if (!compactValue) return;
    // A single isolated letter is usually a false positive between signs.
    if (compactValue.length === 1 && cleaned.length && cleaned[cleaned.length - 1].length > 1) return;
    // Drop long letter runs that are not a known word instead of presenting
    // classifier noise such as "UBREMPJW" as a real result.
    if (token === token.toUpperCase() && compactValue.length > 1 && !knownWords.has(compactValue)) return;
    if (!knownWords.has(compactValue) && compactValue.length > 1) return;
    if (cleaned[cleaned.length - 1] === value) return;
    cleaned.push(value);
  });
  return cleaned;
}

function finalizeSession() {
  var parts = [], letters = "";
  sessionLabels.forEach(function (label) {
    if (label.startsWith("letter_")) {
      letters += label.replace("letter_", "").toUpperCase();
      return;
    }
    if (letters) {
      parts.push(letters);
      letters = "";
    }
    parts.push(labelToDisplay(label));
  });
  if (letters) parts.push(letters);
  var result = cleanSessionTokens(parts).join(" ").replace(/\s+/g, " ").trim();
  sentence = result;
  document.getElementById("recSessionResult").textContent =
    result || "No confident signs were collected.";
  document.getElementById("recSentenceBox").textContent = result;
  setSessionState("completed", "Completed");
}

function startRecognitionSession() {
  if (!handLandmarker) {
    setSessionState("idle", "Load hand tracking first");
    return;
  }
  sessionLabels = [];
  sessionLastLabel = null;
  sessionCandidateLabel = null;
  sessionCandidateCount = 0;
  sentence = "";
  lastSpokenLabel = null;
  document.getElementById("recSentenceBox").textContent = "";
  document.getElementById("recSessionResult").textContent = "";
  renderSessionSigns();
  setSessionState("recording", "Recording");
}

function stopRecognitionSession() {
  if (sessionState !== "recording") return;
  setSessionState("processing", "Processing");
  window.setTimeout(finalizeSession, 120);
}

function resetRecognitionSession() {
  sessionLabels = [];
  sessionLastLabel = null;
  sessionCandidateLabel = null;
  sessionCandidateCount = 0;
  sentence = "";
  lastSpokenLabel = null;
  document.getElementById("recSentenceBox").textContent = "";
  document.getElementById("recSessionResult").textContent = "";
  renderSessionSigns();
  setSessionState("idle", "Idle");
}

// ---- main per-frame loop ----
function tick() {
  if (!running) return;
  rafId = requestAnimationFrame(tick);
  if (videoEl.readyState < 2) return;

  var w = canvasEl.width, h = canvasEl.height;

  // Mirror the frame before detection, matching recognize_live.py's
  // cv2.flip(frame, 1) so handedness ("Left"/"Right") lines up with the
  // hand the person is actually holding up, and with how the training
  // data was captured.
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(videoEl, -w, 0, w, h);
  ctx.restore();

  var result = handLandmarker.detectForVideo(canvasEl, performance.now());
  drawLandmarks(result, w, h);

  var featureVector = extractFeatureVector(result);
  frameBuffer.push(featureVector);
  if (frameBuffer.length > SEQUENCE_LENGTH) frameBuffer.shift();
  document.getElementById("bufferBar").style.width =
    Math.min(100, (frameBuffer.length / SEQUENCE_LENGTH) * 100) + "%";

  var rawLabel = null, currentConfidence = 0;

  if (classifierModel) {
    if (frameBuffer.length === SEQUENCE_LENGTH) {
      var flat = new Float32Array(SEQUENCE_LENGTH * FEATURES_PER_HAND * 2);
      for (var f = 0; f < SEQUENCE_LENGTH; f++) flat.set(frameBuffer[f], f * FEATURES_PER_HAND * 2);
      var predIdx = -1;
      tf.tidy(function () {
        var input = tf.tensor3d(flat, [1, SEQUENCE_LENGTH, FEATURES_PER_HAND * 2]);
        var predictions = classifierModel.predict(input);
        var data = predictions.dataSync();
        var best = 0;
        for (var k = 1; k < data.length; k++) if (data[k] > data[best]) best = k;
        currentConfidence = data[best];
        predIdx = best;
      });
      if (predIdx >= 0 && currentConfidence >= confidenceThreshold && labelNames[predIdx]) {
        rawLabel = labelNames[predIdx];
      }
    }
  } else {
    var fallback = recognizeFallbackFrame(featureVector);
    if (fallback.label) {
      rawLabel = fallback.label;
      currentConfidence = fallback.confidence;
    }
  }

  recentLabels.push(rawLabel);
  if (recentLabels.length > VOTE_WINDOW) recentLabels.shift();
  var votedLabel = majorityVote(recentLabels);

  if (votedLabel !== null) {
    if (sessionState === "recording") {
      if (currentConfidence >= Math.max(confidenceThreshold, 0.5)) {
        collectSessionLabel(votedLabel);
      }
    } else if (sessionState === "idle" && votedLabel !== lastSpokenLabel) {
      sentence += labelToSentenceFragment(votedLabel);
      document.getElementById("recSentenceBox").textContent = sentence;
      speak(labelToDisplay(votedLabel));
      lastSpokenLabel = votedLabel;
    }
  } else if (rawLabel === null && sessionState === "idle") {
    lastSpokenLabel = null;
  }

  if (rawLabel) {
    setPredictionBadge(labelToDisplay(rawLabel) + " (" + Math.round(currentConfidence * 100) + "%)", !!votedLabel);
  } else if (frameBuffer.length < SEQUENCE_LENGTH) {
    setPredictionBadge("Buffering... " + frameBuffer.length + "/" + SEQUENCE_LENGTH, false);
  } else {
    setPredictionBadge("...", false);
  }

  fpsFrames++;
  var now = performance.now();
  if (now - fpsLastTime >= 500) {
    var fps = Math.round((fpsFrames * 1000) / (now - fpsLastTime));
    document.getElementById("recFpsBadge").textContent = fps + " FPS";
    fpsFrames = 0; fpsLastTime = now;
  }
}

// ---- camera lifecycle ----
async function startCamera() {
  if (!handLandmarker || !labelNames) {
    setPredictionBadge("Load hand tracking first", false);
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (err) {
    setPredictionBadge("Camera permission denied", false);
    return;
  }
  videoEl.srcObject = stream;
  await videoEl.play();
  canvasEl.width = videoEl.videoWidth || 640;
  canvasEl.height = videoEl.videoHeight || 480;

  document.getElementById("camIdle").style.display = "none";
  var toggle = document.getElementById("camToggleBtn");
  toggle.textContent = "Stop camera";
  toggle.classList.add("live");
  document.getElementById("camStartBtn").textContent = "Stop camera";

  frameBuffer = []; recentLabels = []; lastSpokenLabel = null;
  running = true;
  fpsFrames = 0; fpsLastTime = performance.now();
  tick();
}

function stopCamera() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); stream = null; }
  videoEl.srcObject = null;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  document.getElementById("bufferBar").style.width = "0%";
  document.getElementById("camIdle").style.display = "flex";
  var toggle = document.getElementById("camToggleBtn");
  toggle.textContent = "Start camera";
  toggle.classList.remove("live");
  document.getElementById("camStartBtn").textContent = "Start camera";
  setPredictionBadge("Camera off", false);
  document.getElementById("recFpsBadge").textContent = "-- FPS";
}

function toggleCamera() { running ? stopCamera() : startCamera(); }

// ---- model loading ----
async function loadModel() {
  var btn = document.getElementById("recLoadBtn");
  btn.disabled = true;
  setModelState("loading…", "");
  try {
    var vision = await FilesetResolver.forVisionTasks(WASM_BASE);
    var handOptions = {
      baseOptions: { modelAssetPath: HAND_LANDMARKER_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5,
    };
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, handOptions);
    } catch (localError) {
      handOptions.baseOptions.modelAssetPath = HAND_LANDMARKER_REMOTE_URL;
      handLandmarker = await HandLandmarker.createFromOptions(vision, handOptions);
    }

    var modelPath = document.getElementById("recModelPath").value.trim();
    var labelPath = document.getElementById("recLabelPath").value.trim();
    try {
      classifierModel = await tf.loadLayersModel(modelPath);
      var labelResp = await fetch(labelPath);
      if (!labelResp.ok) throw new Error("label map fetch failed (" + labelResp.status + ")");
      var labelJson = await labelResp.json();
      if (Array.isArray(labelJson)) {
        labelNames = labelJson;
      } else {
        var entries = Object.entries(labelJson);
        var valuesAreNumeric = entries.every(function (e) { return !isNaN(Number(e[1])); });
        var arr = [];
        entries.forEach(function (e) {
          if (valuesAreNumeric) arr[Number(e[1])] = e[0];
          else arr[Number(e[0])] = e[1];
        });
        labelNames = arr;
      }
    } catch (classifierError) {
      classifierModel = null;
      labelNames = Object.keys(vocab).sort();
      buildFallbackPrototypes();
      document.getElementById("confidenceLabel").textContent =
        "Using built-in landmark matching; add models_web/model.json for trained recognition.";
    }

    setModelState("ready · " + labelNames.length + " signs", "ready");
    document.getElementById("camToggleBtn").disabled = false;
    document.getElementById("camStartBtn").disabled = false;
    setSessionState("idle", "Idle");
  } catch (err) {
    console.error(err);
    setModelState("error: " + err.message, "err");
  } finally {
    btn.disabled = false;
  }
}

// ---- wiring ----
document.getElementById("recLoadBtn").addEventListener("click", loadModel);
document.getElementById("camToggleBtn").addEventListener("click", toggleCamera);
document.getElementById("camStartBtn").addEventListener("click", toggleCamera);
document.getElementById("recSessionStartBtn").addEventListener("click", startRecognitionSession);
document.getElementById("recSessionStopBtn").addEventListener("click", stopRecognitionSession);
document.getElementById("recSessionResetBtn").addEventListener("click", resetRecognitionSession);
document.getElementById("recClearBtn").addEventListener("click", function () {
  sentence = "";
  document.getElementById("recSentenceBox").textContent = "";
  lastSpokenLabel = null;
});
document.getElementById("recCopyBtn").addEventListener("click", function () {
  if (!sentence) return;
  if (navigator.clipboard) navigator.clipboard.writeText(sentence.trim()).catch(function () {});
});
document.getElementById("recSpeakBtn").addEventListener("click", function () {
  if (sentence.trim()) speak(sentence.trim());
});
document.getElementById("recListenBtn").addEventListener("click", function () {
  if (sentence.trim()) speak(sentence.trim());
});
document.getElementById("confThreshRange").addEventListener("input", function () {
  confidenceThreshold = parseInt(this.value, 10) / 100;
  document.getElementById("confThreshValue").textContent = this.value + "%";
});
document.getElementById("recSpeakToggle").addEventListener("change", function () {
  ttsEnabled = this.checked;
  syncTtsBadge();
});
document.getElementById("recDrawToggle").addEventListener("change", function () {
  drawSkeleton = this.checked;
});
function syncTtsBadge() {
  var badge = document.getElementById("ttsBadge");
  badge.classList.toggle("on", ttsEnabled);
  badge.innerHTML = (ttsEnabled ? "&#128264; Speech on" : "&#128263; Speech off");
  document.getElementById("recSpeakToggle").checked = ttsEnabled;
}
document.getElementById("ttsBadge").addEventListener("click", function () {
  ttsEnabled = !ttsEnabled;
  syncTtsBadge();
});
document.addEventListener("keydown", function (e) {
  if (document.activeElement && (document.activeElement.tagName === "INPUT")) return;
  if (document.getElementById("recognizeView").style.display === "none") return;
  if (e.key === "c" || e.key === "C") document.getElementById("recClearBtn").click();
});

syncTtsBadge();
renderSessionSigns();
setSessionState("idle", "Idle");
window.addEventListener("load", function () {
  window.setTimeout(function () {
    var preloader = document.getElementById("preloader");
    if (preloader) preloader.classList.add("hidden");
  }, 350);
});
</script>

</body>
</html>
"""

def hand_slice_to_points(flat_hand_63):
    """
    Convert a flat 63-value array (21 landmarks x 3 coords) into a list
    of [x, y, z] points, or None if the hand was inactive (all zeros)
    in this frame.

    Flips the y-axis so the hand appears right-side-up in a standard
    3D viewer (our recorded data uses image convention where y
    increases downward; 3D viewers expect y increasing upward).
    """
    if not flat_hand_63.any():
        return None

    points = flat_hand_63.reshape(NUM_LANDMARKS, COORDS_PER_LANDMARK)
    return [[float(x), float(-y), float(z)] for x, y, z in points]


def export_sample(sample_array):
    """
    Convert one (30, 126) sample array into a list of per-frame
    {"left": [...] or None, "right": [...] or None} dicts.
    """
    frames = []
    for frame in sample_array:
        left = hand_slice_to_points(frame[0:FEATURES_PER_HAND])
        right = hand_slice_to_points(frame[FEATURES_PER_HAND:FEATURES_PER_HAND * 2])
        frames.append({"left": left, "right": right})
    return frames


def sample_activity_score(sample_array):
    """
    Count how many frames in a sample have at least one hand detected
    (i.e. not all-zero). Used to pick the "best" representative sample
    for a class instead of arbitrarily taking the first file.
    """
    score = 0
    for frame in sample_array:
        left_active = frame[0:FEATURES_PER_HAND].any()
        right_active = frame[FEATURES_PER_HAND:FEATURES_PER_HAND * 2].any()
        if left_active or right_active:
            score += 1
    return score


def pick_sample(label_dir, sample_files, strategy):
    """
    Choose which .npy file represents a class, and return
    (chosen_filename, loaded_array, activity_score) or (None, None, None)
    if no file in sample_files has the expected shape.
    """
    expected_shape = (SEQUENCE_LENGTH, FEATURES_PER_HAND * 2)

    if strategy == "first":
        for fname in sample_files:
            arr = np.load(os.path.join(label_dir, fname))
            if arr.shape == expected_shape:
                return fname, arr, sample_activity_score(arr)
        return None, None, None

    # strategy == "best": score every sample, keep the highest scorer
    best_file, best_arr, best_score = None, None, -1
    for fname in sample_files:
        arr = np.load(os.path.join(label_dir, fname))
        if arr.shape != expected_shape:
            continue
        score = sample_activity_score(arr)
        if score > best_score:
            best_file, best_arr, best_score = fname, arr, score
    return best_file, best_arr, best_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export recorded sign samples into JSON + a standalone 3D HTML viewer."
    )
    parser.add_argument(
        "--strategy", choices=["first", "best"], default="best",
        help="How to pick each class's representative sample. 'best' (default) scores "
             "every sample by how many of its frames have a detected hand and keeps the "
             "top scorer. 'first' restores the old first-file-found behavior.",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Only write vocabulary_export.json; skip generating the standalone HTML viewer.",
    )
    parser.add_argument(
        "--fbx", default=os.path.join("docs", "handbynadevaynoskix.fbx"),
        help="Path to the articulated FBX hand copied beside sign_viewer.html.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = os.path.join(PROJECT_ROOT, DATA_DIR)
    output_json_path = os.path.join(PROJECT_ROOT, OUTPUT_JSON_PATH)
    output_html_path = os.path.join(PROJECT_ROOT, OUTPUT_HTML_PATH)

    if not os.path.isdir(data_dir):
        print(f"[ERROR] No data directory found at {data_dir}")
        return

    label_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    export_data = {}
    skipped = []
    summary_rows = []  # (label, chosen_file, active_frames, total_frames)

    for label in label_names:
        label_dir = os.path.join(data_dir, label)
        sample_files = sorted(f for f in os.listdir(label_dir) if f.endswith(".npy"))

        if not sample_files:
            skipped.append(label)
            continue

        chosen_file, sample_array, activity_score = pick_sample(label_dir, sample_files, args.strategy)

        if sample_array is None:
            print(f"[WARN] Skipping '{label}': no sample with the expected shape "
                  f"{(SEQUENCE_LENGTH, FEATURES_PER_HAND * 2)}")
            skipped.append(label)
            continue

        export_data[label] = export_sample(sample_array)
        summary_rows.append((label, chosen_file, activity_score, SEQUENCE_LENGTH))

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Write the raw JSON too (useful for uploading into other tools).
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f)
    print(f"[SAVED] {output_json_path}")

    if not args.json_only:
        fbx_source = args.fbx
        if not os.path.isabs(fbx_source):
            project_fbx = os.path.join(PROJECT_ROOT, fbx_source)
            fbx_source = project_fbx if os.path.isfile(project_fbx) else os.path.abspath(fbx_source)

        model_url = os.path.basename(fbx_source)
        if not os.path.isfile(fbx_source):
            print(f"[WARN] FBX not copied: {fbx_source} not found.")

        html_content = (
            HTML_TEMPLATE
            .replace(VOCAB_PLACEHOLDER, json.dumps(export_data))
            .replace(MODEL_URL_PLACEHOLDER, model_url)
        )
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        # Keep a copy beside the viewer for users who serve the folder over HTTP.
        fbx_target = os.path.join(os.path.dirname(output_html_path), os.path.basename(fbx_source))
        if os.path.isfile(fbx_source):
            if os.path.abspath(fbx_source) != os.path.abspath(fbx_target):
                shutil.copy2(fbx_source, fbx_target)
                print(f"[SAVED] {fbx_target}")
            else:
                print(f"[READY] {fbx_target}")
        else:
            print(f"[WARN] FBX not copied: {fbx_source} not found.")
        asset_dir = os.path.join(os.path.dirname(output_html_path), "models_web")
        os.makedirs(asset_dir, exist_ok=True)
        if os.path.isfile(HAND_TASK_SOURCE):
            task_target = os.path.join(asset_dir, "hand_landmarker.task")
            shutil.copy2(HAND_TASK_SOURCE, task_target)
            print(f"[SAVED] {task_target}")
        else:
            print(f"[WARN] MediaPipe task not found: {HAND_TASK_SOURCE}")
        if os.path.isfile(LABEL_MAP_SOURCE):
            label_target = os.path.join(asset_dir, "label_map.json")
            shutil.copy2(LABEL_MAP_SOURCE, label_target)
            print(f"[SAVED] {label_target}")
        print(f"[SAVED] {output_html_path} (viewer; serve this folder over HTTP)")

    if summary_rows:
        print(f"\nExported {len(summary_rows)} classes (strategy: {args.strategy}):")
        name_width = max(len(row[0]) for row in summary_rows)
        for label, fname, active, total in sorted(summary_rows):
            print(f"  {label:<{name_width}}  {fname:<20}  {active:>2}/{total} frames with a hand")
    else:
        print("\nExported 0 classes.")

    if skipped:
        print(f"Skipped (no valid samples): {skipped}")


if __name__ == "__main__":
    main()