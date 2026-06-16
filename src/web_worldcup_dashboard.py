r"""
Local web dashboard for World Cup market monitoring.

Run:
  python src\web_worldcup_dashboard.py --date 2026-06-16

Open:
  http://127.0.0.1:5050

The dashboard is read-only. It fetches Sporttery odds, local model signals,
Odds API dispersion, and off-field factors, then renders a local browser view.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import threading
import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def _load_local_env() -> None:
    """Load local .env values before modules read API keys from os.environ."""
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()

from fetch_off_field_sentiment import (
    DEEPSEEK_API_KEY as OFF_FIELD_DEEPSEEK_API_KEY,
    FRESHNESS_RULE_VERSION as OFF_FIELD_FRESHNESS_RULE_VERSION,
    SENTIMENT_JSON_PATH,
    build_payload as build_off_field_sentiment_payload,
    normalize_team_name as normalize_sentiment_team_name,
    save_outputs as save_off_field_sentiment_outputs,
)
from group_motivation_features import MOTIVATION_COLUMNS, motivation_for_single_match
from build_2026_prediction_inputs import build_prediction_input_table
from live_worldcup_dashboard import (
    fetch_recent_polymarket_trades,
    implied_probs_from_had,
    match_trade_to_game,
    tomorrow_shanghai,
    top_scores,
)
from monitor_worldcup_markets import fetch_sporttery_snapshot, safe_float
from market_decision import (
    append_sporttery_history,
    build_market_decision_rows,
    load_sporttery_history,
)
from poisson_model import score_matrix_from_lambdas, top_exact_scores
from recent_form_features import recent_form_for_single_match


app = Flask(__name__)
DEFAULT_DATE = tomorrow_shanghai()
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
SPORTTERY_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "market_monitor" / "sporttery_latest_snapshot.csv"
PREDICTION_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "worldcup_2026_prediction_inputs.csv"
PREDICTION_INPUT_REFRESH_MAX_AGE_SECONDS = 6 * 60 * 60
ENABLE_POLYMARKET = False

POLYMARKET_TEAM_CODES = {
    "西班牙": "esp",
    "佛得角": "cvi",
    "比利时": "bel",
    "埃及": "egy",
    "沙特阿拉伯": "ksa",
    "乌拉圭": "ury",
    "伊朗": "irn",
    "新西兰": "nzl",
    "瑞典": "swe",
    "突尼斯": "tun",
    "法国": "fra",
    "塞内加尔": "sen",
    "阿根廷": "arg",
    "阿尔及利亚": "alg",
    "葡萄牙": "por",
    "刚果(金)": "cod",
    "英格兰": "eng",
    "克罗地亚": "cro",
    "加纳": "gha",
    "巴拿马": "pan",
    "乌兹别克斯坦": "uzb",
    "哥伦比亚": "col",
}

POLYMARKET_EXACT_SCORE_LAST_GOOD: dict[str, list[dict]] = {}
POLYMARKET_EXACT_SCORE_LAST_UPDATED: dict[str, str] = {}
POLYMARKET_EXACT_REFRESHING_KEYS: set[str] = set()
POLYMARKET_EXACT_REFRESH_LOCK = threading.Lock()
OFF_FIELD_REFRESHING_DATES: set[str] = set()
OFF_FIELD_REFRESH_LOCK = threading.Lock()
SPORTTERY_REFRESHING = False
SPORTTERY_REFRESH_LOCK = threading.Lock()
PREDICTION_INPUT_REFRESHING = False
PREDICTION_INPUT_REFRESH_LOCK = threading.Lock()

TEAM_NAME_MAP = {
    "西班牙": "Spain",
    "佛得角": "Cape Verde",
    "比利时": "Belgium",
    "埃及": "Egypt",
    "沙特阿拉伯": "Saudi Arabia",
    "乌拉圭": "Uruguay",
    "伊朗": "Iran",
    "新西兰": "New Zealand",
    "法国": "France",
    "塞内加尔": "Senegal",
    "伊拉克": "Iraq",
    "挪威": "Norway",
    "阿根廷": "Argentina",
    "阿尔及利亚": "Algeria",
    "奥地利": "Austria",
    "约旦": "Jordan",
    "葡萄牙": "Portugal",
    "刚果(金)": "DR Congo",
    "英格兰": "England",
    "克罗地亚": "Croatia",
    "加纳": "Ghana",
    "巴拿马": "Panama",
    "乌兹别克斯坦": "Uzbekistan",
    "哥伦比亚": "Colombia",
    "瑞典": "Sweden",
    "突尼斯": "Tunisia",
    "澳大利亚": "Australia",
    "波黑": "Bosnia and Herzegovina",
    "巴西": "Brazil",
    "加拿大": "Canada",
    "库拉索": "Curacao",
    "捷克": "Czech Republic",
    "厄瓜多尔": "Ecuador",
    "德国": "Germany",
    "海地": "Haiti",
    "科特迪瓦": "Ivory Coast",
    "日本": "Japan",
    "墨西哥": "Mexico",
    "摩洛哥": "Morocco",
    "荷兰": "Netherlands",
    "巴拉圭": "Paraguay",
    "卡塔尔": "Qatar",
    "苏格兰": "Scotland",
    "南非": "South Africa",
    "韩国": "South Korea",
    "瑞士": "Switzerland",
    "土耳其": "Turkey",
    "美国": "United States",
}

TEAM_DISPLAY_MAP = {english: chinese for chinese, english in TEAM_NAME_MAP.items()}

MODEL_LABELS = {0: "主胜", 1: "平局", 2: "客胜"}


PAGE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>World Cup Market Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0e1116;
      --panel: #171b22;
      --panel-2: #1f2530;
      --line: #313947;
      --text: #eef2f7;
      --muted: #99a3b3;
      --good: #4ade80;
      --warn: #fbbf24;
      --bad: #fb7185;
      --blue: #60a5fa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
      letter-spacing: 0;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      border-bottom: 1px solid var(--line);
      background: rgba(14, 17, 22, 0.96);
      backdrop-filter: blur(12px);
    }
    .bar {
      max-width: 1320px;
      margin: 0 auto;
      padding: 16px 20px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 700;
    }
    .sub {
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }
    .controls {
      display: flex;
      gap: 8px;
      align-items: end;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
    }
    input {
      height: 36px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      padding: 0 10px;
      font-size: 14px;
    }
    button {
      height: 36px;
      border: 1px solid #3b82f6;
      background: #2563eb;
      color: white;
      border-radius: 6px;
      padding: 0 14px;
      font-weight: 600;
      cursor: pointer;
    }
    main {
      max-width: 1320px;
      margin: 0 auto;
      padding: 18px 20px 40px;
    }
    .status {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .stat {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .stat .k { color: var(--muted); font-size: 12px; }
    .stat .v { margin-top: 4px; font-size: 18px; font-weight: 700; }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .advice {
      margin-bottom: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .advice-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 10px;
    }
    .advice-head h2 {
      margin: 0;
      font-size: 17px;
    }
    .advice-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .advice-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .plan {
      background: #11151c;
      border: 1px solid #263040;
      border-radius: 8px;
      padding: 12px;
    }
    .plan-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .plan-metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 6px;
      margin-bottom: 8px;
    }
    .metric {
      background: #0e1116;
      border: 1px solid #253044;
      border-radius: 6px;
      padding: 7px;
    }
    .metric .k { color: var(--muted); font-size: 11px; }
    .metric .v { margin-top: 3px; font-weight: 700; }
    .legs {
      display: grid;
      gap: 6px;
    }
    .leg {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      border-top: 1px solid #263040;
      padding-top: 6px;
      color: var(--text);
      font-size: 13px;
    }
    .leg .subline {
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }
    .match {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .match-head {
      padding: 14px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      background: var(--panel-2);
      border-bottom: 1px solid var(--line);
    }
    .teams { font-size: 18px; font-weight: 700; }
    .meta { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .pick {
      min-width: 94px;
      text-align: right;
      font-weight: 700;
      color: var(--good);
    }
    .pick .small {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }
    .body { padding: 14px; display: grid; gap: 12px; }
    .section-title {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .odds-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    .odd {
      background: #11151c;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
    }
    .odd .name { color: var(--muted); font-size: 12px; }
    .odd .num { margin-top: 3px; font-size: 17px; font-weight: 700; }
    .odd .prob { margin-top: 2px; color: var(--blue); font-size: 12px; }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip {
      border: 1px solid var(--line);
      background: #11151c;
      border-radius: 999px;
      padding: 6px 9px;
      font-size: 13px;
    }
    .alerts { display: grid; gap: 8px; }
    .alert {
      border-left: 3px solid var(--warn);
      background: #181510;
      padding: 8px 10px;
      border-radius: 4px;
      font-size: 13px;
    }
    .empty { color: var(--muted); font-size: 13px; }
    .market-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .market-table th,
    .market-table td {
      border-bottom: 1px solid #2a3240;
      padding: 7px 6px;
      text-align: left;
    }
    .market-table th { color: var(--muted); font-weight: 600; }
    .market-table td:nth-child(2),
    .market-table td:nth-child(3),
    .market-table td:nth-child(4) { text-align: right; }
    details {
      border: 1px solid var(--line);
      background: #11151c;
      border-radius: 6px;
      padding: 8px 10px;
    }
    summary {
      cursor: pointer;
      color: var(--text);
      font-weight: 700;
      font-size: 13px;
    }
    .pool-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
      gap: 7px;
      margin-top: 9px;
      max-height: 220px;
      overflow: auto;
      padding-right: 4px;
    }
    .pool-option {
      border: 1px solid #2a3240;
      background: #0e1116;
      border-radius: 5px;
      padding: 7px;
      min-width: 0;
    }
    .pool-option .o { color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .pool-option .p { font-size: 15px; font-weight: 700; margin-top: 2px; }
    .error {
      background: #2a1519;
      border: 1px solid #7f1d1d;
      padding: 12px;
      border-radius: 8px;
      color: #fecdd3;
      display: none;
      margin-bottom: 12px;
    }
    .loading-overlay {
      position: fixed;
      inset: 0;
      z-index: 20;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(7, 10, 15, 0.58);
      backdrop-filter: blur(3px);
    }
    .loading-overlay.active { display: flex; }
    .loading-box {
      width: min(420px, calc(100vw - 32px));
      border: 1px solid #31415a;
      background: #11151c;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    }
    .loading-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      font-weight: 700;
      margin-bottom: 10px;
    }
    .loading-step {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      min-height: 20px;
    }
    .progress-track {
      height: 8px;
      overflow: hidden;
      background: #0e1116;
      border: 1px solid #2a3240;
      border-radius: 999px;
      margin: 10px 0 8px;
    }
    .progress-bar {
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, #60a5fa, #4ade80);
      border-radius: inherit;
      transition: width 220ms ease;
    }
    .top-progress {
      position: fixed;
      top: 0;
      left: 0;
      z-index: 30;
      width: 0%;
      height: 3px;
      background: linear-gradient(90deg, #60a5fa, #4ade80);
      transition: width 220ms ease, opacity 220ms ease;
      opacity: 0;
    }
    .top-progress.active { opacity: 1; }
    body.is-loading button,
    body.is-loading input {
      opacity: 0.72;
    }
    body.is-loading button {
      cursor: wait;
    }
    @media (max-width: 900px) {
      .bar { grid-template-columns: 1fr; }
      .controls { justify-content: flex-start; }
      .status { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .advice-grid { grid-template-columns: 1fr; }
      .plan-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div id="top-progress" class="top-progress"></div>
  <div id="loading-overlay" class="loading-overlay" aria-live="polite" aria-busy="true">
    <div class="loading-box">
      <div class="loading-title">
        <span id="loading-title">正在加载</span>
        <span id="loading-percent">0%</span>
      </div>
      <div class="progress-track"><div id="loading-bar" class="progress-bar"></div></div>
      <div id="loading-step" class="loading-step">准备请求数据...</div>
    </div>
  </div>
  <header>
    <div class="bar">
      <div>
        <h1>World Cup Market Dashboard</h1>
        <div class="sub">中国体彩赔率 + Odds API 分歧 + 本地模型 + 场外因素</div>
      </div>
      <div class="controls">
        <label>日期
          <input id="date" type="date" value="{{ target_date }}">
        </label>
        <label>大单阈值
          <input id="notional" type="number" min="1" step="50" value="500">
        </label>
        <label>刷新秒数
          <input id="refresh" type="number" min="20" step="10" value="60">
        </label>
        <button id="reload">刷新</button>
        <button id="force-reload" type="button">强制刷新</button>
      </div>
    </div>
  </header>
  <main>
    <div id="error" class="error"></div>
    <div class="status">
      <div class="stat"><div class="k">更新时间</div><div class="v" id="updated">-</div></div>
      <div class="stat"><div class="k">比赛数</div><div class="v" id="match-count">-</div></div>
      <div class="stat"><div class="k">体彩快照行</div><div class="v" id="snapshot-rows">-</div></div>
      <div class="stat"><div class="k">缓存状态</div><div class="v" id="cache-mode">-</div></div>
    </div>
    <section id="betting-advice" class="advice"></section>
    <div id="matches" class="grid"></div>
  </main>
  <script>
    const els = {
      date: document.getElementById('date'),
      notional: document.getElementById('notional'),
      refresh: document.getElementById('refresh'),
      reload: document.getElementById('reload'),
      forceReload: document.getElementById('force-reload'),
      matches: document.getElementById('matches'),
      error: document.getElementById('error'),
      updated: document.getElementById('updated'),
      matchCount: document.getElementById('match-count'),
      snapshotRows: document.getElementById('snapshot-rows'),
      alertCount: document.getElementById('alert-count'),
      cacheMode: document.getElementById('cache-mode'),
      bettingAdvice: document.getElementById('betting-advice'),
      loadingOverlay: document.getElementById('loading-overlay'),
      loadingTitle: document.getElementById('loading-title'),
      loadingPercent: document.getElementById('loading-percent'),
      loadingBar: document.getElementById('loading-bar'),
      loadingStep: document.getElementById('loading-step'),
      topProgress: document.getElementById('top-progress')
    };
    let activeRequestId = 0;
    let progressTimer = null;
    let progressStartedAt = 0;
    let hasLoadedOnce = false;

    function setProgress(percent, step) {
      const value = Math.max(0, Math.min(100, Math.round(percent)));
      els.loadingPercent.textContent = value + '%';
      els.loadingBar.style.width = value + '%';
      els.topProgress.style.width = value + '%';
      if (step) els.loadingStep.textContent = step;
    }
    function startLoading(mode, targetDate) {
      progressStartedAt = Date.now();
      const title = mode === 'auto' ? '自动刷新中' : (mode === 'force' ? '强制刷新中' : '正在加载比赛日');
      els.loadingTitle.textContent = title;
      setProgress(8, `请求 ${targetDate} 的模型、体彩盘口和赛前信号...`);
      const showOverlay = !hasLoadedOnce || mode === 'initial';
      els.loadingOverlay.classList.toggle('active', showOverlay);
      els.topProgress.classList.add('active');
      document.body.classList.add('is-loading');
      els.reload.disabled = true;
      els.forceReload.disabled = true;
      els.date.disabled = true;
      els.notional.disabled = true;
      if (progressTimer) clearInterval(progressTimer);
      progressTimer = setInterval(() => {
        const elapsed = (Date.now() - progressStartedAt) / 1000;
        let next = 12 + Math.min(76, elapsed * 7);
        let step = '读取本地模型与赛程输入...';
        if (elapsed > 3) step = '拉取中国体彩盘口并记录盘口快照...';
        if (elapsed > 8) step = '读取 Odds API 分歧、场外因素和购彩方案...';
        if (elapsed > 14) step = '生成串关建议，接口可能正在等待 DeepSeek 或外部盘口数据...';
        setProgress(next, `${step} 已等待 ${elapsed.toFixed(0)} 秒`);
      }, 500);
    }
    function finishLoading(success, message) {
      if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
      }
      const elapsed = ((Date.now() - progressStartedAt) / 1000).toFixed(1);
      setProgress(success ? 100 : 92, message || (success ? `加载完成，用时 ${elapsed} 秒` : `加载失败，用时 ${elapsed} 秒`));
      if (success) hasLoadedOnce = true;
      window.setTimeout(() => {
        els.loadingOverlay.classList.remove('active');
        els.topProgress.classList.remove('active');
        els.topProgress.style.width = '0%';
        document.body.classList.remove('is-loading');
        els.reload.disabled = false;
        els.forceReload.disabled = false;
        els.date.disabled = false;
        els.notional.disabled = false;
      }, success ? 320 : 900);
    }

    function pct(value) {
      if (value === null || value === undefined) return '-';
      return (value * 100).toFixed(1) + '%';
    }
    function num(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      return Number(value).toFixed(2).replace(/\\.00$/, '');
    }
    function escapeHtml(text) {
      return String(text ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    function oddCard(name, odds, prob) {
      return `<div class="odd"><div class="name">${name}</div><div class="num">${num(odds)}</div><div class="prob">${pct(prob)}</div></div>`;
    }
    function signedPct(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      const n = Number(value) * 100;
      return (n >= 0 ? '+' : '') + n.toFixed(1) + '%';
    }
    function signedNum(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      const n = Number(value);
      return (n >= 0 ? '+' : '') + n.toFixed(2);
    }
    function renderPlan(plan) {
      if (!plan || !plan.available) {
        return `<div class="plan">
          <div class="plan-title"><span>${escapeHtml(plan?.title || '方案')}</span><span>不建议</span></div>
          <div class="empty">${escapeHtml(plan?.reason || '盘口或模型数据不足，无法计算。')}</div>
        </div>`;
      }
      const matchGroups = (plan.matches || []).map(group => {
        const options = (group.options || []).map(opt => `
          <div class="leg">
            <div>
              <strong>${escapeHtml(opt.pool)} · ${escapeHtml(opt.selection)}</strong>
              <div class="subline">${escapeHtml(opt.note || '')}</div>
            </div>
            <div>
              <strong>@ ${num(opt.odds)}</strong>
              <div class="subline">p ${pct(opt.probability)} · EV ${signedPct(opt.expected_value)}</div>
            </div>
          </div>`).join('');
        return `<details open>
          <summary>${escapeHtml(group.match)} · ${group.options.length} 项</summary>
          <div class="legs">${options}</div>
        </details>`;
      }).join('');
      return `<div class="plan">
        <div class="plan-title"><span>${escapeHtml(plan.title)}</span><span>${escapeHtml(plan.play_type)}</span></div>
        <div class="plan-metrics">
          <div class="metric"><div class="k">已选 / 注数</div><div class="v">${escapeHtml(plan.selected_count)} / ${escapeHtml(plan.bet_count)}</div></div>
          <div class="metric"><div class="k">金额</div><div class="v">${num(plan.total_cost)} 元</div></div>
          <div class="metric"><div class="k">赔率范围</div><div class="v">${num(plan.min_combined_odds)}-${num(plan.max_combined_odds)}</div></div>
          <div class="metric"><div class="k">最高奖金</div><div class="v">${num(plan.theoretical_max_prize)} 元</div></div>
          <div class="metric"><div class="k">覆盖概率估算</div><div class="v">${pct(plan.package_hit_probability)}</div></div>
          <div class="metric"><div class="k">期望值</div><div class="v">${signedPct(plan.expected_value)}</div></div>
          <div class="metric"><div class="k">平均EV</div><div class="v">${signedPct(plan.average_option_ev)}</div></div>
        </div>
        <div class="legs">${matchGroups}</div>
      </div>`;
    }
    function renderBettingAdvice(advice) {
      if (!advice) {
        return `<div class="advice-head"><h2>串关方案</h2></div><div class="empty">暂无建议数据</div>`;
      }
      return `<div class="advice-head">
        <h2>串关方案</h2>
        <div class="advice-note">仅做模型/盘口分析，不保证收益；串关方差很高，建议固定小本金。</div>
      </div>
      <div class="advice-note">${escapeHtml(advice.rules_note || '')}</div>
      <div class="section-title" style="margin-top:12px;">胜平负串关</div>
      <div class="advice-grid">
        ${renderPlan(advice.wdl?.conservative)}
        ${renderPlan(advice.wdl?.aggressive)}
      </div>
      <div class="section-title" style="margin-top:12px;">比分串关</div>
      <div class="advice-grid">
        ${renderPlan(advice.score?.conservative)}
        ${renderPlan(advice.score?.aggressive)}
      </div>`;
    }
    function renderMatch(match) {
      const had = match.had || {};
      const hhad = match.hhad || {};
      const model = match.local_model || {};
      const poisson = model.poisson || {};
      const fixture = match.fixture_metadata || {};
      const oddsApi = fixture.odds_api || {};
      const motivation = model.group_motivation || match.group_motivation || {};
      const offField = match.off_field_sentiment || {};
      const decisions = (match.market_decision || []).slice(0, 6).map(row => `
        <tr>
          <td>${escapeHtml(row.pool)}</td>
          <td>${escapeHtml(row.selection)}</td>
          <td>${num(row.odds)}</td>
          <td>${pct(row.model_probability)}</td>
          <td>${signedPct(row.value_edge)}</td>
          <td>${signedNum(row.odds_change_30m)}</td>
          <td>${signedNum(row.final_score)}</td>
          <td>${escapeHtml(row.grade)}</td>
        </tr>
        <tr><td colspan="8" class="subline">${escapeHtml(row.note || '')}</td></tr>
      `).join('');
      const topScores = (match.top_scores || []).map(s => `<span class="chip">${escapeHtml(s.label)} @ ${num(s.odds)}</span>`).join('');
      const sourceLabel = match.source === 'fixture' ? '本地赛程 · 盘口待获取' : '中国体彩';
      const allPools = (match.pools || []).map(pool => {
        const options = (pool.options || []).map(opt =>
          `<div class="pool-option"><div class="o">${escapeHtml(opt.outcome)}</div><div class="p">${num(opt.odds)}</div></div>`
        ).join('');
        return `<details ${pool.pool_code === 'crs' ? '' : 'open'}>
          <summary>${escapeHtml(pool.title)} · ${pool.options.length} 项</summary>
          <div class="pool-grid">${options || '<span class="empty">暂无</span>'}</div>
        </details>`;
      }).join('');
      const hadHtml = match.had_available
        ? `<div class="odds-row">
            ${oddCard('主胜', had.home_odds, had.home_prob)}
            ${oddCard('平局', had.draw_odds, had.draw_prob)}
            ${oddCard('客胜', had.away_odds, had.away_prob)}
          </div>${match.had_real_available ? '' : '<div class="empty">普通胜平负未开售，上方概率由比分盘反推</div>'}`
        : `<div class="empty">普通胜平负暂不可用：该比赛暂未从接口获取到开盘数据</div>`;
      const hhadHtml = match.hhad_available
        ? `<div class="chips">
            <span class="chip">让球 ${escapeHtml(hhad.handicap_line)}</span>
            <span class="chip">让胜 ${num(hhad.home_odds)} · ${pct(hhad.home_prob)}</span>
            <span class="chip">让平 ${num(hhad.draw_odds)} · ${pct(hhad.draw_prob)}</span>
            <span class="chip">让负 ${num(hhad.away_odds)} · ${pct(hhad.away_prob)}</span>
          </div>`
        : `<div class="empty">让球盘暂不可用：该比赛暂未从接口获取到开盘数据</div>`;
      const poissonScores = (poisson.top_scores || []).map(row =>
        `<span class="chip">${escapeHtml(row.score)} · ${pct(row.probability)}</span>`
      ).join('');
      const poissonHtml = poisson.available
        ? `<div class="chips">
            <span class="chip">λ主 ${num(poisson.home_lambda)}</span>
            <span class="chip">λ客 ${num(poisson.away_lambda)}</span>
            <span class="chip">泊松倾向 ${escapeHtml(poisson.pick)}</span>
          </div>
          <div class="chips">${poissonScores}</div>`
        : `<div class="empty">${escapeHtml(poisson.reason || '泊松基底模型尚未训练')}</div>`;
      const motivationHtml = motivation.available
        ? `<div class="chips">
            <span class="chip">主队压力 ${pct(motivation.home_group_pressure)}</span>
            <span class="chip">客队压力 ${pct(motivation.away_group_pressure)}</span>
            <span class="chip">同轮先赛 ${escapeHtml(motivation.known_prior_same_round_matches ?? motivation.group_known_prior_same_round_matches ?? 0)} 场</span>
            ${Number(motivation.home_needs_win_flag || 0) ? '<span class="chip">主队必须争胜</span>' : ''}
            ${Number(motivation.away_needs_win_flag || 0) ? '<span class="chip">客队必须争胜</span>' : ''}
          </div>
          <div class="empty">${escapeHtml(model.motivation_note || '')}</div>`
        : `<div class="empty">暂无小组压力数据</div>`;
      const oddsApiHtml = Object.keys(oddsApi).length
        ? `<div class="chips">
            <span class="chip">博彩公司 ${num(oddsApi.odds_api_bookmaker_count)}</span>
            <span class="chip">平均分散 ${num(oddsApi.odds_api_prob_dispersion_mean)}</span>
            <span class="chip">最大分散 ${num(oddsApi.odds_api_prob_dispersion_max)}</span>
            <span class="chip">平局分歧 ${num(oddsApi.odds_api_draw_disagreement_score)}</span>
            <span class="chip">平局概率STD ${num(oddsApi.odds_api_draw_prob_std)}</span>
          </div>`
        : `<div class="empty">暂无 Odds API 多博彩公司分歧数据</div>`;
      const offFieldHtml = offField.available
        ? `<div class="chips">
            <span class="chip">主队场外 ${signedNum(offField.home?.overall)} · 置信 ${pct(offField.home?.confidence)}</span>
            <span class="chip">客队场外 ${signedNum(offField.away?.overall)} · 置信 ${pct(offField.away?.confidence)}</span>
            <span class="chip">加权差 ${signedNum(offField.diff)}</span>
          </div>
          <div class="empty">主队：${escapeHtml(offField.home?.reasoning || '无明显场外信号')}</div>
          <div class="empty">客队：${escapeHtml(offField.away?.reasoning || '无明显场外信号')}</div>
          ${offField.refresh_error ? `<div class="empty">刷新提示：${escapeHtml(offField.refresh_error)}</div>` : ''}`
        : `<div class="empty">${offField.refresh_error ? `场外因素刷新失败：${escapeHtml(offField.refresh_error)}` : '暂无场外因素数据'}</div>`;
      const modelHtml = model.available
        ? `<div class="odds-row">
            ${oddCard('主胜', null, model.home_prob)}
            ${oddCard('平局', null, model.draw_prob)}
            ${oddCard('客胜', null, model.away_prob)}
          </div>
          <div class="chips">
            <span class="chip">模型倾向 ${escapeHtml(model.pick)}</span>
            <span class="chip">底座 ${escapeHtml(model.model_source || '-')}</span>
            <span class="chip">缺失填充 ${escapeHtml(model.missing_count)}</span>
          </div>`
        : `<div class="empty">${escapeHtml(model.reason || '本地模型暂不可用')}</div>`;
      return `<article class="match">
        <div class="match-head">
          <div>
            <div class="teams">${escapeHtml(match.home_team)} vs ${escapeHtml(match.away_team)}</div>
            <div class="meta">${escapeHtml(match.match_time)} | match_id=${escapeHtml(match.match_id)}</div>
            <div class="meta">${escapeHtml(sourceLabel)}</div>
            ${fixture.available ? `<div class="meta">${escapeHtml(fixture.group)}组 第${escapeHtml(fixture.matchday)}轮 | ${escapeHtml(fixture.stadium_name)} · ${escapeHtml(fixture.stadium_region)}</div>` : ''}
          </div>
          <div class="pick">
            ${escapeHtml(match.market_pick || '-')}
            <span class="small">${escapeHtml(match.market_pick_source || '')}</span>
            <span class="small">让球: ${escapeHtml(match.handicap_pick || '-')}</span>
          </div>
        </div>
        <div class="body">
          <section><div class="section-title">本地模型预测</div>${modelHtml}</section>
          <section><div class="section-title">泊松基底比分</div>${poissonHtml}</section>
          <section><div class="section-title">小组出线压力 / 情绪因子</div>${motivationHtml}</section>
          <section><div class="section-title">Odds API 多家公司分歧</div>${oddsApiHtml}</section>
          <section><div class="section-title">场外因素 / DeepSeek</div>${offFieldHtml}</section>
          <section>
            <div class="section-title">盘口变化 / 决策评分</div>
            ${decisions ? `<table class="market-table"><thead><tr><th>玩法</th><th>选项</th><th>赔率</th><th>模型</th><th>价值差</th><th>30m</th><th>评分</th><th>等级</th></tr></thead><tbody>${decisions}</tbody></table>` : '<div class="empty">盘口历史不足：等待网页多刷新几次后会显示 10/30/120 分钟变化</div>'}
          </section>
          <section><div class="section-title">HAD 胜平负</div>${hadHtml}</section>
          <section><div class="section-title">HHAD 让球胜平负</div>${hhadHtml}</section>
          <section><div class="section-title">CRS 比分盘热门</div><div class="chips">${topScores || '<span class="empty">比分盘暂不可用：该比赛暂未从接口获取到开盘数据</span>'}</div></section>
          <section><div class="section-title">全部竞彩开售选项</div>${allPools || '<div class="empty">暂无竞彩开售选项：该比赛暂未从接口获取到盘口数据</div>'}</section>
        </div>
      </article>`;
    }
    async function loadData(force = false, mode = 'manual') {
      const requestId = ++activeRequestId;
      startLoading(force ? 'force' : mode, els.date.value);
      const params = new URLSearchParams({
        date: els.date.value,
        min_notional: els.notional.value,
        trade_limit: '500',
        _: Date.now().toString()
      });
      if (force) params.set('force_refresh', '1');
      els.error.style.display = 'none';
      try {
        const res = await fetch('/api/dashboard?' + params.toString(), { cache: 'no-store' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || '请求失败');
        if (requestId !== activeRequestId) return;
        setProgress(90, '渲染页面和购彩方案...');
        els.updated.textContent = data.updated_at_display;
        els.matchCount.textContent = data.matches.length;
        els.snapshotRows.textContent = data.snapshot_rows;
        if (els.alertCount) els.alertCount.textContent = data.polymarket_alert_count;
        els.cacheMode.textContent = data.cache_mode || '-';
        els.bettingAdvice.innerHTML = renderBettingAdvice(data.betting_advice);
        els.matches.innerHTML = data.matches.map(renderMatch).join('') || '<div class="empty">这一天没有找到体彩世界杯比赛。</div>';
        finishLoading(true);
      } catch (err) {
        if (requestId !== activeRequestId) return;
        els.error.textContent = err.message;
        els.error.style.display = 'block';
        finishLoading(false, `加载失败：${err.message}`);
      }
    }
    let timer = null;
    function resetTimer() {
      if (timer) clearInterval(timer);
      timer = setInterval(() => loadData(false, 'auto'), Math.max(20, Number(els.refresh.value || 60)) * 1000);
    }
    els.reload.addEventListener('click', () => { loadData(false, 'manual'); resetTimer(); });
    els.forceReload.addEventListener('click', () => { loadData(true, 'force'); resetTimer(); });
    els.date.addEventListener('change', () => { loadData(false, 'date'); resetTimer(); });
    els.notional.addEventListener('change', () => { loadData(false, 'manual'); resetTimer(); });
    els.refresh.addEventListener('change', resetTimer);
    loadData(false, 'initial');
    resetTimer();
  </script>
</body>
</html>
"""


def _odds_map(frame: pd.DataFrame) -> dict[str, float]:
    return {row.outcome: float(row.odds) for row in frame.itertuples()}


def _load_cached_sporttery_snapshot() -> pd.DataFrame:
    if not SPORTTERY_CACHE_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(SPORTTERY_CACHE_PATH)
    except Exception:
        return pd.DataFrame()


def _refresh_sporttery_snapshot_background() -> None:
    """Refresh Sporttery odds in the background so date switching stays fast."""
    global SPORTTERY_REFRESHING
    try:
        snapshot = fetch_sporttery_snapshot(["had", "hhad", "crs", "ttg", "hafu"])
        if not snapshot.empty:
            SPORTTERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            snapshot.to_csv(SPORTTERY_CACHE_PATH, index=False, encoding="utf-8-sig")
            append_sporttery_history(snapshot)
    except Exception as exc:
        print(f"[warn] Sporttery background refresh failed: {exc}")
    finally:
        with SPORTTERY_REFRESH_LOCK:
            SPORTTERY_REFRESHING = False


def _schedule_sporttery_snapshot_refresh() -> bool:
    """Start at most one background Sporttery refresh."""
    global SPORTTERY_REFRESHING
    with SPORTTERY_REFRESH_LOCK:
        if SPORTTERY_REFRESHING:
            return False
        SPORTTERY_REFRESHING = True
    worker = threading.Thread(
        target=_refresh_sporttery_snapshot_background,
        name="sporttery-snapshot-refresh",
        daemon=True,
    )
    worker.start()
    return True


def _prediction_input_is_stale() -> bool:
    if not PREDICTION_INPUT_PATH.exists():
        return True
    age_seconds = time.time() - PREDICTION_INPUT_PATH.stat().st_mtime
    return age_seconds > PREDICTION_INPUT_REFRESH_MAX_AGE_SECONDS


def _refresh_prediction_inputs_background(force: bool = False) -> None:
    """Refresh 2026 schedule/prediction input table without blocking the page."""
    global PREDICTION_INPUT_REFRESHING
    try:
        if not force and not _prediction_input_is_stale():
            return
        print("[info] Refreshing 2026 prediction input table...")
        table = build_prediction_input_table(
            include_completed=True,
            use_live_sporttery=False,
            add_predictions=True,
            refresh_worldcup2026_repo=True,
        )
        PREDICTION_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(PREDICTION_INPUT_PATH, index=False, encoding="utf-8")
        _load_fixture_metadata_table.cache_clear()
        print(f"[info] Refreshed prediction inputs: {PREDICTION_INPUT_PATH} rows={len(table)}")
    except Exception as exc:
        print(f"[warn] prediction input refresh failed: {exc}")
    finally:
        with PREDICTION_INPUT_REFRESH_LOCK:
            PREDICTION_INPUT_REFRESHING = False


def _schedule_prediction_input_refresh(force: bool = False) -> bool:
    """Start at most one background fixture/prediction-input refresh."""
    global PREDICTION_INPUT_REFRESHING
    if not force and not _prediction_input_is_stale():
        return False
    with PREDICTION_INPUT_REFRESH_LOCK:
        if PREDICTION_INPUT_REFRESHING:
            return False
        PREDICTION_INPUT_REFRESHING = True
    worker = threading.Thread(
        target=_refresh_prediction_inputs_background,
        args=(force,),
        name="prediction-input-refresh",
        daemon=True,
    )
    worker.start()
    return True


@lru_cache(maxsize=1)
def _load_local_model_bundle() -> dict:
    ensemble_model_path = PROJECT_ROOT / "models" / "ensemble_model.pkl"
    ensemble_features_path = PROJECT_ROOT / "models" / "ensemble_features.json"
    model_path = PROJECT_ROOT / "models" / "football_model.pkl"
    poisson_model_path = PROJECT_ROOT / "models" / "poisson_base_model.pkl"
    features_path = PROJECT_ROOT / "models" / "features.json"
    training_path = PROJECT_ROOT / "data" / "raw" / "matches.csv"
    squad_path = PROJECT_ROOT / "data" / "processed" / "current_squad_team_features.csv"
    elo_path = PROJECT_ROOT / "data" / "raw" / "national_team_elo.csv"
    fifa_path = PROJECT_ROOT / "data" / "raw" / "fifa_ranking.csv"
    results_path = PROJECT_ROOT / "data" / "raw" / "results.csv"

    model_source = "single_model"
    if ensemble_model_path.exists() and ensemble_features_path.exists():
        model_path = ensemble_model_path
        features_path = ensemble_features_path
        model_source = "ensemble"
    elif not model_path.exists() or not features_path.exists():
        return {"available": False, "reason": "模型文件不存在"}

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    poisson_model = None
    if poisson_model_path.exists():
        with open(poisson_model_path, "rb") as f:
            poisson_model = pickle.load(f)

    medians = pd.Series(0.0, index=feature_names)
    if training_path.exists():
        training = pd.read_csv(training_path, usecols=lambda col: col in feature_names)
        medians = training.reindex(columns=feature_names).median(numeric_only=True).reindex(feature_names).fillna(0)

    squad = pd.DataFrame()
    if squad_path.exists():
        squad = pd.read_csv(squad_path)

    elo = pd.DataFrame()
    if elo_path.exists():
        elo = pd.read_csv(elo_path)
        if "date" in elo.columns:
            elo["date"] = pd.to_datetime(elo["date"], errors="coerce")
        elo["elo"] = pd.to_numeric(elo["elo"], errors="coerce")

    fifa = pd.DataFrame()
    if fifa_path.exists():
        fifa = pd.read_csv(fifa_path)
        if "date" in fifa.columns:
            fifa["date"] = pd.to_datetime(fifa["date"], errors="coerce")
        fifa["fifa_points"] = pd.to_numeric(fifa["fifa_points"], errors="coerce")

    results = pd.DataFrame()
    if results_path.exists():
        results = pd.read_csv(results_path)
        if "date" in results.columns:
            results["date"] = pd.to_datetime(results["date"], errors="coerce")
        for col in ["home_score", "away_score"]:
            if col in results.columns:
                results[col] = pd.to_numeric(results[col], errors="coerce")

    return {
        "available": True,
        "model": model,
        "model_source": model_source,
        "poisson_model": poisson_model,
        "feature_names": feature_names,
        "medians": medians,
        "squad": squad,
        "elo": elo,
        "fifa": fifa,
        "results": results,
    }


def _team_en(team_name: str) -> str:
    return TEAM_NAME_MAP.get(str(team_name), str(team_name))


def _team_display(team_name: str) -> str:
    """Prefer Chinese display names, but keep English when no mapping exists yet."""
    text = str(team_name)
    return TEAM_DISPLAY_MAP.get(text, text)


@lru_cache(maxsize=1)
def _load_fixture_metadata_table() -> pd.DataFrame:
    _schedule_prediction_input_refresh(force=False)
    path = PREDICTION_INPUT_PATH
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _lookup_fixture_metadata(home_team_cn: str, away_team_cn: str, target_date: str) -> dict:
    table = _load_fixture_metadata_table()
    if table.empty:
        return {"available": False}

    home_team = _team_en(home_team_cn)
    away_team = _team_en(away_team_cn)
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target):
        return {"available": False}

    candidate_dates = [target.normalize(), (target - pd.Timedelta(days=1)).normalize()]
    rows = table[
        table["date"].isin(candidate_dates)
        & table["home_team"].astype(str).eq(home_team)
        & table["away_team"].astype(str).eq(away_team)
    ]
    if rows.empty:
        rows = table[
            table["date"].isin(candidate_dates)
            & table["home_team"].astype(str).eq(away_team)
            & table["away_team"].astype(str).eq(home_team)
        ]
    if rows.empty:
        return {"available": False}

    row = rows.iloc[0]
    odds_api = {}
    for key in [
        "odds_api_bookmaker_count",
        "odds_api_prob_dispersion_mean",
        "odds_api_prob_dispersion_max",
        "odds_api_draw_disagreement_score",
        "odds_api_draw_prob_std",
        "odds_api_draw_prob_range",
        "odds_api_draw_odds_cv",
    ]:
        value = row.get(key)
        if pd.notna(value):
            odds_api[key] = float(value)
    return {
        "available": True,
        "group": str(row.get("worldcup_group") or ""),
        "matchday": int(row.get("worldcup_matchday")) if pd.notna(row.get("worldcup_matchday")) else "",
        "local_kickoff": str(row.get("local_kickoff") or ""),
        "stadium_name": str(row.get("stadium_name") or ""),
        "stadium_city": str(row.get("stadium_city") or ""),
        "stadium_country": str(row.get("stadium_country") or ""),
        "stadium_capacity": int(row.get("stadium_capacity")) if pd.notna(row.get("stadium_capacity")) else "",
        "stadium_region": str(row.get("stadium_region") or ""),
        "is_host_country_match": bool(row.get("is_host_country_match") == 1),
        "odds_api": odds_api,
    }


def _lookup_group_motivation(home_team_name: str, away_team_name: str, target_date: str) -> dict:
    table = _load_fixture_metadata_table()
    if table.empty:
        return {"available": False}
    return motivation_for_single_match(
        table,
        _team_en(home_team_name),
        _team_en(away_team_name),
        target_date,
    )


def _fixture_rows_for_dashboard(target_date: str, existing_keys: set[tuple[str, str]]) -> list[dict]:
    """
    Return local fixture rows for a dashboard date when live odds are missing.

    Sporttery uses the China viewing date. For matches hosted in North America,
    the local fixture date in our prediction table is usually one day earlier
    than the China date shown in the dashboard, so we try target_date - 1 first.
    """
    table = _load_fixture_metadata_table()
    if table.empty:
        return []

    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target):
        return []

    preferred_date = (target - pd.Timedelta(days=1)).normalize()
    fallback_date = target.normalize()
    rows = table[table["date"].eq(preferred_date)].copy()
    if rows.empty:
        rows = table[table["date"].eq(fallback_date)].copy()

    if rows.empty:
        return []

    sort_cols = [col for col in ["local_kickoff", "fixture_id", "repo_match_id"] if col in rows.columns]
    if sort_cols:
        rows = rows.sort_values(sort_cols)

    fixtures = []
    for row in rows.itertuples(index=False):
        home_en = str(getattr(row, "home_team"))
        away_en = str(getattr(row, "away_team"))
        key = (home_en.casefold(), away_en.casefold())
        reverse_key = (away_en.casefold(), home_en.casefold())
        if key in existing_keys or reverse_key in existing_keys:
            continue

        match_id = getattr(row, "repo_match_id", None)
        if pd.isna(match_id):
            match_id = getattr(row, "fixture_id", "")
        match_time = getattr(row, "local_kickoff", "")
        fixtures.append(
            {
                "match_id": f"fixture-{match_id}",
                "match_time": str(match_time or "未开盘"),
                "home_team": _team_display(home_en),
                "away_team": _team_display(away_en),
                "source": "fixture",
            }
        )
    return fixtures


def _latest_team_value(frame: pd.DataFrame, team: str, value_col: str, target_date: str) -> float | None:
    if frame.empty or "team" not in frame.columns or value_col not in frame.columns:
        return None
    rows = frame[frame["team"].astype(str).str.casefold() == team.casefold()].copy()
    if rows.empty:
        return None
    if "date" in rows.columns:
        target = pd.to_datetime(target_date, errors="coerce")
        if pd.notna(target):
            rows = rows[rows["date"] <= target]
            if rows.empty:
                return None
            rows = rows.sort_values("date")
    value = rows.iloc[-1][value_col]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_squad_features(row: dict, bundle: dict, home_team: str, away_team: str) -> None:
    squad = bundle.get("squad", pd.DataFrame())
    if squad.empty or "team" not in squad.columns:
        return

    home = squad[squad["team"].astype(str).str.casefold() == home_team.casefold()]
    away = squad[squad["team"].astype(str).str.casefold() == away_team.casefold()]
    home_row = home.iloc[-1].to_dict() if not home.empty else {}
    away_row = away.iloc[-1].to_dict() if not away.empty else {}

    for feature in bundle["feature_names"]:
        if feature.startswith("home_squad_"):
            base = feature.removeprefix("home_squad_")
            row[feature] = home_row.get(base)
        elif feature.startswith("away_squad_"):
            base = feature.removeprefix("away_squad_")
            row[feature] = away_row.get(base)
        elif feature.startswith("squad_") and feature.endswith("_diff"):
            base = feature.removeprefix("squad_").removesuffix("_diff")
            home_value = home_row.get(base)
            away_value = away_row.get(base)
            try:
                row[feature] = float(home_value) - float(away_value)
            except (TypeError, ValueError):
                row[feature] = None


def _equivalent_odds_from_probs(probs: dict[str, float]) -> dict[str, float]:
    mapping = {
        "home_win": "closing_home_odds",
        "draw": "closing_draw_odds",
        "away_win": "closing_away_odds",
    }
    odds = {}
    for key, col in mapping.items():
        prob = probs.get(key)
        if prob and prob > 0:
            odds[col] = 1 / prob
    return odds


def _renormalize_probs(values) -> list[float]:
    clipped = [max(0.01, float(value)) for value in values]
    total = sum(clipped)
    if total <= 0:
        return [1 / 3, 1 / 3, 1 / 3]
    return [value / total for value in clipped]


def _apply_group_motivation_adjustment(probabilities, motivation: dict) -> tuple[list[float], str]:
    """
    Apply a small transparent adjustment for group qualification pressure.

    This is intentionally conservative: it nudges the local model, but does not
    replace the trained model's ELO/FIFA/odds/squad signal.
    """
    if not motivation or not motivation.get("available"):
        return list(probabilities), "暂无小组出线压力修正"

    home_pressure = float(motivation.get("home_group_pressure", 0) or 0)
    away_pressure = float(motivation.get("away_group_pressure", 0) or 0)
    pressure_diff = max(-1.0, min(1.0, home_pressure - away_pressure))
    shared_pressure = max(0.0, min(home_pressure, away_pressure))

    home, draw, away = [float(value) for value in probabilities]
    if abs(pressure_diff) > 0:
        shift = 0.04 * pressure_diff
        home += shift
        away -= shift
        draw -= 0.015 * abs(pressure_diff)

    if shared_pressure > 0:
        # When both teams are under pressure, the draw becomes a little less
        # attractive because both sides have a stronger reason to chase a win.
        draw -= 0.025 * shared_pressure
        home += 0.0125 * shared_pressure
        away += 0.0125 * shared_pressure

    adjusted = _renormalize_probs([home, draw, away])
    if home_pressure == 0 and away_pressure == 0:
        note = "小组压力中性，未明显修正"
    elif pressure_diff > 0:
        note = "主队出线/净胜球压力更高，模型小幅上调主胜并压低平局"
    elif pressure_diff < 0:
        note = "客队出线/净胜球压力更高，模型小幅上调客胜并压低平局"
    else:
        note = "双方压力接近，模型仅小幅压低平局"
    return adjusted, note


def _predict_local_model(
    home_team_cn: str,
    away_team_cn: str,
    target_date: str,
    had_odds: dict[str, float],
    market_probs: dict[str, float],
    group_motivation: dict | None = None,
) -> dict:
    bundle = _load_local_model_bundle()
    if not bundle.get("available"):
        return {"available": False, "reason": bundle.get("reason", "模型不可用")}

    home_team = _team_en(home_team_cn)
    away_team = _team_en(away_team_cn)
    feature_names = bundle["feature_names"]
    row = {}

    row.update(
        {
            "closing_home_odds": had_odds.get("home_win"),
            "closing_draw_odds": had_odds.get("draw"),
            "closing_away_odds": had_odds.get("away_win"),
            "is_neutral": 1,
            "is_world_cup": 1,
            "is_friendly": 0,
        }
    )
    if not all(row.get(col) for col in ["closing_home_odds", "closing_draw_odds", "closing_away_odds"]):
        row.update(_equivalent_odds_from_probs(market_probs))

    home_elo = _latest_team_value(bundle["elo"], home_team, "elo", target_date)
    away_elo = _latest_team_value(bundle["elo"], away_team, "elo", target_date)
    row["home_elo"] = home_elo
    row["away_elo"] = away_elo
    if home_elo is not None and away_elo is not None:
        row["elo_diff"] = home_elo - away_elo
        row["elo_ratio"] = home_elo / away_elo if away_elo else None

    home_fifa = _latest_team_value(bundle["fifa"], home_team, "fifa_points", target_date)
    away_fifa = _latest_team_value(bundle["fifa"], away_team, "fifa_points", target_date)
    row["home_fifa_points"] = home_fifa
    row["away_fifa_points"] = away_fifa
    if home_fifa is not None and away_fifa is not None:
        row["fifa_points_diff"] = home_fifa - away_fifa

    _add_squad_features(row, bundle, home_team, away_team)
    if not bundle["results"].empty:
        row.update(recent_form_for_single_match(bundle["results"], home_team, away_team, target_date, window=5))
    if group_motivation and group_motivation.get("available"):
        for feature in MOTIVATION_COLUMNS:
            if feature in group_motivation:
                row[feature] = group_motivation[feature]

    values = []
    missing = []
    for feature in feature_names:
        value = row.get(feature)
        if value in ("", None) or pd.isna(value):
            value = bundle["medians"].get(feature, 0)
            missing.append(feature)
        values.append(float(value))

    X = pd.DataFrame([values], columns=feature_names)
    raw_probabilities = bundle["model"].predict_proba(X)[0]
    probabilities, motivation_note = _apply_group_motivation_adjustment(raw_probabilities, group_motivation or {})
    try:
        prediction = int(bundle["model"].predict(X)[0])
    except Exception:
        prediction = max(range(len(probabilities)), key=lambda index: probabilities[index])
    poisson_result = {"available": False}
    poisson_model = bundle.get("poisson_model")
    if poisson_model is not None:
        try:
            poisson_rows = poisson_model.predict_probability_rows(X)
            poisson_row = poisson_rows.iloc[0]
            poisson_result = {
                "available": True,
                "home_lambda": float(poisson_row["home_lambda"]),
                "away_lambda": float(poisson_row["away_lambda"]),
                "home_win_prob": float(poisson_row["home_win_prob"]),
                "draw_prob": float(poisson_row["draw_prob"]),
                "away_win_prob": float(poisson_row["away_win_prob"]),
                "pick": str(poisson_row["pick"]),
                "top_scores": poisson_row["top_scores"],
            }
        except Exception as exc:
            poisson_result = {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "model_source": bundle.get("model_source", "single_model"),
        "home_prob": float(probabilities[0]),
        "draw_prob": float(probabilities[1]),
        "away_prob": float(probabilities[2]),
        "raw_home_prob": float(raw_probabilities[0]),
        "raw_draw_prob": float(raw_probabilities[1]),
        "raw_away_prob": float(raw_probabilities[2]),
        "pick": MODEL_LABELS[prediction],
        "missing_count": len(missing),
        "missing_features": missing[:8],
        "group_motivation": group_motivation or {"available": False},
        "motivation_note": motivation_note,
        "poisson": poisson_result,
    }


def _match_pick(probs: dict[str, float]) -> str:
    labels = {"home_win": "主胜", "draw": "平局", "away_win": "客胜"}
    if not probs:
        return "-"
    key = max(probs, key=probs.get)
    return labels.get(key, key)


def _implied_probs_from_crs(crs: pd.DataFrame) -> dict[str, float]:
    """Infer WDL probabilities from exact-score odds when HAD is unavailable."""
    if crs.empty:
        return {}
    totals = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
    for row in crs.itertuples():
        outcome = str(row.outcome)
        if "other" in outcome or "-" not in outcome:
            continue
        try:
            home_goals, away_goals = [int(part) for part in outcome.split("-", 1)]
            odds = float(row.odds)
        except (TypeError, ValueError):
            continue
        if odds <= 0:
            continue
        key = "home_win" if home_goals > away_goals else "away_win" if away_goals > home_goals else "draw"
        totals[key] += 1 / odds
    total = sum(totals.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in totals.items()}


def _normalized_probs_from_odds(odds: dict[str, float]) -> dict[str, float]:
    inv = {key: 1 / value for key, value in odds.items() if value and value > 0}
    total = sum(inv.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in inv.items()}


def _handicap_pick(hhad: pd.DataFrame) -> tuple[str, dict[str, float]]:
    labels = {"home_win": "让胜", "draw": "让平", "away_win": "让负"}
    if hhad.empty:
        return "-", {}
    odds = _odds_map(hhad)
    probs = _normalized_probs_from_odds(odds)
    if not probs:
        return "-", {}
    key = max(probs, key=probs.get)
    return labels.get(key, key), probs


def _clean_alert_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title or "")).strip()


POOL_TITLES = {
    "had": "HAD 胜平负",
    "hhad": "HHAD 让球胜平负",
    "crs": "CRS 比分",
    "ttg": "TTG 总进球",
    "hafu": "HAFU 半全场",
}


POOL_ORDER = ["had", "hhad", "ttg", "hafu", "crs"]

OUTCOME_LABELS = {
    "home_win": "主胜",
    "draw": "平局",
    "away_win": "客胜",
    "home_other": "主胜其他",
    "draw_other": "平局其他",
    "away_other": "客胜其他",
}


def _display_outcome(value: str) -> str:
    return OUTCOME_LABELS.get(str(value), str(value))


def _parse_json_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _polymarket_date_for_match(target_date: str) -> str:
    """Polymarket match slugs use US-local date, usually one day before China time."""
    try:
        return (datetime.fromisoformat(target_date).date() - timedelta(days=1)).isoformat()
    except ValueError:
        return target_date


def _score_from_question(question: str) -> str:
    text = str(question or "")
    match = re.search(r"Exact Score:\s*.*?(\d+)\s*-\s*(\d+).*?\?", text)
    if not match:
        match = re.search(r"\b(\d+)\s*-\s*(\d+)\b", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    if "Any Other Score" in text:
        return "其他比分"
    return text


def _parse_polymarket_exact_score_markets(markets: list[dict]) -> list[dict]:
    rows = []
    for market in markets or []:
        question = str(market.get("question") or "")
        if "Exact Score:" not in question and "Any Other Score" not in question:
            continue

        outcomes = _parse_json_list(market.get("outcomes"))
        prices = _parse_json_list(market.get("outcomePrices"))
        if "Yes" not in outcomes:
            continue
        yes_index = outcomes.index("Yes")
        try:
            yes_prob = float(prices[yes_index])
        except (IndexError, TypeError, ValueError):
            continue

        rows.append(
            {
                "score": _score_from_question(question),
                "yes_probability": yes_prob,
                "yes_price_decimal": (1 / yes_prob) if yes_prob > 0 else None,
                "volume": float(market.get("volume") or market.get("volumeNum") or 0),
                "liquidity": float(market.get("liquidity") or market.get("liquidityNum") or 0),
                "question": question,
                "slug": market.get("slug"),
            }
        )
    rows.sort(key=lambda item: item["yes_probability"], reverse=True)
    return rows


def _extract_field(text: str, field: str) -> str | None:
    match = re.search(rf'"{re.escape(field)}":"((?:\\.|[^"\\])*)"', text)
    if not match:
        return None
    return match.group(1).encode("utf-8").decode("unicode_escape")


def _extract_number_field(text: str, field: str) -> float:
    match = re.search(rf'"{re.escape(field)}":(?:"([^"]*)"|([0-9.]+))', text)
    if not match:
        return 0.0
    raw = match.group(1) if match.group(1) is not None else match.group(2)
    try:
        return float(raw or 0)
    except (TypeError, ValueError):
        return 0.0


@lru_cache(maxsize=128)
def _fetch_polymarket_exact_scores_from_page(event_slug: str) -> tuple[dict, ...]:
    """Fallback for markets embedded in Polymarket's sports page but absent from Gamma event API."""
    response = None
    last_error = None
    urls = [
        f"https://polymarket.com/sports/world-cup/{event_slug}",
        f"https://polymarket.com/event/{event_slug}",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "close",
    }
    for attempt in range(1):
        for url in urls:
            try:
                response = requests.get(url, timeout=6, headers=headers)
                if response.text and "Exact Score:" in response.text:
                    break
            except requests.RequestException as exc:
                last_error = exc
                response = None
        if response is not None and response.text and "Exact Score:" in response.text:
            break
        time.sleep(0.8 * (attempt + 1))
    if response is None:
        raise last_error or requests.RequestException("Polymarket page request failed")
    response.raise_for_status()
    html_text = response.text

    rows = []
    seen_slugs = set()
    for match in re.finditer(r'"question":"Exact Score:[^"]+\?"', html_text):
        start = max(0, html_text.rfind('{"id"', 0, match.start()))
        end = html_text.find(',"clobTokenIds"', match.end())
        if end == -1:
            end = min(len(html_text), match.end() + 5000)
        block = html_text[start:end]

        question = _extract_field(block, "question")
        slug = _extract_field(block, "slug")
        if not question or not slug or slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        prices_match = re.search(r'"outcomePrices":\["([^"]+)","([^"]+)"\]', block)
        if not prices_match:
            continue
        try:
            yes_prob = float(prices_match.group(1))
        except ValueError:
            continue

        rows.append(
            {
                "score": _score_from_question(question),
                "yes_probability": yes_prob,
                "yes_price_decimal": (1 / yes_prob) if yes_prob > 0 else None,
                "volume": _extract_number_field(block, "volume"),
                "liquidity": _extract_number_field(block, "liquidity"),
                "question": question,
                "slug": slug,
            }
        )

    rows.sort(key=lambda item: item["yes_probability"], reverse=True)
    return tuple(rows)


def _fetch_polymarket_exact_scores(
    home_team: str,
    away_team: str,
    target_date: str,
    allow_page_fallback: bool = True,
) -> list[dict]:
    home_code = POLYMARKET_TEAM_CODES.get(home_team) or POLYMARKET_TEAM_CODES.get(_team_display(home_team))
    away_code = POLYMARKET_TEAM_CODES.get(away_team) or POLYMARKET_TEAM_CODES.get(_team_display(away_team))
    if not home_code or not away_code:
        return []

    slug_date = _polymarket_date_for_match(target_date)
    event_slug = f"fifwc-{home_code}-{away_code}-{slug_date}"
    response = requests.get(
        GAMMA_EVENTS_URL,
        params={"slug": event_slug},
        timeout=3,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    response.raise_for_status()
    events = response.json()
    if not isinstance(events, list) or not events:
        return list(_fetch_polymarket_exact_scores_from_page(event_slug)) if allow_page_fallback else []

    rows = _parse_polymarket_exact_score_markets(events[0].get("markets", []) or [])
    if rows:
        return rows
    return list(_fetch_polymarket_exact_scores_from_page(event_slug)) if allow_page_fallback else []


def _polymarket_exact_cache_key(home_team: str, away_team: str, target_date: str) -> str:
    return f"{target_date}|{home_team}|{away_team}"


def _refresh_polymarket_exact_scores_background(
    key: str,
    home_team: str,
    away_team: str,
    target_date: str,
) -> None:
    """Refresh slow Polymarket page fallback without blocking dashboard rendering."""
    try:
        rows = _fetch_polymarket_exact_scores(
            home_team,
            away_team,
            target_date,
            allow_page_fallback=True,
        )
        if rows:
            POLYMARKET_EXACT_SCORE_LAST_GOOD[key] = rows
            POLYMARKET_EXACT_SCORE_LAST_UPDATED[key] = datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
                "%H:%M:%S"
            )
    except requests.RequestException as exc:
        print(f"[warn] Polymarket exact-score background refresh failed for {key}: {exc}")
    finally:
        with POLYMARKET_EXACT_REFRESH_LOCK:
            POLYMARKET_EXACT_REFRESHING_KEYS.discard(key)


def _schedule_polymarket_exact_refresh(home_team: str, away_team: str, target_date: str) -> bool:
    key = _polymarket_exact_cache_key(home_team, away_team, target_date)
    with POLYMARKET_EXACT_REFRESH_LOCK:
        if key in POLYMARKET_EXACT_REFRESHING_KEYS:
            return False
        POLYMARKET_EXACT_REFRESHING_KEYS.add(key)
    worker = threading.Thread(
        target=_refresh_polymarket_exact_scores_background,
        args=(key, home_team, away_team, target_date),
        name=f"polymarket-exact-{key}",
        daemon=True,
    )
    worker.start()
    return True


def _get_polymarket_exact_scores_stable(home_team: str, away_team: str, target_date: str) -> tuple[list[dict], str]:
    """Return live Exact Score rows, falling back to the last successful refresh."""
    key = _polymarket_exact_cache_key(home_team, away_team, target_date)
    try:
        rows = _fetch_polymarket_exact_scores(
            home_team,
            away_team,
            target_date,
            allow_page_fallback=False,
        )
        if rows:
            POLYMARKET_EXACT_SCORE_LAST_GOOD[key] = rows
            POLYMARKET_EXACT_SCORE_LAST_UPDATED[key] = datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
                "%H:%M:%S"
            )
            return rows, "实时"
    except requests.RequestException:
        pass

    cached = POLYMARKET_EXACT_SCORE_LAST_GOOD.get(key, [])
    if cached:
        updated = POLYMARKET_EXACT_SCORE_LAST_UPDATED.get(key, "")
        suffix = f"缓存 {updated}" if updated else "缓存"
        _schedule_polymarket_exact_refresh(home_team, away_team, target_date)
        return cached, suffix
    started = _schedule_polymarket_exact_refresh(home_team, away_team, target_date)
    return [], "后台刷新" if started else "后台刷新中"


def _large_exact_score_buys_from_trades(
    trades: list[dict],
    home_team: str,
    away_team: str,
    target_date: str,
    min_notional: float,
) -> list[dict]:
    alerts = []
    for trade in trades:
        if str(trade.get("side", "")).upper() != "BUY":
            continue
        if not match_trade_to_game(trade, home_team, away_team, target_date):
            continue
        size = safe_float(trade.get("size")) or 0.0
        price = safe_float(trade.get("price")) or 0.0
        notional = size * price
        if notional < min_notional:
            continue
        alerts.append(
            {
                "notional": notional,
                "price": price,
                "size": size,
                "title": trade.get("title", ""),
                "outcome": trade.get("outcome", ""),
                "trader": trade.get("name") or trade.get("pseudonym") or "",
            }
        )
    alerts.sort(key=lambda item: item["notional"], reverse=True)
    return alerts


OFF_FIELD_SENTIMENT_TTL_SECONDS = 6 * 60 * 60


def _sentiment_refresh_message(target_date: str) -> str:
    return f"场外因素正在后台刷新 DeepSeek 联网搜索，日期 {target_date}；本次先使用缓存或中性值。"


def _load_off_field_sentiment_file(target_date: str) -> dict:
    if not SENTIMENT_JSON_PATH.exists():
        return {}
    try:
        payload = json.loads(SENTIMENT_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if str(payload.get("date")) != str(target_date):
        return {}
    return payload


def _sentiment_payload_is_fresh(payload: dict) -> bool:
    if payload.get("freshness_rule_version") != OFF_FIELD_FRESHNESS_RULE_VERSION:
        return False
    if OFF_FIELD_DEEPSEEK_API_KEY and _payload_has_missing_key_fallback(payload):
        return False
    generated_at = payload.get("generated_at")
    if not generated_at:
        return False
    try:
        generated = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
    except ValueError:
        return False
    now = datetime.now(generated.tzinfo or ZoneInfo("UTC"))
    return (now - generated).total_seconds() <= OFF_FIELD_SENTIMENT_TTL_SECONDS


def _payload_has_missing_key_fallback(payload: dict) -> bool:
    teams = payload.get("teams") or {}
    for item in teams.values():
        reason = str(item.get("reasoning", ""))
        if "DEEPSEEK_API_KEY" in reason and "not set" in reason:
            return True
    return False


def _clear_runtime_caches() -> None:
    """Clear in-process caches after code/data changes or explicit force refresh."""
    _load_local_model_bundle.cache_clear()
    _load_fixture_metadata_table.cache_clear()
    _fetch_polymarket_exact_scores_from_page.cache_clear()
    POLYMARKET_EXACT_SCORE_LAST_GOOD.clear()
    POLYMARKET_EXACT_SCORE_LAST_UPDATED.clear()
    with POLYMARKET_EXACT_REFRESH_LOCK:
        POLYMARKET_EXACT_REFRESHING_KEYS.clear()


def _refresh_off_field_sentiment_background(target_date: str) -> None:
    """Refresh slow DeepSeek web-search sentiment without blocking the dashboard."""
    try:
        payload = build_off_field_sentiment_payload(
            target_date,
            source="both",
            max_results=4,
            use_api=True,
            search_mode="deepseek",
        )
        save_off_field_sentiment_outputs(payload)
    except Exception as exc:
        print(f"[warn] off-field sentiment background refresh failed for {target_date}: {exc}")
    finally:
        with OFF_FIELD_REFRESH_LOCK:
            OFF_FIELD_REFRESHING_DATES.discard(target_date)


def _schedule_off_field_sentiment_refresh(target_date: str) -> bool:
    """Start at most one DeepSeek sentiment refresh per date."""
    with OFF_FIELD_REFRESH_LOCK:
        if target_date in OFF_FIELD_REFRESHING_DATES:
            return False
        OFF_FIELD_REFRESHING_DATES.add(target_date)

    worker = threading.Thread(
        target=_refresh_off_field_sentiment_background,
        args=(target_date,),
        name=f"off-field-sentiment-{target_date}",
        daemon=True,
    )
    worker.start()
    return True


def _get_off_field_sentiment_payload(target_date: str, force_refresh: bool = False) -> dict:
    """
    Load or refresh off-field sentiment.

    The web dashboard refreshes often, so this uses a 6-hour cache. If the file
    is missing or stale, DeepSeek sentiment refreshes in the background because
    web search can be slow or unstable.
    """
    payload = _load_off_field_sentiment_file(target_date)
    if payload and not force_refresh and _sentiment_payload_is_fresh(payload):
        return payload

    started = _schedule_off_field_sentiment_refresh(target_date)
    message = _sentiment_refresh_message(target_date)
    if not started:
        message = f"{message} 当前已有后台任务在运行。"

    if payload:
        payload = dict(payload)
        payload["refresh_error"] = message
        return payload

    return {
        "date": target_date,
        "teams": {},
        "refresh_error": message,
        "generated_at": "",
        "background_refreshing": True,
    }


def _team_sentiment(sentiment_payload: dict, team_name: str) -> dict:
    teams = sentiment_payload.get("teams") or {}
    team_en = normalize_sentiment_team_name(_team_en(team_name))
    item = teams.get(team_en) or teams.get(str(team_name)) or {}
    dimensions = item.get("dimensions") if isinstance(item.get("dimensions"), dict) else {}
    return {
        "available": bool(item),
        "team": team_en,
        "overall": float(item.get("overall", 0) or 0),
        "confidence": float(item.get("confidence", 0) or 0),
        "reasoning": str(item.get("reasoning", "")),
        "dimensions": {
            "morale": float(dimensions.get("morale", 0) or 0),
            "external": float(dimensions.get("external", 0) or 0),
            "media": float(dimensions.get("media", 0) or 0),
            "momentum": float(dimensions.get("momentum", 0) or 0),
        },
    }


def _match_off_field_sentiment(sentiment_payload: dict, home_team: str, away_team: str) -> dict:
    home = _team_sentiment(sentiment_payload, home_team)
    away = _team_sentiment(sentiment_payload, away_team)
    diff = (home["overall"] * home["confidence"]) - (away["overall"] * away["confidence"])
    return {
        "available": home["available"] or away["available"],
        "home": home,
        "away": away,
        "diff": float(max(-3.0, min(3.0, diff))),
        "refresh_error": sentiment_payload.get("refresh_error", ""),
    }


def _adjust_probability_by_sentiment(
    probability: float,
    candidate: dict,
    match_sentiment: dict,
) -> tuple[float, str]:
    """Slightly nudge package probabilities using off-field sentiment."""
    if not match_sentiment or not match_sentiment.get("available"):
        return probability, ""

    diff = float(match_sentiment.get("diff", 0) or 0)
    selection = str(candidate.get("selection", ""))
    pool = str(candidate.get("pool", ""))

    direction = 0.0
    if "主胜" in selection or "让胜" in selection:
        direction = 1.0
    elif "客胜" in selection or "让负" in selection:
        direction = -1.0
    elif "平" in selection:
        direction = -0.25 * abs(diff)

    if direction == 0:
        return probability, ""

    # Max move is small by design: off-field factors should influence betting
    # selection, not overpower model and market probabilities.
    if "HAD" in pool or "HHAD" in pool:
        multiplier = 1.0 + 0.035 * diff * direction
    else:
        multiplier = 1.0 + 0.02 * diff * direction
    adjusted = max(0.001, min(0.98, probability * multiplier))
    note = "场外因素小幅加权" if abs(adjusted - probability) >= 0.001 else ""
    return adjusted, note


def _kelly_fraction(probability: float, decimal_odds: float) -> float:
    """Full Kelly fraction for a decimal-odds bet."""
    if probability <= 0 or decimal_odds <= 1:
        return 0.0
    edge = probability * decimal_odds - 1
    if edge <= 0:
        return 0.0
    return max(0.0, edge / (decimal_odds - 1))


def _selection_edge(probability: float, decimal_odds: float) -> dict:
    edge = probability * decimal_odds - 1 if probability and decimal_odds else 0.0
    return {
        "probability": float(probability or 0),
        "odds": float(decimal_odds or 0),
        "expected_value": float(edge),
        "kelly": float(_kelly_fraction(float(probability or 0), float(decimal_odds or 0))),
    }


def _normalize_market_selection(pool: str, selection: str) -> tuple[str, str]:
    """Return a comparable pool/selection key for advice and market rows."""
    pool_text = str(pool or "")
    text = str(selection or "").strip()
    if "让球" in pool_text or text.startswith("让"):
        pool_code = "hhad"
        if text.startswith("让胜"):
            result = "让胜"
        elif text.startswith("让平"):
            result = "让平"
        elif text.startswith("让负"):
            result = "让负"
        else:
            result = text
    else:
        pool_code = "had"
        if text in {"平", "平局"}:
            result = "平"
        elif "主胜" in text:
            result = "主胜"
        elif "客胜" in text:
            result = "客胜"
        else:
            result = text
    return pool_code, result


def _market_signal_for_candidate(match: dict, pool: str, selection: str) -> dict | None:
    target_pool, target_selection = _normalize_market_selection(pool, selection)
    for row in match.get("market_decision") or []:
        row_pool, row_selection = _normalize_market_selection(
            str(row.get("pool_code") or row.get("pool") or ""),
            str(row.get("selection") or ""),
        )
        if row_pool == target_pool and row_selection == target_selection:
            return row
    return None


def _apply_market_signal_to_candidate(candidate: dict, match: dict) -> dict:
    """
    Nudge advice candidates with the live odds-movement decision score.

    The nudge is intentionally bounded: market movement should help rank similar
    options, not override the base model by itself.
    """
    signal = _market_signal_for_candidate(
        match,
        str(candidate.get("pool") or ""),
        str(candidate.get("selection") or ""),
    )
    if not signal:
        candidate["market_signal_score"] = 0.0
        return candidate

    score = float(signal.get("final_score") or 0)
    multiplier = 1.0 + max(-0.15, min(0.18, score))
    adjusted_probability = max(0.001, min(0.98, float(candidate.get("probability") or 0) * multiplier))
    odds = float(candidate.get("odds") or 0)
    candidate.update(_selection_edge(adjusted_probability, odds))
    candidate["market_signal_score"] = score
    candidate["market_signal_grade"] = str(signal.get("grade") or "")
    note = str(candidate.get("note") or "")
    signal_note = str(signal.get("note") or "")
    suffix = f"盘口评分{score:+.2f}"
    if signal_note:
        suffix += f"（{signal_note}）"
    candidate["note"] = f"{note}；{suffix}" if note else suffix
    return candidate


def _handicap_probs_from_poisson(model: dict, handicap_line: str | float | int | None) -> dict[str, float]:
    """
    Estimate HHAD probabilities from the Poisson score matrix.

    Sporttery HHAD uses home_score + handicap versus away_score:
    - home_score + handicap > away_score: 让胜
    - home_score + handicap == away_score: 让平
    - home_score + handicap < away_score: 让负
    """
    poisson = model.get("poisson") or {}
    if not poisson.get("available"):
        return {}
    handicap = safe_float(handicap_line)
    if handicap is None:
        return {}

    matrix = score_matrix_from_lambdas(
        float(poisson.get("home_lambda") or 0),
        float(poisson.get("away_lambda") or 0),
        max_goals=8,
    )
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            probability = float(matrix[home_goals, away_goals])
            adjusted_margin = home_goals + handicap - away_goals
            if adjusted_margin > 1e-9:
                home_win += probability
            elif adjusted_margin < -1e-9:
                away_win += probability
            else:
                draw += probability

    total = home_win + draw + away_win
    if total <= 0:
        return {}
    return {
        "home_prob": home_win / total,
        "draw_prob": draw / total,
        "away_prob": away_win / total,
    }


def _build_parlay_plan(
    candidates: list[dict],
    label: str,
    max_legs: int,
    min_kelly: float,
    kelly_multiplier: float,
    bankroll_cap: float,
    min_probability: float = 0.0,
    require_model_pick: bool = False,
) -> dict:
    raw_usable = [
        item
        for item in candidates
        if item.get("kelly", 0) >= min_kelly
        and item.get("probability", 0) >= min_probability
        and (not require_model_pick or item.get("is_model_pick"))
        and item.get("probability", 0) > 0
        and item.get("odds", 0) > 1
    ]
    best_by_match = {}
    for item in raw_usable:
        key = item.get("match", "")
        current = best_by_match.get(key)
        if current is None or (
            item["kelly"],
            item["expected_value"],
            item["probability"],
        ) > (
            current["kelly"],
            current["expected_value"],
            current["probability"],
        ):
            best_by_match[key] = item
    usable = list(best_by_match.values())
    usable = sorted(
        usable,
        key=lambda item: (item["kelly"], item["expected_value"], item["probability"]),
        reverse=True,
    )
    selected = usable[:max_legs]
    if len(selected) < 2:
        return {
            "available": False,
            "title": label,
            "reason": "可用正期望选项不足 2 场，按中国体彩串关口径暂不建议强行组串。",
            "legs": [],
        }

    combined_odds = 1.0
    combined_probability = 1.0
    for item in selected:
        combined_odds *= item["odds"]
        combined_probability *= item["probability"]

    combined_ev = combined_probability * combined_odds - 1
    full_kelly = _kelly_fraction(combined_probability, combined_odds)
    stake_fraction = min(bankroll_cap, full_kelly * kelly_multiplier)
    return {
        "available": combined_ev > 0 and stake_fraction > 0,
        "title": label,
        "play_type": f"{len(selected)}串1",
        "combined_odds": float(combined_odds),
        "combined_probability": float(combined_probability),
        "expected_value": float(combined_ev),
        "full_kelly": float(full_kelly),
        "suggested_bankroll_fraction": float(stake_fraction),
        "legs": selected,
        "reason": "" if combined_ev > 0 else "组合期望值不为正，不建议购买。",
    }


def _build_betting_package(
    candidates: list[dict],
    matches: list[dict],
    title: str,
    min_probability: float,
    max_options_per_match: int,
    aggressive: bool,
    stake_per_bet: float = 2.0,
    max_total_cost: float | None = None,
) -> dict:
    """
    Build a real Sporttery-style package: one or more options per match.

    A 4-match day with option counts [2, 2, 2, 1] becomes 8 tickets of 4串1.
    """
    match_order = [str(match["match_id"]) for match in matches]
    match_names = {
        str(match["match_id"]): f"{match['home_team']} vs {match['away_team']}"
        for match in matches
    }
    by_match = {match_id: [] for match_id in match_order}
    for item in candidates:
        match_id = str(item.get("match_id", ""))
        selection_text = str(item.get("selection", ""))
        if "平" in selection_text and not aggressive:
            min_draw_edge = 0.03
            if item.get("expected_value", 0) < min_draw_edge:
                continue
        if match_id in by_match and item.get("odds", 0) > 1 and item.get("probability", 0) >= min_probability:
            by_match[match_id].append(item)

    selected_by_match = []
    skipped_matches = []
    for match_id in match_order:
        items = by_match.get(match_id, [])
        if not items:
            skipped_matches.append(match_names.get(match_id, match_id))
            continue

        if aggressive:
            ranked = sorted(
                items,
                key=lambda item: (
                    item.get("market_signal_score", 0),
                    item.get("expected_value", 0),
                    item.get("probability", 0) * item.get("odds", 0),
                    item.get("expected_value", 0),
                    item.get("odds", 0),
                    item.get("probability", 0),
                ),
                reverse=True,
            )
        else:
            model_pick_item = next((item for item in items if item.get("is_model_pick")), None)
            model_direction = None
            if model_pick_item:
                pick_selection = str(model_pick_item.get("selection", ""))
                if "主胜" in pick_selection:
                    model_direction = "home"
                elif "客胜" in pick_selection:
                    model_direction = "away"
                elif "平" in pick_selection:
                    model_direction = "draw"
            if model_direction is None:
                model_pick_label = str((items[0] or {}).get("model_pick", ""))
                if model_pick_label == "主胜":
                    model_direction = "home"
                elif model_pick_label == "客胜":
                    model_direction = "away"
                elif model_pick_label == "平局":
                    model_direction = "draw"

            def same_direction(item: dict) -> bool:
                if model_direction is None:
                    return True
                selection = str(item.get("selection", ""))
                if model_direction == "home":
                    return "主胜" in selection or "让胜" in selection
                if model_direction == "away":
                    return "客胜" in selection or "让负" in selection
                return "平" in selection or "让平" in selection

            qualified = []
            for min_odds in [1.5, 1.3]:
                qualified = [item for item in items if item.get("odds", 0) >= min_odds]
                if qualified:
                    break
            if not qualified:
                skipped_matches.append(match_names.get(match_id, match_id))
                continue

            broad_qualified = list(qualified)
            directional = [item for item in qualified if same_direction(item)]
            if directional:
                qualified = directional
            positive_edge = [item for item in qualified if item.get("expected_value", 0) > 0]
            if positive_edge:
                qualified = positive_edge
            else:
                # Practical parlay mode: keep one reasonable same-direction leg
                # when EV is only mildly negative. This preserves 4-leg tickets
                # without buying the whole handicap board.
                soft_fallback = [
                    item
                    for item in qualified
                    if item.get("expected_value", 0) >= -0.18
                    and item.get("probability", 0) >= 0.35
                ]
                if soft_fallback:
                    qualified = soft_fallback
                    for item in qualified:
                        item["note"] = str(item.get("note", "")) + "；补足串关的小幅负EV腿"
                else:
                    value_fallback = [
                        item
                        for item in broad_qualified
                        if item.get("pool") == "HHAD 让球胜平负"
                        if item.get("expected_value", 0) >= 0.08
                        and item.get("probability", 0) >= 0.25
                    ]
                    if value_fallback:
                        qualified = value_fallback[:1]
                        for item in qualified:
                            item["note"] = str(item.get("note", "")) + "；非模型方向的高EV保护腿"
                    else:
                        skipped_matches.append(match_names.get(match_id, match_id))
                        continue

            ranked = sorted(
                qualified,
                key=lambda item: (
                    bool(item.get("is_model_pick")),
                    same_direction(item),
                    item.get("market_signal_score", 0),
                    item.get("probability", 0) * item.get("odds", 0),
                    item.get("kelly", 0),
                    item.get("expected_value", 0),
                ),
                reverse=True,
            )
            chosen = []
            for item in ranked:
                if len(chosen) >= max_options_per_match:
                    break
                item["_same_dir_conservative"] = same_direction(item)
                chosen.append(item)

        if aggressive:
            chosen = ranked[:max_options_per_match]

        selected_by_match.append(
            {
                "match_id": match_id,
                "match": match_names.get(match_id, match_id),
                "options": chosen,
            }
        )

    if len(selected_by_match) < 2:
        reason = "可用选项不足 2 场，不强行组串。"
        if skipped_matches:
            reason += f" 已跳过：{'、'.join(skipped_matches)}"
        return {
            "available": False,
            "title": title,
            "reason": reason,
            "matches": [],
        }

    def aggregate(groups: list[dict]) -> dict:
        bet_count = 1
        min_combined_odds = 1.0
        max_combined_odds = 1.0
        package_hit_probability = 1.0
        expected_return_sum_factor = 1.0
        selected_count = 0
        kelly_values = []
        option_evs = []
        for group in groups:
            options = group["options"]
            bet_count *= len(options)
            selected_count += len(options)
            min_combined_odds *= min(item["odds"] for item in options)
            max_combined_odds *= max(item["odds"] for item in options)
            package_hit_probability *= min(0.98, sum(item["probability"] for item in options))
            expected_return_sum_factor *= sum(item["probability"] * item["odds"] for item in options)
            kelly_values.extend(item.get("kelly", 0) for item in options)
            option_evs.extend(item.get("expected_value", 0) for item in options)
        return {
            "bet_count": bet_count,
            "min_combined_odds": min_combined_odds,
            "max_combined_odds": max_combined_odds,
            "package_hit_probability": package_hit_probability,
            "expected_return_sum_factor": expected_return_sum_factor,
            "selected_count": selected_count,
            "average_kelly": sum(kelly_values) / len(kelly_values) if kelly_values else 0,
            "average_option_ev": sum(option_evs) / len(option_evs) if option_evs else 0,
            "total_cost": bet_count * stake_per_bet,
        }

    metrics = aggregate(selected_by_match)
    if max_total_cost is not None and metrics["total_cost"] > max_total_cost:
        trimmed = True
        while trimmed and metrics["total_cost"] > max_total_cost:
            trimmed = False
            remove_group = None
            remove_index = -1
            remove_score = float("inf")
            for group in selected_by_match:
                options = group["options"]
                if len(options) <= 1:
                    continue
                for index, item in enumerate(options):
                    if item.get("is_model_pick"):
                        continue
                    score = item.get("probability", 0) * item.get("odds", 0)
                    if item.get("_same_dir_conservative"):
                        score *= 2.0
                    if score < remove_score:
                        remove_score = score
                        remove_group = group
                        remove_index = index
            if remove_group is not None:
                remove_group["options"].pop(remove_index)
                trimmed = True
                metrics = aggregate(selected_by_match)

    bet_count = metrics["bet_count"]
    expected_value = metrics["expected_return_sum_factor"] / bet_count - 1

    return {
        "available": bool(expected_value > -0.12),
        "title": title,
        "play_type": f"{len(selected_by_match)}串1",
        "selected_count": int(metrics["selected_count"]),
        "bet_count": int(bet_count),
        "stake_per_bet": float(stake_per_bet),
        "total_cost": float(metrics["total_cost"]),
        "min_combined_odds": float(metrics["min_combined_odds"]),
        "max_combined_odds": float(metrics["max_combined_odds"]),
        "theoretical_max_prize": float(metrics["max_combined_odds"] * stake_per_bet),
        "package_hit_probability": float(metrics["package_hit_probability"]),
        "expected_value": float(expected_value),
        "average_kelly": float(metrics["average_kelly"]),
        "average_option_ev": float(metrics["average_option_ev"]),
        "matches": selected_by_match,
        "reason": (
            f"组合期望值偏低，属于娱乐型多串，不是严格价值投注。"
            if expected_value <= 0
            else (f"已跳过：{'、'.join(skipped_matches)}" if skipped_matches else "")
        ),
    }


def _score_result_key(label: str) -> str | None:
    text = str(label or "")
    match = re.search(r"(\d+)\s*-\s*(\d+)", text)
    if not match:
        return None
    home_goals = int(match.group(1))
    away_goals = int(match.group(2))
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def _exact_score_candidates(match: dict) -> list[dict]:
    """
    Estimate exact-score value from CRS odds plus local WDL probabilities.

    CRS alone only gives market-implied probabilities. We rescale each exact
    score by the local model's win/draw/loss probability for the same outcome.
    This is a pragmatic approximation until the score-regression model is wired
    into the dashboard as a calibrated exact-score model.
    """
    model = match.get("local_model") or {}
    if not model.get("available"):
        return []

    poisson = model.get("poisson") or {}
    if poisson.get("available"):
        market_odds = {
            str(score.get("label")): float(score.get("odds") or 0)
            for score in match.get("top_scores") or []
        }
        candidates = []
        for row in poisson.get("top_scores") or []:
            label = str(row.get("score") or "")
            probability = float(row.get("probability") or 0)
            odds = market_odds.get(label)
            if not odds or odds <= 1:
                continue
            edge = _selection_edge(probability, odds)
            candidates.append(
                {
                    **edge,
                    "match_id": str(match.get("match_id", "")),
                    "match": f"{match['home_team']} vs {match['away_team']}",
                    "selection": label,
                    "pool": "CRS 比分",
                    "note": "概率来自泊松比分基底模型",
                    "is_model_pick": label == (poisson.get("top_scores") or [{}])[0].get("score"),
                }
            )
        if candidates:
            return candidates

    outcome_model_probs = {
        "home_win": float(model.get("home_prob") or 0),
        "draw": float(model.get("draw_prob") or 0),
        "away_win": float(model.get("away_prob") or 0),
    }
    model_pick_key = {"主胜": "home_win", "平局": "draw", "客胜": "away_win"}.get(
        str(model.get("pick") or "")
    )
    scores = match.get("top_scores") or []
    if not scores:
        return []

    implied = []
    for score in scores:
        odds = float(score.get("odds") or 0)
        label = str(score.get("label") or "")
        outcome_key = _score_result_key(label)
        if odds <= 1 or outcome_key is None:
            continue
        implied.append(
            {
                "label": label,
                "odds": odds,
                "outcome_key": outcome_key,
                "implied_raw": 1 / odds,
            }
        )

    total_implied = sum(item["implied_raw"] for item in implied)
    if total_implied <= 0:
        return []

    market_outcome_probs = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
    for item in implied:
        item["market_probability"] = item["implied_raw"] / total_implied
        market_outcome_probs[item["outcome_key"]] += item["market_probability"]

    candidates = []
    for item in implied:
        market_outcome_prob = market_outcome_probs.get(item["outcome_key"], 0)
        if market_outcome_prob <= 0:
            continue
        adjusted_probability = item["market_probability"] * (
            outcome_model_probs[item["outcome_key"]] / market_outcome_prob
        )
        adjusted_probability = max(0.001, min(0.35, adjusted_probability))
        edge = _selection_edge(adjusted_probability, item["odds"])
        candidates.append(
            {
                **edge,
                "match_id": str(match.get("match_id", "")),
                "match": f"{match['home_team']} vs {match['away_team']}",
                "selection": item["label"],
                "pool": "CRS 比分",
                "note": "比分概率为CRS隐含概率按本地胜平负方向修正",
                "is_model_pick": item["outcome_key"] == model_pick_key,
            }
        )
    return candidates


def _build_daily_betting_advice(matches: list[dict]) -> dict:
    """Create Sporttery-style multi-selection packages for the selected day."""
    wdl_candidates = []
    score_candidates = []
    result_labels = {
        "home_win": "主胜",
        "draw": "平局",
        "away_win": "客胜",
    }
    model_pick_to_key = {"主胜": "home_win", "平局": "draw", "客胜": "away_win"}

    for match in matches:
        model = match.get("local_model") or {}
        had = match.get("had") or {}
        hhad = match.get("hhad") or {}
        match_sentiment = match.get("off_field_sentiment") or {}
        match_id = str(match.get("match_id", ""))
        match_name = f"{match['home_team']} vs {match['away_team']}"
        if model.get("available") and match.get("had_real_available"):
            model_pick_key = model_pick_to_key.get(str(model.get("pick") or ""))
            for key, prob_key, odds_key in [
                ("home_win", "home_prob", "home_odds"),
                ("draw", "draw_prob", "draw_odds"),
                ("away_win", "away_prob", "away_odds"),
            ]:
                probability = float(model.get(prob_key) or 0)
                odds = float(had.get(odds_key) or 0)
                temp_candidate = {"selection": result_labels[key], "pool": "HAD 胜平负"}
                probability, sentiment_note = _adjust_probability_by_sentiment(
                    probability,
                    temp_candidate,
                    match_sentiment,
                )
                edge = _selection_edge(probability, odds)
                if odds > 1:
                    candidate = {
                        **edge,
                        "match_id": match_id,
                        "match": match_name,
                        "selection": result_labels[key],
                        "pool": "HAD 胜平负",
                        "note": "本地模型概率 vs 中国体彩固定赔率" + (f"；{sentiment_note}" if sentiment_note else ""),
                        "is_model_pick": key == model_pick_key,
                        "model_pick": str(model.get("pick") or ""),
                    }
                    wdl_candidates.append(
                        _apply_market_signal_to_candidate(candidate, match)
                    )

        if match.get("hhad_available"):
            poisson_handicap_probs = _handicap_probs_from_poisson(
                model,
                hhad.get("handicap_line"),
            )
            for key, prob_key, odds_key, label in [
                ("home_win", "home_prob", "home_odds", "让胜"),
                ("draw", "draw_prob", "draw_odds", "让平"),
                ("away_win", "away_prob", "away_odds", "让负"),
            ]:
                probability_source = "泊松比分分布"
                probability = float(poisson_handicap_probs.get(prob_key) or 0)
                if probability <= 0:
                    probability_source = "体彩盘口隐含概率"
                    probability = float(hhad.get(prob_key) or 0)
                odds = float(hhad.get(odds_key) or 0)
                selection = f"{label}({hhad.get('handicap_line') or ''})"
                temp_candidate = {"selection": selection, "pool": "HHAD 让球胜平负"}
                probability, sentiment_note = _adjust_probability_by_sentiment(
                    probability,
                    temp_candidate,
                    match_sentiment,
                )
                edge = _selection_edge(probability, odds)
                if odds > 1:
                    candidate = {
                        **edge,
                        "match_id": match_id,
                        "match": match_name,
                        "selection": selection,
                        "pool": "HHAD 让球胜平负",
                        "note": f"让球概率来自{probability_source}，用于覆盖胜平负方向"
                        + (f"；{sentiment_note}" if sentiment_note else ""),
                        "is_model_pick": False,
                        "model_pick": str(model.get("pick") or ""),
                    }
                    wdl_candidates.append(
                        _apply_market_signal_to_candidate(candidate, match)
                    )

        score_candidates.extend(_exact_score_candidates(match))

    return {
        "rules_note": (
            "按中国体彩混合过关理解：每场可选多个玩法/结果，系统按每场选项数自动展开为多注。"
            "例如 4 场分别选 2、2、2、1 个选项，就是 8 注 4串1；默认按每注 2 元估算金额。"
            "覆盖概率和EV是模型估算，尤其多玩法同场相关性较强，只作为筛选信号，不等于确定收益。"
        ),
        "wdl": {
            "conservative": _build_betting_package(
                wdl_candidates,
                matches,
                "胜平负/让球保守包",
                min_probability=0.15,
                max_options_per_match=3,
                aggressive=False,
                max_total_cost=50,
            ),
            "aggressive": _build_betting_package(
                wdl_candidates,
                matches,
                "胜平负/让球激进包",
                min_probability=0.12,
                max_options_per_match=3,
                aggressive=True,
                max_total_cost=80,
            ),
        },
        "score": {
            "conservative": _build_betting_package(
                score_candidates,
                matches,
                "比分保守包",
                min_probability=0.06,
                max_options_per_match=2,
                aggressive=False,
            ),
            "aggressive": _build_betting_package(
                score_candidates,
                matches,
                "比分激进包",
                min_probability=0.02,
                max_options_per_match=3,
                aggressive=True,
            ),
        },
        "wdl_candidates_count": len(wdl_candidates),
        "score_candidates_count": len(score_candidates),
    }


def build_dashboard_payload(
    target_date: str,
    min_notional: float,
    trade_limit: int = 500,
    force_refresh: bool = False,
) -> dict:
    if force_refresh:
        _schedule_prediction_input_refresh(force=True)
    if force_refresh:
        _clear_runtime_caches()

    snapshot_error = ""
    cached_snapshot = _load_cached_sporttery_snapshot()
    if not cached_snapshot.empty:
        cached_snapshot = cached_snapshot.copy()
        cached_snapshot["snapshot_source"] = "cached_latest"
        snapshot = cached_snapshot
        started = _schedule_sporttery_snapshot_refresh()
        snapshot_error = "体彩盘口后台刷新中，本次先使用本地最新快照。" if started else "体彩盘口后台刷新中，本次先使用本地最新快照。"
    else:
        try:
            snapshot = fetch_sporttery_snapshot(["had", "hhad", "crs", "ttg", "hafu"])
            snapshot["snapshot_source"] = "live"
            append_sporttery_history(snapshot)
        except Exception as exc:
            snapshot_error = str(exc)
            snapshot = pd.DataFrame(
                columns=[
                    "match_id",
                    "match_date",
                    "match_time",
                    "home_team",
                    "away_team",
                    "pool_code",
                    "outcome",
                    "outcome_key",
                    "odds",
                    "handicap_line",
                    "update_time",
                ]
            )
    odds_history = load_sporttery_history()
    day = snapshot[snapshot["match_date"].astype(str) == target_date].copy()
    sporttery_match_keys = (
        day[["match_id", "match_time", "home_team", "away_team"]]
        .drop_duplicates()
        .sort_values("match_time")
    )
    match_records = [
        {
            "match_id": str(row.match_id),
            "match_time": str(row.match_time),
            "home_team": str(row.home_team),
            "away_team": str(row.away_team),
            "source": "sporttery",
        }
        for row in sporttery_match_keys.itertuples(index=False)
    ]
    existing_keys = {
        (_team_en(record["home_team"]).casefold(), _team_en(record["away_team"]).casefold())
        for record in match_records
    }
    match_records.extend(_fixture_rows_for_dashboard(target_date, existing_keys))

    recent_trades = []
    if ENABLE_POLYMARKET:
        try:
            recent_trades = fetch_recent_polymarket_trades(trade_limit)
        except requests.RequestException:
            recent_trades = []

    off_field_sentiment_payload = _get_off_field_sentiment_payload(
        target_date,
        force_refresh=force_refresh,
    )

    matches = []
    alert_count = 0
    for match in match_records:
        mid = str(match["match_id"])
        home_team = str(match["home_team"])
        away_team = str(match["away_team"])
        game = day[day["match_id"].astype(str) == mid]
        had = game[game["pool_code"] == "had"]
        hhad = game[game["pool_code"] == "hhad"]
        crs = game[game["pool_code"] == "crs"]
        probs = implied_probs_from_had(had)
        prob_source = "HAD"
        if not probs:
            probs = _implied_probs_from_crs(crs)
            prob_source = "CRS 比分盘反推" if probs else ""
        had_odds = _odds_map(had) if not had.empty else {}
        hhad_odds = _odds_map(hhad) if not hhad.empty else {}
        handicap_pick, handicap_probs = _handicap_pick(hhad)
        fixture_metadata = _lookup_fixture_metadata(home_team, away_team, target_date)
        group_motivation = _lookup_group_motivation(home_team, away_team, target_date)
        off_field_sentiment = _match_off_field_sentiment(
            off_field_sentiment_payload,
            home_team,
            away_team,
        )
        local_model = _predict_local_model(
            home_team,
            away_team,
            target_date,
            had_odds,
            probs,
            group_motivation,
        )

        alerts = []
        if ENABLE_POLYMARKET:
            alerts = _large_exact_score_buys_from_trades(
                recent_trades,
                home_team,
                away_team,
                target_date,
                min_notional,
            )
        alert_count += len(alerts)
        polymarket_exact_scores, polymarket_exact_source = [], ""
        if ENABLE_POLYMARKET:
            polymarket_exact_scores, polymarket_exact_source = _get_polymarket_exact_scores_stable(
                home_team,
                away_team,
                target_date,
            )

        pools = []
        for pool_code in POOL_ORDER:
            pool_frame = game[game["pool_code"] == pool_code].copy()
            if pool_frame.empty:
                continue
            pool_frame["odds_num"] = pd.to_numeric(pool_frame["odds"], errors="coerce")
            if pool_code in {"crs", "ttg", "hafu"}:
                pool_frame = pool_frame.sort_values(["odds_num", "outcome_key"], na_position="last")
            options = [
                {
                    "outcome": _display_outcome(str(row.outcome)),
                    "odds": float(row.odds_num) if pd.notna(row.odds_num) else None,
                    "handicap_line": str(row.handicap_line or ""),
                    "update_time": str(row.update_time or ""),
                }
                for row in pool_frame.itertuples()
            ]
            pools.append(
                {
                    "pool_code": pool_code,
                    "title": POOL_TITLES.get(pool_code, pool_code.upper()),
                    "options": options,
                }
            )

        match_payload = {
            "match_id": mid,
            "match_time": str(match["match_time"]),
            "home_team": home_team,
            "away_team": away_team,
            "source": str(match.get("source", "")),
            "market_pick": _match_pick(probs),
            "market_pick_source": prob_source,
            "handicap_pick": handicap_pick,
            "local_model": local_model,
            "fixture_metadata": fixture_metadata,
            "group_motivation": group_motivation,
            "off_field_sentiment": off_field_sentiment,
            "had_available": bool(probs),
            "had_real_available": not had.empty,
            "had": {
                "home_odds": had_odds.get("home_win"),
                "draw_odds": had_odds.get("draw"),
                "away_odds": had_odds.get("away_win"),
                "home_prob": probs.get("home_win"),
                "draw_prob": probs.get("draw"),
                "away_prob": probs.get("away_win"),
            },
            "hhad_available": not hhad.empty,
            "hhad": {
                "handicap_line": str(hhad.iloc[0]["handicap_line"]) if not hhad.empty else "",
                "home_odds": hhad_odds.get("home_win"),
                "draw_odds": hhad_odds.get("draw"),
                "away_odds": hhad_odds.get("away_win"),
                "home_prob": handicap_probs.get("home_win"),
                "draw_prob": handicap_probs.get("draw"),
                "away_prob": handicap_probs.get("away_win"),
            },
            "top_scores": [
                {"label": item.split("@")[0], "odds": float(item.split("@")[1])}
                for item in top_scores(crs, n=5)
                if "@" in item
            ],
            "polymarket_exact_scores": polymarket_exact_scores[:8],
            "polymarket_exact_source": polymarket_exact_source,
            "pools": pools,
            "polymarket_large_buys": [
                {
                    "notional": alert["notional"],
                    "price": alert["price"],
                    "size": alert["size"],
                    "title": _clean_alert_title(alert["title"]),
                    "outcome": alert["outcome"],
                    "trader": alert["trader"],
                }
                for alert in alerts[:8]
            ],
        }
        match_payload["market_decision"] = build_market_decision_rows(
            match_payload,
            game,
            odds_history,
        )
        matches.append(match_payload)

    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    betting_advice = _build_daily_betting_advice(matches)
    return {
        "updated_at": now.isoformat(),
        "updated_at_display": now.strftime("%Y-%m-%d %H:%M:%S"),
        "target_date": target_date,
        "snapshot_rows": int(len(snapshot)),
        "snapshot_error": snapshot_error,
        "cache_mode": "强制刷新" if force_refresh else "普通缓存",
        "off_field_sentiment_generated_at": off_field_sentiment_payload.get("generated_at", ""),
        "off_field_sentiment_refresh_error": off_field_sentiment_payload.get("refresh_error", ""),
        "polymarket_alert_count": int(alert_count),
        "betting_advice": betting_advice,
        "matches": matches,
    }


@app.get("/")
def index():
    target_date = request.args.get("date", DEFAULT_DATE)
    return render_template_string(PAGE_TEMPLATE, target_date=target_date)


@app.after_request
def add_no_store_headers(response):
    """Keep browser-side caching out of the development feedback loop."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.post("/api/clear-cache")
def api_clear_cache():
    _clear_runtime_caches()
    return jsonify({"ok": True, "cleared_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})


@app.get("/api/dashboard")
def api_dashboard():
    target_date = request.args.get("date", DEFAULT_DATE)
    min_notional = float(request.args.get("min_notional", "500"))
    trade_limit = int(request.args.get("trade_limit", "500"))
    force_refresh = request.args.get("force_refresh", "0") in {"1", "true", "yes"}
    try:
        return jsonify(build_dashboard_payload(target_date, min_notional, trade_limit, force_refresh))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local World Cup web dashboard.")
    parser.add_argument("--date", default=DEFAULT_DATE, help="Default date shown by page, YYYY-MM-DD.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug reloader while developing the dashboard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DEFAULT_DATE
    DEFAULT_DATE = args.date
    _schedule_prediction_input_refresh(force=False)
    _schedule_sporttery_snapshot_refresh()
    print(f"Open http://{args.host}:{args.port}?date={args.date}")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug)


if __name__ == "__main__":
    main()
