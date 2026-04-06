#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import io
import re
import zipfile
import time
import warnings
from datetime import datetime
from typing import List, Optional

import pandas as pd
import pandas_ta as ta
import requests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager

# =========================
# Font / matplotlib setup
# =========================
preferred_fonts = [
    "Microsoft JhengHei",
    "Noto Sans CJK TC",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
selected_font = next((f for f in preferred_fonts if f in available_fonts), "DejaVu Sans")

matplotlib.rcParams["font.family"] = selected_font
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph .* missing from font")
warnings.filterwarnings("ignore", message="findfont: Font family .* not found")

# =========================
# Screening settings
# =========================
INPUT_FOLDER = "stockList"
OUTPUT_FOLDER = "stockData"
SUMMARY_FILE = "summary.xlsx"
SUMMARY_WITH_3INST_FILE = "summary.xlsx"
BAD_FILE_LOG = "bad_files.txt"

PRICE_THD = 15
VOLUME_MIN_THD = 800
VOLUME_THD = 1000
KD_MAX_THD = 50

CHART_DIRS = {
    '布林上軌突破+量增': "stockData/BL",
    #'MACD 柱狀體首次轉正': "stockData/MACD",
    #'均線糾結': "stockData/均線糾結",
    #'月季均線首向上': "stockData/月季均線首向上",
    #'布林壓縮': "stockData/BLC",
    #'回測季線': "stockData/backSeason",
    #'KD 黃金交叉': "stockData/KD_GC"
}

# =========================
# Institutional settings
# =========================
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.twse.com.tw/",
}

OUT_COLS = [
    "外資淨買賣",
    "投信淨買賣",
    "自營淨買賣",
    "三大法人合計",
    "外資近10日",
    "投信近10日",
    "自營近10日",
    "市場",
]

LOOKBACK_TRADING_DAYS = 10
MAX_BACKTRACK_CALENDAR_DAYS = 25


def log(msg: str) -> None:
    print(msg, flush=True)


# =========================
# Common helpers
# =========================
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", str(name))


def normalize_code(v) -> str:
    s = str(v).strip()
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"[^\d]", "", s)
    if s.isdigit() and len(s) < 4:
        s = s.zfill(4)
    return s


def convert_roc_date(roc_str):
    y, m, d = str(roc_str).split('/')
    return f"{int(y)+1911}-{m.zfill(2)}-{d.zfill(2)}"


def parse_any_date(value):
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value

    if isinstance(value, datetime):
        return pd.Timestamp(value)

    s = str(value).strip()
    if not s:
        return pd.NaT

    s = s.replace('.', '/').replace('-', '/')
    parts = s.split('/')

    try:
        if len(parts) == 3:
            if len(parts[0]) == 3 and parts[0].isdigit():
                return pd.to_datetime(convert_roc_date(s), errors='coerce')
            if len(parts[0]) == 4 and parts[0].isdigit():
                yyyy, mm, dd = parts
                return pd.to_datetime(f"{yyyy}-{mm.zfill(2)}-{dd.zfill(2)}", errors='coerce')

        return pd.to_datetime(s, errors='coerce')
    except Exception:
        return pd.NaT


def to_num(v) -> int:
    if pd.isna(v):
        return 0
    s = str(v).strip()
    if s in ("", "-", "--", "nan", "None"):
        return 0
    s = s.replace(",", "").replace("\u3000", "").replace(" ", "")
    s = s.replace("(", "-").replace(")", "")
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else 0


def to_lot(v) -> int:
    return int(v) // 1000


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        cols = []
        for tup in x.columns:
            parts = [str(c).strip() for c in tup if str(c).strip() and "Unnamed" not in str(c)]
            cols.append("".join(parts))
        x.columns = cols
    else:
        x.columns = [str(c).strip() for c in x.columns]
    return x


# =========================
# Screening helpers
# =========================
def calc_trend(col):
    return col.ge(col.shift(1)).map({True: "向上", False: "向下"})


def calc_cross_zone(df):
    result = []
    for i in df.index:
        try:
            five = df.loc[i, '5日平均價格']
            month = df.loc[i, '月平均價格']
            season = df.loc[i, '季平均價格']
            year = df.loc[i, '年平均價格']
            base = sorted([five, month, season])[1]
            ok = all(abs(base - val) <= base * 0.01 for val in [five, month, season, year])
            result.append("糾結" if ok else "")
        except Exception:
            result.append("")
    return result


def find_first_two_up(df):
    if len(df) < 2:
        return None
    if df.iloc[-1]['月均線'] == '向上' and df.iloc[-1]['季均線'] == '向上':
        if df.iloc[-2]['月均線'] != '向上' or df.iloc[-2]['季均線'] != '向上':
            return 1
    return None


def get_bollinger_cols(df):
    upper_col = None
    lower_col = None
    for c in df.columns:
        c_str = str(c)
        if c_str.startswith("BBU_"):
            upper_col = c
        elif c_str.startswith("BBL_"):
            lower_col = c
    return upper_col, lower_col


def has_macd_hist(df):
    return "MACD_HIST" in df.columns


def compute_macd_safe(df, price_col='收盤價', fast=12, slow=26, signal=9):
    close = pd.to_numeric(df[price_col], errors='coerce').astype(float)
    close = close.replace([float("inf"), float("-inf")], pd.NA)

    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line

    df["MACD_LINE"] = macd_line
    df["MACD_SIGNAL"] = signal_line
    df["MACD_HIST"] = hist
    return df


def is_bollinger_constricted(df, days=15, threshold=0.02):
    upper_col, lower_col = get_bollinger_cols(df)
    if upper_col is None or lower_col is None:
        return False

    recent = df.tail(days)
    if recent.empty:
        return False

    if recent[upper_col].isnull().any() or recent[lower_col].isnull().any():
        return False

    diff = recent[upper_col] - recent[lower_col]
    center = (recent[upper_col] + recent[lower_col]) / 2
    ratio = diff / center
    return (ratio < threshold).all()


def is_first_bollinger_upper_break(df, price_col='收盤價'):
    upper_col, _ = get_bollinger_cols(df)
    if df is None or len(df) < 2 or upper_col is None:
        return False

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_p = prev.get(price_col)
    curr_p = curr.get(price_col)
    prev_u = prev.get(upper_col)
    curr_u = curr.get(upper_col)

    if pd.isna(prev_p) or pd.isna(curr_p) or pd.isna(prev_u) or pd.isna(curr_u):
        return False

    return float(curr_p) > float(curr_u) and float(prev_p) <= float(prev_u)


def compute_kd_tw(df, length=9):
    high = df['最高價'].astype(float)
    low = df['最低價'].astype(float)
    close = df['收盤價'].astype(float)

    rsv_list = []
    for i in range(len(df)):
        start = max(0, i - length + 1)
        hh = high[start:i+1].max()
        ll = low[start:i+1].min()
        if hh == ll:
            rsv = 0.0
        else:
            rsv = (close.iloc[i] - ll) / (hh - ll) * 100
        rsv_list.append(rsv)

    K = []
    D = []
    k_prev = 50.0
    d_prev = 50.0
    for rsv in rsv_list:
        k_today = (2.0 / 3.0) * k_prev + (1.0 / 3.0) * rsv
        d_today = (2.0 / 3.0) * d_prev + (1.0 / 3.0) * k_today
        K.append(k_today)
        D.append(d_today)
        k_prev = k_today
        d_prev = d_today

    df['K值'] = K
    df['D值'] = D
    return df


def is_first_kd_golden_cross(df, k_col='K值', d_col='D值'):
    if len(df) < 2:
        return False

    prev_k = df.iloc[-2].get(k_col)
    prev_d = df.iloc[-2].get(d_col)
    curr_k = df.iloc[-1].get(k_col)
    curr_d = df.iloc[-1].get(d_col)

    if pd.isna(prev_k) or pd.isna(prev_d) or pd.isna(curr_k) or pd.isna(curr_d):
        return False

    return prev_k <= prev_d and curr_k > curr_d and curr_k < KD_MAX_THD


def plot_chart(df, name, code, condition):
    df = df.sort_values(by='日期').tail(240)
    upper_col, lower_col = get_bollinger_cols(df)

    if condition == 'MACD 柱狀體首次轉正':
        plt.figure(figsize=(12, 8))
        gs = GridSpec(3, 1, height_ratios=[3, 1.2, 1.2])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)

        ax1.plot(df['日期'], df['收盤價'], label='Close Price', color='black')
        ax1.plot(df['日期'], df['5日平均價格'], label='MA5')
        ax1.plot(df['日期'], df['月平均價格'], label='MA20')
        ax1.plot(df['日期'], df['季平均價格'], label='MA60')
        ax1.plot(df['日期'], df['年平均價格'], label='MA240')
        ax1.set_title(f"{name} ({code}) - MACD Turned Positive")
        ax1.grid(True)
        ax1.legend()

        if has_macd_hist(df):
            ax2.bar(df['日期'], df['MACD_HIST'], label='MACD')
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_ylabel("MACD")
        ax2.grid(True)
        ax2.legend(loc='upper right')

        ax3.plot(df['日期'], df['K值'], label='K')
        ax3.plot(df['日期'], df['D值'], label='D')
        ax3.axhline(50, color='gray', linestyle='--', linewidth=0.8)
        ax3.set_ylabel("KD")
        ax3.set_title("KD")
        ax3.grid(True)
        ax3.legend(loc='upper right')

        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.tight_layout()

        safe_name = sanitize_filename(name)
        output_path = os.path.join(CHART_DIRS[condition], f"{safe_name}_{code}.png")
        plt.savefig(output_path)
        plt.close()
        return

    if condition == 'KD 黃金交叉':
        plt.figure(figsize=(12, 8))
        gs = GridSpec(3, 1, height_ratios=[3, 1.3, 1.2])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)

        ax1.plot(df['日期'], df['收盤價'], label='Close Price', color='black')
        ax1.plot(df['日期'], df['5日平均價格'], label='MA5')
        ax1.plot(df['日期'], df['月平均價格'], label='MA20')
        ax1.plot(df['日期'], df['季平均價格'], label='MA60')
        ax1.plot(df['日期'], df['年平均價格'], label='MA240')
        ax1.set_title(f"{name} ({code}) - KD Golden Cross")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(df['日期'], df['K值'], label='K')
        ax2.plot(df['日期'], df['D值'], label='D')
        ax2.axhline(50, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_ylabel("KD")
        ax2.set_title("KD")
        ax2.grid(True)

        ax2b = ax2.twinx()
        ax2b.bar(df['日期'], df['成交量'], alpha=0.25)
        ax2b.set_ylabel("Volume (Lots)")
        ax2.legend(loc='upper left')

        if has_macd_hist(df):
            ax3.bar(df['日期'], df['MACD_HIST'], label='MACD')
        ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax3.set_ylabel("MACD")
        ax3.set_title("MACD")
        ax3.grid(True)
        ax3.legend(loc='upper right')

        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.tight_layout()

        safe_name = sanitize_filename(name)
        output_path = os.path.join(CHART_DIRS[condition], f"{safe_name}_{code}.png")
        plt.savefig(output_path)
        plt.close()
        return

    if condition == '回測季線':
        plt.figure(figsize=(12, 8))
        gs = GridSpec(3, 1, height_ratios=[3, 1.3, 1.2])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)

        ax1.plot(df['日期'], df['收盤價'], label='Close Price', color='black')
        #ax1.plot(df['日期'], df['5日平均價格'], label='MA5')
        ax1.plot(df['日期'], df['月平均價格'], label='MA20')
        ax1.plot(df['日期'], df['季平均價格'], label='MA60')
        ax1.plot(df['日期'], df['年平均價格'], label='MA240')
        ax1.set_title(f"{name} ({code}) - 回測季線")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(df['日期'], df['K值'], label='K')
        ax2.plot(df['日期'], df['D值'], label='D')
        ax2.axhline(50, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_ylabel("KD")
        ax2.set_title("KD")
        ax2.grid(True)

        ax2b = ax2.twinx()
        ax2b.bar(df['日期'], df['成交量'], alpha=0.25)
        ax2b.set_ylabel("Volume (Lots)")
        ax2.legend(loc='upper left')

        if has_macd_hist(df):
            ax3.bar(df['日期'], df['MACD_HIST'], label='MACD')
        ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax3.set_ylabel("MACD")
        ax3.set_title("MACD")
        ax3.grid(True)
        ax3.legend(loc='upper right')

        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.tight_layout()

        safe_name = sanitize_filename(name)
        output_path = os.path.join(CHART_DIRS[condition], f"{safe_name}_{code}.png")
        plt.savefig(output_path)
        plt.close()
        return

    plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    #if condition == '回測季線':
     #   ax1.plot(df['日期'], df['收盤價'], label='Close Price', color='black')
     #   ax1.plot(df['日期'], df['5日平均價格'], label='MA5')
     #   ax1.plot(df['日期'], df['月平均價格'], label='MA20')
     #   ax1.plot(df['日期'], df['季平均價格'], label='MA60')
     #   ax1.plot(df['日期'], df['年平均價格'], label='MA240')
     #   ax1.set_title(f"{name} ({code}) - 回測季線")
    if condition in ['布林上軌突破+量增', '布林壓縮'] and upper_col and lower_col:
        ax1.plot(df['日期'], df['收盤價'], label='Price', color='black')
        ax1.plot(df['日期'], df[upper_col], label='Bollinger Upper', linestyle='--')
        ax1.plot(df['日期'], df[lower_col], label='Bollinger Lower', linestyle='--')
        ax1.set_title(f"{name} ({code}) - Bollinger Pattern")
    else:
        ax1.plot(df['日期'], df['收盤價'], label='Close Price', color='black')
        ax1.plot(df['日期'], df['5日平均價格'], label='MA5')
        ax1.plot(df['日期'], df['月平均價格'], label='MA20')
        ax1.plot(df['日期'], df['季平均價格'], label='MA60')
        ax1.plot(df['日期'], df['年平均價格'], label='MA240')
        ax1.set_title(f"{name} ({code}) - Moving Averages")

    ax1.grid(True)
    ax1.legend()

    ax2.bar(df['日期'], df['成交量'], label='Volume', color='gray')
    ax2.set_title("Last 240 Days Volume")
    ax2.grid(True)

    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.tight_layout()

    safe_name = sanitize_filename(name)
    output_path = os.path.join(CHART_DIRS[condition], f"{safe_name}_{code}.png")
    plt.savefig(output_path)
    plt.close()


def is_valid_xlsx(path):
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False


def prepare_output_dirs():
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for path in CHART_DIRS.values():
        os.makedirs(path, exist_ok=True)


def process_all_stocks() -> bool:
    prepare_output_dirs()

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".xlsx")]
    summary_rows = []
    bad_files = []

    print(f"🟡 Found {len(files)} Excel files.")

    for filename in files:
        print(f"🔍 Processing file: {filename}")
        path = os.path.join(INPUT_FOLDER, filename)

        if not is_valid_xlsx(path):
            bad_files.append(f"{filename} => invalid xlsx zip structure")
            continue

        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            bad_files.append(f"{filename} => read_excel failed: {e}")
            continue

        required_cols = {'股票名稱', '股票代號', '日期', '最高價', '最低價', '收盤價', '成交量'}
        if not required_cols.issubset(df.columns):
            bad_files.append(f"{filename} => missing required columns")
            continue

        df['日期'] = df['日期'].apply(parse_any_date)

        for col in ['最高價', '最低價', '收盤價', '成交量']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['日期', '最高價', '最低價', '收盤價', '成交量'])
        df = df.sort_values(by='日期', ascending=True).reset_index(drop=True)

        if df.empty or len(df) < 20:
            bad_files.append(f"{filename} => insufficient valid rows")
            continue

        df['成交量'] = df['成交量'] / 1000

        df['5日平均價格'] = df['收盤價'].rolling(window=5).mean()
        df['5日均線'] = calc_trend(df['5日平均價格'])
        df['月平均價格'] = df['收盤價'].rolling(window=20).mean()
        df['月均線'] = calc_trend(df['月平均價格'])
        df['季平均價格'] = df['收盤價'].rolling(window=60).mean()
        df['季均線'] = calc_trend(df['季平均價格'])
        df['年平均價格'] = df['收盤價'].rolling(window=240).mean()
        df['年均線'] = calc_trend(df['年平均價格'])
        df['均線糾結'] = calc_cross_zone(df)
        df['20日均量'] = df['成交量'].rolling(window=5).mean()

        bb = ta.bbands(df['收盤價'], length=20, std=2.0)
        if bb is not None and not bb.empty:
            df = pd.concat([df, bb], axis=1)

        df = compute_macd_safe(df, price_col='收盤價', fast=12, slow=26, signal=9)
        df = compute_kd_tw(df, length=9)

        df.to_excel(os.path.join(OUTPUT_FOLDER, filename), index=False)

        latest = df.iloc[-1]
        stock_name = latest.get('股票名稱')
        stock_id = latest.get('股票代號')
        date = latest.get('日期')
        price = latest.get('收盤價')
        volume = latest.get('成交量')

        upper_col, _ = get_bollinger_cols(df)

        try:
            if upper_col and all(pd.notna([latest.get(upper_col), latest.get('20日均量'), price, volume])):
                if is_first_bollinger_upper_break(df) and float(price) > float(latest[upper_col]) and float(volume) > VOLUME_THD:
                    summary_rows.append([stock_name, stock_id, date, price, '布林上軌突破+量增'])
                    plot_chart(df.copy(), stock_name, stock_id, '布林上軌突破+量增')
        except Exception:
            pass

        #try:
            #if all(pd.notna([latest.get('季平均價格'), price, volume])):
                #if price < latest['季平均價格'] and latest['月均線'] == '向上' and latest['季均線'] == '向上' and float(volume) > VOLUME_THD:
                    #summary_rows.append([stock_name, stock_id, date, price, '回測季線'])
                    #plot_chart(df.copy(), stock_name, stock_id, '回測季線')
        #except Exception:
            #pass

        #try:
            #if latest.get('均線糾結') == '糾結' and float(volume) > VOLUME_MIN_THD:
                #summary_rows.append([stock_name, stock_id, date, price, '均線糾結'])
                #plot_chart(df.copy(), stock_name, stock_id, '均線糾結')
        #except Exception:
            #pass

        #try:
            #if find_first_two_up(df) == 1 and float(volume) > VOLUME_THD:
                #summary_rows.append([stock_name, stock_id, date, price, '月季均線首向上'])
                #plot_chart(df.copy(), stock_name, stock_id, '月季均線首向上')
        #except Exception:
            #pass

        #try:
            #if has_macd_hist(df) and len(df) >= 2:
                #if df.iloc[-2]['MACD_HIST'] <= 0 and df.iloc[-1]['MACD_HIST'] > 0 and latest['月均線'] == '向上' and latest['季均線'] == '向上' and float(volume) > VOLUME_THD:
                    #summary_rows.append([stock_name, stock_id, date, price, 'MACD 柱狀體首次轉正'])
                    #plot_chart(df.copy(), stock_name, stock_id, 'MACD 柱狀體首次轉正')
        #except Exception:
            #pass

        #try:
            #if is_bollinger_constricted(df) and float(volume) > VOLUME_MIN_THD and float(price) > PRICE_THD:
                #summary_rows.append([stock_name, stock_id, date, price, '布林壓縮'])
                #plot_chart(df.copy(), stock_name, stock_id, '布林壓縮')
        #except Exception:
            #pass

        #try:
            #if is_first_kd_golden_cross(df) and float(volume) > VOLUME_MIN_THD:
                #summary_rows.append([stock_name, stock_id, date, price, 'KD 黃金交叉'])
                #plot_chart(df.copy(), stock_name, stock_id, 'KD 黃金交叉')
        #except Exception:
            #pass

    if summary_rows:
        df_summary = pd.DataFrame(
            summary_rows,
            columns=['股票名稱', '股票代號', '日期', '收盤價', '篩選條件']
        )

        df_summary = (
            df_summary
            .sort_values(['股票代號', '日期'])
            .groupby(['股票名稱', '股票代號', '日期'], as_index=False)
            .agg({
                '收盤價': 'last',
                '篩選條件': lambda s: '、'.join(pd.unique(s.dropna()))
            })
        )

        df_summary.to_excel(SUMMARY_FILE, index=False)
        print(f"📊 Screening result saved to: {SUMMARY_FILE}")
    else:
        print("📭 No stocks matched the screening conditions.")

    if bad_files:
        with open(BAD_FILE_LOG, "w", encoding="utf-8") as f:
            for line in bad_files:
                f.write(line + "\n")
        print(f"⚠️ Bad file log saved to: {BAD_FILE_LOG}")

    return os.path.exists(SUMMARY_FILE)


# =========================
# Institutional fetch helpers
# =========================
def pick_twse_table(tables: List[pd.DataFrame]) -> pd.DataFrame:
    best_tbl = None
    best_score = -1
    best_idx = -1

    for idx, tbl in enumerate(tables):
        t = flatten_columns(tbl)
        cols = list(t.columns)
        score = 0

        if any("證券代號" in c or c == "代號" for c in cols):
            score += 2
        if any("外資及陸資" in c or "外陸資" in c for c in cols):
            score += 2
        if any("投信" in c for c in cols):
            score += 2
        if any("自營商(自行買賣)" in c or "自營商自行買賣" in c for c in cols):
            score += 2
        if any("自營商(避險)" in c or "自營商避險" in c for c in cols):
            score += 2
        if any("三大法人" in c for c in cols):
            score += 2

        if score > best_score:
            best_tbl = tbl
            best_score = score
            best_idx = idx

    if best_tbl is None or best_score < 4:
        raise ValueError("TWSE candidate table not found")

    return best_tbl


def parse_twse(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    x = flatten_columns(df)
    cols = list(x.columns)

    def find_col_endswith(cols: List[str], suffixes: List[str]) -> Optional[str]:
        for c in cols:
            for s in suffixes:
                if str(c).strip().endswith(s):
                    return c
        return None

    code_col = find_col_endswith(cols, ["證券代號", "代號"])
    foreign_col = find_col_endswith(
        cols,
        [
            "外陸資買賣超股數(不含外資自營商)",
            "外資及陸資買賣超股數(不含外資自營商)",
        ],
    )
    trust_col = find_col_endswith(cols, ["投信買賣超股數"])

    dealer_total_col = None
    for c in cols:
        c_str = str(c).strip()
        if c_str.endswith("自營商買賣超股數") and not c_str.endswith("外資自營商買賣超股數"):
            dealer_total_col = c
            break

    dealer_self_col = find_col_endswith(cols, ["自營商買賣超股數(自行買賣)"])
    dealer_hedge_col = find_col_endswith(cols, ["自營商買賣超股數(避險)"])
    total_col = find_col_endswith(cols, ["三大法人買賣超股數"])

    if not code_col:
        raise ValueError(f"TWSE code column not found: {cols}")

    x = x[x[code_col].astype(str).str.contains(r"\d", na=False)].copy()
    x["股票代號"] = x[code_col].map(normalize_code)

    foreign_val = x[foreign_col].map(to_num) if foreign_col else 0
    trust_val = x[trust_col].map(to_num) if trust_col else 0

    if dealer_self_col or dealer_hedge_col:
        dealer_self = x[dealer_self_col].map(to_num) if dealer_self_col else 0
        dealer_hedge = x[dealer_hedge_col].map(to_num) if dealer_hedge_col else 0
        dealer_val = dealer_self + dealer_hedge
    elif dealer_total_col:
        dealer_val = x[dealer_total_col].map(to_num)
    else:
        dealer_val = 0

    if total_col:
        total_val = x[total_col].map(to_num)
    else:
        total_val = foreign_val + trust_val + dealer_val

    x["外資淨買賣"] = foreign_val.map(to_lot) if hasattr(foreign_val, "map") else 0
    x["投信淨買賣"] = trust_val.map(to_lot) if hasattr(trust_val, "map") else 0
    x["自營淨買賣"] = dealer_val.map(to_lot) if hasattr(dealer_val, "map") else 0
    x["三大法人合計"] = total_val.map(to_lot) if hasattr(total_val, "map") else 0

    x["日期"] = pd.to_datetime(target_date).strftime("%Y-%m-%d")
    x["市場"] = "上市"

    return (
        x[
            ["股票代號", "日期", "外資淨買賣", "投信淨買賣", "自營淨買賣", "三大法人合計", "市場"]
        ]
        .drop_duplicates(["股票代號", "日期"])
        .reset_index(drop=True)
    )


def fetch_twse(date_obj: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    ymd = pd.to_datetime(date_obj).strftime("%Y%m%d")
    url = f"https://www.twse.com.tw/fund/T86?response=html&date={ymd}&selectType=ALLBUT0999"
    r = session.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    if not tables:
        raise ValueError(f"TWSE no table for {pd.to_datetime(date_obj).strftime('%Y-%m-%d')}")
    return parse_twse(pick_twse_table(tables), pd.to_datetime(date_obj))


def roc_date_str(date_obj: pd.Timestamp) -> str:
    y = date_obj.year - 1911
    return f"{y}/{date_obj.month:02d}/{date_obj.day:02d}"


def fetch_tpex(date_obj: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    target_date = pd.to_datetime(date_obj).normalize()
    roc_str = roc_date_str(target_date)

    url = (
        "https://www.tpex.org.tw/web/stock/3insti/daily_trade/"
        f"3itrade_hedge_result.php?l=zh-tw&o=htm&d={roc_str}&s=0,asc,0&_={int(time.time())}"
    )

    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    if "代號" not in r.text or "名稱" not in r.text:
        raise ValueError("TPEx response does not contain expected table header")

    tables = pd.read_html(io.StringIO(r.text))
    if not tables:
        raise ValueError(f"TPEx no table for {target_date.strftime('%Y-%m-%d')}")

    best_tbl = None
    best_score = -1

    for tbl in tables:
        t = flatten_columns(tbl)
        cols = list(t.columns)

        score = 0
        if any("代號" in c or "證券代號" in c for c in cols):
            score += 2
        if any("外資及陸資" in c for c in cols):
            score += 2
        if any("投信" in c for c in cols):
            score += 2
        if any("自營商" in c for c in cols):
            score += 2
        if any("三大法人" in c for c in cols):
            score += 2

        if score > best_score:
            best_tbl = tbl
            best_score = score

    if best_tbl is None or best_score < 4:
        raise ValueError("TPEx candidate table not found")

    x = flatten_columns(best_tbl)
    cols = list(x.columns)

    def find_col_endswith(cols: List[str], suffixes: List[str]) -> Optional[str]:
        for c in cols:
            for s in suffixes:
                if str(c).strip().endswith(s):
                    return c
        return None

    code_col = find_col_endswith(cols, ["代號", "證券代號"])
    foreign_col = find_col_endswith(
        cols,
        [
            "外資及陸資買賣超股數",
            "外資及陸資淨買股數",
            "外資及陸資(不含外資自營商)買賣超股數",
            "外資及陸資(不含外資自營商)淨買股數",
        ],
    )
    trust_col = find_col_endswith(cols, ["投信買賣超股數", "投信淨買股數"])

    dealer_total_col = None
    for c in cols:
        c_str = str(c).strip()
        if (
            c_str.endswith("自營商買賣超股數")
            or c_str.endswith("自營商淨買股數")
        ) and ("外資" not in c_str):
            dealer_total_col = c
            break

    dealer_self_col = find_col_endswith(cols, ["自營商(自行買賣)買賣超股數", "自營商(自行買賣)淨買股數"])
    dealer_hedge_col = find_col_endswith(cols, ["自營商(避險)買賣超股數", "自營商(避險)淨買股數"])
    total_col = find_col_endswith(cols, ["三大法人買賣超股數合計", "三大法人買賣超股數", "三大法人淨買股數"])

    if not code_col:
        raise ValueError(f"TPEx code column not found: {cols}")

    x = x[x[code_col].astype(str).str.contains(r"\d", na=False)].copy()
    x["股票代號"] = x[code_col].map(normalize_code)

    foreign_val = x[foreign_col].map(to_num) if foreign_col else 0
    trust_val = x[trust_col].map(to_num) if trust_col else 0

    if dealer_total_col:
        dealer_val = x[dealer_total_col].map(to_num)
    elif dealer_self_col or dealer_hedge_col:
        dealer_self = x[dealer_self_col].map(to_num) if dealer_self_col else 0
        dealer_hedge = x[dealer_hedge_col].map(to_num) if dealer_hedge_col else 0
        dealer_val = dealer_self + dealer_hedge
    else:
        dealer_val = 0

    if total_col:
        total_val = x[total_col].map(to_num)
    else:
        total_val = foreign_val + trust_val + dealer_val

    x["外資淨買賣"] = foreign_val.map(to_lot) if hasattr(foreign_val, "map") else 0
    x["投信淨買賣"] = trust_val.map(to_lot) if hasattr(trust_val, "map") else 0
    x["自營淨買賣"] = dealer_val.map(to_lot) if hasattr(dealer_val, "map") else 0
    x["三大法人合計"] = total_val.map(to_lot) if hasattr(total_val, "map") else 0

    x["日期"] = pd.to_datetime(target_date).strftime("%Y-%m-%d")
    x["市場"] = "上櫃"

    return (
        x[
            ["股票代號", "日期", "外資淨買賣", "投信淨買賣", "自營淨買賣", "三大法人合計", "市場"]
        ]
        .drop_duplicates(["股票代號", "日期"])
        .reset_index(drop=True)
    )


def read_summary_for_inst(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={"股票代號": str})
    df.columns = [str(c).strip() for c in df.columns]

    if "股票代號" not in df.columns or "日期" not in df.columns:
        raise ValueError("summary.xlsx must contain '股票代號' and '日期' columns")

    df["股票代號"] = df["股票代號"].map(normalize_code)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.strftime("%Y-%m-%d")

    if df["日期"].isna().any():
        bad_rows = df[df["日期"].isna()]
        raise ValueError(f"summary.xlsx contains invalid 日期 values, rows={len(bad_rows)}")

    return df


def build_inst_data(summary_df: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(summary_df["日期"].dropna().unique()))
    log("[INFO] Unique dates in summary: " + ", ".join(pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates))

    session = requests.Session()
    frames = []
    base_cols = ["股票代號", "日期", "外資淨買賣", "投信淨買賣", "自營淨買賣", "三大法人合計", "市場"]

    for d in dates:
        d = pd.to_datetime(d)
        log(f"[INFO] Fetch institutional data for summary date: {d.strftime('%Y-%m-%d')}")

        try:
            twse_df = fetch_twse(d, session)
            log(f"[INFO] TWSE {d.strftime('%Y-%m-%d')}: {len(twse_df)} rows")
        except Exception as e:
            log(f"[WARN] TWSE {d.strftime('%Y-%m-%d')} failed: {e}")
            twse_df = pd.DataFrame(columns=base_cols)

        try:
            tpex_df = fetch_tpex(d, session)
            log(f"[INFO] TPEx {d.strftime('%Y-%m-%d')}: {len(tpex_df)} rows")
        except Exception as e:
            log(f"[WARN] TPEx {d.strftime('%Y-%m-%d')} failed: {e}")
            tpex_df = pd.DataFrame(columns=base_cols)

        non_empty_parts = []
        if not twse_df.empty:
            non_empty_parts.append(twse_df)
        if not tpex_df.empty:
            non_empty_parts.append(tpex_df)

        if non_empty_parts:
            frames.append(pd.concat(non_empty_parts, ignore_index=True))

    if not frames:
        return pd.DataFrame(columns=base_cols)

    return pd.concat(frames, ignore_index=True).drop_duplicates(["股票代號", "日期"], keep="first")


def build_recent_10d_sum(summary_df: pd.DataFrame) -> pd.DataFrame:
    unique_dates = sorted(pd.to_datetime(summary_df["日期"].dropna().unique()))
    session = requests.Session()
    records = []

    for ref_date in unique_dates:
        found_days = 0
        checked_days = 0
        cursor = pd.to_datetime(ref_date).normalize()

        tmp_list = []

        while found_days < LOOKBACK_TRADING_DAYS and checked_days < MAX_BACKTRACK_CALENDAR_DAYS:
            day_frames = []

            try:
                twse_df = fetch_twse(cursor, session)
                if not twse_df.empty:
                    day_frames.append(twse_df)
            except Exception:
                pass

            try:
                tpex_df = fetch_tpex(cursor, session)
                if not tpex_df.empty:
                    day_frames.append(tpex_df)
            except Exception:
                pass

            if day_frames:
                day_df = pd.concat(day_frames, ignore_index=True)
                tmp_list.append(day_df)
                found_days += 1

            cursor = cursor - pd.Timedelta(days=1)
            checked_days += 1

        if tmp_list:
            df_all = pd.concat(tmp_list, ignore_index=True)
            df_all["基準日期"] = pd.to_datetime(ref_date).strftime("%Y-%m-%d")

            agg = (
                df_all.groupby(["股票代號", "基準日期"], as_index=False)[
                    ["外資淨買賣", "投信淨買賣", "自營淨買賣"]
                ]
                .sum()
            )
            records.append(agg)

    if not records:
        return pd.DataFrame(columns=["股票代號", "日期", "外資近10日", "投信近10日", "自營近10日"])

    result = pd.concat(records, ignore_index=True)
    result = result.rename(columns={
        "基準日期": "日期",
        "外資淨買賣": "外資近10日",
        "投信淨買賣": "投信近10日",
        "自營淨買賣": "自營近10日",
    })

    for col in ["外資近10日", "投信近10日", "自營近10日"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype("int64")

    return result


def reorder_columns(merged: pd.DataFrame) -> pd.DataFrame:
    ordered_cols = list(merged.columns)
    for c in OUT_COLS:
        if c in ordered_cols:
            ordered_cols.remove(c)

    insert_after = "篩選條件" if "篩選條件" in ordered_cols else ordered_cols[-1]
    idx = ordered_cols.index(insert_after) + 1
    ordered_cols[idx:idx] = [c for c in OUT_COLS if c in merged.columns]

    return merged[ordered_cols]


def write_inst_excel(df: pd.DataFrame, output_path: str) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
        ws = writer.sheets["Sheet1"]

        for idx, col_name in enumerate(df.columns, start=1):
            letter = ws.cell(row=1, column=idx).column_letter
            max_len = max(len(str(col_name)), 12)
            sample_series = df[col_name].astype(str).head(50)
            if not sample_series.empty:
                max_len = max(max_len, int(sample_series.map(len).max()))
            ws.column_dimensions[letter].width = min(max_len + 2, 18)


def enrich_summary_with_institutional(summary_path: str = SUMMARY_FILE,
                                      output_path: str = SUMMARY_FILE) -> bool:
    if not os.path.exists(summary_path):
        log(f"[WARN] Summary file not found: {summary_path}")
        return False

    log(f"[INFO] Read input file: {summary_path}")
    summary_df = read_summary_for_inst(summary_path)
    log(f"[INFO] Input rows: {len(summary_df)}")

    inst_df = build_inst_data(summary_df)
    hist10_df = build_recent_10d_sum(summary_df)

    merged = summary_df.merge(inst_df, on=["股票代號", "日期"], how="left")
    merged = merged.merge(hist10_df, on=["股票代號", "日期"], how="left")

    for col in ["外資淨買賣", "投信淨買賣", "自營淨買賣", "三大法人合計"]:
        merged[col] = pd.to_numeric(merged.get(col), errors="coerce").fillna(0).astype("int64")

    for col in ["外資近10日", "投信近10日", "自營近10日"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype("int64")
        else:
            merged[col] = 0

    merged["市場"] = merged.get("市場")
    merged["市場"] = merged["市場"].fillna("未知")
    merged["日期"] = pd.to_datetime(merged["日期"]).dt.strftime("%Y-%m-%d")

    merged = reorder_columns(merged)
    write_inst_excel(merged, output_path)

    matched = merged["市場"].ne("未知").sum()
    unmatched = len(merged) - matched

    log(f"[INFO] Saved: {output_path}")
    log(f"[INFO] Matched rows: {matched}")
    log(f"[INFO] Unmatched rows: {unmatched}")
    return True


if __name__ == "__main__":
    has_summary = process_all_stocks()
    if has_summary:
        enrich_summary_with_institutional(SUMMARY_FILE, SUMMARY_WITH_3INST_FILE)
    print("🎉 All stock processing, chart generation, and institutional merge completed.")
