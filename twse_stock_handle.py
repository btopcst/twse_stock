import os
import shutil
import pandas as pd
import pandas_ta as ta
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import re
import zipfile
from datetime import datetime

# Set font to support Traditional Chinese
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

INPUT_FOLDER = "stockList"
OUTPUT_FOLDER = "stockData"
SUMMARY_FILE = "summary.xlsx"
BAD_FILE_LOG = "bad_files.txt"

PRICE_THD = 15
VOLUME_MIN_THD = 800
VOLUME_THD = 1000
KD_MAX_THD = 50
KD_MIN_THD = 5

CHART_DIRS = {
    '布林上軌突破+量增': "stockData/BL",
    'MACD 柱狀體首次轉正': "stockData/MACD",
    '均線糾結': "stockData/均線糾結",
    '月季均線首向上': "stockData/月季年均線首向上",
    '布林壓縮': "stockData/BLC",
    '回測季線': "stockData/backSeason",
    'KD 黃金交叉': "stockData/KD_GC"
}


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", str(name))


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
    df = df.sort_values(by='日期').tail(120)
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
            ax2.bar(df['日期'], df['MACD_HIST'], label='MACD Histogram')
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
            ax3.bar(df['日期'], df['MACD_HIST'], label='MACD Histogram')
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

    if condition == '回測季線':
        ax1.plot(df['日期'], df['收盤價'], label='Close Price', color='black')
        ax1.plot(df['日期'], df['5日平均價格'], label='MA5')
        ax1.plot(df['日期'], df['月平均價格'], label='MA20')
        ax1.plot(df['日期'], df['季平均價格'], label='MA60')
        ax1.plot(df['日期'], df['年平均價格'], label='MA240')
        ax1.set_title(f"{name} ({code}) - 回測季線")
    elif condition in ['布林上軌突破+量增', '布林壓縮'] and upper_col and lower_col:
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
    ax2.set_title("Last 120 Days Volume")
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


# Reset output folders
if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
for path in CHART_DIRS.values():
    os.makedirs(path, exist_ok=True)


def process_all_stocks():
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

        try:
            if all(pd.notna([latest.get('季平均價格'), price, volume])):
                if price < latest['季平均價格'] and latest['月均線'] == '向上' and latest['季均線'] == '向上' and float(volume) > VOLUME_THD:
                    summary_rows.append([stock_name, stock_id, date, price, '回測季線'])
                    plot_chart(df.copy(), stock_name, stock_id, '回測季線')
        except Exception:
            pass

        try:
            if latest.get('均線糾結') == '糾結' and float(volume) > VOLUME_MIN_THD:
                summary_rows.append([stock_name, stock_id, date, price, '均線糾結'])
                plot_chart(df.copy(), stock_name, stock_id, '均線糾結')
        except Exception:
            pass

        try:
            if find_first_two_up(df) == 1 and float(volume) > VOLUME_THD:
                summary_rows.append([stock_name, stock_id, date, price, '月季均線首向上'])
                plot_chart(df.copy(), stock_name, stock_id, '月季均線首向上')
        except Exception:
            pass

        try:
            if has_macd_hist(df) and len(df) >= 2:
                if df.iloc[-2]['MACD_HIST'] <= 0 and df.iloc[-1]['MACD_HIST'] > 0 and latest['月均線'] == '向上' and latest['季均線'] == '向上' and float(volume) > VOLUME_THD:
                    summary_rows.append([stock_name, stock_id, date, price, 'MACD 柱狀體首次轉正'])
                    plot_chart(df.copy(), stock_name, stock_id, 'MACD 柱狀體首次轉正')
        except Exception:
            pass

        try:
            if is_bollinger_constricted(df) and float(volume) > VOLUME_MIN_THD and float(price) > PRICE_THD:
                summary_rows.append([stock_name, stock_id, date, price, '布林壓縮'])
                plot_chart(df.copy(), stock_name, stock_id, '布林壓縮')
        except Exception:
            pass

        try:
            if is_first_kd_golden_cross(df) and float(volume) > VOLUME_MIN_THD:
                summary_rows.append([stock_name, stock_id, date, price, 'KD 黃金交叉'])
                plot_chart(df.copy(), stock_name, stock_id, 'KD 黃金交叉')
        except Exception:
            pass

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


if __name__ == "__main__":
    process_all_stocks()
    print("🎉 All stock processing and chart generation completed.")