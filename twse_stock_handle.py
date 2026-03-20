import os
import shutil
import pandas as pd
import pandas_ta as ta
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import re
import numpy as np

# Set font to support Traditional Chinese (for chart rendering)
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

INPUT_FOLDER = "stockList"
OUTPUT_FOLDER = "stockData"
SUMMARY_FILE = "summary.xlsx"

PRICE_THD = 15
VOLUME_MIN_THD = 800
VOLUME_THD = 1000
KD_MAX_THD = 50
KD_MIN_THD = 5

# Chart output directories
CHART_DIRS = {
    '布林上軌突破+量增': "stockData/BL",
    'MACD 柱狀體首次轉正': "stockData/MACD",
    '均線糾結': "stockData/均線糾結",
    '月季均線首向上': "stockData/月季年均線首向上" if False else "stockData/月季均線首向上",
    #'月季年均線首向上': "stockData/月季年均線首向上",
    '布林壓縮': "stockData/BLC",
    #'W底': "stockData/W_Style",
    '回測季線': "stockData/backSeason",
    #'低檔出量': "stockData/lowPoint",
    'KD 黃金交叉': "stockData/KD_GC"
}

# Sanitize filenames (remove invalid characters)
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

# Clean and recreate stockData and subfolders
if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
for path in CHART_DIRS.values():
    os.makedirs(path, exist_ok=True)

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
        except:
            result.append("")
    return result

def find_first_two_up(df):
    if len(df) < 2:
        return None
    if df.iloc[-1]['月均線'] == '向上' and df.iloc[-1]['季均線'] == '向上':
        if df.iloc[-2]['月均線'] != '向上' or df.iloc[-2]['季均線'] != '向上':
            return 1
    return None

def is_bollinger_constricted(df, days=15, threshold=0.02):
    recent = df.tail(days)
    if recent['BBU_20_2.0'].isnull().any() or recent['BBL_20_2.0'].isnull().any():
        return False
    diff = recent['BBU_20_2.0'] - recent['BBL_20_2.0']
    center = (recent['BBU_20_2.0'] + recent['BBL_20_2.0']) / 2
    ratio = diff / center
    return (ratio < threshold).all()

# -------------------------------------------------
# 只允許「第一次頂到/突破布林上軌」
# -------------------------------------------------
def is_first_bollinger_upper_break(df, price_col='收盤價', upper_col='BBU_20_2.0'):
    if df is None or len(df) < 2:
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

# -------------------------------------------------
# 台股版 KD (日線)
# -------------------------------------------------
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
        k_today = (2.0/3.0) * k_prev + (1.0/3.0) * rsv
        d_today = (2.0/3.0) * d_prev + (1.0/3.0) * k_today
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
    if prev_k <= prev_d and curr_k > curr_d and curr_k < KD_MAX_THD:
        return True
    return False


def plot_chart(df, name, code, condition):
    df = df.sort_values(by='日期')
    df = df.tail(240)

    # =========================================================
    # 1) MACD 柱狀體首次轉正：價格 / MACD / KD （三格）
    # =========================================================
    if condition == 'MACD 柱狀體首次轉正':
        plt.figure(figsize=(12, 8))
        gs = GridSpec(3, 1, height_ratios=[3, 1.2, 1.2])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)  # MACD
        ax3 = plt.subplot(gs[2], sharex=ax1)  # KD

        # 上圖：價格 + MA
        ax1.plot(df['日期'], df['收盤價'], label='Close Price', color='black')
        ax1.plot(df['日期'], df['5日平均價格'], label='MA5')
        ax1.plot(df['日期'], df['月平均價格'], label='MA20')
        ax1.plot(df['日期'], df['季平均價格'], label='MA60')
        ax1.plot(df['日期'], df['年平均價格'], label='MA240')
        ax1.set_title(f"{name} ({code}) - MACD Turned Positive")
        ax1.grid(True)
        ax1.legend()

        # 中圖：MACD Histogram
        ax2.bar(df['日期'], df['MACDh_12_26_9'], label='MACD Histogram')
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_ylabel("MACD")
        ax2.grid(True)
        ax2.legend(loc='upper right')

        # 下圖：KD
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

    # =========================================================
    # 2) KD 黃金交叉：價格 / KD(+右軸量) / MACD（三格）
    # =========================================================
    if condition in ['KD 黃金交叉']:
        plt.figure(figsize=(12, 8))
        gs = GridSpec(3, 1, height_ratios=[3, 1.3, 1.2])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)  # KD
        ax3 = plt.subplot(gs[2], sharex=ax1)  # MACD

        # 上圖：價格 + MA
        ax1.plot(df['日期'], df['收盤價'], label='Close Price', color='black')
        ax1.plot(df['日期'], df['5日平均價格'], label='MA5')
        ax1.plot(df['日期'], df['月平均價格'], label='MA20')
        ax1.plot(df['日期'], df['季平均價格'], label='MA60')
        ax1.plot(df['日期'], df['年平均價格'], label='MA240')
        ax1.set_title(f"{name} ({code}) - {condition}")
        ax1.grid(True)
        ax1.legend()

        # 中圖：KD（保留你原本 KD + 50線）
        ax2.plot(df['日期'], df['K值'], label='K')
        ax2.plot(df['日期'], df['D值'], label='D')
        ax2.axhline(50, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_ylabel("KD")
        ax2.set_title("KD")
        ax2.grid(True)

        # 右軸：Volume（保留你原本 twin axis）
        ax2b = ax2.twinx()
        ax2b.bar(df['日期'], df['成交量'], alpha=0.25)
        ax2b.set_ylabel("Volume (張)")

        # KD legend 放左上
        ax2.legend(loc='upper left')

        # 下圖：MACD Histogram
        ax3.bar(df['日期'], df['MACDh_12_26_9'], label='MACD Histogram')
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

    # =========================================================
    # 3) 其他條件：維持你原本 2 子圖（價格 / Volume）
    # =========================================================
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
    elif condition in ['布林上軌突破+量增', '布林壓縮']:
        ax1.plot(df['日期'], df['收盤價'], label='Price', color='black')
        ax1.plot(df['日期'], df['BBU_20_2.0'], label='Bollinger Upper', linestyle='--')
        ax1.plot(df['日期'], df['BBL_20_2.0'], label='Bollinger Lower', linestyle='--')
        ax1.set_title(f"{name} ({code}) - 布林上軌突破")
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


def process_all_stocks():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".xlsx")]
    summary_rows = []
    print(f"🟡 Found {len(files)} Excel files.")

    for filename in files:
        print(f"🔍 Processing file: {filename}")
        path = os.path.join(INPUT_FOLDER, filename)
        df = pd.read_excel(path).sort_values(by='日期', ascending=True).reset_index(drop=True)

        df['成交量'] = df['成交量'] / 1000  # 轉換為張數

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

        bb = ta.bbands(df['收盤價'], length=20)
        macd = ta.macd(df['收盤價'])
        df = pd.concat([df, bb, macd], axis=1)

        # 日 KD
        df = compute_kd_tw(df, length=9)

        df.to_excel(os.path.join(OUTPUT_FOLDER, filename), index=False)

        if df.empty:
            continue

        latest = df.iloc[-1]
        stock_name = latest.get('股票名稱')
        stock_id = latest.get('股票代號')
        date = latest.get('日期')
        price = latest.get('收盤價')
        volume = latest.get('成交量')

        try:
            if all(pd.notna([latest['BBU_20_2.0'], latest['20日均量'], price, volume])):
                if is_first_bollinger_upper_break(df) and float(price) > float(latest['BBU_20_2.0']) and float(volume) > VOLUME_THD:
                    summary_rows.append([stock_name, stock_id, date, price, '布林上軌突破+量增'])
                    plot_chart(df.copy(), stock_name, stock_id, '布林上軌突破+量增')
        except:
            pass

        try:
            if all(pd.notna([latest['季平均價格'], price, volume])):
                if abs(price - latest['季平均價格']) < 0 and latest['月均線'] == '向上' and latest['季均線'] == '向上' and volume > VOLUME_THD:
                    summary_rows.append([stock_name, stock_id, date, price, '回測季線'])
                    plot_chart(df.copy(), stock_name, stock_id, '回測季線')
        except Exception as e:
            print(f"回測季線錯誤：{filename} => {e}")

        try:
            df_tail20 = df.tail(20)
            if len(df_tail20) == 20 and df_tail20['收盤價'].min() == price and float(volume) > 5 * float(latest['20日均量']):
                summary_rows.append([stock_name, stock_id, date, price, '低檔出量'])
                plot_chart(df.copy(), stock_name, stock_id, '低檔出量')
        except Exception as e:
            print(f"低檔出量錯誤：{filename} => {e}")

        if latest.get('均線糾結') == '糾結' and float(volume) > VOLUME_MIN_THD:
            summary_rows.append([stock_name, stock_id, date, price, '均線糾結'])
            plot_chart(df.copy(), stock_name, stock_id, '均線糾結')

        if find_first_two_up(df) == 1 and float(volume) > VOLUME_THD:
            summary_rows.append([stock_name, stock_id, date, price, '月季均線首向上'])
            plot_chart(df.copy(), stock_name, stock_id, '月季均線首向上')

        if df.iloc[-2]['MACDh_12_26_9'] <= 0 and df.iloc[-1]['MACDh_12_26_9'] > 0 and latest['月均線'] == '向上' and latest['季均線'] == '向上' and volume > VOLUME_THD:
            summary_rows.append([stock_name, stock_id, date, price, 'MACD 柱狀體首次轉正'])
            plot_chart(df.copy(), stock_name, stock_id, 'MACD 柱狀體首次轉正')

        if is_bollinger_constricted(df) and float(volume) > VOLUME_MIN_THD and float(price) > PRICE_THD:
            summary_rows.append([stock_name, stock_id, date, price, '布林壓縮'])
            plot_chart(df.copy(), stock_name, stock_id, '布林壓縮')

        # 10. 日KD 黃金交叉
        try:
            if is_first_kd_golden_cross(df) and float(volume) > VOLUME_MIN_THD:
                summary_rows.append([stock_name, stock_id, date, price, 'KD 黃金交叉'])
                plot_chart(df.copy(), stock_name, stock_id, 'KD 黃金交叉')
        except Exception as e:
            print(f"KD 判斷錯誤：{filename} => {e}")

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


if __name__ == "__main__":
    process_all_stocks()
    print("🎉 All stock processing and chart generation completed.")