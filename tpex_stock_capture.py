# tpex_stock_capture.py（修正非法檔名 + 多執行緒 + 進度條 + 支援上櫃）
import os
import re
import time
import threading
import requests
import pandas as pd
from datetime import datetime
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MAX_WORKERS = 8
SAVE_FOLDER = "stockList"
INDEX_FOLDER = "stockTse"

# TPEx 容易被 403，限制同時併發與最小請求間隔
TPEX_MAX_CONCURRENT = 4
TPEX_MIN_INTERVAL_SEC = 0.2

session = requests.Session()
tpex_session = requests.Session()

tpex_semaphore = threading.Semaphore(TPEX_MAX_CONCURRENT)
tpex_rate_lock = threading.Lock()
tpex_last_request_ts = 0.0


# --------------------
# 將民國年轉西元年
# --------------------
def convert_roc_date(roc_str):
    y, m, d = roc_str.split('/')
    return f"{int(y)+1911}-{m.zfill(2)}-{d.zfill(2)}"


# --------------------
# 支援民國/西元日期字串轉 datetime
# --------------------
def parse_any_date(date_str):
    if pd.isna(date_str):
        return pd.NaT

    s = str(date_str).strip().replace('-', '/')
    parts = s.split('/')

    try:
        if len(parts) == 3 and len(parts[0]) == 3:
            s = convert_roc_date(s)
        elif len(parts) == 3 and len(parts[0]) == 4:
            s = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
        return pd.to_datetime(s, errors='coerce')
    except:
        return pd.NaT


# --------------------
# 清理檔名非法字元
# --------------------
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)


# --------------------
# TPEx 節流
# --------------------
def tpex_throttle():
    global tpex_last_request_ts
    with tpex_rate_lock:
        now = time.time()
        wait_sec = TPEX_MIN_INTERVAL_SEC - (now - tpex_last_request_ts)
        if wait_sec > 0:
            time.sleep(wait_sec)
        tpex_last_request_ts = time.time()


# --------------------
# 擷取普通股清單（含市場別）
# --------------------
def fetch_stock_list():
    urls = {
        '上櫃': 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=4'
    }
    stock_list = []

    for board, url in urls.items():
        try:
            res = session.get(url, timeout=20)
            res.encoding = 'big5'
            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]
            df.columns = df.iloc[0]
            df = df[1:]
            df = df[df['有價證券代號及名稱'].notna()]

            for entry in df['有價證券代號及名稱']:
                try:
                    code, name = entry.split('\u3000')
                    if re.match(r'^\d{4}$', code.strip()):
                        stock_list.append({
                            "stock_id": code.strip(),
                            "stock_name": name.strip(),
                            "market": board
                        })
                except:
                    continue
        except:
            continue

    return pd.DataFrame(stock_list).drop_duplicates()


# --------------------
# 擷取上櫃單月資料（含重試 + 節流）
# --------------------
def fetch_monthly_stock_data_tpex(stock_id, year, month, retries=3):
    url = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://www.tpex.org.tw',
        'Referer': 'https://www.tpex.org.tw/zh-tw/mainboard/trading/info/stock-pricing.html'
    }
    payload = {
        'code': stock_id,
        'date': f"{year}/{month:02d}/01",
        'id': '',
        'response': 'json'
    }

    for attempt in range(retries):
        try:
            with tpex_semaphore:
                tpex_throttle()
                res = tpex_session.post(url, data=payload, headers=headers, timeout=10)

            # 403 通常是被擋，稍微退避一下
            if res.status_code == 403:
                time.sleep(attempt * 0.8)
                continue

            res.raise_for_status()
            data = res.json()

            if 'tables' not in data or not data['tables']:
                return None

            rows = data['tables'][0].get('data')
            if not rows:
                return None

            # TPEx 結構：
            # 0日期, 1成交張數, 2成交仟元, 3開盤, 4最高, 5最低, 6收盤, 7漲跌, 8筆數
            df = pd.DataFrame(rows)
            df = df[[0, 4, 5, 6, 1]]
            df.columns = ['日期', '最高價', '最低價', '收盤價', '成交量']

            for col in ['最高價', '最低價', '收盤價', '成交量']:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '').replace('--', pd.NA),
                    errors='coerce'
                )

            # 成交張數 -> 股數
            df['成交量'] = df['成交量'] * 1000
            df['排序用日期'] = df['日期'].apply(parse_any_date)
            return df.dropna(subset=['排序用日期'])
        except:
            time.sleep(attempt * 0.5)

    return None


# --------------------
# 統一擷取單月資料
# --------------------
def fetch_monthly_stock_data(stock_id, year, month, market, retries=1):
    if market == '上櫃':
        return fetch_monthly_stock_data_tpex(stock_id, year, month, retries=retries)
    return None

# --------------------
# 處理單一股票（主邏輯）
# --------------------
def process_single_stock(row):
    stock_id = row['stock_id']
    raw_name = row['stock_name'].replace(' ', '_')
    stock_name = sanitize_filename(raw_name)
    market = row['market']
    output_file = os.path.join(SAVE_FOLDER, f"{stock_name}_{stock_id}.xlsx")
    today = datetime.today()

    # 已存在檔案：只更新最新月份
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_excel(output_file)

            if '日期' not in existing_df.columns:
                return f"⚠️ {stock_name}_{stock_id} 舊檔欄位異常"

            existing_df['排序用日期'] = existing_df['日期'].apply(parse_any_date)
            latest_date = existing_df['排序用日期'].max()

            if pd.isna(latest_date):
                return f"⚠️ {stock_name}_{stock_id} 舊檔日期異常"

            new_data = fetch_monthly_stock_data(stock_id, latest_date.year, latest_date.month, market)

            if new_data is None or new_data.empty:
                return f"⏩ {stock_name}_{stock_id} 已是最新"

            new_data['排序用日期'] = new_data['日期'].apply(parse_any_date)
            new_data = new_data[~new_data['排序用日期'].isin(existing_df['排序用日期'])]

            if new_data.empty:
                return f"⏩ {stock_name}_{stock_id} 無新增資料"

            new_data.insert(0, '股票名稱', raw_name)
            new_data.insert(1, '股票代號', stock_id)

            merged_df = pd.concat([existing_df, new_data], ignore_index=True)
            merged_df = merged_df.sort_values(by='排序用日期', ascending=False).head(280)

            if '排序用日期' in merged_df.columns:
                merged_df.drop(columns=['排序用日期'], inplace=True)

            merged_df.to_excel(output_file, index=False)
            return f"🔄 {stock_name}_{stock_id} 已更新最新資料"
        except:
            return f"⚠️ {stock_name}_{stock_id} 更新失敗"

    # 新檔案：往前回補到 280 筆
    all_data = []
    total_rows = 0
    months_back = 0
    consecutive_fails = 0

    while total_rows < 280:
        if consecutive_fails >= 1:
            return f"❌ {stock_name}_{stock_id} 連續失敗 1 次，跳過"

        query_date = today - pd.DateOffset(months=months_back)
        df = fetch_monthly_stock_data(stock_id, query_date.year, query_date.month, market)

        if df is not None and not df.empty:
            df.insert(0, '股票名稱', raw_name)
            df.insert(1, '股票代號', stock_id)
            all_data.append(df)
            total_rows += len(df)
            consecutive_fails = 0
        else:
            consecutive_fails += 1

        months_back += 1

        # 保護：避免無限回補
        if months_back > 18:
            break

    if all_data:
        try:
            merged = pd.concat(all_data, ignore_index=True)
            merged = merged.sort_values(by='排序用日期', ascending=False).head(280)

            if '排序用日期' in merged.columns:
                merged.drop(columns=['排序用日期'], inplace=True)

            merged.to_excel(output_file, index=False)
            return f"✅ {stock_name}_{stock_id} 儲存完成"
        except:
            return f"⚠️ {stock_name}_{stock_id} 儲存失敗"

    return f"⚠️ {stock_name}_{stock_id} 無有效資料"


# --------------------
# 主程式執行點
# --------------------
if __name__ == "__main__":
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    print("[INFO] 開始擷取 TPEx 普通股（含多執行緒 + 進度條）")

    all_stocks = fetch_stock_list()
    print(f"[INFO] 共取得 {len(all_stocks)} 檔股票")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_stock, row): row for _, row in all_stocks.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="📊 處理中"):
            results.append(future.result())

    print("\n📋 執行摘要：")
    for line in results:
        print(line)
    print("🎉 全部完成！")