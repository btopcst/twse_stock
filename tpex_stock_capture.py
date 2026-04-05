import os
import re
import time
import threading
import argparse
import requests
import pandas as pd
from datetime import datetime
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MAX_WORKERS = 8
SAVE_FOLDER = "stockList"

TPEX_MAX_CONCURRENT = 4
TPEX_MIN_INTERVAL_SEC = 0.2

session = requests.Session()
tpex_session = requests.Session()

tpex_semaphore = threading.Semaphore(TPEX_MAX_CONCURRENT)
tpex_rate_lock = threading.Lock()
tpex_last_request_ts = 0.0


# --------------------
# date convert
# --------------------
def convert_roc_date(roc_str):
    y, m, d = roc_str.split('/')
    return f"{int(y)+1911}-{m.zfill(2)}-{d.zfill(2)}"


def parse_any_date(date_str):
    if pd.isna(date_str):
        return pd.NaT

    s = str(date_str).strip().replace('-', '/')
    parts = s.split('/')

    try:
        if len(parts) == 3 and len(parts[0]) == 3:
            s = convert_roc_date(s)
        elif len(parts) == 3:
            s = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
        return pd.to_datetime(s, errors='coerce')
    except:
        return pd.NaT


def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)


# --------------------
# throttle
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
# stock list
# --------------------
def fetch_stock_list():
    url = 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=4'

    stock_list = []

    res = session.get(url)
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
                    "stock_name": name.strip()
                })
        except:
            continue

    return pd.DataFrame(stock_list).drop_duplicates()


# --------------------
# fetch monthly
# --------------------
def fetch_monthly(stock_id, year, month):

    url = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"

    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Content-Type': 'application/x-www-form-urlencoded',
        'X-Requested-With': 'XMLHttpRequest'
    }

    payload = {
        'code': stock_id,
        'date': f"{year}/{month:02d}/01",
        'response': 'json'
    }

    try:
        with tpex_semaphore:
            tpex_throttle()
            res = tpex_session.post(url, data=payload, headers=headers, timeout=10)

        if res.status_code != 200:
            return None

        data = res.json()

        if 'tables' not in data or not data['tables']:
            return None

        rows = data['tables'][0]['data']

        df = pd.DataFrame(rows)
        df = df[[0, 4, 5, 6, 1]]
        df.columns = ['日期', '最高價', '最低價', '收盤價', '成交量']

        for col in ['最高價', '最低價', '收盤價', '成交量']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

        df['成交量'] = df['成交量'] * 1000
        df['sort_date'] = df['日期'].apply(parse_any_date)

        return df.dropna(subset=['sort_date'])

    except:
        return None


# --------------------
# process stock
# --------------------
def process_stock(row):

    stock_id = row['stock_id']
    name = sanitize_filename(row['stock_name'])

    output_file = os.path.join(SAVE_FOLDER, f"{name}_{stock_id}.xlsx")

    today = datetime.today()

    all_data = []
    total = 0
    months_back = 0

    while total < 280:

        query_date = today - pd.DateOffset(months=months_back)

        df = fetch_monthly(stock_id, query_date.year, query_date.month)

        if df is not None and not df.empty:
            df.insert(0, '股票名稱', name)
            df.insert(1, '股票代號', stock_id)
            all_data.append(df)
            total += len(df)
        else:
            break

        months_back += 1

        if months_back > 18:
            break

    if all_data:
        merged = pd.concat(all_data)
        merged = merged.sort_values('sort_date', ascending=False).head(280)
        merged.drop(columns=['sort_date'], inplace=True)

        merged.to_excel(output_file, index=False)

        return f"[OK] {name}_{stock_id}"

    return f"[FAIL] {name}_{stock_id}"


# --------------------
# main
# --------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(SAVE_FOLDER, exist_ok=True)

    print(f"[INFO] batch={args.batch}, batch_size={args.batch_size}")

    stocks = fetch_stock_list()

    start = args.batch * args.batch_size
    end = start + args.batch_size

    stocks = stocks.iloc[start:end]

    print(f"[INFO] processing {len(stocks)} stocks")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_stock, row): row for _, row in stocks.iterrows()}

        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

    print("\n[SUMMARY]")
    for r in results:
        print(r)

    print("[DONE]")
