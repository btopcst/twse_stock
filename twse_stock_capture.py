# -*- coding: utf-8 -*-

import os
import re
import time
import argparse
import requests
import pandas as pd
from datetime import datetime
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MAX_WORKERS = 8
SAVE_FOLDER = "stockList"
INDEX_FOLDER = "stockIndex"

# --------------------
# Convert ROC date to Gregorian
# --------------------
def convert_roc_date(roc_str):
    y, m, d = roc_str.split('/')
    return f"{int(y)+1911}-{m.zfill(2)}-{d.zfill(2)}"

# --------------------
# Sanitize filename
# --------------------
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)

# --------------------
# Fetch stock list (TWSE + TPEx)
# --------------------
def fetch_stock_list():
    urls = {
        'TWSE': 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2',
        'TPEx': 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=4'
    }

    stock_list = []

    for board, url in urls.items():
        try:
            res = requests.get(url, timeout=20)
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

        except Exception as e:
            print(f"[WARN] Failed to fetch {board} list: {e}")

    return pd.DataFrame(stock_list).drop_duplicates()

# --------------------
# Fetch monthly stock data
# --------------------
def fetch_monthly_stock_data(stock_id, year, month, retries=2):
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"

    params = {
        'response': 'json',
        'date': f"{year}{month:02d}01",
        'stockNo': stock_id
    }

    headers = {'User-Agent': 'Mozilla/5.0'}

    for _ in range(retries):
        try:
            res = requests.get(url, params=params, headers=headers, timeout=15)
            res.raise_for_status()
            data = res.json()

            if data.get('stat') != 'OK':
                return None

            df = pd.DataFrame(data['data'], columns=data['fields'])

            df = df[['日期', '最高價', '最低價', '收盤價', '成交股數']]
            df.columns = ['date', 'high', 'low', 'close', 'volume']

            for col in ['high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '').replace('--', pd.NA),
                    errors='coerce'
                )

            df['sort_date'] = pd.to_datetime(
                df['date'].apply(convert_roc_date),
                errors='coerce'
            )

            return df.dropna(subset=['sort_date'])

        except:
            time.sleep(2)

    return None

# --------------------
# Fetch index data
# --------------------
def fetch_index_data(market='tse'):
    today = datetime.today().strftime("%Y%m%d")

    url = f"https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={today}&market={market}"

    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        data = res.json()

        if 'data' in data:
            df = pd.DataFrame(data['data'], columns=data['fields'])

            os.makedirs(INDEX_FOLDER, exist_ok=True)
            file_name = f"{market}_index_{today}.xlsx"

            df.to_excel(os.path.join(INDEX_FOLDER, file_name), index=False)
            return f"[OK] Index saved: {file_name}"

        return f"[WARN] No index data: {market}"

    except Exception as e:
        return f"[ERROR] Index fetch failed: {e}"

# --------------------
# Process single stock
# --------------------
def process_single_stock(row):
    stock_id = row['stock_id']
    raw_name = row['stock_name'].replace(' ', '_')
    stock_name = sanitize_filename(raw_name)

    output_file = os.path.join(SAVE_FOLDER, f"{stock_name}_{stock_id}.xlsx")
    today = datetime.today()

    try:
        if os.path.exists(output_file):
            existing_df = pd.read_excel(output_file)

            existing_df['sort_date'] = pd.to_datetime(
                existing_df['date'].apply(convert_roc_date),
                errors='coerce'
            )

            latest_date = existing_df['sort_date'].max()

            if pd.isna(latest_date):
                return f"[WARN] Invalid existing date: {stock_id}"

            new_data = fetch_monthly_stock_data(stock_id, latest_date.year, latest_date.month)

            if new_data is None or new_data.empty:
                return f"[SKIP] {stock_id} already up-to-date"

            new_data['sort_date'] = pd.to_datetime(
                new_data['date'].apply(convert_roc_date),
                errors='coerce'
            )

            new_data = new_data[~new_data['sort_date'].isin(existing_df['sort_date'])]

            if new_data.empty:
                return f"[SKIP] {stock_id} no new data"

            new_data.insert(0, 'stock_name', raw_name)
            new_data.insert(1, 'stock_id', stock_id)

            merged_df = pd.concat([existing_df, new_data], ignore_index=True)
            merged_df = merged_df.sort_values(by='sort_date', ascending=False).head(280)

            merged_df.drop(columns=['sort_date'], inplace=True)

            merged_df.to_excel(output_file, index=False)
            return f"[UPDATE] {stock_id} updated"

        all_data = []
        total_rows = 0
        months_back = 0
        fail_count = 0

        while total_rows < 280 and months_back < 24:

            if fail_count >= 2:
                return f"[FAIL] {stock_id} consecutive failures"

            query_date = today - pd.DateOffset(months=months_back)
            df = fetch_monthly_stock_data(stock_id, query_date.year, query_date.month)

            if df is not None and not df.empty:
                df.insert(0, 'stock_name', raw_name)
                df.insert(1, 'stock_id', stock_id)

                all_data.append(df)
                total_rows += len(df)
                fail_count = 0
            else:
                fail_count += 1

            months_back += 1
            time.sleep(0.2)

        if all_data:
            merged = pd.concat(all_data, ignore_index=True)
            merged = merged.sort_values(by='sort_date', ascending=False).head(280)

            merged.drop(columns=['sort_date'], inplace=True)
            merged.to_excel(output_file, index=False)

            return f"[OK] {stock_id} saved"

        return f"[WARN] {stock_id} no data"

    except Exception as e:
        return f"[ERROR] {stock_id}: {e}"

# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    os.makedirs(SAVE_FOLDER, exist_ok=True)

    print(f"[INFO] Start batch={args.batch}")

    if not args.skip_index and args.batch == 0:
        print(fetch_index_data('tse'))
        print(fetch_index_data('otc'))

    all_stocks = fetch_stock_list()
    print(f"[INFO] Total stocks: {len(all_stocks)}")

    start = args.batch * args.batch_size
    end = start + args.batch_size

    batch_stocks = all_stocks.iloc[start:end].reset_index(drop=True)

    print(f"[INFO] Processing range: {start} ~ {end}")
    print(f"[INFO] Batch size: {len(batch_stocks)}")

    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_stock, row): row
            for _, row in batch_stocks.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    summary_file = f"summary_batch_{args.batch}.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")

    print("[INFO] Done")


if __name__ == "__main__":
    main()