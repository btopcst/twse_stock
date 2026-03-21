# twse_stock_capture.py
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
INDEX_FOLDER = "stockTse"

# --------------------
# 將民國年轉西元年
# --------------------
def convert_roc_date(roc_str):
    y, m, d = roc_str.split('/')
    return f"{int(y)+1911}-{m.zfill(2)}-{d.zfill(2)}"

# --------------------
# 清理檔名非法字元
# --------------------
def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)

# --------------------
# 擷取普通股清單
# --------------------
def fetch_stock_list():
    urls = {
        '上市': 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2'
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
                except Exception:
                    continue
        except Exception as e:
            print(f"[WARN] 擷取 {board} 股票清單失敗: {e}")

    return pd.DataFrame(stock_list).drop_duplicates()

# --------------------
# 擷取單月資料（含重試）
# --------------------
def fetch_monthly_stock_data(stock_id, year, month, retries=2):
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
    params = {
        'response': 'json',
        'date': f"{year}{month:02d}01",
        'stockNo': stock_id
    }
    headers = {'User-Agent': 'Mozilla/5.0'}

    for attempt in range(retries):
        try:
            res = requests.get(url, params=params, headers=headers, timeout=15)
            res.raise_for_status()
            data = res.json()

            if data.get('stat') != 'OK':
                return None

            df = pd.DataFrame(data['data'], columns=data['fields'])
            df = df[['日期', '最高價', '最低價', '收盤價', '成交股數']]
            df.columns = ['日期', '最高價', '最低價', '收盤價', '成交量']

            for col in ['最高價', '最低價', '收盤價', '成交量']:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '').replace('--', pd.NA),
                    errors='coerce'
                )

            df['排序用日期'] = pd.to_datetime(
                df['日期'].apply(convert_roc_date),
                errors='coerce'
            )
            return df.dropna(subset=['排序用日期'])

        except Exception:
            time.sleep(2)

    return None

# --------------------
# 擷取大盤指數資料
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
            file_name = f"{'上市' if market == 'tse' else '上櫃'}指數_{today}.xlsx"
            os.makedirs(INDEX_FOLDER, exist_ok=True)
            df.to_excel(os.path.join(INDEX_FOLDER, file_name), index=False)
            return f"✅ 指數資料已儲存：{file_name}"

        return f"⚠️ 指數資料無內容：{market}"

    except Exception as e:
        return f"❌ 指數擷取失敗：{market} => {e}"

# --------------------
# 處理單一股票
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
            existing_df['排序用日期'] = pd.to_datetime(
                existing_df['日期'].apply(convert_roc_date),
                errors='coerce'
            )
            latest_date = existing_df['排序用日期'].max()

            if pd.isna(latest_date):
                return f"⚠️ {stock_name}_{stock_id} 舊檔日期異常"

            new_data = fetch_monthly_stock_data(stock_id, latest_date.year, latest_date.month)

            if new_data is None or new_data.empty:
                return f"⏩ {stock_name}_{stock_id} 已是最新"

            new_data['排序用日期'] = pd.to_datetime(
                new_data['日期'].apply(convert_roc_date),
                errors='coerce'
            )
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

        all_data = []
        total_rows = 0
        months_back = 0
        consecutive_fails = 0

        while total_rows < 280 and months_back < 24:
            if consecutive_fails >= 2:
                return f"❌ {stock_name}_{stock_id} 連續失敗 2 次，跳過"

            query_date = today - pd.DateOffset(months=months_back)
            df = fetch_monthly_stock_data(stock_id, query_date.year, query_date.month)

            if df is not None and not df.empty:
                df.insert(0, '股票名稱', raw_name)
                df.insert(1, '股票代號', stock_id)
                all_data.append(df)
                total_rows += len(df)
                consecutive_fails = 0
            else:
                consecutive_fails += 1

            months_back += 1
            time.sleep(0.2)

        if all_data:
            merged = pd.concat(all_data, ignore_index=True)
            merged = merged.sort_values(by='排序用日期', ascending=False).head(280)
            merged.drop(columns=['排序用日期'], inplace=True)
            merged.to_excel(output_file, index=False)
            return f"✅ {stock_name}_{stock_id} 儲存完成"

        return f"⚠️ {stock_name}_{stock_id} 無有效資料"

    except Exception as e:
        return f"❌ {stock_name}_{stock_id} 發生錯誤: {e}"

# --------------------
# 主程式
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=0, help="批次編號")
    parser.add_argument("--batch-size", type=int, default=400, help="每批股票數量")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="執行緒數")
    parser.add_argument("--skip-index", action="store_true", help="略過指數擷取")
    args = parser.parse_args()

    os.makedirs(SAVE_FOLDER, exist_ok=True)

    print(f"[INFO] 開始執行 batch={args.batch}, batch_size={args.batch_size}, workers={args.workers}")

    all_stocks = fetch_stock_list()
    print(f"[INFO] 共取得 {len(all_stocks)} 檔股票")

    start_idx = args.batch * args.batch_size
    end_idx = start_idx + args.batch_size
    batch_stocks = all_stocks.iloc[start_idx:end_idx].reset_index(drop=True)

    print(f"[INFO] 本批處理區間: {start_idx} ~ {end_idx - 1}")
    print(f"[INFO] 本批共 {len(batch_stocks)} 檔股票")

    if batch_stocks.empty:
        print("[WARN] 本批沒有股票可處理")
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_stock, row): row
            for _, row in batch_stocks.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {args.batch} 處理中"):
            results.append(future.result())

    summary_file = f"capture_summary_batch_{args.batch}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")

    print("\n📋 執行摘要：")
    for line in results[:20]:
        print(line)

    print(f"[INFO] 摘要已儲存至 {summary_file}")
    print("🎉 本批完成！")


if __name__ == "__main__":
    main()