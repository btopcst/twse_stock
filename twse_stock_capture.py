# twse_stock_capture.py（修正非法檔名 + 多執行緒 + 進度條）
import os
import re
import time
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
        '上市': 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2',
        '上櫃': 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=4'
    }
    stock_list = []
    for board, url in urls.items():
        res = requests.get(url)
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
                    stock_list.append({"stock_id": code.strip(), "stock_name": name.strip()})
            except:
                continue
    return pd.DataFrame(stock_list).drop_duplicates()

# --------------------
# 擷取單月資料（含重試）
# --------------------
def fetch_monthly_stock_data(stock_id, year, month, retries=1):
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
    params = {'response': 'json', 'date': f"{year}{month:02d}01", 'stockNo': stock_id}
    headers = {'User-Agent': 'Mozilla/5.0'}
    for attempt in range(retries):
        try:
            res = requests.get(url, params=params, headers=headers, timeout=10)
            res.raise_for_status()
            data = res.json()
            if data['stat'] != 'OK':
                return None
            df = pd.DataFrame(data['data'], columns=data['fields'])
            df = df[['日期', '最高價', '最低價', '收盤價', '成交股數']]
            df.columns = ['日期', '最高價', '最低價', '收盤價', '成交量']
            for col in ['最高價', '最低價', '收盤價', '成交量']:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').replace('--', pd.NA), errors='coerce')
            df['排序用日期'] = pd.to_datetime(df['日期'].apply(convert_roc_date), errors='coerce')
            return df.dropna(subset=['排序用日期'])
        except:
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
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        data = res.json()
        if 'data' in data:
            df = pd.DataFrame(data['data'], columns=data['fields'])
            file_name = f"{'上市' if market == 'tse' else '上櫃'}指數_{today}.xlsx"
            os.makedirs(INDEX_FOLDER, exist_ok=True)
            df.to_excel(os.path.join(INDEX_FOLDER, file_name), index=False)
            return f"✅ 指數資料已儲存：{file_name}"
    except Exception as e:
        return f"❌ 指數擷取失敗：{market} => {e}"

# --------------------
# 處理單一股票（主邏輯）
# --------------------
def process_single_stock(row):
    stock_id = row['stock_id']
    raw_name = row['stock_name'].replace(' ', '_')
    stock_name = sanitize_filename(raw_name)
    output_file = os.path.join(SAVE_FOLDER, f"{stock_name}_{stock_id}.xlsx")
    today = datetime.today()

    if os.path.exists(output_file):
        existing_df = pd.read_excel(output_file)
        existing_df['排序用日期'] = pd.to_datetime(existing_df['日期'].apply(convert_roc_date), errors='coerce')
        latest_date = existing_df['排序用日期'].max()
        new_data = fetch_monthly_stock_data(stock_id, latest_date.year, latest_date.month)

        if new_data is None or new_data.empty:
            return f"⏩ {stock_name}_{stock_id} 已是最新"

        new_data['排序用日期'] = pd.to_datetime(new_data['日期'].apply(convert_roc_date), errors='coerce')
        new_data = new_data[~new_data['排序用日期'].isin(existing_df['排序用日期'])]

        if new_data.empty:
            return f"⏩ {stock_name}_{stock_id} 無新增資料"

        new_data.insert(0, '股票名稱', raw_name)
        new_data.insert(1, '股票代號', stock_id)
        merged_df = pd.concat([existing_df, new_data], ignore_index=True)
        merged_df = merged_df.sort_values(by='排序用日期', ascending=False).head(280)
        merged_df.drop(columns=['排序用日期'], inplace=True)
        merged_df.to_excel(output_file, index=False)
        return f"🔄 {stock_name}_{stock_id} 已更新最新資料"

    all_data, total_rows, months_back, consecutive_fails = [], 0, 0, 0
    while total_rows < 280:
        if consecutive_fails >= 1:
            return f"❌ {stock_name}_{stock_id} 連續失敗 1 次，跳過"
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
        time.sleep(0.5)

    if all_data:
        merged = pd.concat(all_data, ignore_index=True)
        merged = merged.sort_values(by='排序用日期', ascending=False).head(280)
        merged.drop(columns=['排序用日期'], inplace=True)
        merged.to_excel(output_file, index=False)
        return f"✅ {stock_name}_{stock_id} 儲存完成"
    return f"⚠️ {stock_name}_{stock_id} 無有效資料"

# --------------------
# 主程式執行點
# --------------------
if __name__ == "__main__":
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    print("[INFO] 開始擷取 TWSE 普通股（含多執行緒 + 進度條）")

    print("[INFO] 擷取上市大盤指數...")
    print(fetch_index_data('tse'))
    print("[INFO] 擷取上櫃大盤指數...")
    print(fetch_index_data('otc'))

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
