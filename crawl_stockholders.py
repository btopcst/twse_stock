# -*- coding: utf-8 -*-

import os
import re
import time
import random
import warnings
import argparse
from io import StringIO
from typing import List, Tuple, Optional, Iterable, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# 關閉 pandas 未來版提醒（僅針對 read_html 的 literal html 警告）
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Passing literal html to 'read_html'"
)

# ---------- 常數 ----------
URL_VARIANTS = [
    "https://norway.twsthr.info/stockholders.aspx?stock={code}",
    "https://norway.twsthr.info/StockHolders.aspx?stock={code}",
    "https://norway.twsthr.info/StockHolders.aspx?STOCK={code}",
    "https://norway.twsthr.info/stockholders.aspx?STOCK={code}",
]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": "https://norway.twsthr.info/stockholders.aspx",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}
OUT_DIR = "stockholders"
LOG_DIR = "logs"
HTML_SAMPLES_DIR = os.path.join(LOG_DIR, "html_samples")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(HTML_SAMPLES_DIR, exist_ok=True)

# ---------- 節流器 ----------
class RateLimiter:
    """全域節流器：確保整體請求速率不超過 rate req/sec。"""
    def __init__(self, rate: float):
        self.rate = max(rate, 0.1)
        self.lock = Lock()
        self.allow_at = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.allow_at:
                time.sleep(self.allow_at - now)
                now = time.time()
            step = (1.0 / self.rate) * (0.9 + random.random() * 0.2)
            self.allow_at = now + step

# ---------- 共用工具 ----------
def _decode_try(b: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp950", "big5"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("cp950", errors="ignore")

def _get_with_retries(url: str, session: requests.Session, limiter: RateLimiter, max_retry: int = 3) -> requests.Response:
    last_exc = None
    for attempt in range(1, max_retry + 1):
        try:
            limiter.wait()
            r = session.get(url, headers=HEADERS, timeout=25, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(1.0 * attempt + random.random())
    raise last_exc

# ---------- 取得 TWSE/TPEx 普通股清單 ----------
def fetch_isin_table(strMode: int, session: requests.Session, limiter: RateLimiter) -> pd.DataFrame:
    """
    strMode=2 -> TWSE 上市；strMode=4 -> TPEx 上櫃
    回傳 DataFrame（第一列升為欄名）
    """
    url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={strMode}"
    r = _get_with_retries(url, session=session, limiter=limiter)
    # ISIN 網頁常見 cp950/big5，做容錯解碼
    html_txt = _decode_try(r.content)
    try:
        dfs = pd.read_html(StringIO(html_txt))
    except Exception:
        dfs = pd.read_html(r.content)
    if not dfs:
        raise RuntimeError("No table parsed from TWSE ISIN page.")
    df = dfs[0].copy()
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    return df

def get_all_tickers(limiter: RateLimiter) -> List[Tuple[str, str]]:
    """
    回傳 [(code, name), ...]；以「4~5位數字開頭 + 名稱」判斷，不依賴「備註」欄。
    """
    all_list: List[Tuple[str, str]] = []
    with requests.Session() as session:
        for mode in (2, 4):
            df = fetch_isin_table(mode, session=session, limiter=limiter)
            col0_candidates = [c for c in df.columns if "代號" in str(c)]
            col0 = col0_candidates[0] if col0_candidates else df.columns[0]
            pat = re.compile(r"^(\d{4,5})\s+(.+)$")
            def split_code_name(s) -> Optional[Tuple[str, str]]:
                s = "" if pd.isna(s) else str(s).strip()
                m = pat.match(s)
                if m:
                    return m.group(1), m.group(2)
                return None
            pairs = df[col0].apply(split_code_name).dropna().tolist()
            all_list.extend(pairs)
    # 去重
    seen = set()
    unique: List[Tuple[str, str]] = []
    for code, name in all_list:
        if code not in seen:
            seen.add(code)
            unique.append((code, name))
    return unique

# ---------- 解析個股頁 ----------
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {c: str(c).replace("\xa0", "").replace(" ", "") for c in df.columns}
    return df.rename(columns=rename_map)

def _normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    def to_num(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).replace(",", "").replace("%", "").strip()
        if s == "":
            return pd.NA
        try:
            return float(s)
        except Exception:
            return pd.NA
    for c in df.columns:
        if c == "資料日期":
            continue
        df[c] = df[c].apply(to_num)
    return df

def _parse_table_details(soup: BeautifulSoup) -> Optional[pd.DataFrame]:
    """優先解析 <table id='Details'>；若找不到，再用 read_html 掃包含「資料日期」列的表；最後文字 fallback。"""
    # 1) 直接抓 id=Details
    tbl = soup.find("table", id="Details")
    if tbl is not None:
        raw = pd.read_html(StringIO(str(tbl)))[0]
        # 若第一列含標頭，把第一列升為欄名
        if raw.shape[0] > 0 and raw.iloc[0].astype(str).str.contains("資料日期").any():
            raw.columns = raw.iloc[0]
            raw = raw.iloc[1:].reset_index(drop=True)

        # 有些頁可能在最左有1~2欄樣式/色塊，試著移除
        # 目標：確保欄名包含「資料日期」且欄數 >= 12
        def try_strip_left(df_in: pd.DataFrame) -> pd.DataFrame:
            df = df_in.copy()
            for drop_n in (1, 2):
                if df.shape[1] > drop_n:
                    tmp = df.drop(columns=list(df.columns)[:drop_n])
                    if tmp.shape[0] > 0 and tmp.iloc[0].astype(str).str.contains("資料日期").any():
                        tmp.columns = tmp.iloc[0]
                        tmp = tmp.iloc[1:].reset_index(drop=True)
                        return tmp
            return df_in

        if "資料日期" not in "".join(map(str, raw.columns)):
            raw = try_strip_left(raw)

        df = _clean_columns(raw)
        # 標準欄名修正
        if "資料日期" not in df.columns:
            for c in list(df.columns):
                if "資料日期" in str(c):
                    df = df.rename(columns={c: "資料日期"})
                    break
        if "資料日期" in df.columns:
            df = df[df["資料日期"].astype(str).str.match(r"^20\d{2}", na=False)]
        df = df.dropna(how="all").reset_index(drop=True)
        return df if not df.empty else None

    # 2) read_html 掃所有表，找包含「資料日期」的列，將該列升為表頭
    try:
        tables = pd.read_html(StringIO(str(soup)))
        candidate = None
        header_row_idx = None
        for t in tables:
            if t.empty:
                continue
            mask = t.apply(lambda row: row.astype(str).str.contains("資料日期", na=False).any(), axis=1)
            idxs = mask[mask].index.tolist()
            if idxs:
                candidate = t.copy()
                header_row_idx = idxs[0]
                break
        if candidate is not None:
            df = candidate.copy()
            df.columns = df.iloc[header_row_idx]
            df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
            df = _clean_columns(df)
            if "資料日期" in df.columns:
                df = df[df["資料日期"].astype(str).str.match(r"^20\d{2}", na=False)]
            df = df.dropna(how="all").reset_index(drop=True)
            return df if not df.empty else None
    except Exception:
        pass

    # 3) 文字 fallback
    text = soup.get_text("\n")
    lines = []
    for line in text.splitlines():
        s = re.sub(r"\s+", " ", line).strip()
        if re.match(r"^(20\d{2})(?:[-/ ]?\d{2})(?:[-/ ]?\d{2})?\b", s):
            lines.append(s)
    if not lines:
        return None
    cols = [
        "資料日期", "集保總張數", "總股東人數", "平均張數/人",
        ">400張大股東持有張數", ">400張大股東持有百分比",
        "大股東人數", "400~600張人數", "600~800張人數",
        "800~1000張人數", ">1000張人數", ">1000張大股東持有百分比",
        "收盤價",
    ]
    records: List[List[str]] = []
    for s in lines:
        parts = s.split()
        if len(parts) >= 5:
            row = [parts[0]] + parts[1:]
            if len(row) < len(cols):
                row += [None] * (len(cols) - len(row))
            elif len(row) > len(cols):
                row = row[:len(cols)]
            records.append(row)
    return pd.DataFrame(records, columns=cols) if records else None

def parse_stockholders_html(html: str) -> Optional[pd.DataFrame]:
    """封裝解析 + 基本阻擋偵測。"""
    soup = BeautifulSoup(html, "lxml")
    probe = soup.get_text(" ", strip=True)
    if ("系統偵測到廣告阻擋" in probe) and not re.search(r"\b20\d{2}[/-]?\d{2}", probe):
        return None
    df = _parse_table_details(soup)
    if df is None or df.empty:
        return None
    # 標準欄位集合（缺欄補 NA）
    standard_cols = [
        "資料日期", "集保總張數", "總股東人數", "平均張數/人",
        ">400張大股東持有張數", ">400張大股東持有百分比",
        "大股東人數", "400~600張人數", "600~800張人數",
        "800~1000張人數", ">1000張人數", ">1000張大股東持有百分比",
        "收盤價",
    ]
    for col in standard_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[[c for c in standard_cols if c in df.columns]]
    df = _normalize_numeric_columns(df)
    return df

# ---------- 擷取單檔 ----------
def _fetch_html_for_code(code: str, session: requests.Session, limiter: RateLimiter) -> Optional[str]:
    """嘗試多個 URL 變體，回傳成功取得的 HTML；否則 None。"""
    for u in URL_VARIANTS:
        url = u.format(code=code)
        try:
            r = _get_with_retries(url, session=session, limiter=limiter)
            txt = _decode_try(r.content)
            # 粗略判斷是否像資料頁
            if ("資料日期" in txt) or re.search(r"\b20\d{2}[/-]?\d{2}", txt):
                return txt
        except Exception:
            continue
    return None

def fetch_one(code: str, name: str, session: requests.Session, limiter: RateLimiter,
              skip_existing: bool = False, save_html_samples: int = 10) -> Tuple[str, bool]:
    """
    抓取單一代號；回傳 (code, success)
    成功則輸出 CSV；失敗會在前 N 檔保留 HTML 樣本。
    """
    safe_name = re.sub(r"[\\/:*?\"<>| ]+", "_", name)
    out_path = os.path.join(OUT_DIR, f"{code}_{safe_name}.csv")

    if skip_existing and os.path.exists(out_path):
        return code, True

    html_txt: Optional[str] = None
    for attempt in range(1, 4):
        html_txt = _fetch_html_for_code(code, session, limiter)
        if html_txt:
            break
        time.sleep(1.0 * attempt + random.random())

    if not html_txt:
        samples = [f for f in os.listdir(HTML_SAMPLES_DIR) if f.endswith(".html")]
        if len(samples) < save_html_samples:
            with open(os.path.join(HTML_SAMPLES_DIR, f"{code}.html"), "w", encoding="utf-8") as f:
                f.write("<!-- fetch failed: no usable html -->")
        return code, False

    df = parse_stockholders_html(html_txt)
    if df is None or df.empty:
        samples = [f for f in os.listdir(HTML_SAMPLES_DIR) if f.endswith(".html")]
        if len(samples) < save_html_samples:
            with open(os.path.join(HTML_SAMPLES_DIR, f"{code}.html"), "w", encoding="utf-8") as f:
                f.write(html_txt)
        return code, False

    df.insert(0, "代號", code)
    df.insert(1, "名稱", name)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return code, True

# ---------- 其它 ----------
def iter_codes_from_file(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if re.fullmatch(r"\d{4,5}", s):
                yield s

# ---------- 入口 ----------
def main():
    parser = argparse.ArgumentParser(description="神秘金字塔 股權分散表 - 全市場/指定清單")
    parser.add_argument("--workers", type=int, default=8, help="併發執行緒數 (default: 8)")
    parser.add_argument("--rate", type=float, default=4.0, help="全程節流速率 req/sec (default: 4)")
    parser.add_argument("--skip-existing", action="store_true", help="已存在檔案就略過 (可斷點續抓)")
    parser.add_argument("--codes", type=str, default=None, help="只抓指定清單（每行一個代號）")
    parser.add_argument("--no-merge", action="store_true", help="抓完不合併輸出大檔")
    args = parser.parse_args()

    limiter = RateLimiter(rate=args.rate)

    # 代號來源
    if args.codes:
        codes = list(iter_codes_from_file(args.codes))
        if not codes:
            print("指定 --codes 檔案沒有有效代號。")
            return
        tickers = [(c, c) for c in codes]
        print(f"使用指定清單，代號數量：{len(tickers)}")
    else:
        tickers = get_all_tickers(limiter=limiter)
        tickers = [(c, n) for c, n in tickers if re.fullmatch(r"\d{4,5}", c)]
        print(f"取得代號數量：{len(tickers)}")
        if not tickers:
            print("未取得任何代號，請稍後重試（可能是 TWSE ISIN 臨時異常或網路/防火牆影響）。")
            return

    success_codes: List[str] = []
    fail_codes: List[str] = []

    with requests.Session() as session:
        adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut2code: Dict = {}
            for code, name in tickers:
                fut = ex.submit(fetch_one, code, name, session, limiter, args.skip_existing)
                fut2code[fut] = code

            for fut in tqdm(as_completed(fut2code), total=len(fut2code), desc="Crawling", ncols=100):
                code, ok = fut.result()
                if ok:
                    success_codes.append(code)
                else:
                    fail_codes.append(code)

    # 輸出 missing.txt
    with open(os.path.join(LOG_DIR, "missing.txt"), "w", encoding="utf-8") as f:
        for code in fail_codes:
            f.write(f"{code}\n")

    # 合併（可選）
    if not args.no_merge:
        merged: List[pd.DataFrame] = []
        for fname in os.listdir(OUT_DIR):
            if not fname.lower().endswith(".csv"):
                continue
            try:
                df = pd.read_csv(os.path.join(OUT_DIR, fname), dtype={"代號": str})
                merged.append(df)
            except Exception:
                continue
        if merged:
            all_df = pd.concat(merged, ignore_index=True)
            all_df.to_parquet("all_stockholders.parquet", index=False)
            sample_rows = min(len(all_df), 200_000)
            all_df.iloc[:sample_rows].to_csv("all_stockholders_sample.csv", index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"成功：{len(success_codes)} 檔；失敗：{len(fail_codes)} 檔")
    print(f"檔案輸出資料夾：{OUT_DIR}")
    if not args.no_merge:
        print("合併檔：all_stockholders.parquet、all_stockholders_sample.csv")
    print("抓不到或無明細的清單：logs/missing.txt")
    print("若仍有大量失敗，請檢查 logs/html_samples/ 內的 .html 樣本以了解實際頁面型態。")

if __name__ == "__main__":
    main()
