# -*- coding: utf-8 -*-

import os
import re
import io
import time
import argparse
from typing import Optional, Tuple, List

import pandas as pd
import requests
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.twse.com.tw/",
}

LOOKBACK_TRADING_DAYS = 10
MAX_BACKTRACK_CALENDAR_DAYS = 25


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_code(v) -> str:
    s = str(v).strip()
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"[^\d]", "", s)
    if s.isdigit() and len(s) < 4:
        s = s.zfill(4)
    return s


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


def _parse_date(s: str) -> Optional[pd.Timestamp]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    if not s:
        return None
    s = s.replace("/", "-")
    if re.fullmatch(r"\d{8}", s):
        s = f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    try:
        return pd.to_datetime(s, errors="raise", format="%Y-%m-%d")
    except Exception:
        try:
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return None


def _find_percent_col(columns: List[str]) -> Optional[str]:
    norm_map = {}
    for c in columns:
        cc = str(c).replace(" ", "").replace("\u3000", "")
        cc = cc.replace("＞", ">")
        norm_map[c] = cc

    preferred = [
        ">1000張大股東持有百分比",
        ">1000張大股東持有比率",
        "1000張以上大股東持有百分比",
    ]
    for orig, cc in norm_map.items():
        if cc in preferred:
            return orig

    candidates = []
    for orig, cc in norm_map.items():
        if "1000" in cc and ("百分比" in cc or "比率" in cc):
            candidates.append(orig)

    if len(candidates) > 1:
        ranked = sorted(
            candidates,
            key=lambda x: (
                -int(">" in norm_map[x]),
                -int("張" in norm_map[x]),
                -int("大股東" in norm_map[x]),
                -int("百分比" in norm_map[x]),
            ),
        )
        return ranked[0] if ranked else candidates[0]
    return candidates[0] if candidates else None


def _to_float_percent(x) -> Optional[float]:
    if pd.isna(x):
        return None
    s = str(x).replace(",", "").replace("%", "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def compare_last_two_weeks(
    df: pd.DataFrame, code: str, name: str, pct_col: str
) -> Optional[Tuple[str, str, str, float, str, float, float]]:
    if "資料日期" not in df.columns:
        return None

    df = df.copy()
    df["__dt"] = df["資料日期"].apply(_parse_date)
    df = df[~df["__dt"].isna()]
    if df.empty:
        return None

    df["__pct"] = df[pct_col].apply(_to_float_percent)
    df = df[~df["__pct"].isna()]
    if df.empty:
        return None

    iso = df["__dt"].dt.isocalendar()
    df["__iso_year"] = iso["year"]
    df["__iso_week"] = iso["week"]

    weekly = (
        df.sort_values("__dt")
        .groupby(["__iso_year", "__iso_week"], as_index=False)
        .tail(1)
        .sort_values(["__iso_year", "__iso_week"])
        .reset_index(drop=True)
    )

    if len(weekly) < 2:
        return None

    last = weekly.iloc[-1]
    prev = weekly.iloc[-2]

    last_date = last["__dt"].date().isoformat()
    prev_date = prev["__dt"].date().isoformat()
    last_pct = float(last["__pct"])
    prev_pct = float(prev["__pct"])
    delta = last_pct - prev_pct

    return (code, name, last_date, last_pct, prev_date, prev_pct, delta)


def pick_twse_table(tables: List[pd.DataFrame]) -> pd.DataFrame:
    best_tbl = None
    best_score = -1

    for tbl in tables:
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
        ["外陸資買賣超股數(不含外資自營商)", "外資及陸資買賣超股數(不含外資自營商)"],
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

    x["外資淨買賣"] = foreign_val.map(to_lot) if hasattr(foreign_val, "map") else 0
    x["投信淨買賣"] = trust_val.map(to_lot) if hasattr(trust_val, "map") else 0
    x["自營淨買賣"] = dealer_val.map(to_lot) if hasattr(dealer_val, "map") else 0
    x["日期"] = pd.to_datetime(target_date).strftime("%Y-%m-%d")
    x["市場"] = "上市"

    return x[["股票代號", "日期", "外資淨買賣", "投信淨買賣", "自營淨買賣", "市場"]].drop_duplicates(["股票代號", "日期"])


def fetch_twse(date_obj: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    ymd = pd.to_datetime(date_obj).strftime("%Y%m%d")
    url = f"https://www.twse.com.tw/fund/T86?response=html&date={ymd}&selectType=ALLBUT0999"
    r = session.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    if not tables:
        raise ValueError("TWSE no table")
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
        raise ValueError("TPEx no table")

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
        ["外資及陸資買賣超股數", "外資及陸資淨買股數", "外資及陸資(不含外資自營商)買賣超股數", "外資及陸資(不含外資自營商)淨買股數"],
    )
    trust_col = find_col_endswith(cols, ["投信買賣超股數", "投信淨買股數"])

    dealer_total_col = None
    for c in cols:
        c_str = str(c).strip()
        if (c_str.endswith("自營商買賣超股數") or c_str.endswith("自營商淨買股數")) and ("外資" not in c_str):
            dealer_total_col = c
            break

    dealer_self_col = find_col_endswith(cols, ["自營商(自行買賣)買賣超股數", "自營商(自行買賣)淨買股數"])
    dealer_hedge_col = find_col_endswith(cols, ["自營商(避險)買賣超股數", "自營商(避險)淨買股數"])

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

    x["外資淨買賣"] = foreign_val.map(to_lot) if hasattr(foreign_val, "map") else 0
    x["投信淨買賣"] = trust_val.map(to_lot) if hasattr(trust_val, "map") else 0
    x["自營淨買賣"] = dealer_val.map(to_lot) if hasattr(dealer_val, "map") else 0
    x["日期"] = pd.to_datetime(target_date).strftime("%Y-%m-%d")
    x["市場"] = "上櫃"

    return x[["股票代號", "日期", "外資淨買賣", "投信淨買賣", "自營淨買賣", "市場"]].drop_duplicates(["股票代號", "日期"])


def build_recent_10d_sum(base_df: pd.DataFrame) -> pd.DataFrame:
    unique_dates = sorted(pd.to_datetime(base_df["日期"].dropna().unique()))
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
                tmp_list.append(pd.concat(day_frames, ignore_index=True))
                found_days += 1

            cursor = cursor - pd.Timedelta(days=1)
            checked_days += 1

        if tmp_list:
            df_all = pd.concat(tmp_list, ignore_index=True)
            df_all["基準日期"] = pd.to_datetime(ref_date).strftime("%Y-%m-%d")
            agg = (
                df_all.groupby(["股票代號", "基準日期"], as_index=False)[["外資淨買賣", "投信淨買賣", "自營淨買賣"]]
                .sum()
            )
            records.append(agg)

    if not records:
        return pd.DataFrame(columns=["股票代號", "日期", "外資近10日", "投信近10日", "自營近10日"])

    result = pd.concat(records, ignore_index=True).rename(
        columns={"基準日期": "日期", "外資淨買賣": "外資近10日", "投信淨買賣": "投信近10日", "自營淨買賣": "自營近10日"}
    )
    for col in ["外資近10日", "投信近10日", "自營近10日"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype("int64")
    return result


def build_inst_snapshot(base_df: pd.DataFrame) -> pd.DataFrame:
    unique_dates = sorted(pd.to_datetime(base_df["日期"].dropna().unique()))
    session = requests.Session()
    frames = []

    for d in unique_dates:
        parts = []
        try:
            twse_df = fetch_twse(pd.to_datetime(d), session)[["股票代號", "日期", "市場"]].copy()
            if not twse_df.empty:
                parts.append(twse_df)
        except Exception:
            pass

        try:
            tpex_df = fetch_tpex(pd.to_datetime(d), session)[["股票代號", "日期", "市場"]].copy()
            if not tpex_df.empty:
                parts.append(tpex_df)
        except Exception:
            pass

        if parts:
            frames.append(pd.concat(parts, ignore_index=True))

    if not frames:
        return pd.DataFrame(columns=["股票代號", "日期", "市場"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["股票代號", "日期"], keep="first")


def enrich_with_3inst_10d(report_df: pd.DataFrame) -> pd.DataFrame:
    if report_df.empty:
        return report_df

    base_df = pd.DataFrame({
        "股票代號": report_df["代號"].map(normalize_code),
        "日期": pd.to_datetime(report_df["最近週日期"], errors="coerce").dt.strftime("%Y-%m-%d"),
    }).dropna(subset=["股票代號", "日期"]).drop_duplicates()

    hist10_df = build_recent_10d_sum(base_df)
    market_df = build_inst_snapshot(base_df)

    merged = report_df.copy()
    merged["代號"] = merged["代號"].map(normalize_code)
    merged["最近週日期"] = pd.to_datetime(merged["最近週日期"], errors="coerce").dt.strftime("%Y-%m-%d")

    merged = merged.merge(
        hist10_df.rename(columns={"股票代號": "代號", "日期": "最近週日期"}),
        on=["代號", "最近週日期"],
        how="left",
    )
    merged = merged.merge(
        market_df.rename(columns={"股票代號": "代號", "日期": "最近週日期"}),
        on=["代號", "最近週日期"],
        how="left",
    )

    for col in ["外資近10日", "投信近10日", "自營近10日"]:
        merged[col] = pd.to_numeric(merged.get(col), errors="coerce").fillna(0).astype("int64")

    merged["市場"] = merged.get("市場", "未知")
    merged["市場"] = merged["市場"].fillna("未知")
    return merged


def main():
    parser = argparse.ArgumentParser(description="找出最近一週 > 前一週之 >1000張大股東持有百分比 超過閾值的股票，並整合近10日三大法人累計")
    parser.add_argument("--dir", type=str, default="stockholders", help="CSV 目錄 (default: stockholders)")
    parser.add_argument("--threshold", type=float, default=2.0, help="門檻：增加的百分點 (default: 2.0)")
    parser.add_argument("--out", type=str, default="report_stockholders_weekly_increase.csv", help="輸出彙整檔名")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"找不到目錄：{args.dir}")
        return

    results = []
    files = [f for f in os.listdir(args.dir) if f.lower().endswith(".csv")]
    if not files:
        print(f"{args.dir} 沒有 CSV 檔。")
        return

    for fname in tqdm(files, desc="Scanning", ncols=100):
        fpath = os.path.join(args.dir, fname)
        try:
            df = pd.read_csv(fpath, dtype={"代號": str})
        except Exception:
            continue

        if "代號" not in df.columns or "名稱" not in df.columns or "資料日期" not in df.columns:
            continue

        direct = [c for c in df.columns if str(c).replace(" ", "").replace("＞", ">") == ">1000張大股東持有百分比"]
        pct_col = direct[0] if direct else _find_percent_col(list(df.columns))
        if pct_col is None:
            continue

        code = str(df["代號"].iloc[0])
        name = str(df["名稱"].iloc[0])

        comp = compare_last_two_weeks(df, code, name, pct_col)
        if comp is None:
            continue

        code, name, last_date, last_pct, prev_date, prev_pct, delta = comp
        if delta > args.threshold:
            results.append({
                "代號": normalize_code(code),
                "名稱": name,
                "前週日期": prev_date,
                "前週>1000張大股東持有百分比(%)": prev_pct,
                "最近週日期": last_date,
                "最近週>1000張大股東持有百分比(%)": last_pct,
                "增加(百分點)": round(delta, 3),
            })

    if not results:
        print(f"沒有股票在最近一週相較前一週增加超過 {args.threshold} 個百分點。")
        return

    out_df = pd.DataFrame(results).sort_values(["增加(百分點)", "代號"], ascending=[False, True])
    out_df = enrich_with_3inst_10d(out_df)

    ordered_cols = [
        "代號",
        "名稱",
        "前週日期",
        "前週>1000張大股東持有百分比(%)",
        "最近週日期",
        "最近週>1000張大股東持有百分比(%)",
        "增加(百分點)",
        "外資近10日",
        "投信近10日",
        "自營近10日",
        "市場",
    ]
    out_df = out_df[[c for c in ordered_cols if c in out_df.columns]]

    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("\n符合條件的股票（依增幅由大到小）：")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(out_df.to_string(index=False))
    print(f"\n已輸出彙整：{args.out}")


if __name__ == "__main__":
    main()
