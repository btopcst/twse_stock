import os
import time
import requests
from pathlib import Path

# =========================
# Telegram settings
# =========================
BOT_TOKEN = "8633158077:AAHkAOGGGbweEK_SpLOE4FmEKaXCQc0WUMU"
CHAT_ID = "8713102570"

# =========================
# Report path settings
# =========================
BASE_DIR = Path(".")
SUMMARY_FILE = BASE_DIR / "summary.xlsx"

CATEGORY_DIRS = {
    "布林突破": BASE_DIR / "stockData" / "BL",
    "KD黃金交叉": BASE_DIR / "stockData" / "KD_GC",
    "MACD轉強": BASE_DIR / "stockData" / "MACD",
    "回測季線": BASE_DIR / "stockData" / "backSeason",
    "月季年均線首向上": BASE_DIR / "stockData" / "月季年均線首向上",
}

# 每個分類最多傳幾張，避免洗版
MAX_IMAGES_PER_CATEGORY = 30

# 傳圖之間的延遲，避免 Telegram API 太快
SEND_DELAY_SEC = 1.0


def tg_api_url(method: str) -> str:
    return f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"


def send_message(text: str) -> bool:
    url = tg_api_url("sendMessage")
    data = {
        "chat_id": CHAT_ID,
        "text": text
    }
    r = requests.post(url, data=data, timeout=30)
    print("sendMessage:", r.status_code, r.text)
    return r.ok


def send_document(file_path: Path, caption: str = "") -> bool:
    if not file_path.exists():
        print(f"[WARN] File not found: {file_path}")
        return False

    url = tg_api_url("sendDocument")
    with open(file_path, "rb") as f:
        data = {
            "chat_id": CHAT_ID,
            "caption": caption
        }
        files = {
            "document": f
        }
        r = requests.post(url, data=data, files=files, timeout=120)

    print("sendDocument:", r.status_code, r.text)
    return r.ok


def send_photo(photo_path: Path, caption: str = "") -> bool:
    if not photo_path.exists():
        print(f"[WARN] Photo not found: {photo_path}")
        return False

    url = tg_api_url("sendPhoto")
    with open(photo_path, "rb") as f:
        data = {
            "chat_id": CHAT_ID,
            "caption": caption
        }
        files = {
            "photo": f
        }
        r = requests.post(url, data=data, files=files, timeout=120)

    print("sendPhoto:", r.status_code, r.text)
    return r.ok


def collect_images(folder: Path):
    if not folder.exists() or not folder.is_dir():
        return []

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]

    # 依檔名排序
    files.sort(key=lambda x: x.name.lower())
    return files


def build_daily_summary_text(category_files: dict) -> str:
    lines = []
    lines.append("📊 每日股票分析結果")
    lines.append("")

    if SUMMARY_FILE.exists():
        lines.append("✅ 已附上 summary.xlsx")
    else:
        lines.append("⚠️ 找不到 summary.xlsx")

    lines.append("")
    lines.append("各分類圖檔數量：")

    total = 0
    for category, files in category_files.items():
        count = len(files)
        total += count
        lines.append(f"- {category}: {count} 張")

    lines.append("")
    lines.append(f"總圖檔數: {total} 張")

    return "\n".join(lines)


def send_category_images(category: str, images: list[Path], max_count: int):
    if not images:
        send_message(f"📂 {category}\n今天沒有圖檔。")
        return

    send_message(f"📂 {category}\n共 {len(images)} 張，開始傳送前 {min(len(images), max_count)} 張。")

    for idx, img in enumerate(images[:max_count], start=1):
        caption = f"{category} ({idx}/{min(len(images), max_count)})\n{img.name}"
        ok = send_photo(img, caption=caption)
        if not ok:
            print(f"[ERROR] Failed to send image: {img}")
        time.sleep(SEND_DELAY_SEC)

    if len(images) > max_count:
        send_message(f"ℹ️ {category} 尚有 {len(images) - max_count} 張未傳送，避免洗版先略過。")


def main():
    # 1. 收集各分類圖檔
    category_files = {}
    for category, folder in CATEGORY_DIRS.items():
        category_files[category] = collect_images(folder)

    # 2. 傳送總結訊息
    summary_text = build_daily_summary_text(category_files)
    send_message(summary_text)
    time.sleep(SEND_DELAY_SEC)

    # 3. 傳送 summary.xlsx
    if SUMMARY_FILE.exists():
        send_document(SUMMARY_FILE, caption="📎 今日分析 summary.xlsx")
        time.sleep(SEND_DELAY_SEC)

    # 4. 逐類傳圖
    for category, images in category_files.items():
        send_category_images(category, images, MAX_IMAGES_PER_CATEGORY)
        time.sleep(SEND_DELAY_SEC)

    # 5. 結束通知
    send_message("✅ 今日 Telegram 報表傳送完成")


if __name__ == "__main__":
    main()