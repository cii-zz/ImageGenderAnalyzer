import os
import pandas as pd
import requests
from urllib.parse import urlparse
import random

INPUT_EXCEL = "input.xlsx"
OUTPUT_DIR = "downloads"
MAX_IMAGES = 2000

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("读取 Excel ...")
df = pd.read_excel(INPUT_EXCEL)

# 收集所有 URL（打散前）
all_urls = []

for idx, row in df.iterrows():
    url_str = str(row.iloc[1]) if row.iloc[1] is not None else ""
    urls = [u.strip() for u in url_str.split(",") if u.strip().startswith("http")]
    all_urls.extend(urls)

# 去重
all_urls = list(set(all_urls))

print(f"去重后共有 {len(all_urls)} 个 URL")

# 随机打乱
random.shuffle(all_urls)

count = 0

for url in all_urls:
    if count >= MAX_IMAGES:
        break

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # 确定文件扩展名
            ext = os.path.splitext(urlparse(url).path)[1]
            if not ext:
                ext = ".jpg"

            filename = f"img_{count}{ext}"
            path = os.path.join(OUTPUT_DIR, filename)

            with open(path, "wb") as f:
                f.write(response.content)

            print(f"[{count}] 下载成功: {url}")
            count += 1
        else:
            print(f"下载失败（{response.status_code}）: {url}")

    except Exception as e:
        print(f"下载错误: {url} - {e}")

print(f"\n完成！共下载 {count} 张图片。")
