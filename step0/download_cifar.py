import os
import sys
import time
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TAR_PATH = os.path.join(DEST_DIR, "cifar-10-python.tar.gz")
NUM_CHUNKS = 8


def get_file_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return int(resp.headers["Content-Length"])


def download_chunk_with_retry(url, start, end, chunk_file, max_retries=10):
    expected_len = end - start + 1
    for attempt in range(max_retries):
        cur_pos = 0
        if os.path.exists(chunk_file):
            cur_pos = os.path.getsize(chunk_file)
            if cur_pos == expected_len:
                return
            elif cur_pos > expected_len:
                os.remove(chunk_file)
                cur_pos = 0

        req_start = start + cur_pos
        req = urllib.request.Request(url, headers={"Range": f"bytes={req_start}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp, open(chunk_file, "ab") as f:
                while True:
                    buf = resp.read(65536)
                    if not buf:
                        break
                    f.write(buf)
            if os.path.getsize(chunk_file) == expected_len:
                return
        except Exception as e:
            time.sleep(1)
    if os.path.getsize(chunk_file) != expected_len:
        raise RuntimeError(f"Failed chunk {chunk_file} after {max_retries} attempts")


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    extracted_marker = os.path.join(DEST_DIR, "cifar-10-batches-py")
    if os.path.exists(extracted_marker):
        print(f"[INFO] Dataset already extracted at: {extracted_marker}")
        return

    print(f"[INFO] Connecting to {URL}...")
    total_size = get_file_size(URL)
    print(f"[INFO] Total size: {total_size / (1024*1024):.2f} MB. Downloading using {NUM_CHUNKS} parallel chunks with auto-resume...")

    chunk_size = total_size // NUM_CHUNKS
    ranges = []
    chunk_files = []

    for i in range(NUM_CHUNKS):
        start = i * chunk_size
        end = total_size - 1 if i == NUM_CHUNKS - 1 else (start + chunk_size - 1)
        chunk_file = os.path.join(DEST_DIR, f"part_{i}.tmp")
        ranges.append((start, end, chunk_file))
        chunk_files.append(chunk_file)

    with ThreadPoolExecutor(max_workers=NUM_CHUNKS) as executor:
        futures = [executor.submit(download_chunk_with_retry, URL, start, end, cf) for start, end, cf in ranges]
        for idx, f in enumerate(futures):
            f.result()
            print(f"  -> Completed chunk {idx + 1}/{NUM_CHUNKS}")

    print(f"[INFO] Combining chunks into {TAR_PATH}...")
    with open(TAR_PATH, "wb") as outfile:
        for cf in chunk_files:
            with open(cf, "rb") as infile:
                outfile.write(infile.read())
            try:
                os.remove(cf)
            except Exception:
                pass

    print(f"[INFO] Extracting {TAR_PATH}...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(path=DEST_DIR)

    print("[SUCCESS] CIFAR-10 downloaded and extracted successfully!")


if __name__ == "__main__":
    main()
