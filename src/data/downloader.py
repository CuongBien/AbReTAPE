import os
import time
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CIFAR10_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"


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
        except Exception:
            time.sleep(1)
    if os.path.getsize(chunk_file) != expected_len:
        raise RuntimeError(f"Failed chunk {chunk_file} after {max_retries} attempts")


def download_cifar10(dest_dir="./data", num_chunks=8):
    os.makedirs(dest_dir, exist_ok=True)
    extracted_marker = os.path.join(dest_dir, "cifar-10-batches-py")
    if os.path.exists(extracted_marker):
        print(f"[INFO] Dataset already extracted at: {extracted_marker}")
        return

    tar_path = os.path.join(dest_dir, "cifar-10-python.tar.gz")
    print(f"[INFO] Connecting to {CIFAR10_URL}...")
    total_size = get_file_size(CIFAR10_URL)
    print(f"[INFO] Total size: {total_size / (1024*1024):.2f} MB. Downloading using {num_chunks} parallel chunks with auto-resume...")

    chunk_size = total_size // num_chunks
    ranges = []
    chunk_files = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = total_size - 1 if i == num_chunks - 1 else (start + chunk_size - 1)
        chunk_file = os.path.join(dest_dir, f"part_{i}.tmp")
        ranges.append((start, end, chunk_file))
        chunk_files.append(chunk_file)

    with ThreadPoolExecutor(max_workers=num_chunks) as executor:
        futures = [executor.submit(download_chunk_with_retry, CIFAR10_URL, start, end, cf) for start, end, cf in ranges]
        for idx, f in enumerate(futures):
            f.result()
            print(f"  -> Completed chunk {idx + 1}/{num_chunks}")

    print(f"[INFO] Combining chunks into {tar_path}...")
    with open(tar_path, "wb") as outfile:
        for cf in chunk_files:
            with open(cf, "rb") as infile:
                outfile.write(infile.read())
            try:
                os.remove(cf)
            except Exception:
                pass

    print(f"[INFO] Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)

    print("[SUCCESS] CIFAR-10 downloaded and extracted successfully!")


if __name__ == "__main__":
    download_cifar10()
