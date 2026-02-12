#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception as exc:  # pragma: no cover - hard dependency
    raise SystemExit(f"requests is required: {exc}")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "modelsweights"


def get_hf_mirror() -> str | None:
    value = os.getenv("HF_MIRROR") or os.getenv("HF_ENDPOINT") or os.getenv("HUGGINGFACE_HUB_ENDPOINT")
    if not value:
        return None
    value = value.strip().rstrip("/")
    if not value:
        return None
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    return value


def hf_hosts() -> set[str]:
    hosts = {"huggingface.co"}
    mirror = get_hf_mirror()
    if mirror:
        hosts.add(urlparse(mirror).netloc)
    return hosts


def rewrite_hf_url(url: str) -> str:
    mirror = get_hf_mirror()
    if not mirror:
        return url
    parsed = urlparse(url)
    if parsed.netloc == "huggingface.co":
        rebuilt = mirror + parsed.path
        if parsed.query:
            rebuilt += f"?{parsed.query}"
        return rebuilt
    return url


def _get_timeout() -> tuple[int, int]:
    connect_timeout = int(os.getenv("DOWNLOAD_CONNECT_TIMEOUT", "10"))
    read_timeout = int(os.getenv("DOWNLOAD_READ_TIMEOUT", "120"))
    return (connect_timeout, read_timeout)


def _get_chunk_size() -> int:
    mb = int(os.getenv("DOWNLOAD_CHUNK_MB", "1"))
    return max(1, mb) * 1024 * 1024


def _make_session() -> requests.Session:
    retries = int(os.getenv("DOWNLOAD_RETRIES", "3"))
    backoff = float(os.getenv("DOWNLOAD_BACKOFF", "0.5"))
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def load_env() -> None:
    if load_dotenv is None:
        return
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def format_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def head_size(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> int | None:
    url = rewrite_hf_url(url)
    try:
        resp = requests.head(url, allow_redirects=True, headers=headers, timeout=timeout)
        if resp.ok and resp.headers.get("content-length"):
            return int(resp.headers["content-length"])
    except Exception:
        pass

    try:
        range_headers = {"Range": "bytes=0-0"}
        if headers:
            range_headers.update(headers)
        resp = requests.get(url, stream=True, headers=range_headers, timeout=timeout)
        if resp.status_code in (200, 206):
            content_range = resp.headers.get("content-range")
            if content_range and "/" in content_range:
                return int(content_range.split("/")[-1])
    except Exception:
        pass

    return None


def default_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or "download.bin"


def hf_repo_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.netloc not in hf_hosts():
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None
    return "/".join(parts[:2])


def hf_repo_name(repo_id: str | None) -> str | None:
    if not repo_id:
        return None
    return repo_id.split("/")[-1]

def download_stream(url: str, dest: Path, headers: dict[str, str] | None = None, force: bool = False) -> None:
    url = rewrite_hf_url(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return

    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    resume_from = 0
    if tmp_dest.exists() and not force:
        resume_from = tmp_dest.stat().st_size
    elif tmp_dest.exists():
        tmp_dest.unlink()

    request_headers = dict(headers or {})
    if resume_from > 0:
        request_headers["Range"] = f"bytes={resume_from}-"

    timeout = _get_timeout()
    chunk_size = _get_chunk_size()
    session = _make_session()

    with session.get(url, stream=True, headers=request_headers, timeout=timeout) as resp:
        if resp.status_code == 416 and resume_from > 0:
            tmp_dest.rename(dest)
            return
        if resume_from > 0 and resp.status_code == 200:
            resume_from = 0
            tmp_dest.unlink(missing_ok=True)
        resp.raise_for_status()

        total = None
        content_range = resp.headers.get("content-range")
        if content_range and "/" in content_range:
            try:
                total = int(content_range.split("/")[-1])
            except Exception:
                total = None
        if total is None and resp.headers.get("content-length"):
            total = int(resp.headers["content-length"])
            if resume_from > 0:
                total += resume_from

        if tqdm is not None and total and total > 0:
            pbar = tqdm(total=total, initial=resume_from, unit="B", unit_scale=True, desc=dest.name)
        else:
            pbar = None

        mode = "ab" if resume_from > 0 else "wb"
        with open(tmp_dest, mode) as handle:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                if pbar is not None:
                    pbar.update(len(chunk))

        if pbar is not None:
            pbar.close()

    tmp_dest.rename(dest)


def extract_gdrive_id(value: str) -> str | None:
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", value or ""):
        return value
    parsed = urlparse(value)
    if parsed.netloc.endswith("drive.google.com"):
        if parsed.path.startswith("/file/"):
            parts = parsed.path.split("/")
            if len(parts) > 3:
                return parts[3]
        qs = parse_qs(parsed.query)
        if "id" in qs:
            return qs["id"][0]
    return None


def download_gdrive(file_id: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return

    session = _make_session()
    base_url = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}
    resp = session.get(base_url, params=params, stream=True, timeout=_get_timeout())
    token = None
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        params["confirm"] = token
        resp = session.get(base_url, params=params, stream=True, timeout=_get_timeout())

    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    if tqdm is not None and total > 0:
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
    else:
        pbar = None

    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    if tmp_dest.exists():
        tmp_dest.unlink()

    chunk_size = _get_chunk_size()
    with open(tmp_dest, "wb") as handle:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            handle.write(chunk)
            if pbar is not None:
                pbar.update(len(chunk))

    if pbar is not None:
        pbar.close()

    tmp_dest.rename(dest)


def maybe_snapshot_hf(repo_id: str, target_dir: Path, token: str | None = None, force: bool = False) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print("huggingface_hub is not installed; cannot snapshot Hugging Face repos.")
        return False

    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        return True

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    return True


def download_meddinov3(models_dir: Path, check_size_only: bool, force: bool) -> list[tuple[str, Path, int | None]]:
    results: list[tuple[str, Path, int | None]] = []
    med_dir = models_dir / "meddinov3"
    default_subdir = os.getenv("MEDDINOV3_SUBDIR") or "MedDINOv3-ViTB-16-CT-3M"

    gdrive_value = os.getenv("MEDDINOV3_GDRIVE_FILE_ID") or os.getenv("MEDDINOV3_GDRIVE_URL")
    if gdrive_value:
        file_id = extract_gdrive_id(gdrive_value)
        if file_id:
            filename = os.getenv("MEDDINOV3_GDRIVE_FILENAME") or "meddinov3_vitb16_ct3m.pth"
            dest = med_dir / default_subdir / filename
            if check_size_only:
                size = None
            else:
                download_gdrive(file_id, dest, force=force)
                size = path_size_bytes(dest)
            results.append(("meddinov3:gdrive", dest, size))

    url = os.getenv("MEDDINOV3_URL")
    if url:
        repo_id = hf_repo_from_url(url)
        subdir = hf_repo_name(repo_id) or default_subdir
        filename = os.getenv("MEDDINOV3_FILENAME") or default_filename_from_url(url)
        dest = med_dir / subdir / filename
        size = head_size(url)
        if check_size_only:
            results.append(("meddinov3:url", dest, size))
        else:
            download_stream(url, dest, force=force)
            results.append(("meddinov3:url", dest, path_size_bytes(dest)))

    hf_repo = os.getenv("MEDDINOV3_HF_REPO")
    if hf_repo:
        token = os.getenv("HUGGINGFACE_TOKEN")
        repo_dir = med_dir / (hf_repo_name(hf_repo) or hf_repo.replace("/", "__"))
        ok = maybe_snapshot_hf(hf_repo, repo_dir, token=token, force=force)
        if ok:
            results.append((f"meddinov3:hf:{hf_repo}", repo_dir, path_size_bytes(repo_dir)))
        else:
            results.append((f"meddinov3:hf:{hf_repo}", repo_dir, None))

    dino_url = os.getenv("DINOV3_URL")
    if dino_url:
        repo_id = hf_repo_from_url(dino_url)
        subdir = hf_repo_name(repo_id) or "dinov3"
        filename = os.getenv("DINOV3_FILENAME") or default_filename_from_url(dino_url)
        dest = med_dir / subdir / filename
        size = head_size(dino_url)
        if check_size_only:
            results.append(("dinov3:url", dest, size))
        else:
            download_stream(dino_url, dest, force=force)
            results.append(("dinov3:url", dest, path_size_bytes(dest)))

    return results


def render_results(results: Iterable[tuple[str, Path, int | None]]) -> None:
    for name, path, size in results:
        if size is None:
            size_str = "unknown"
        else:
            size_str = format_bytes(size)
        print(f"{name}: {path} ({size_str})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download third-party model weights into modelsweights/")
    parser.add_argument("--package", action="append", choices=["meddinov3"], help="Package to download")
    parser.add_argument("--check-size", action="store_true", help="Only check remote size (no download)")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    load_env()

    models_dir = Path(os.getenv("MODELS_DIR") or DEFAULT_MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    packages = args.package or ["meddinov3"]
    all_results: list[tuple[str, Path, int | None]] = []

    if "meddinov3" in packages:
        all_results.extend(download_meddinov3(models_dir, args.check_size, args.force))

    render_results(all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
