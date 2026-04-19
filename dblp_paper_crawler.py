from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from openai import OpenAI
from rapidfuzz import fuzz
from tqdm import tqdm

NA = "N/A"
DEFAULT_CATEGORIES = [
    "大语言模型安全",
    "隐私保护",
    "强化学习对齐",
    "智能体安全",
    "数据泄露攻击",
    "提示注入与越狱",
    "模型隐私攻击",
    "差分隐私",
    "联邦学习安全",
    "其他",
]
USER_AGENT = (
    "dblp-paper-crawler/1.0 "
    "(https://dblp.org/; academic metadata enrichment; contact: local-user)"
)
DEFAULT_RANDOM_USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Edg/124.0.2478.80 Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
]
DBLP_DEFAULT_BASE_URL = "https://dblp.org"
CROSSREF_WORKS_API = "https://api.crossref.org/works"
OPENALEX_WORKS_API = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_PAPER_API = "https://api.semanticscholar.org/graph/v1/paper"
SEMANTIC_SCHOLAR_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API = "https://export.arxiv.org/api/query"
RESTART_STAGE_ORDER = {
    "fetch": 0,
    "match": 1,
    "detail": 2,
    "abstract": 3,
    "affiliation": 4,
    "llm": 5,
}
RESTART_STAGE_ALIASES = {
    "fetch": "fetch",
    "crawl": "fetch",
    "dblp": "fetch",
    "match": "match",
    "matching": "match",
    "filter": "match",
    "detail": "detail",
    "metadata": "detail",
    "abstract": "abstract",
    "summary-input": "abstract",
    "affiliation": "affiliation",
    "affiliations": "affiliation",
    "llm": "llm",
    "ai": "llm",
}


def parse_restart_stage(value: str) -> str:
    stage = RESTART_STAGE_ALIASES.get(str(value).strip().lower())
    if not stage:
        supported = ", ".join(RESTART_STAGE_ORDER.keys())
        raise argparse.ArgumentTypeError(
            f"Unsupported restart stage: {value!r}. Use one of: {supported}."
        )
    return stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl DBLP papers, match titles, enrich metadata, and export CSV."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file. Defaults to ./config.yaml.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Do not call the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--test-ai",
        action="store_true",
        help="Only test whether the OpenAI-compatible API configuration works, then exit.",
    )
    parser.add_argument(
        "--restart-from",
        type=parse_restart_stage,
        default=None,
        help=(
            "Restart from a specific stage and rewrite cache for that stage and later. "
            "Supported stages: fetch, match, detail, abstract, affiliation, llm. "
            "Aliases such as crawl->fetch are also accepted."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N matched papers.",
    )
    parser.add_argument(
        "--randomize-ua",
        "--randomize-user-agent",
        action="store_true",
        help=(
            "Pick a random browser-like User-Agent, write it back to request.user_agent "
            "in the config file, and use it for this run."
        ),
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_language_code(value: Any, default: str = "zh") -> str:
    text = clean_text(value).lower()
    if text in {"zh", "zh-cn", "zh_hans", "cn", "chinese", "中文"}:
        return "zh"
    if text in {"en", "en-us", "en_gb", "english", "英文"}:
        return "en"
    return default


def extract_override_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("stream_query", "stream", "venue_url", "url", "override"):
            candidate = value.get(key)
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text:
                return text
        return ""
    if isinstance(value, bool):
        return ""
    text = str(value).strip()
    return text


def normalize_venues_config(raw_venues: Any) -> tuple[List[str], Dict[str, str]]:
    venues: List[str] = []
    implicit_overrides: Dict[str, str] = {}

    if isinstance(raw_venues, dict):
        for raw_name, raw_value in raw_venues.items():
            venue_name = str(raw_name).strip()
            if not venue_name:
                continue

            enabled = True
            if isinstance(raw_value, bool):
                enabled = raw_value
            elif isinstance(raw_value, dict):
                enabled = bool(raw_value.get("enabled", True))

            if not enabled:
                continue

            venues.append(venue_name)
            override_value = extract_override_value(raw_value)
            if override_value:
                implicit_overrides[venue_name] = override_value
        return venues, implicit_overrides

    if isinstance(raw_venues, str):
        text = raw_venues.strip()
        return ([text] if text else []), {}

    if isinstance(raw_venues, list):
        for item in raw_venues:
            if isinstance(item, dict) and len(item) == 1:
                raw_name, raw_value = next(iter(item.items()))
                venue_name = str(raw_name).strip()
                if not venue_name:
                    continue
                venues.append(venue_name)
                override_value = extract_override_value(raw_value)
                if override_value:
                    implicit_overrides[venue_name] = override_value
                continue

            venue_name = str(item).strip()
            if venue_name:
                venues.append(venue_name)
        return venues, implicit_overrides

    return [], {}


def normalize_venue_stream_overrides(raw_overrides: Any) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not isinstance(raw_overrides, dict):
        return overrides

    for raw_name, raw_value in raw_overrides.items():
        venue_name = str(raw_name).strip()
        if not venue_name:
            continue
        override_value = extract_override_value(raw_value)
        if override_value:
            overrides[venue_name] = override_value
    return overrides


def normalize_base_url(value: Any, default: str = DBLP_DEFAULT_BASE_URL) -> str:
    text = clean_text(value).strip() or default
    if not re.match(r"^https?://", text, flags=re.IGNORECASE):
        text = f"https://{text.lstrip('/')}"
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("dblp.base_url must be a valid http(s) URL")
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def build_dblp_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def get_dblp_base_path(dblp_base_url: str) -> str:
    text = clean_text(dblp_base_url)
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else f"https://{text.lstrip('/')}")
    return parsed.path.rstrip("/")


def strip_dblp_base_path(path: str, dblp_base_url: str) -> str:
    current_path = clean_text(path)
    base_path = get_dblp_base_path(dblp_base_url)
    if not current_path or not base_path:
        return current_path
    if current_path == base_path:
        return "/"
    if current_path.startswith(f"{base_path}/"):
        stripped = current_path[len(base_path) :]
        return stripped or "/"
    return current_path


def rewrite_dblp_url(url: Any, dblp_base_url: str) -> str:
    text = clean_text(url)
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        if not is_dblp_url(text, dblp_base_url):
            return text
        rewritten = build_dblp_url(dblp_base_url, strip_dblp_base_path(parsed.path, dblp_base_url))
        if parsed.query:
            rewritten = f"{rewritten}?{parsed.query}"
        if parsed.fragment:
            rewritten = f"{rewritten}#{parsed.fragment}"
        return rewritten
    return build_dblp_url(dblp_base_url, strip_dblp_base_path(text, dblp_base_url))


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config: Dict[str, Any] = {
        "dblp": raw.get("dblp", {}) or {},
        "match_rules": raw.get("match_rules", []) or [],
        "classification": raw.get("classification", {}) or {},
        "openai": raw.get("openai", {}) or {},
        "llm_output": raw.get("llm_output", {}) or {},
        "output": raw.get("output", {}) or {},
        "cache": raw.get("cache", {}) or {},
        "request": raw.get("request", {}) or {},
    }

    dblp_cfg = config["dblp"]
    dblp_cfg["base_url"] = normalize_base_url(dblp_cfg.get("base_url", DBLP_DEFAULT_BASE_URL))
    dblp_cfg["venue_api_url"] = build_dblp_url(dblp_cfg["base_url"], "search/venue/api")
    dblp_cfg["publ_api_url"] = build_dblp_url(dblp_cfg["base_url"], "search/publ/api")
    venues, implicit_overrides = normalize_venues_config(dblp_cfg.get("venues", []))
    dblp_cfg["venues"] = venues
    dblp_cfg["year_start"] = int(dblp_cfg.get("year_start", 2000))
    dblp_cfg["year_end"] = int(dblp_cfg.get("year_end", dblp_cfg["year_start"]))
    explicit_overrides = normalize_venue_stream_overrides(
        dblp_cfg.get("venue_stream_overrides", {}) or {}
    )
    dblp_cfg["venue_stream_overrides"] = {**implicit_overrides, **explicit_overrides}
    if dblp_cfg["year_start"] > dblp_cfg["year_end"]:
        raise ValueError("dblp.year_start cannot be greater than dblp.year_end")
    if not dblp_cfg["venues"]:
        raise ValueError("dblp.venues cannot be empty")

    config["match_rules"] = [
        [str(term) for term in group if str(term).strip()]
        for group in config["match_rules"]
        if isinstance(group, list) and group
    ]

    classification_cfg = config["classification"]
    classification_cfg["categories"] = list(
        classification_cfg.get("categories") or DEFAULT_CATEGORIES
    )
    if "其他" not in classification_cfg["categories"]:
        classification_cfg["categories"].append("其他")
    classification_cfg["allow_new_category"] = bool(
        classification_cfg.get("allow_new_category", True)
    )

    openai_cfg = config["openai"]
    openai_cfg["host"] = str(openai_cfg.get("host", "https://api.openai.com/v1")).rstrip("/")
    openai_cfg["api_key"] = str(openai_cfg.get("api_key", "") or "").strip()
    openai_cfg["model"] = str(openai_cfg.get("model", "gpt-4o-mini"))
    openai_cfg["temperature"] = float(openai_cfg.get("temperature", 0.2))
    openai_cfg["max_tokens"] = int(openai_cfg.get("max_tokens", 800))
    openai_cfg["max_retries"] = int(openai_cfg.get("max_retries", 3))

    llm_output_cfg = config["llm_output"]
    llm_output_cfg["title_translation_enabled"] = bool(
        llm_output_cfg.get("title_translation_enabled", False)
    )
    llm_output_cfg["summary_language"] = normalize_language_code(
        llm_output_cfg.get("summary_language", "zh"),
        default="zh",
    )

    output_cfg = config["output"]
    raw_csv_dir = output_cfg.get("csv_dir")
    raw_csv_path = output_cfg.get("csv_path")
    if raw_csv_dir:
        output_cfg["csv_dir"] = str(resolve_path(path.parent, raw_csv_dir))
    elif raw_csv_path:
        legacy_path = resolve_path(path.parent, raw_csv_path)
        output_cfg["csv_dir"] = str(legacy_path.parent if legacy_path.suffix else legacy_path)
    else:
        output_cfg["csv_dir"] = str(resolve_path(path.parent, "./outputs"))

    cache_cfg = config["cache"]
    cache_cfg["enabled"] = bool(cache_cfg.get("enabled", True))
    cache_cfg["path"] = str(resolve_path(path.parent, cache_cfg.get("path", "./cache/papers_cache.jsonl")))
    cache_cfg["publ_query_enabled"] = bool(cache_cfg.get("publ_query_enabled", cache_cfg["enabled"]))
    cache_cfg["publ_query_path"] = str(
        resolve_path(path.parent, cache_cfg.get("publ_query_path", "./cache/dblp_publ_cache.json"))
    )
    cache_cfg["publ_query_current_year_ttl_hours"] = float(
        cache_cfg.get("publ_query_current_year_ttl_hours", 24)
    )
    cache_cfg["publ_query_max_refetch_rounds"] = max(
        0,
        int(cache_cfg.get("publ_query_max_refetch_rounds", 2)),
    )

    request_cfg = config["request"]
    request_cfg["user_agent"] = clean_text(request_cfg.get("user_agent")) or USER_AGENT
    request_cfg["sleep_seconds"] = float(request_cfg.get("sleep_seconds", 1))
    jitter_min, jitter_max = normalize_sleep_jitter_range(
        request_cfg.get("sleep_jitter_range", [0.0, 0.25])
    )
    request_cfg["sleep_jitter_range"] = [jitter_min, jitter_max]
    request_cfg["sleep_jitter_min_seconds"] = jitter_min
    request_cfg["sleep_jitter_max_seconds"] = jitter_max
    request_cfg["timeout_seconds"] = float(request_cfg.get("timeout_seconds", 20))
    request_cfg["max_retries"] = int(request_cfg.get("max_retries", 3))

    return config


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def normalize_sleep_jitter_range(value: Any) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.25

    lower: float
    upper: float
    if isinstance(value, dict):
        lower = float(value.get("min", 0.0))
        upper = float(value.get("max", lower))
    elif isinstance(value, (list, tuple)):
        if len(value) < 2:
            raise ValueError("request.sleep_jitter_range must contain two numeric values")
        lower = float(value[0])
        upper = float(value[1])
    else:
        upper = float(value)
        lower = 0.0 if upper >= 0 else upper

    return (lower, upper) if lower <= upper else (upper, lower)


def choose_random_user_agent(current_user_agent: str = "") -> str:
    current = clean_text(current_user_agent)
    candidates = [item for item in DEFAULT_RANDOM_USER_AGENTS if clean_text(item) != current]
    if not candidates:
        candidates = list(DEFAULT_RANDOM_USER_AGENTS)
    return random.choice(candidates)


def yaml_quote_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def persist_request_user_agent_to_config(config_path: str, user_agent: str) -> Path:
    path = Path(config_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    serialized = yaml_quote_string(user_agent)

    request_idx: Optional[int] = None
    request_indent = ""
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^request\s*:\s*(?:#.*)?$", stripped):
            request_idx = idx
            request_indent = line[: len(line) - len(line.lstrip())]
            break

    if request_idx is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(
            [
                "request:",
                f"  user_agent: {serialized}",
            ]
        )
    else:
        child_indent = f"{request_indent}  "
        section_end = len(lines)
        for idx in range(request_idx + 1, len(lines)):
            stripped = lines[idx].strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
            if len(indent) <= len(request_indent):
                section_end = idx
                break

        replaced = False
        for idx in range(request_idx + 1, section_end):
            stripped = lines[idx].strip()
            if re.match(r"^user_agent\s*:\s*.*$", stripped):
                lines[idx] = f"{child_indent}user_agent: {serialized}"
                replaced = True
                break

        if not replaced:
            insert_idx = request_idx + 1
            while insert_idx < section_end and not lines[insert_idx].strip():
                insert_idx += 1
            lines.insert(insert_idx, f"{child_indent}user_agent: {serialized}")

    trailing_newline = "\n" if text.endswith("\n") or not text else ""
    path.write_text("\n".join(lines) + trailing_newline, encoding="utf-8")
    return path


def build_requests_session(request_cfg: Dict[str, Any]) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": request_cfg["user_agent"],
            "Accept": "application/json, text/html, application/xml, text/xml;q=0.9",
        }
    )
    return session


def build_openai_client(config: Dict[str, Any]) -> Optional[OpenAI]:
    openai_cfg = config["openai"]
    if not openai_cfg.get("api_key") or openai_cfg["api_key"] == "YOUR_API_KEY":
        return None
    return OpenAI(
        api_key=openai_cfg["api_key"],
        base_url=openai_cfg["host"],
        max_retries=0,
    )


def ensure_parent_directory(path_str: str) -> None:
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def pause_request(request_cfg: Dict[str, Any], attempt: int = 0) -> None:
    base_sleep = max(0.0, float(request_cfg.get("sleep_seconds", 0)))
    jitter_min = float(request_cfg.get("sleep_jitter_min_seconds", 0.0))
    jitter_max = float(request_cfg.get("sleep_jitter_max_seconds", 0.25))
    jitter = random.uniform(jitter_min, jitter_max)
    delay = max(0.0, base_sleep + min(attempt, 3) * 0.5 + jitter)
    if delay > 0:
        time.sleep(delay)


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
    **kwargs: Any,
) -> Optional[requests.Response]:
    timeout = kwargs.pop("timeout", request_cfg.get("timeout_seconds", 20))
    max_retries = int(request_cfg.get("max_retries", 3))
    retriable_codes = {408, 409, 425, 429, 500, 502, 503, 504}
    response: Optional[requests.Response] = None

    for attempt in range(max_retries + 1):
        try:
            response = session.request(method=method, url=url, timeout=timeout, **kwargs)
            if response.status_code < 400:
                pause_request(request_cfg, attempt)
                return response
            if response.status_code not in retriable_codes or attempt >= max_retries:
                logger.warning("Request failed: %s %s -> HTTP %s", method, url, response.status_code)
                pause_request(request_cfg, attempt)
                return response
            logger.warning(
                "Transient HTTP error for %s %s -> %s, retry %s/%s",
                method,
                url,
                response.status_code,
                attempt + 1,
                max_retries,
            )
        except requests.RequestException as exc:
            logger.warning(
                "Request exception for %s %s on attempt %s/%s: %s",
                method,
                url,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            if attempt >= max_retries:
                pause_request(request_cfg, attempt)
                return None
        pause_request(request_cfg, attempt)
    return response


def request_json(
    session: requests.Session,
    url: str,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    response = request_with_retries(session, "GET", url, request_cfg, logger, **kwargs)
    if response is None or response.status_code >= 400:
        return None
    try:
        return response.json()
    except ValueError:
        logger.warning("Failed to parse JSON response from %s", url)
        return None


def request_json_with_status(
    session: requests.Session,
    url: str,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
    **kwargs: Any,
) -> tuple[Optional[Dict[str, Any]], str]:
    response = request_with_retries(session, "GET", url, request_cfg, logger, **kwargs)
    if response is None:
        return None, "request_failed"
    if response.status_code >= 400:
        if response.status_code == 404:
            return None, "not_found"
        return None, "request_failed"
    try:
        return response.json(), "success"
    except ValueError:
        logger.warning("Failed to parse JSON response from %s", url)
        return None, "parse_failed"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "lxml").get_text(" ", strip=True)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_title(title: str) -> str:
    text = clean_text(title).lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_person_name(name: str) -> str:
    text = clean_text(name).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_doi(doi: Optional[str]) -> str:
    if not doi:
        return ""
    value = clean_text(doi).strip()
    value = value.replace("https://doi.org/", "").replace("http://doi.org/", "")
    value = value.replace("doi:", "")
    value = value.strip().lower()
    if value in {"", "n/a", "na", "none", "null", "pending"}:
        return ""
    return value


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"\d{4}", str(value))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = clean_text(value)
        if text and text != NA:
            return text
    return NA


def make_fingerprint(*parts: Any) -> str:
    hasher = hashlib.sha1()
    for part in parts:
        hasher.update(clean_text(part).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def extract_dblp_record_key(url: Any) -> str:
    text = clean_text(url)
    if not text or text == NA:
        return ""
    path = urlparse(text).path or text
    if "/rec/" in path:
        key = path.split("/rec/", 1)[1]
    elif path.startswith("rec/"):
        key = path[len("rec/") :]
    else:
        return ""
    key = key.strip("/")
    if key.endswith(".xml"):
        key = key[:-4]
    return key


def compute_primary_dedupe_key(record: Dict[str, Any]) -> str:
    doi = normalize_doi(record.get("doi"))
    if doi:
        return f"doi:{doi}"
    dblp_record_key = extract_dblp_record_key(record.get("dblp_url"))
    if dblp_record_key:
        return f"dblp_key:{dblp_record_key}"
    dblp_url = clean_text(record.get("dblp_url"))
    if dblp_url and dblp_url != NA:
        return f"dblp:{dblp_url}"
    normalized_title = normalize_title(record.get("title", ""))
    if normalized_title:
        return f"title:{normalized_title}"
    return ""


def make_record_keys(record: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    doi = normalize_doi(record.get("doi"))
    dblp_record_key = extract_dblp_record_key(record.get("dblp_url"))
    dblp_url = clean_text(record.get("dblp_url"))
    normalized_title = normalize_title(record.get("title", ""))
    if doi:
        keys.append(f"doi:{doi}")
    if dblp_record_key:
        keys.append(f"dblp_key:{dblp_record_key}")
    if dblp_url and dblp_url != NA:
        keys.append(f"dblp:{dblp_url}")
    if normalized_title:
        keys.append(f"title:{normalized_title}")
    return keys


def compute_record_identity(record: Dict[str, Any]) -> str:
    return compute_primary_dedupe_key(record)


def compute_abstract_signature(record: Dict[str, Any]) -> str:
    return make_fingerprint(
        compute_record_identity(record),
        record.get("title"),
        record.get("paper_url"),
    )


def compute_affiliation_signature(record: Dict[str, Any]) -> str:
    return make_fingerprint(
        compute_record_identity(record),
        *to_list(record.get("authors")),
    )


def compute_llm_signature(record: Dict[str, Any], config: Dict[str, Any]) -> str:
    return make_fingerprint(
        compute_record_identity(record),
        record.get("title"),
        record.get("abstract"),
        config["llm_output"]["summary_language"],
        str(bool(config["llm_output"]["title_translation_enabled"])),
        json.dumps(config["classification"]["categories"], ensure_ascii=False),
        str(bool(config["classification"]["allow_new_category"])),
    )


def rewrite_record_dblp_urls(record: Dict[str, Any], dblp_base_url: str) -> Dict[str, Any]:
    if not dblp_base_url:
        return record
    for key in ("dblp_url", "paper_url", "venue_url"):
        value = clean_text(record.get(key))
        if not value or not is_dblp_url(value, dblp_base_url):
            continue
        record[key] = rewrite_dblp_url(value, dblp_base_url)
    return record


def sanitize_record_for_cache(record: Dict[str, Any], dblp_base_url: str = "") -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in record.items():
        if key.startswith("_"):
            continue
        if isinstance(value, Path):
            clean[key] = str(value)
        else:
            clean[key] = value
    rewrite_record_dblp_urls(clean, dblp_base_url)
    clean["normalized_title"] = normalize_title(clean.get("title", ""))
    clean["doi"] = normalize_doi(clean.get("doi"))
    clean["dedupe_key"] = compute_primary_dedupe_key(clean)
    return clean


def load_cache(cache_path: str, dblp_base_url: str = "") -> Dict[str, Dict[str, Any]]:
    path = Path(cache_path).expanduser().resolve()
    if not path.exists():
        return {}

    index: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            normalized = sanitize_record_for_cache(record, dblp_base_url)
            for key in make_record_keys(normalized):
                index[key] = normalized
    return index


def lookup_cached_record(
    cache_index: Dict[str, Dict[str, Any]],
    record: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    for key in make_record_keys(record):
        if key in cache_index:
            return dict(cache_index[key])
    return None


def append_cache_record(
    record: Dict[str, Any],
    cache_path: str,
    cache_index: Dict[str, Dict[str, Any]],
    enabled: bool,
    dblp_base_url: str = "",
) -> None:
    if not enabled:
        return
    normalized = sanitize_record_for_cache(record, dblp_base_url)
    if not normalized.get("dedupe_key"):
        return

    ensure_parent_directory(cache_path)
    with Path(cache_path).expanduser().resolve().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    for key in make_record_keys(normalized):
        cache_index[key] = normalized


def make_publ_query_cache_key(venue_name: str, stream_query: str, year: int) -> str:
    return make_fingerprint(normalize_title(venue_name), clean_text(stream_query), str(year))


def sanitize_publ_query_cache_entry(
    entry: Dict[str, Any],
    dblp_base_url: str = "",
) -> Optional[Dict[str, Any]]:
    venue_name = clean_text(entry.get("venue_name"))
    stream_query = clean_text(entry.get("stream_query"))
    year = safe_int(entry.get("year"))
    if not venue_name or not stream_query or year is None:
        return None

    records = [
        sanitize_record_for_cache(record, dblp_base_url)
        for record in to_list(entry.get("records"))
        if isinstance(record, dict)
    ]
    return {
        "venue_name": venue_name,
        "stream_query": stream_query,
        "year": year,
        "records": records,
        "complete": bool(entry.get("complete", False)),
        "updated_at": safe_float(entry.get("updated_at")) or 0.0,
    }


def load_publ_query_cache(cache_path: str, dblp_base_url: str = "") -> Dict[str, Dict[str, Any]]:
    path = Path(cache_path).expanduser().resolve()
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}

    entries: Iterable[Any]
    if isinstance(payload, dict):
        entries = payload.values()
    elif isinstance(payload, list):
        entries = payload
    else:
        return {}

    index: Dict[str, Dict[str, Any]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        normalized = sanitize_publ_query_cache_entry(raw_entry, dblp_base_url)
        if not normalized:
            continue
        key = make_publ_query_cache_key(
            normalized["venue_name"],
            normalized["stream_query"],
            normalized["year"],
        )
        index[key] = normalized
    return index


def persist_publ_query_cache(cache_path: str, cache_index: Dict[str, Dict[str, Any]]) -> None:
    path = Path(cache_path).expanduser().resolve()
    ensure_parent_directory(str(path))
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(cache_index, handle, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path.replace(path)


def lookup_publ_query_cache_entry(
    cache_index: Dict[str, Dict[str, Any]],
    venue_name: str,
    stream_query: str,
    year: int,
) -> Optional[Dict[str, Any]]:
    key = make_publ_query_cache_key(venue_name, stream_query, year)
    entry = cache_index.get(key)
    return dict(entry) if entry else None


def clone_publ_query_cache_records(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(record) for record in entry.get("records", []) if isinstance(record, dict)]


def save_publ_query_cache_entry(
    entry: Dict[str, Any],
    cache_path: str,
    cache_index: Dict[str, Dict[str, Any]],
    enabled: bool,
    dblp_base_url: str = "",
) -> None:
    if not enabled:
        return
    normalized = sanitize_publ_query_cache_entry(entry, dblp_base_url)
    if not normalized:
        return
    key = make_publ_query_cache_key(
        normalized["venue_name"],
        normalized["stream_query"],
        normalized["year"],
    )
    cache_index[key] = normalized
    persist_publ_query_cache(cache_path, cache_index)


def can_reuse_publ_query_cache_entry(
    entry: Optional[Dict[str, Any]],
    ttl_hours: float,
    now: Optional[float] = None,
) -> bool:
    if not entry or not entry.get("complete"):
        return False

    year = safe_int(entry.get("year"))
    if year is None:
        return False

    current_time = time.time() if now is None else now
    current_year = time.localtime(current_time).tm_year
    if year < current_year:
        return True

    if ttl_hours <= 0:
        return True

    updated_at = safe_float(entry.get("updated_at"))
    if updated_at is None or updated_at <= 0:
        return False
    return (current_time - updated_at) <= ttl_hours * 3600


def normalize_hostname(url: str) -> str:
    text = clean_text(url)
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else f"https://{text}")
    host = (parsed.netloc or parsed.path).lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def is_dblp_host(value: str, configured_base_url: str = "") -> bool:
    host = normalize_hostname(value)
    if not host:
        return False
    configured_host = normalize_hostname(configured_base_url)
    known_hosts = {"dblp.org", "dblp.uni-trier.de"}
    if configured_host:
        known_hosts.add(configured_host)
    return host in known_hosts or host.endswith(".dblp.org") or host.startswith("dblp.")


def is_dblp_url(value: str, configured_base_url: str = "") -> bool:
    text = clean_text(value)
    if not text:
        return False

    parsed = urlparse(text if "://" in text else f"https://example.invalid/{text.lstrip('/')}")
    host = normalize_hostname(parsed.netloc)
    if is_dblp_host(host):
        return True

    configured_host = normalize_hostname(configured_base_url)
    if not configured_host or host != configured_host:
        return False

    base_path = get_dblp_base_path(configured_base_url)
    if not base_path:
        return True
    current_path = parsed.path or "/"
    return current_path == base_path or current_path.startswith(f"{base_path}/")


def is_doi_url(url: str) -> bool:
    host = normalize_hostname(url)
    return host == "doi.org" or host.endswith(".doi.org")


def is_metadata_url(url: str, dblp_base_url: str = "") -> bool:
    host = normalize_hostname(url)
    if is_dblp_url(url, dblp_base_url):
        return True
    metadata_hosts = {
        "wikidata.org",
        "openalex.org",
        "crossref.org",
        "semanticscholar.org",
        "api.openalex.org",
    }
    return any(host == candidate or host.endswith(f".{candidate}") for candidate in metadata_hosts)


def is_fuzzy_match(text: str, keyword: str) -> bool:
    normalized_text = normalize_title(text)
    normalized_keyword = normalize_title(keyword)
    if not normalized_text or not normalized_keyword:
        return False

    if normalized_keyword in normalized_text:
        return True

    token_count = len(normalized_keyword.split())
    keyword_length = len(normalized_keyword)

    partial_score = fuzz.partial_ratio(normalized_keyword, normalized_text)
    token_score = fuzz.token_set_ratio(normalized_keyword, normalized_text)
    ratio_score = fuzz.ratio(normalized_keyword, normalized_text)

    if token_count >= 2:
        return token_score >= 95
    if keyword_length <= 4:
        return partial_score >= 96 or ratio_score >= 94
    if keyword_length <= 8:
        return partial_score >= 92 or ratio_score >= 88
    return partial_score >= 88 or token_score >= 86


def match_title(title: str, rules: List[List[str]]) -> bool:
    if not rules:
        return True
    return all(any(is_fuzzy_match(title, keyword) for keyword in group) for group in rules)


def extract_search_hits(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not payload:
        return []
    result = payload.get("result", {})
    hits = result.get("hits", {}).get("hit", [])
    if isinstance(hits, dict):
        return [hits]
    if isinstance(hits, list):
        return hits
    return []


def score_venue_candidate(venue_name: str, candidate: Dict[str, Any], candidate_url: str) -> int:
    target = normalize_title(venue_name)
    values: List[str] = []

    def collect_strings(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect_strings(child)
        elif isinstance(value, list):
            for child in value:
                collect_strings(child)
        elif value is not None:
            values.append(clean_text(value))

    collect_strings(candidate)
    values.append(candidate_url)
    best = 0
    for value in values:
        normalized_value = normalize_title(value)
        if not normalized_value:
            continue
        score = fuzz.token_set_ratio(target, normalized_value)
        if target == normalized_value:
            score = 100
        elif target and target in normalized_value:
            score = max(score, 96)
        best = max(best, int(score))
    return best


def resolve_stream_query_from_html(html_text: str) -> Optional[str]:
    soup = BeautifulSoup(html_text, "lxml")
    for anchor in soup.select('a[href*="/search?q="]'):
        if "dblp search" not in anchor.get_text(" ", strip=True).lower():
            continue
        href = anchor.get("href", "")
        if not href:
            continue
        parsed = urlparse(href)
        query_value = parse_qs(parsed.query).get("q", [])
        if query_value:
            return unquote(query_value[0])

    stream_match = re.search(r"streams/([A-Za-z0-9_./-]+)", html_text)
    if stream_match:
        return f"streamid:{stream_match.group(1)}:"
    return None


def resolve_override_stream_query(
    override: str,
    dblp_base_url: str,
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, str]]:
    value = clean_text(override)
    if not value:
        return None

    if value.startswith("streamid:"):
        return {"stream_query": value, "venue_url": value}

    if value.startswith("streams/"):
        return {"stream_query": f"streamid:{value.replace('streams/', '', 1)}:", "venue_url": value}

    if re.match(r"^(conf|journals|series|books)/", value):
        return {"stream_query": f"streamid:{value.strip(':')}:", "venue_url": value}

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        if is_dblp_url(value, dblp_base_url):
            value = rewrite_dblp_url(value, dblp_base_url)
            parsed = urlparse(value)
            if parsed.path.startswith("/search"):
                q_value = parse_qs(parsed.query).get("q", [])
                if q_value:
                    return {"stream_query": unquote(q_value[0]), "venue_url": value}
            response = request_with_retries(session, "GET", value, request_cfg, logger)
            if response is None or response.status_code >= 400:
                return None
            query = resolve_stream_query_from_html(response.text)
            if query:
                return {"stream_query": query, "venue_url": value}
        return None
    return None


def resolve_venue(
    venue_name: str,
    overrides: Dict[str, str],
    dblp_base_url: str,
    dblp_venue_api_url: str,
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, str]]:
    override_value = overrides.get(venue_name)
    if override_value:
        resolved_override = resolve_override_stream_query(
            override_value, dblp_base_url, session, request_cfg, logger
        )
        if resolved_override:
            resolved_override["input_venue"] = venue_name
            logger.info("Resolved venue override for %s -> %s", venue_name, resolved_override["stream_query"])
            return resolved_override
        logger.warning("Failed to resolve venue override for %s: %s", venue_name, override_value)

    payload = request_json(
        session,
        dblp_venue_api_url,
        request_cfg,
        logger,
        params={"q": venue_name, "format": "json", "h": 10, "c": 0},
    )
    hits = extract_search_hits(payload)
    candidates: List[Dict[str, Any]] = []
    for hit in hits:
        info = hit.get("info", hit)
        candidate_url = rewrite_dblp_url(info.get("url"), dblp_base_url)
        if not candidate_url or candidate_url == NA:
            key = clean_text(info.get("key"))
            if key:
                candidate_url = build_dblp_url(dblp_base_url, key)
        if not candidate_url or candidate_url == NA:
            continue
        score = score_venue_candidate(venue_name, info, candidate_url)
        candidates.append({"url": candidate_url, "score": score})

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0] if candidates else None
    if not best or best["score"] < 70:
        logger.warning("Unable to resolve DBLP venue page for %s", venue_name)
        return None

    response = request_with_retries(session, "GET", best["url"], request_cfg, logger)
    if response is None or response.status_code >= 400:
        logger.warning("Failed to fetch DBLP venue page for %s: %s", venue_name, best["url"])
        return None

    stream_query = resolve_stream_query_from_html(response.text)
    if not stream_query:
        logger.warning("Failed to extract stream query for venue %s from %s", venue_name, best["url"])
        return None

    logger.info("Resolved venue %s -> %s", venue_name, stream_query)
    return {"input_venue": venue_name, "venue_url": best["url"], "stream_query": stream_query}


def extract_authors_from_search_info(info: Dict[str, Any]) -> List[str]:
    authors_container = info.get("authors", {})
    raw_authors = []
    if isinstance(authors_container, dict):
        raw_authors = to_list(authors_container.get("author"))
    elif authors_container:
        raw_authors = to_list(authors_container)
    authors: List[str] = []
    for author in raw_authors:
        if isinstance(author, dict):
            name = clean_text(author.get("text") or author.get("#text") or author.get("name"))
        else:
            name = clean_text(author)
        if name:
            authors.append(name)
    return authors


def should_skip_search_hit(info: Dict[str, Any]) -> bool:
    type_text = clean_text(info.get("type")).lower()
    skip_markers = [
        "editorship",
        "proceedings",
        "books and theses",
        "informal and other publications",
        "reference works",
        "parts in books or collections",
    ]
    return any(marker in type_text for marker in skip_markers)


def parse_paper_from_search_hit(
    hit: Dict[str, Any],
    venue_hint: str,
    dblp_base_url: str,
) -> Optional[Dict[str, Any]]:
    info = hit.get("info", hit)
    if should_skip_search_hit(info):
        return None

    title = clean_text(info.get("title"))
    dblp_url = rewrite_dblp_url(info.get("url"), dblp_base_url)
    key = clean_text(info.get("key"))
    if not dblp_url and key:
        dblp_url = build_dblp_url(dblp_base_url, f"rec/{key}")
    if not title or not dblp_url:
        return None

    ee_value = info.get("ee")
    ee_list = [clean_text(item) for item in to_list(ee_value) if clean_text(item)]
    paper_url = choose_paper_url(ee_list, normalize_doi(info.get("doi")), dblp_base_url)
    record = {
        "title": title,
        "authors": extract_authors_from_search_info(info),
        "year": safe_int(info.get("year")) or NA,
        "venue": first_non_empty(info.get("venue"), venue_hint),
        "source_venue": venue_hint,
        "dblp_url": dblp_url,
        "doi": normalize_doi(info.get("doi")) or NA,
        "paper_url": paper_url,
        "match_checked": True,
        "detail_status": "pending",
        "abstract": NA,
        "abstract_source": NA,
        "abstract_status": "pending",
        "summary_text": NA,
        "summary_language": NA,
        "summary_zh": NA,
        "title_translation": NA,
        "title_translation_status": "pending",
        "category": NA,
        "ai_suggested_category": NA,
        "reason": NA,
        "llm_status": "pending",
        "affiliations": [NA],
        "affiliation_source": NA,
        "affiliation_mode": NA,
        "affiliation_status": "pending",
        "paper_type": clean_text(info.get("type")) or NA,
        "matched": False,
        "completed": False,
        "skip_export": False,
    }
    return record


def make_publ_query_task_id(venue_name: str, year: int) -> str:
    return make_fingerprint(clean_text(venue_name), str(year))


def summarize_publ_query_tasks(tasks: List[Dict[str, Any]], limit: int = 8) -> str:
    labels = [f"{task['venue_name']}:{task['year']}" for task in tasks[:limit]]
    if len(tasks) > limit:
        labels.append(f"...(+{len(tasks) - limit} more)")
    return ", ".join(labels) if labels else "none"


def fetch_dblp_papers_for_venue_year(
    venue_name: str,
    stream_query: str,
    year: int,
    dblp_base_url: str,
    dblp_publ_api_url: str,
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> tuple[List[Dict[str, Any]], bool]:
    logger.info("Fetching DBLP papers for venue=%s year=%s", venue_name, year)
    offset = 0
    page_size = 1000
    year_papers: List[Dict[str, Any]] = []

    while True:
        payload = request_json(
            session,
            dblp_publ_api_url,
            request_cfg,
            logger,
            params={
                "q": f"{stream_query} {year}$",
                "format": "json",
                "h": page_size,
                "f": offset,
                "c": 0,
            },
        )
        if payload is None:
            logger.warning(
                "DBLP publ query incomplete for venue=%s year=%s offset=%s; this venue/year will be retried later",
                venue_name,
                year,
                offset,
            )
            return year_papers, False

        hits = extract_search_hits(payload)
        if not hits:
            return year_papers, True

        for hit in hits:
            paper = parse_paper_from_search_hit(hit, venue_name, dblp_base_url)
            if paper:
                year_papers.append(paper)

        if len(hits) < page_size:
            return year_papers, True
        offset += len(hits)


def fetch_papers_from_dblp(
    venues: List[str],
    year_start: int,
    year_end: int,
    venue_overrides: Dict[str, str],
    dblp_base_url: str,
    dblp_venue_api_url: str,
    dblp_publ_api_url: str,
    publ_query_cache_path: str,
    publ_query_cache_index: Dict[str, Dict[str, Any]],
    publ_query_cache_enabled: bool,
    publ_query_current_year_ttl_hours: float,
    publ_query_max_refetch_rounds: int,
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for venue_name in venues:
        for year in range(year_start, year_end + 1):
            tasks.append(
                {
                    "task_id": make_publ_query_task_id(venue_name, year),
                    "venue_name": venue_name,
                    "year": year,
                }
            )

    total_rounds = publ_query_max_refetch_rounds + 1
    pending_tasks = list(tasks)
    pair_results: Dict[str, List[Dict[str, Any]]] = {}
    resolved_venues: Dict[str, Dict[str, str]] = {}

    for round_index in range(total_rounds):
        if not pending_tasks:
            break

        logger.info(
            "Starting DBLP publ fetch round %s/%s pending_tasks=%s",
            round_index + 1,
            total_rounds,
            len(pending_tasks),
        )
        if round_index > 0:
            logger.info(
                "Retrying incomplete DBLP publ tasks: %s",
                summarize_publ_query_tasks(pending_tasks),
            )

        grouped_tasks: Dict[str, List[Dict[str, Any]]] = {}
        for task in pending_tasks:
            grouped_tasks.setdefault(task["venue_name"], []).append(task)

        next_pending: List[Dict[str, Any]] = []
        for venue_name in venues:
            venue_tasks = grouped_tasks.get(venue_name, [])
            if not venue_tasks:
                continue

            resolved = resolved_venues.get(venue_name)
            if not resolved:
                resolved = resolve_venue(
                    venue_name,
                    venue_overrides,
                    dblp_base_url,
                    dblp_venue_api_url,
                    session,
                    request_cfg,
                    logger,
                )
                if resolved:
                    resolved_venues[venue_name] = resolved
            if not resolved:
                logger.warning(
                    "Still unable to resolve venue=%s in DBLP publ round %s/%s; will retry all pending years for this venue",
                    venue_name,
                    round_index + 1,
                    total_rounds,
                )
                next_pending.extend(venue_tasks)
                continue

            stream_query = resolved["stream_query"]
            for task in venue_tasks:
                year = task["year"]
                task_id = task["task_id"]
                cached_year_entry = lookup_publ_query_cache_entry(
                    publ_query_cache_index,
                    venue_name,
                    stream_query,
                    year,
                )
                if can_reuse_publ_query_cache_entry(
                    cached_year_entry,
                    publ_query_current_year_ttl_hours,
                ):
                    cached_records = clone_publ_query_cache_records(cached_year_entry)
                    pair_results[task_id] = cached_records
                    logger.info(
                        "Reusing DBLP publ cache for venue=%s year=%s cached_records=%s",
                        venue_name,
                        year,
                        len(cached_records),
                    )
                    continue

                year_papers, year_complete = fetch_dblp_papers_for_venue_year(
                    venue_name,
                    stream_query,
                    year,
                    dblp_base_url,
                    dblp_publ_api_url,
                    session,
                    request_cfg,
                    logger,
                )
                if year_complete:
                    pair_results[task_id] = year_papers
                    save_publ_query_cache_entry(
                        {
                            "venue_name": venue_name,
                            "stream_query": stream_query,
                            "year": year,
                            "records": year_papers,
                            "complete": True,
                            "updated_at": time.time(),
                        },
                        publ_query_cache_path,
                        publ_query_cache_index,
                        publ_query_cache_enabled,
                        dblp_base_url,
                    )
                    continue

                if cached_year_entry and cached_year_entry.get("complete"):
                    stale_records = clone_publ_query_cache_records(cached_year_entry)
                    pair_results[task_id] = stale_records
                    logger.warning(
                        "Using stale DBLP publ cache for venue=%s year=%s cached_records=%s; this venue/year remains pending for later retry",
                        venue_name,
                        year,
                        len(stale_records),
                    )
                else:
                    pair_results.pop(task_id, None)
                    logger.warning(
                        "No complete DBLP publ cache available for venue=%s year=%s; this venue/year remains pending for later retry",
                        venue_name,
                        year,
                    )
                next_pending.append(task)

        pending_tasks = next_pending
        if pending_tasks:
            logger.warning(
                "DBLP publ fetch round %s/%s finished with %s incomplete venue/year tasks: %s",
                round_index + 1,
                total_rounds,
                len(pending_tasks),
                summarize_publ_query_tasks(pending_tasks),
            )

    if pending_tasks:
        logger.warning(
            "DBLP publ fetch stopped with %s incomplete venue/year tasks after %s total rounds: %s",
            len(pending_tasks),
            total_rounds,
            summarize_publ_query_tasks(pending_tasks),
        )
    else:
        logger.info("DBLP publ fetch completed for all venue/year tasks within %s rounds", total_rounds)

    papers: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    for task in tasks:
        for paper in pair_results.get(task["task_id"], []):
            dblp_url = clean_text(paper.get("dblp_url"))
            if not dblp_url or dblp_url in seen_urls:
                continue
            seen_urls.add(dblp_url)
            papers.append(paper)

    logger.info("Fetched %s candidate DBLP records before title matching", len(papers))
    return papers


def choose_paper_url(ee_urls: List[str], doi: str, dblp_base_url: str = "") -> str:
    clean_urls = [clean_text(url) for url in ee_urls if clean_text(url)]
    for url in clean_urls:
        if is_doi_url(url) or is_metadata_url(url, dblp_base_url):
            continue
        if urlparse(url).scheme in {"http", "https"}:
            return url
    if doi:
        return f"https://doi.org/{doi}"
    for url in clean_urls:
        if is_doi_url(url):
            return url
    return NA


def fetch_paper_detail(
    dblp_url: str,
    dblp_base_url: str,
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    if not dblp_url or dblp_url == NA:
        return {"detail_status": "missing_identifier"}

    xml_url = dblp_url.rstrip("/")
    if not xml_url.endswith(".xml"):
        xml_url = f"{xml_url}.xml"

    response = request_with_retries(session, "GET", xml_url, request_cfg, logger)
    if response is None:
        return {"detail_status": "request_failed"}
    if response.status_code >= 400:
        return {"detail_status": "request_failed"}

    try:
        soup = BeautifulSoup(response.text, "xml")
        record_tag = soup.find(
            [
                "article",
                "inproceedings",
                "incollection",
                "proceedings",
                "book",
                "phdthesis",
                "mastersthesis",
                "www",
            ]
        )
        if record_tag is None:
            return {"detail_status": "parse_failed"}
    except Exception:
        return {"detail_status": "parse_failed"}

    paper_type = record_tag.name
    authors = [
        clean_text(author.get_text(" ", strip=True))
        for author in record_tag.find_all("author")
        if clean_text(author.get_text(" ", strip=True))
    ]
    title = clean_text(record_tag.find("title").get_text(" ", strip=True) if record_tag.find("title") else "")
    year = safe_int(record_tag.find("year").get_text(" ", strip=True) if record_tag.find("year") else None)
    venue = first_non_empty(
        record_tag.find("booktitle").get_text(" ", strip=True) if record_tag.find("booktitle") else "",
        record_tag.find("journal").get_text(" ", strip=True) if record_tag.find("journal") else "",
        record_tag.find("series").get_text(" ", strip=True) if record_tag.find("series") else "",
    )
    doi = normalize_doi(
        record_tag.find("doi").get_text(" ", strip=True) if record_tag.find("doi") else ""
    )
    ee_urls = [
        clean_text(ee_tag.get_text(" ", strip=True))
        for ee_tag in record_tag.find_all("ee")
        if clean_text(ee_tag.get_text(" ", strip=True))
    ]
    paper_url = choose_paper_url(ee_urls, doi, dblp_base_url)

    return {
        "title": title or NA,
        "authors": authors or [],
        "year": year or NA,
        "venue": venue,
        "doi": doi or NA,
        "paper_url": paper_url,
        "detail_status": "success",
        "paper_type": paper_type,
        "dblp_url": dblp_url,
    }


def title_similarity(title_a: str, title_b: str) -> int:
    return int(fuzz.token_set_ratio(normalize_title(title_a), normalize_title(title_b)))


def is_supported_paper_type(paper_type: str) -> bool:
    type_text = clean_text(paper_type).lower()
    return type_text in {"article", "inproceedings"} or any(
        marker in type_text
        for marker in [
            "journal articles",
            "journal article",
            "conference and workshop papers",
            "conference paper",
        ]
    )


def extract_candidate_title(item: Dict[str, Any]) -> str:
    return first_non_empty(
        item.get("title"),
        item.get("display_name"),
        item.get("name"),
    )


def extract_candidate_year(item: Dict[str, Any]) -> Optional[int]:
    for key in ("year", "publication_year"):
        year = safe_int(item.get(key))
        if year is not None:
            return year

    for key in ("issued", "published", "published-print", "published-online", "created", "deposited"):
        value = item.get(key)
        if isinstance(value, dict):
            date_parts = value.get("date-parts", [])
            if isinstance(date_parts, list) and date_parts and date_parts[0]:
                year = safe_int(date_parts[0][0])
                if year is not None:
                    return year
        year = safe_int(value)
        if year is not None:
            return year
    return None


def extract_candidate_authors(item: Dict[str, Any]) -> List[str]:
    authors: List[str] = []
    seen: set[str] = set()

    def add_author(name: Any) -> None:
        text = clean_text(name)
        if not text:
            return
        normalized = normalize_person_name(text)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        authors.append(text)

    for author in item.get("authors", []) or []:
        if not isinstance(author, dict):
            add_author(author)
            continue
        add_author(
            author.get("name")
            or author.get("display_name")
            or ((author.get("author") or {}).get("display_name"))
        )

    for author in item.get("author", []) or []:
        if not isinstance(author, dict):
            add_author(author)
            continue
        add_author(" ".join(filter(None, [author.get("given"), author.get("family")])))

    for authorship in item.get("authorships", []) or []:
        if not isinstance(authorship, dict):
            continue
        add_author(((authorship.get("author") or {}).get("display_name")))

    return authors


def count_author_overlaps(paper_authors: List[str], candidate_authors: List[str]) -> int:
    if not paper_authors or not candidate_authors:
        return 0

    normalized_candidates = [normalize_person_name(name) for name in candidate_authors]
    overlaps = 0
    for author in paper_authors:
        normalized_author = normalize_person_name(author)
        if not normalized_author:
            continue
        if normalized_author in normalized_candidates:
            overlaps += 1
            continue
        best_score = max(
            (fuzz.ratio(normalized_author, candidate_name) for candidate_name in normalized_candidates),
            default=0,
        )
        if best_score >= 90:
            overlaps += 1
    return overlaps


def select_best_title_candidate(
    candidates: Iterable[Dict[str, Any]],
    paper: Dict[str, Any],
    min_score: int = 80,
) -> Optional[Dict[str, Any]]:
    title = clean_text(paper.get("title"))
    paper_year = safe_int(paper.get("year"))
    paper_authors = [clean_text(author) for author in paper.get("authors") or [] if clean_text(author)]
    best_item: Optional[Dict[str, Any]] = None
    best_score = min_score
    for item in candidates:
        candidate_title = extract_candidate_title(item)
        if candidate_title == NA:
            continue
        title_score = title_similarity(title, candidate_title)
        if title_score <= min_score:
            continue

        candidate_year = extract_candidate_year(item)
        if (
            paper_year is not None
            and candidate_year is not None
            and abs(paper_year - candidate_year) > 1
        ):
            continue

        candidate_authors = extract_candidate_authors(item)
        author_overlap = count_author_overlaps(paper_authors, candidate_authors)
        if paper_authors and candidate_authors and author_overlap == 0:
            continue

        score = title_score
        if paper_year is not None and candidate_year is not None:
            score += 4 if paper_year == candidate_year else 1
        if author_overlap:
            score += min(author_overlap, 3) * 3

        if score > best_score:
            best_score = score
            best_item = item
    return best_item


def get_crossref_by_doi(
    paper: Dict[str, Any],
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    source_cache = paper.setdefault("_source_cache", {})
    if "crossref_doi" in source_cache:
        return source_cache["crossref_doi"]

    doi = normalize_doi(paper.get("doi"))
    if not doi:
        source_cache["crossref_doi"] = None
        source_cache["crossref_doi_status"] = "missing_identifier"
        return None

    payload, status = request_json_with_status(
        session,
        f"{CROSSREF_WORKS_API}/{quote(doi, safe='')}",
        request_cfg,
        logger,
    )
    message = payload.get("message") if payload else None
    source_cache["crossref_doi"] = message
    source_cache["crossref_doi_status"] = status
    return message


def search_crossref_by_title(
    paper: Dict[str, Any],
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    source_cache = paper.setdefault("_source_cache", {})
    if "crossref_title" in source_cache:
        return source_cache["crossref_title"]

    title = clean_text(paper.get("title"))
    if not title:
        source_cache["crossref_title"] = None
        source_cache["crossref_title_status"] = "missing_identifier"
        return None

    payload, status = request_json_with_status(
        session,
        CROSSREF_WORKS_API,
        request_cfg,
        logger,
        params={"query.title": title, "rows": 5},
    )
    items = payload.get("message", {}).get("items", []) if payload else []
    best_item = select_best_title_candidate(
        (
            {
                **item,
                "title": extract_crossref_title(item),
            }
            for item in items
        ),
        paper,
        min_score=78,
    )
    source_cache["crossref_title"] = best_item
    source_cache["crossref_title_status"] = status
    return best_item


def clean_crossref_abstract(raw_abstract: str) -> str:
    text = clean_text(raw_abstract)
    return text or NA


def extract_crossref_title(item: Dict[str, Any]) -> str:
    title_value = item.get("title")
    if isinstance(title_value, list):
        return clean_text(title_value[0] if title_value else "")
    return clean_text(title_value)


def get_openalex_by_doi(
    paper: Dict[str, Any],
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    source_cache = paper.setdefault("_source_cache", {})
    if "openalex_doi" in source_cache:
        return source_cache["openalex_doi"]

    doi = normalize_doi(paper.get("doi"))
    if not doi:
        source_cache["openalex_doi"] = None
        source_cache["openalex_doi_status"] = "missing_identifier"
        return None

    payload, status = request_json_with_status(
        session,
        OPENALEX_WORKS_API,
        request_cfg,
        logger,
        params={
            "filter": f"doi:{doi}",
            "per-page": 1,
        },
    )
    results = payload.get("results", []) if payload else []
    source_cache["openalex_doi"] = results[0] if results else None
    source_cache["openalex_doi_status"] = status
    return source_cache["openalex_doi"]


def search_openalex_by_title(
    paper: Dict[str, Any],
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    source_cache = paper.setdefault("_source_cache", {})
    if "openalex_title" in source_cache:
        return source_cache["openalex_title"]

    title = clean_text(paper.get("title"))
    if not title:
        source_cache["openalex_title"] = None
        source_cache["openalex_title_status"] = "missing_identifier"
        return None

    payload, status = request_json_with_status(
        session,
        OPENALEX_WORKS_API,
        request_cfg,
        logger,
        params={"search": title, "per-page": 5},
    )
    results = payload.get("results", []) if payload else []
    best_item = select_best_title_candidate(results, paper, min_score=80)
    source_cache["openalex_title"] = best_item
    source_cache["openalex_title_status"] = status
    return best_item


def reconstruct_openalex_abstract(work: Dict[str, Any]) -> str:
    inverted_index = work.get("abstract_inverted_index")
    if not isinstance(inverted_index, dict) or not inverted_index:
        return NA
    positions: Dict[int, str] = {}
    for token, indexes in inverted_index.items():
        for index in indexes or []:
            positions[int(index)] = token
    words = [positions[idx] for idx in sorted(positions)]
    return clean_text(" ".join(words)) or NA


def get_semantic_scholar_by_doi(
    paper: Dict[str, Any],
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    source_cache = paper.setdefault("_source_cache", {})
    if "semantic_doi" in source_cache:
        return source_cache["semantic_doi"]

    doi = normalize_doi(paper.get("doi"))
    if not doi:
        source_cache["semantic_doi"] = None
        source_cache["semantic_doi_status"] = "missing_identifier"
        return None

    payload, status = request_json_with_status(
        session,
        f"{SEMANTIC_SCHOLAR_PAPER_API}/DOI:{quote(doi, safe='')}",
        request_cfg,
        logger,
        params={"fields": "title,abstract,authors.name,authors.affiliations,venue,year,url,externalIds"},
    )
    source_cache["semantic_doi"] = payload
    source_cache["semantic_doi_status"] = status
    return payload


def search_semantic_scholar_by_title(
    paper: Dict[str, Any],
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    source_cache = paper.setdefault("_source_cache", {})
    if "semantic_title" in source_cache:
        return source_cache["semantic_title"]

    title = clean_text(paper.get("title"))
    if not title:
        source_cache["semantic_title"] = None
        source_cache["semantic_title_status"] = "missing_identifier"
        return None

    payload, status = request_json_with_status(
        session,
        SEMANTIC_SCHOLAR_SEARCH_API,
        request_cfg,
        logger,
        params={
            "query": title,
            "limit": 5,
            "fields": "title,abstract,authors.name,authors.affiliations,venue,year,url,externalIds",
        },
    )
    data = payload.get("data", []) if payload else []
    best_item = select_best_title_candidate(data, paper, min_score=80)
    source_cache["semantic_title"] = best_item
    source_cache["semantic_title_status"] = status
    return best_item


def extract_arxiv_id(paper: Dict[str, Any]) -> str:
    paper_url = clean_text(paper.get("paper_url"))
    doi = normalize_doi(paper.get("doi"))
    candidates = [paper_url, doi]
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/([A-Za-z0-9.\-]+)",
        r"10\.48550/arxiv\.([A-Za-z0-9.\-]+)",
        r"arxiv:([A-Za-z0-9.\-]+)",
    ]
    for candidate in candidates:
        lower_candidate = candidate.lower()
        for pattern in patterns:
            match = re.search(pattern, lower_candidate)
            if match:
                return match.group(1).replace(".pdf", "")
    return ""


def fetch_arxiv_abstract(
    paper: Dict[str, Any],
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[str]:
    source_cache = paper.setdefault("_source_cache", {})
    if "arxiv_abstract" in source_cache:
        return source_cache["arxiv_abstract"]

    arxiv_id = extract_arxiv_id(paper)
    if not arxiv_id:
        source_cache["arxiv_abstract"] = None
        source_cache["arxiv_abstract_status"] = "missing_identifier"
        return None

    response = request_with_retries(
        session,
        "GET",
        ARXIV_API,
        request_cfg,
        logger,
        params={"id_list": arxiv_id},
    )
    if response is None or response.status_code >= 400:
        source_cache["arxiv_abstract"] = None
        source_cache["arxiv_abstract_status"] = "request_failed"
        return None

    try:
        soup = BeautifulSoup(response.text, "xml")
        summary = soup.find("summary")
        abstract = clean_text(summary.get_text(" ", strip=True) if summary else "")
        source_cache["arxiv_abstract"] = abstract or None
        source_cache["arxiv_abstract_status"] = "success"
        return source_cache["arxiv_abstract"]
    except Exception:
        source_cache["arxiv_abstract"] = None
        source_cache["arxiv_abstract_status"] = "parse_failed"
        return None


def resolve_source_failure_status(paper: Dict[str, Any], status_keys: List[str]) -> str:
    source_cache = paper.get("_source_cache", {}) or {}
    statuses = [
        clean_text(source_cache.get(key)).lower()
        for key in status_keys
        if clean_text(source_cache.get(key))
    ]
    if any(status == "request_failed" for status in statuses):
        return "request_failed"
    if any(status == "parse_failed" for status in statuses):
        return "parse_failed"
    return "not_found"


def fetch_abstract(
    paper: Dict[str, Any],
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, str]:
    abstract_signature = compute_abstract_signature(paper)
    doi = normalize_doi(paper.get("doi"))
    title = clean_text(paper.get("title"))
    paper_url = clean_text(paper.get("paper_url"))
    if not any([doi, title, paper_url]):
        return {
            "abstract": NA,
            "abstract_source": NA,
            "abstract_status": "missing_identifier",
            "abstract_signature": abstract_signature,
        }

    try:
        crossref_doi = get_crossref_by_doi(paper, session, request_cfg, logger)
        if crossref_doi and crossref_doi.get("abstract"):
            return {
                "abstract": clean_crossref_abstract(crossref_doi["abstract"]),
                "abstract_source": "Crossref",
                "abstract_status": "success",
                "abstract_signature": abstract_signature,
            }

        openalex_doi = get_openalex_by_doi(paper, session, request_cfg, logger)
        openalex_abstract = reconstruct_openalex_abstract(openalex_doi or {})
        if openalex_abstract != NA:
            return {
                "abstract": openalex_abstract,
                "abstract_source": "OpenAlex",
                "abstract_status": "success",
                "abstract_signature": abstract_signature,
            }

        semantic_doi = get_semantic_scholar_by_doi(paper, session, request_cfg, logger)
        semantic_abstract = clean_text((semantic_doi or {}).get("abstract"))
        if semantic_abstract:
            return {
                "abstract": semantic_abstract,
                "abstract_source": "Semantic Scholar",
                "abstract_status": "success",
                "abstract_signature": abstract_signature,
            }

        arxiv_abstract = fetch_arxiv_abstract(paper, session, request_cfg, logger)
        if arxiv_abstract:
            return {
                "abstract": arxiv_abstract,
                "abstract_source": "arXiv",
                "abstract_status": "success",
                "abstract_signature": abstract_signature,
            }

        crossref_title = search_crossref_by_title(paper, session, request_cfg, logger)
        if crossref_title and crossref_title.get("abstract"):
            return {
                "abstract": clean_crossref_abstract(crossref_title["abstract"]),
                "abstract_source": "Crossref",
                "abstract_status": "success",
                "abstract_signature": abstract_signature,
            }

        openalex_title = search_openalex_by_title(paper, session, request_cfg, logger)
        openalex_title_abstract = reconstruct_openalex_abstract(openalex_title or {})
        if openalex_title_abstract != NA:
            return {
                "abstract": openalex_title_abstract,
                "abstract_source": "OpenAlex",
                "abstract_status": "success",
                "abstract_signature": abstract_signature,
            }

        semantic_title = search_semantic_scholar_by_title(paper, session, request_cfg, logger)
        semantic_title_abstract = clean_text((semantic_title or {}).get("abstract"))
        if semantic_title_abstract:
            return {
                "abstract": semantic_title_abstract,
                "abstract_source": "Semantic Scholar",
                "abstract_status": "success",
                "abstract_signature": abstract_signature,
            }

        abstract_status = resolve_source_failure_status(
            paper,
            [
                "crossref_doi_status",
                "openalex_doi_status",
                "semantic_doi_status",
                "arxiv_abstract_status",
                "crossref_title_status",
                "openalex_title_status",
                "semantic_title_status",
            ],
        )
        return {
            "abstract": NA,
            "abstract_source": NA,
            "abstract_status": abstract_status,
            "abstract_signature": abstract_signature,
        }
    except requests.RequestException:
        return {
            "abstract": NA,
            "abstract_source": NA,
            "abstract_status": "request_failed",
            "abstract_signature": abstract_signature,
        }
    except Exception:
        logger.exception("Unexpected error while fetching abstract for %s", title)
        return {
            "abstract": NA,
            "abstract_source": NA,
            "abstract_status": "parse_failed",
            "abstract_signature": abstract_signature,
        }


def align_affiliations_by_order(
    authors: List[str],
    entries: List[Dict[str, Any]],
) -> Optional[List[str]]:
    if not authors or not entries:
        return None
    if len(entries) < len(authors):
        return None

    affiliations: List[str] = []
    for idx, author in enumerate(authors):
        if idx >= len(entries):
            return None
        candidate_name = normalize_person_name(entries[idx].get("name", ""))
        author_name = normalize_person_name(author)
        if author_name and candidate_name and fuzz.ratio(author_name, candidate_name) < 70:
            return None
        affiliation_text = clean_text(entries[idx].get("affiliation"))
        affiliations.append(affiliation_text or NA)
    return affiliations


def align_affiliations_by_name(
    authors: List[str],
    entries: List[Dict[str, Any]],
) -> Optional[List[str]]:
    if not authors or not entries:
        return None

    mapping: Dict[str, str] = {}
    for entry in entries:
        normalized_name = normalize_person_name(entry.get("name", ""))
        affiliation_text = clean_text(entry.get("affiliation"))
        if normalized_name and affiliation_text:
            mapping[normalized_name] = affiliation_text
    if not mapping:
        return None

    aligned: List[str] = []
    matched_any = False
    for author in authors:
        normalized_author = normalize_person_name(author)
        best_affiliation = mapping.get(normalized_author)
        if best_affiliation:
            matched_any = True
            aligned.append(best_affiliation)
            continue

        best_score = 0
        best_value = NA
        for candidate_name, affiliation in mapping.items():
            score = fuzz.ratio(normalized_author, candidate_name)
            if score > best_score:
                best_score = score
                best_value = affiliation
        if best_score >= 85:
            matched_any = True
            aligned.append(best_value)
        else:
            aligned.append(NA)

    return aligned if matched_any else None


def dedupe_affiliations(entries: List[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    values: List[str] = []
    for entry in entries:
        affiliation = clean_text(entry.get("affiliation"))
        if not affiliation:
            continue
        if affiliation not in seen:
            seen.add(affiliation)
            values.append(affiliation)
    return values


def parse_crossref_affiliations(message: Dict[str, Any], authors: List[str]) -> Optional[Dict[str, Any]]:
    author_entries = message.get("author", []) if isinstance(message, dict) else []
    parsed: List[Dict[str, Any]] = []
    for item in author_entries:
        if not isinstance(item, dict):
            continue
        name = clean_text(" ".join(filter(None, [item.get("given"), item.get("family")])))
        affiliations = [
            clean_text(aff.get("name"))
            for aff in to_list(item.get("affiliation"))
            if isinstance(aff, dict) and clean_text(aff.get("name"))
        ]
        parsed.append({"name": name, "affiliation": "; ".join(affiliations) if affiliations else NA})

    order_aligned = align_affiliations_by_order(authors, parsed)
    if order_aligned:
        return {"authors": authors, "affiliations": order_aligned, "mode": "aligned_by_order"}

    name_aligned = align_affiliations_by_name(authors, parsed)
    if name_aligned:
        return {"authors": authors, "affiliations": name_aligned, "mode": "aligned_by_name"}

    merged = dedupe_affiliations(parsed)
    if merged:
        return {"authors": authors, "affiliations": merged, "mode": "merged_deduped"}
    return None


def parse_openalex_affiliations(work: Dict[str, Any], authors: List[str]) -> Optional[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for authorship in work.get("authorships", []) or []:
        if not isinstance(authorship, dict):
            continue
        author_name = clean_text(((authorship.get("author") or {}).get("display_name")))
        institutions = []
        for institution in authorship.get("institutions", []) or []:
            institution_name = clean_text((institution or {}).get("display_name"))
            if institution_name:
                institutions.append(institution_name)
        parsed.append(
            {
                "name": author_name,
                "affiliation": "; ".join(dict.fromkeys(institutions)) if institutions else NA,
            }
        )

    order_aligned = align_affiliations_by_order(authors, parsed)
    if order_aligned:
        return {"authors": authors, "affiliations": order_aligned, "mode": "aligned_by_order"}

    name_aligned = align_affiliations_by_name(authors, parsed)
    if name_aligned:
        return {"authors": authors, "affiliations": name_aligned, "mode": "aligned_by_name"}

    merged = dedupe_affiliations(parsed)
    if merged:
        return {"authors": authors, "affiliations": merged, "mode": "merged_deduped"}
    return None


def parse_semantic_scholar_affiliations(work: Dict[str, Any], authors: List[str]) -> Optional[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for author in work.get("authors", []) or []:
        if not isinstance(author, dict):
            continue
        affiliation_value = author.get("affiliations", [])
        if isinstance(affiliation_value, str):
            affiliations = [clean_text(affiliation_value)] if clean_text(affiliation_value) else []
        else:
            affiliations = [clean_text(item) for item in to_list(affiliation_value) if clean_text(item)]
        parsed.append(
            {
                "name": clean_text(author.get("name")),
                "affiliation": "; ".join(dict.fromkeys(affiliations)) if affiliations else NA,
            }
        )

    order_aligned = align_affiliations_by_order(authors, parsed)
    if order_aligned:
        return {"authors": authors, "affiliations": order_aligned, "mode": "aligned_by_order"}

    name_aligned = align_affiliations_by_name(authors, parsed)
    if name_aligned:
        return {"authors": authors, "affiliations": name_aligned, "mode": "aligned_by_name"}

    merged = dedupe_affiliations(parsed)
    if merged:
        return {"authors": authors, "affiliations": merged, "mode": "merged_deduped"}
    return None


def fetch_affiliations(
    paper: Dict[str, Any],
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    affiliation_signature = compute_affiliation_signature(paper)
    authors = paper.get("authors") or []
    if not authors:
        return {
            "authors": [],
            "affiliations": [NA],
            "affiliation_source": NA,
            "affiliation_mode": NA,
            "affiliation_status": "not_found",
            "affiliation_signature": affiliation_signature,
        }

    try:
        crossref_doi = get_crossref_by_doi(paper, session, request_cfg, logger)
        crossref_data = parse_crossref_affiliations(crossref_doi or {}, authors)
        if crossref_data:
            return {
                "authors": authors,
                "affiliations": crossref_data["affiliations"],
                "affiliation_source": "Crossref",
                "affiliation_mode": crossref_data["mode"],
                "affiliation_status": "success",
                "affiliation_signature": affiliation_signature,
            }

        crossref_title = search_crossref_by_title(paper, session, request_cfg, logger)
        crossref_title_data = parse_crossref_affiliations(crossref_title or {}, authors)
        if crossref_title_data:
            return {
                "authors": authors,
                "affiliations": crossref_title_data["affiliations"],
                "affiliation_source": "Crossref",
                "affiliation_mode": crossref_title_data["mode"],
                "affiliation_status": "success",
                "affiliation_signature": affiliation_signature,
            }

        openalex_doi = get_openalex_by_doi(paper, session, request_cfg, logger)
        openalex_data = parse_openalex_affiliations(openalex_doi or {}, authors)
        if openalex_data:
            return {
                "authors": authors,
                "affiliations": openalex_data["affiliations"],
                "affiliation_source": "OpenAlex",
                "affiliation_mode": openalex_data["mode"],
                "affiliation_status": "success",
                "affiliation_signature": affiliation_signature,
            }

        openalex_title = search_openalex_by_title(paper, session, request_cfg, logger)
        openalex_title_data = parse_openalex_affiliations(openalex_title or {}, authors)
        if openalex_title_data:
            return {
                "authors": authors,
                "affiliations": openalex_title_data["affiliations"],
                "affiliation_source": "OpenAlex",
                "affiliation_mode": openalex_title_data["mode"],
                "affiliation_status": "success",
                "affiliation_signature": affiliation_signature,
            }

        semantic_doi = get_semantic_scholar_by_doi(paper, session, request_cfg, logger)
        semantic_data = parse_semantic_scholar_affiliations(semantic_doi or {}, authors)
        if semantic_data:
            return {
                "authors": authors,
                "affiliations": semantic_data["affiliations"],
                "affiliation_source": "Semantic Scholar",
                "affiliation_mode": semantic_data["mode"],
                "affiliation_status": "success",
                "affiliation_signature": affiliation_signature,
            }

        semantic_title = search_semantic_scholar_by_title(paper, session, request_cfg, logger)
        semantic_title_data = parse_semantic_scholar_affiliations(semantic_title or {}, authors)
        if semantic_title_data:
            return {
                "authors": authors,
                "affiliations": semantic_title_data["affiliations"],
                "affiliation_source": "Semantic Scholar",
                "affiliation_mode": semantic_title_data["mode"],
                "affiliation_status": "success",
                "affiliation_signature": affiliation_signature,
            }

        affiliation_status = resolve_source_failure_status(
            paper,
            [
                "crossref_doi_status",
                "crossref_title_status",
                "openalex_doi_status",
                "openalex_title_status",
                "semantic_doi_status",
                "semantic_title_status",
            ],
        )
        return {
            "authors": authors,
            "affiliations": [NA],
            "affiliation_source": NA,
            "affiliation_mode": NA,
            "affiliation_status": affiliation_status,
            "affiliation_signature": affiliation_signature,
        }
    except requests.RequestException:
        return {
            "authors": authors,
            "affiliations": [NA],
            "affiliation_source": NA,
            "affiliation_mode": NA,
            "affiliation_status": "request_failed",
            "affiliation_signature": affiliation_signature,
        }
    except Exception:
        logger.exception("Unexpected error while fetching affiliations for %s", paper.get("title"))
        return {
            "authors": authors,
            "affiliations": [NA],
            "affiliation_source": NA,
            "affiliation_mode": NA,
            "affiliation_status": "parse_failed",
            "affiliation_signature": affiliation_signature,
        }


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def get_summary_text(record: Dict[str, Any]) -> str:
    summary_text = clean_text(record.get("summary_text"))
    if summary_text:
        return summary_text
    legacy_summary = clean_text(record.get("summary_zh"))
    if legacy_summary:
        return legacy_summary
    return NA


def get_export_link(record: Dict[str, Any]) -> str:
    doi = normalize_doi(record.get("doi"))
    return first_non_empty(
        f"https://doi.org/{doi}" if doi else "",
        record.get("paper_url"),
        record.get("dblp_url"),
    )


def build_llm_default_result(
    paper: Dict[str, Any],
    config: Dict[str, Any],
    llm_status: str,
) -> Dict[str, str]:
    title_translation_enabled = bool(config["llm_output"]["title_translation_enabled"])
    summary_language = config["llm_output"]["summary_language"]
    return {
        "title_translation": NA,
        "title_translation_status": "disabled" if not title_translation_enabled else llm_status,
        "summary_text": NA,
        "summary_language": summary_language,
        "summary_zh": NA,
        "category": NA,
        "ai_suggested_category": NA,
        "reason": NA,
        "llm_signature": compute_llm_signature(paper, config),
        "llm_status": llm_status,
    }


def summarize_and_classify(
    paper: Dict[str, Any],
    config: Dict[str, Any],
    client: Optional[OpenAI],
    logger: logging.Logger,
) -> Dict[str, str]:
    title = clean_text(paper.get("title"))
    abstract = clean_text(paper.get("abstract"))
    title_translation_enabled = bool(config["llm_output"]["title_translation_enabled"])
    summary_language = config["llm_output"]["summary_language"]
    has_abstract = bool(abstract and abstract != NA)

    if not title and not has_abstract:
        return build_llm_default_result(paper, config, "missing_input")

    if not has_abstract and not title_translation_enabled:
        return build_llm_default_result(paper, config, "no_abstract")

    if client is None:
        return build_llm_default_result(paper, config, "llm_unavailable")

    categories = config["classification"]["categories"]
    allow_new_category = bool(config["classification"]["allow_new_category"])
    openai_cfg = config["openai"]
    summary_language_name = "中文" if summary_language == "zh" else "英文"
    llm_signature = compute_llm_signature(paper, config)

    system_prompt = (
        "你是一名严谨的计算机领域论文分析助手。"
        "你必须只输出一个 JSON 对象，不能输出 Markdown、解释或多余文本。"
    )
    prompt_lines = [
        "请根据论文标题和摘要，完成标题翻译、摘要总结和研究方向归类。",
        f"候选类别：{json.dumps(categories, ensure_ascii=False)}",
        f"是否允许提出新类别：{allow_new_category}",
        "输出 JSON，且必须包含以下字符串字段："
        "title_translation, summary_text, category, ai_suggested_category, reason。",
        "规则：",
    ]
    if title_translation_enabled:
        prompt_lines.append('1. title_translation 填写论文标题的中文翻译。')
    else:
        prompt_lines.append('1. title_translation 必须写 "N/A"。')
    if has_abstract:
        prompt_lines.extend(
            [
                f"2. summary_text 用{summary_language_name}简洁总结论文核心内容。",
                "3. category 必须从候选类别中选择一个。",
                '4. 如果候选类别中没有合适项，则 category 必须为 "其他"。',
                '5. 只有在允许提出新类别且 category 为 "其他" 时，ai_suggested_category 才能填写简短新类别，否则必须写 "N/A"。',
                '6. 如果已有候选类别合适，则 ai_suggested_category 必须写 "N/A"。',
                "7. reason 用中文简要说明分类依据。",
            ]
        )
    else:
        prompt_lines.extend(
            [
                '2. 当前没有摘要，因此 summary_text、category、ai_suggested_category、reason 都必须写 "N/A"。',
                "3. 仅完成标题翻译，不要基于标题猜测分类。",
            ]
        )
    prompt_lines.extend(
        [
            "",
            f"标题：{title or NA}",
            f"摘要：{abstract if has_abstract else NA}",
        ]
    )
    user_prompt = "\n".join(prompt_lines)

    max_retries = int(openai_cfg.get("max_retries", 3))
    errors: List[str] = []

    for attempt in range(max_retries + 1):
        try:
            request_kwargs = {
                "model": openai_cfg["model"],
                "temperature": float(openai_cfg["temperature"]),
                "max_tokens": int(openai_cfg["max_tokens"]),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }
            if attempt > 0:
                request_kwargs.pop("response_format", None)

            response = client.chat.completions.create(**request_kwargs)
            content = ""
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
            payload = extract_json_object(content)
            if not payload:
                raise ValueError("Model did not return valid JSON content.")

            title_translation = clean_text(payload.get("title_translation")) or NA
            summary_text = clean_text(payload.get("summary_text")) or NA
            category = clean_text(payload.get("category")) or ("其他" if has_abstract else NA)
            ai_suggested_category = clean_text(payload.get("ai_suggested_category")) or NA
            reason = clean_text(payload.get("reason")) or NA
            title_translation_status = "disabled"

            if title_translation_enabled:
                title_translation_status = "success" if title_translation != NA else "missing"
            else:
                title_translation = NA

            if not has_abstract:
                summary_text = NA
                category = NA
                ai_suggested_category = NA
                reason = NA
                llm_status = "translated_only" if title_translation_enabled else "no_abstract"
                return {
                    "title_translation": title_translation,
                    "title_translation_status": title_translation_status,
                    "summary_text": summary_text,
                    "summary_language": summary_language,
                    "summary_zh": NA,
                    "category": category,
                    "ai_suggested_category": ai_suggested_category,
                    "reason": reason,
                    "llm_signature": llm_signature,
                    "llm_status": llm_status,
                }

            if category not in categories:
                logger.warning(
                    "Model returned category outside candidates for '%s': %s",
                    paper.get("title", NA),
                    category,
                )
                category = "其他"

            if category != "其他":
                ai_suggested_category = NA
            elif not allow_new_category:
                ai_suggested_category = NA
            elif ai_suggested_category == NA:
                ai_suggested_category = NA

            return {
                "title_translation": title_translation,
                "title_translation_status": title_translation_status,
                "summary_text": summary_text,
                "summary_language": summary_language,
                "summary_zh": summary_text if summary_language == "zh" else NA,
                "category": category,
                "ai_suggested_category": ai_suggested_category,
                "reason": reason,
                "llm_signature": llm_signature,
                "llm_status": "success",
            }
        except Exception as exc:
            errors.append(str(exc))
            logger.warning(
                "OpenAI-compatible call failed for '%s' on attempt %s/%s: %s",
                paper.get("title", NA),
                attempt + 1,
                max_retries + 1,
                exc,
            )
            time.sleep(1 + attempt * 0.5)

    logger.error(
        "OpenAI-compatible summarization failed for '%s': %s",
        paper.get("title", NA),
        " | ".join(errors[-3:]) if errors else "unknown error",
    )
    return build_llm_default_result(paper, config, "failed")


def test_ai_configuration(
    config: Dict[str, Any],
    client: Optional[OpenAI],
    logger: logging.Logger,
) -> bool:
    if client is None:
        logger.error(
            "AI API test failed: OpenAI client could not be initialized. "
            "Please check openai.host, openai.api_key, and openai.model."
        )
        return False

    openai_cfg = config["openai"]
    max_retries = int(openai_cfg.get("max_retries", 3))
    max_tokens = max(64, min(int(openai_cfg.get("max_tokens", 800)), 128))
    errors: List[str] = []

    system_prompt = (
        "You are validating an OpenAI-compatible API configuration. "
        "Return exactly one compact JSON object and nothing else."
    )
    user_prompt = (
        'Return JSON like {"status":"ok","message":"api works"} '
        "to confirm the chat completion endpoint is usable."
    )

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=openai_cfg["model"],
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = ""
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
            payload = extract_json_object(content)
            if not payload:
                raise ValueError("Model did not return valid JSON content.")

            logger.info(
                "AI API test succeeded | host=%s | model=%s | response=%s",
                openai_cfg["host"],
                openai_cfg["model"],
                json.dumps(payload, ensure_ascii=False),
            )
            return True
        except Exception as exc:
            errors.append(str(exc))
            logger.warning(
                "AI API test failed on attempt %s/%s: %s",
                attempt + 1,
                max_retries + 1,
                exc,
            )
            time.sleep(1 + attempt * 0.5)

    logger.error(
        "AI API test failed after retries | host=%s | model=%s | errors=%s",
        openai_cfg["host"],
        openai_cfg["model"],
        " | ".join(errors[-3:]) if errors else "unknown error",
    )
    return False


def should_refresh_abstract(record: Dict[str, Any]) -> bool:
    expected_signature = compute_abstract_signature(record)
    if clean_text(record.get("abstract_signature")) != expected_signature:
        return True
    status = clean_text(record.get("abstract_status")).lower()
    if status == "success" and clean_text(record.get("abstract")) not in {"", NA}:
        return False
    if status in {"not_found", "missing_identifier"}:
        return False
    return True


def should_refresh_affiliations(record: Dict[str, Any]) -> bool:
    expected_signature = compute_affiliation_signature(record)
    if clean_text(record.get("affiliation_signature")) != expected_signature:
        return True
    status = clean_text(record.get("affiliation_status")).lower()
    if status == "success":
        return False
    if status == "not_found":
        return False
    return True


def should_refresh_detail(record: Dict[str, Any], dblp_base_url: str = "") -> bool:
    status = clean_text(record.get("detail_status")).lower()
    if status != "success":
        return True
    if not record.get("authors"):
        return True
    paper_url = clean_text(record.get("paper_url"))
    if paper_url in {"", NA}:
        return False
    if is_metadata_url(paper_url, dblp_base_url):
        return True
    return False


def should_refresh_llm(record: Dict[str, Any], no_llm: bool, config: Dict[str, Any]) -> bool:
    expected_signature = compute_llm_signature(record, config)
    if clean_text(record.get("llm_signature")) != expected_signature:
        return True
    status = clean_text(record.get("llm_status")).lower()
    if no_llm:
        return status != "no_llm"
    title_translation_enabled = bool(config["llm_output"]["title_translation_enabled"])
    target_summary_language = config["llm_output"]["summary_language"]
    has_abstract = clean_text(record.get("abstract")) not in {"", NA}
    title_translation_missing = (
        title_translation_enabled
        and clean_text(record.get("title_translation")) in {"", NA}
    )
    summary_missing = has_abstract and get_summary_text(record) in {"", NA}
    summary_language_mismatch = (
        has_abstract
        and clean_text(record.get("summary_language")).lower() not in {"", target_summary_language}
    )

    if title_translation_missing or summary_missing or summary_language_mismatch:
        return True
    if status in {"success", "translated_only", "no_abstract", "no_llm"}:
        return False
    return True


def is_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        text = clean_text(value)
        return text not in {"", NA, "pending"}
    if isinstance(value, list):
        return any(is_meaningful_value(item) for item in value)
    if isinstance(value, dict):
        return bool(value)
    return True


def merge_record(preferred: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(fallback)
    for key, value in preferred.items():
        if key not in merged:
            merged[key] = value
            continue
        if isinstance(value, bool):
            if value or not isinstance(merged.get(key), bool):
                merged[key] = value
            continue
        if is_meaningful_value(value):
            merged[key] = value
    return merged


def sanitize_filename(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._")
    return text or "unknown_venue"


def build_csv_rows(records: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if record.get("skip_export"):
            continue
        primary_key = compute_primary_dedupe_key(record)
        if not primary_key:
            continue
        deduped[primary_key] = record

    ordered = sorted(
        deduped.values(),
        key=lambda item: (
            -(safe_int(item.get("year")) or 0),
            clean_text(item.get("venue")),
            normalize_title(item.get("title", "")),
        ),
    )
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(ordered, start=1):
        authors = record.get("authors") or []
        affiliations = record.get("affiliations") or [NA]
        row = {
            "序号": idx,
            "标题": clean_text(record.get("title")) or NA,
            "标题翻译": clean_text(record.get("title_translation")) or NA,
            "链接": get_export_link(record),
            "作者": "; ".join(clean_text(author) for author in authors if clean_text(author)) or NA,
            "作者单位": "; ".join(
                clean_text(affiliation)
                for affiliation in affiliations
                if clean_text(affiliation)
            )
            or NA,
            "年份": safe_int(record.get("year")) or NA,
            "期刊/会议": clean_text(record.get("venue")) or NA,
            "类别": clean_text(record.get("category")) or NA,
            "AI建议新类别": clean_text(record.get("ai_suggested_category")) or NA,
            "摘要总结": get_summary_text(record),
        }
        rows.append(row)
    return rows


def export_csv(records: List[Dict[str, Any]], csv_dir: str, config: Dict[str, Any], logger: logging.Logger) -> int:
    output_dir = Path(csv_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_records: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        if record.get("skip_export"):
            continue
        source_venue = first_non_empty(record.get("source_venue"), record.get("venue"))
        grouped_records.setdefault(source_venue, []).append(record)

    fieldnames = [
        "序号",
        "标题",
        "标题翻译",
        "链接",
        "作者",
        "作者单位",
        "年份",
        "期刊/会议",
        "类别",
        "AI建议新类别",
        "摘要总结",
    ]

    for venue_name in config["dblp"]["venues"]:
        stale_path = output_dir / f"{sanitize_filename(venue_name)}.csv"
        if stale_path.exists():
            stale_path.unlink()

    total_rows = 0
    for source_venue, venue_records in sorted(grouped_records.items(), key=lambda item: clean_text(item[0])):
        rows = build_csv_rows(venue_records, config)
        file_path = output_dir / f"{sanitize_filename(source_venue)}.csv"
        with file_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        total_rows += len(rows)
        logger.info("Exported %s rows to CSV: %s", len(rows), file_path)

    logger.info("Exported %s total rows across %s CSV files in %s", total_rows, len(grouped_records), output_dir)
    return total_rows


def collect_unique_cached_records(cache_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique_records: Dict[str, Dict[str, Any]] = {}
    for cached_record in cache_index.values():
        normalized = sanitize_record_for_cache(cached_record)
        dedupe_key = normalized.get("dedupe_key") or compute_primary_dedupe_key(normalized)
        if not dedupe_key:
            continue
        unique_records[dedupe_key] = normalized
    return list(unique_records.values())


def is_record_in_config_scope(record: Dict[str, Any], config: Dict[str, Any]) -> bool:
    venues = set(config["dblp"]["venues"])
    source_venue = first_non_empty(record.get("source_venue"), record.get("venue"))
    if source_venue not in venues:
        return False

    year = safe_int(record.get("year"))
    if year is None:
        return True

    return config["dblp"]["year_start"] <= year <= config["dblp"]["year_end"]


def is_publ_query_entry_in_config_scope(entry: Dict[str, Any], config: Dict[str, Any]) -> bool:
    venue_name = clean_text(entry.get("venue_name"))
    year = safe_int(entry.get("year"))
    if venue_name not in set(config["dblp"]["venues"]) or year is None:
        return False
    return config["dblp"]["year_start"] <= year <= config["dblp"]["year_end"]


def persist_cache_records(
    cache_path: str,
    records: List[Dict[str, Any]],
    dblp_base_url: str = "",
) -> None:
    path = Path(cache_path).expanduser().resolve()
    ensure_parent_directory(str(path))

    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            normalized = sanitize_record_for_cache(record, dblp_base_url)
            if not normalized.get("dedupe_key"):
                continue
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def rebuild_cache_index(
    records: List[Dict[str, Any]],
    dblp_base_url: str = "",
) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        normalized = sanitize_record_for_cache(record, dblp_base_url)
        if not normalized.get("dedupe_key"):
            continue
        for key in make_record_keys(normalized):
            index[key] = normalized
    return index


def restart_includes_stage(restart_stage: str, stage_name: str) -> bool:
    return RESTART_STAGE_ORDER[restart_stage] <= RESTART_STAGE_ORDER[stage_name]


def reset_record_from_stage(record: Dict[str, Any], restart_stage: str) -> Dict[str, Any]:
    updated = sanitize_record_for_cache(record)

    if restart_includes_stage(restart_stage, "match"):
        updated["match_checked"] = False
        updated["matched"] = False

    if restart_includes_stage(restart_stage, "detail"):
        updated["detail_status"] = "pending"
        updated["completed"] = False
        updated["skip_export"] = False

    if restart_includes_stage(restart_stage, "abstract"):
        updated["abstract"] = NA
        updated["abstract_source"] = NA
        updated["abstract_status"] = "pending"
        updated["abstract_signature"] = ""
        updated["completed"] = False
        updated["skip_export"] = False

    if restart_includes_stage(restart_stage, "affiliation"):
        updated["affiliations"] = [NA]
        updated["affiliation_source"] = NA
        updated["affiliation_mode"] = NA
        updated["affiliation_status"] = "pending"
        updated["affiliation_signature"] = ""
        updated["completed"] = False
        updated["skip_export"] = False

    if restart_includes_stage(restart_stage, "llm"):
        updated["title_translation"] = NA
        updated["title_translation_status"] = "pending"
        updated["summary_text"] = NA
        updated["summary_language"] = NA
        updated["summary_zh"] = NA
        updated["category"] = NA
        updated["ai_suggested_category"] = NA
        updated["reason"] = NA
        updated["llm_signature"] = ""
        updated["llm_status"] = "pending"
        updated["completed"] = False
        updated["skip_export"] = False

    return updated


def apply_restart_from_stage(
    restart_stage: str,
    config: Dict[str, Any],
    cache_path: str,
    cache_index: Dict[str, Dict[str, Any]],
    cache_enabled: bool,
    publ_query_cache_path: str,
    publ_query_cache_index: Dict[str, Dict[str, Any]],
    publ_query_cache_enabled: bool,
    dblp_base_url: str,
    logger: logging.Logger,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if restart_stage == "fetch":
        original_records = collect_unique_cached_records(cache_index)
        kept_records = [
            record
            for record in original_records
            if not is_record_in_config_scope(record, config)
        ]
        if cache_enabled:
            persist_cache_records(cache_path, kept_records, dblp_base_url)
        cache_index = rebuild_cache_index(kept_records, dblp_base_url)

        original_publ_entry_count = len(publ_query_cache_index)
        kept_publ_entries = {
            key: entry
            for key, entry in publ_query_cache_index.items()
            if not is_publ_query_entry_in_config_scope(entry, config)
        }
        if publ_query_cache_enabled:
            persist_publ_query_cache(publ_query_cache_path, kept_publ_entries)
        publ_query_cache_index = kept_publ_entries

        logger.info(
            "Restarting from fetch: cleared %s paper cache records and %s DBLP publ cache entries in the current scope",
            max(len(original_records) - len(kept_records), 0),
            max(original_publ_entry_count - len(kept_publ_entries), 0),
        )
        return cache_index, publ_query_cache_index

    if not cache_enabled:
        logger.warning(
            "--restart-from=%s requested, but cache.enabled=false so there is no paper cache to rewrite",
            restart_stage,
        )
        return cache_index, publ_query_cache_index

    rewritten_records: List[Dict[str, Any]] = []
    affected_count = 0
    for record in collect_unique_cached_records(cache_index):
        if is_record_in_config_scope(record, config):
            rewritten_records.append(reset_record_from_stage(record, restart_stage))
            affected_count += 1
        else:
            rewritten_records.append(sanitize_record_for_cache(record, dblp_base_url))

    persist_cache_records(cache_path, rewritten_records, dblp_base_url)
    cache_index = rebuild_cache_index(rewritten_records, dblp_base_url)

    logger.info(
        "Restarting from %s: rewrote %s paper cache records in the current scope",
        restart_stage,
        affected_count,
    )
    return cache_index, publ_query_cache_index


def collect_cached_candidates(
    cache_index: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for record in collect_unique_cached_records(cache_index):
        if not is_record_in_config_scope(record, config):
            continue

        title = clean_text(record.get("title"))
        if not title:
            continue

        record["matched"] = match_title(title, config["match_rules"])
        record["normalized_title"] = normalize_title(title)
        if record["matched"]:
            candidates.append(record)

    logger.info(
        "Loaded %s matched papers from cache for venues=%s years=%s-%s",
        len(candidates),
        config["dblp"]["venues"],
        config["dblp"]["year_start"],
        config["dblp"]["year_end"],
    )
    return candidates


def main() -> int:
    args = parse_args()
    configure_logging()
    logger = logging.getLogger("dblp_paper_crawler")

    if args.no_llm and args.test_ai:
        logger.error("--no-llm and --test-ai cannot be used together.")
        return 1
    try:
        config = load_config(args.config)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    if args.randomize_ua:
        chosen_user_agent = choose_random_user_agent(config["request"].get("user_agent", ""))
        try:
            config_path = persist_request_user_agent_to_config(args.config, chosen_user_agent)
        except Exception as exc:
            logger.error("Failed to persist randomized User-Agent to config: %s", exc)
            return 1
        config["request"]["user_agent"] = chosen_user_agent
        logger.info(
            "Randomized request.user_agent in %s -> %s",
            config_path,
            chosen_user_agent,
        )

    session = build_requests_session(config["request"])
    cache_enabled = bool(config["cache"]["enabled"])
    cache_path = config["cache"]["path"]
    cache_index = load_cache(cache_path, config["dblp"]["base_url"]) if cache_enabled else {}
    publ_query_cache_enabled = bool(config["cache"]["publ_query_enabled"])
    publ_query_cache_path = config["cache"]["publ_query_path"]
    publ_query_cache_index = (
        load_publ_query_cache(publ_query_cache_path, config["dblp"]["base_url"])
        if publ_query_cache_enabled
        else {}
    )
    publ_query_current_year_ttl_hours = float(
        config["cache"]["publ_query_current_year_ttl_hours"]
    )
    publ_query_max_refetch_rounds = int(config["cache"]["publ_query_max_refetch_rounds"])
    client = None if args.no_llm else build_openai_client(config)

    if not args.no_llm and client is None:
        logger.warning(
            "LLM is enabled but OpenAI client could not be initialized. "
            "Papers will be processed without successful summarization."
        )

    if args.test_ai:
        logger.info("Testing AI API configuration using config: %s", args.config)
        return 0 if test_ai_configuration(config, client, logger) else 1

    if args.restart_from:
        cache_index, publ_query_cache_index = apply_restart_from_stage(
            restart_stage=args.restart_from,
            config=config,
            cache_path=cache_path,
            cache_index=cache_index,
            cache_enabled=cache_enabled,
            publ_query_cache_path=publ_query_cache_path,
            publ_query_cache_index=publ_query_cache_index,
            publ_query_cache_enabled=publ_query_cache_enabled,
            dblp_base_url=config["dblp"]["base_url"],
            logger=logger,
        )

    logger.info(
        "Starting crawl for dblp_base_url=%s venues=%s years=%s-%s no_llm=%s restart_from=%s limit=%s cache_enabled=%s publ_query_cache_enabled=%s publ_query_max_refetch_rounds=%s user_agent=%s sleep_seconds=%s sleep_jitter_range=%s..%s",
        config["dblp"]["base_url"],
        config["dblp"]["venues"],
        config["dblp"]["year_start"],
        config["dblp"]["year_end"],
        args.no_llm,
        args.restart_from or NA,
        args.limit,
        cache_enabled,
        publ_query_cache_enabled,
        publ_query_max_refetch_rounds,
        config["request"]["user_agent"],
        config["request"]["sleep_seconds"],
        config["request"]["sleep_jitter_min_seconds"],
        config["request"]["sleep_jitter_max_seconds"],
    )

    matched_candidate_map: Dict[str, Dict[str, Any]] = {}
    if cache_enabled and cache_index:
        for record in collect_cached_candidates(cache_index, config, logger):
            dedupe_key = compute_primary_dedupe_key(record)
            if dedupe_key:
                matched_candidate_map[dedupe_key] = record

    papers = fetch_papers_from_dblp(
        venues=config["dblp"]["venues"],
        year_start=config["dblp"]["year_start"],
        year_end=config["dblp"]["year_end"],
        venue_overrides=config["dblp"].get("venue_stream_overrides", {}),
        dblp_base_url=config["dblp"]["base_url"],
        dblp_venue_api_url=config["dblp"]["venue_api_url"],
        dblp_publ_api_url=config["dblp"]["publ_api_url"],
        publ_query_cache_path=publ_query_cache_path,
        publ_query_cache_index=publ_query_cache_index,
        publ_query_cache_enabled=publ_query_cache_enabled,
        publ_query_current_year_ttl_hours=publ_query_current_year_ttl_hours,
        publ_query_max_refetch_rounds=publ_query_max_refetch_rounds,
        session=session,
        request_cfg=config["request"],
        logger=logger,
    )

    for paper in papers:
        cached = lookup_cached_record(cache_index, paper) or {}
        record = merge_record(paper, cached)
        record["matched"] = match_title(record.get("title", ""), config["match_rules"])
        record["normalized_title"] = normalize_title(record.get("title", ""))
        logger.info(
            "Title match=%s | venue=%s | year=%s | title=%s",
            record["matched"],
            record.get("venue", NA),
            record.get("year", NA),
            record.get("title", NA),
        )
        append_cache_record(
            record,
            cache_path,
            cache_index,
            cache_enabled,
            config["dblp"]["base_url"],
        )
        if not record["matched"]:
            continue

        dedupe_key = compute_primary_dedupe_key(record)
        if dedupe_key:
            matched_candidate_map[dedupe_key] = record

    matched_candidates = list(matched_candidate_map.values())

    if args.limit is not None:
        matched_candidates = matched_candidates[: max(args.limit, 0)]

    logger.info("Matched %s papers after applying title rules", len(matched_candidates))

    processed_records: List[Dict[str, Any]] = []
    exportable_count = 0

    for candidate in tqdm(matched_candidates, desc="Processing matched papers", unit="paper"):
        cached = lookup_cached_record(cache_index, candidate) or {}
        record = merge_record(candidate, cached)
        current_title = record.get("title", NA)

        need_detail = should_refresh_detail(record, config["dblp"]["base_url"])
        if record.get("completed") and record.get("skip_export") and not need_detail:
            logger.info("Skipping cached non-exportable record: %s", current_title)
            continue
        if (
            record.get("completed")
            and not record.get("skip_export")
            and not need_detail
            and not should_refresh_abstract(record)
            and not should_refresh_affiliations(record)
            and not should_refresh_llm(record, args.no_llm, config)
        ):
            processed_records.append(record)
            exportable_count += 1
            logger.info("Reusing completed cached paper: %s", current_title)
            continue

        logger.info("Processing paper: %s", current_title)

        if need_detail:
            detail = fetch_paper_detail(
                record.get("dblp_url", ""),
                config["dblp"]["base_url"],
                session,
                config["request"],
                logger,
            )
            record = merge_record(detail, record)
            append_cache_record(
                record,
                cache_path,
                cache_index,
                cache_enabled,
                config["dblp"]["base_url"],
            )

        logger.info(
            "Identifiers | doi=%s | paper_url=%s | dblp_url=%s",
            record.get("doi", NA),
            record.get("paper_url", NA),
            record.get("dblp_url", NA),
        )

        if not is_supported_paper_type(record.get("paper_type", "")):
            logger.info(
                "Skipping non-paper record type=%s title=%s",
                record.get("paper_type", NA),
                current_title,
            )
            record["skip_export"] = True
            record["completed"] = True
            append_cache_record(
                record,
                cache_path,
                cache_index,
                cache_enabled,
                config["dblp"]["base_url"],
            )
            continue

        if should_refresh_abstract(record):
            abstract_info = fetch_abstract(record, session, config["request"], logger)
            record = merge_record(abstract_info, record)
            append_cache_record(
                record,
                cache_path,
                cache_index,
                cache_enabled,
                config["dblp"]["base_url"],
            )
        logger.info(
            "Abstract status=%s source=%s title=%s",
            record.get("abstract_status", NA),
            record.get("abstract_source", NA),
            current_title,
        )

        if should_refresh_affiliations(record):
            affiliation_info = fetch_affiliations(record, session, config["request"], logger)
            record = merge_record(affiliation_info, record)
            append_cache_record(
                record,
                cache_path,
                cache_index,
                cache_enabled,
                config["dblp"]["base_url"],
            )
        logger.info(
            "Affiliation status=%s source=%s mode=%s title=%s",
            record.get("affiliation_status", NA),
            record.get("affiliation_source", NA),
            record.get("affiliation_mode", NA),
            current_title,
        )

        if args.no_llm:
            record.update(
                {
                    "title_translation": NA,
                    "title_translation_status": (
                        "disabled"
                        if not config["llm_output"]["title_translation_enabled"]
                        else "no_llm"
                    ),
                    "summary_text": NA,
                    "summary_language": config["llm_output"]["summary_language"],
                    "summary_zh": NA,
                    "category": NA,
                    "ai_suggested_category": NA,
                    "reason": NA,
                    "llm_signature": compute_llm_signature(record, config),
                    "llm_status": "no_llm",
                }
            )
            append_cache_record(
                record,
                cache_path,
                cache_index,
                cache_enabled,
                config["dblp"]["base_url"],
            )
        elif should_refresh_llm(record, args.no_llm, config):
            llm_info = summarize_and_classify(record, config, client, logger)
            record = merge_record(llm_info, record)
            append_cache_record(
                record,
                cache_path,
                cache_index,
                cache_enabled,
                config["dblp"]["base_url"],
            )
        logger.info("LLM status=%s title=%s", record.get("llm_status", NA), current_title)

        record["completed"] = True
        record["skip_export"] = False
        append_cache_record(
            record,
            cache_path,
            cache_index,
            cache_enabled,
            config["dblp"]["base_url"],
        )

        processed_records.append(record)
        exportable_count += 1
        logger.info("Current exportable paper count=%s", exportable_count)

    csv_count = export_csv(processed_records, config["output"]["csv_dir"], config, logger)
    logger.info("Done. CSV rows written=%s", csv_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
