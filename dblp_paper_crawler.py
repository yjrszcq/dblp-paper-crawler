from __future__ import annotations

import argparse
import csv
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
DBLP_VENUE_API = "https://dblp.org/search/venue/api"
DBLP_PUBL_API = "https://dblp.org/search/publ/api"
CROSSREF_WORKS_API = "https://api.crossref.org/works"
OPENALEX_WORKS_API = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_PAPER_API = "https://api.semanticscholar.org/graph/v1/paper"
SEMANTIC_SCHOLAR_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API = "https://export.arxiv.org/api/query"


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
        "--resume-only",
        action="store_true",
        help="Do not fetch new DBLP records. Resume processing only from existing cache records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N matched papers.",
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

    request_cfg = config["request"]
    request_cfg["sleep_seconds"] = float(request_cfg.get("sleep_seconds", 1))
    request_cfg["timeout_seconds"] = float(request_cfg.get("timeout_seconds", 20))
    request_cfg["max_retries"] = int(request_cfg.get("max_retries", 3))

    return config


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
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
    jitter = random.uniform(0, 0.25)
    delay = base_sleep + min(attempt, 3) * 0.5 + jitter
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
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_doi(doi: Optional[str]) -> str:
    if not doi:
        return ""
    value = clean_text(doi).strip()
    value = value.replace("https://doi.org/", "").replace("http://doi.org/", "")
    value = value.replace("doi:", "")
    return value.strip().lower()


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


def compute_primary_dedupe_key(record: Dict[str, Any]) -> str:
    doi = normalize_doi(record.get("doi"))
    if doi:
        return f"doi:{doi}"
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
    dblp_url = clean_text(record.get("dblp_url"))
    normalized_title = normalize_title(record.get("title", ""))
    if doi:
        keys.append(f"doi:{doi}")
    if dblp_url and dblp_url != NA:
        keys.append(f"dblp:{dblp_url}")
    if normalized_title:
        keys.append(f"title:{normalized_title}")
    return keys


def sanitize_record_for_cache(record: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in record.items():
        if key.startswith("_"):
            continue
        if isinstance(value, Path):
            clean[key] = str(value)
        else:
            clean[key] = value
    clean["normalized_title"] = normalize_title(clean.get("title", ""))
    clean["doi"] = normalize_doi(clean.get("doi"))
    clean["dedupe_key"] = compute_primary_dedupe_key(clean)
    return clean


def load_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
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
            normalized = sanitize_record_for_cache(record)
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
) -> None:
    if not enabled:
        return
    normalized = sanitize_record_for_cache(record)
    if not normalized.get("dedupe_key"):
        return

    ensure_parent_directory(cache_path)
    with Path(cache_path).expanduser().resolve().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    for key in make_record_keys(normalized):
        cache_index[key] = normalized


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
        return partial_score >= 90 or token_score >= 90
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
        if "dblp.org" in parsed.netloc or "dblp.uni-trier.de" in parsed.netloc:
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
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Dict[str, str]]:
    override_value = overrides.get(venue_name)
    if override_value:
        resolved_override = resolve_override_stream_query(
            override_value, session, request_cfg, logger
        )
        if resolved_override:
            resolved_override["input_venue"] = venue_name
            logger.info("Resolved venue override for %s -> %s", venue_name, resolved_override["stream_query"])
            return resolved_override
        logger.warning("Failed to resolve venue override for %s: %s", venue_name, override_value)

    payload = request_json(
        session,
        DBLP_VENUE_API,
        request_cfg,
        logger,
        params={"q": venue_name, "format": "json", "h": 10, "c": 0},
    )
    hits = extract_search_hits(payload)
    candidates: List[Dict[str, Any]] = []
    for hit in hits:
        info = hit.get("info", hit)
        candidate_url = clean_text(info.get("url"))
        if not candidate_url or candidate_url == NA:
            key = clean_text(info.get("key"))
            if key:
                candidate_url = f"https://dblp.org/{key}"
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


def parse_paper_from_search_hit(hit: Dict[str, Any], venue_hint: str) -> Optional[Dict[str, Any]]:
    info = hit.get("info", hit)
    if should_skip_search_hit(info):
        return None

    title = clean_text(info.get("title"))
    dblp_url = clean_text(info.get("url"))
    key = clean_text(info.get("key"))
    if not dblp_url and key:
        dblp_url = f"https://dblp.org/rec/{key}"
    if not title or not dblp_url:
        return None

    ee_value = info.get("ee")
    ee_list = [clean_text(item) for item in to_list(ee_value) if clean_text(item)]
    paper_url = choose_paper_url(ee_list, normalize_doi(info.get("doi")))
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


def fetch_papers_from_dblp(
    venues: List[str],
    year_start: int,
    year_end: int,
    venue_overrides: Dict[str, str],
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    papers: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for venue_name in venues:
        resolved = resolve_venue(venue_name, venue_overrides, session, request_cfg, logger)
        if not resolved:
            continue

        stream_query = resolved["stream_query"]
        for year in range(year_start, year_end + 1):
            logger.info("Fetching DBLP papers for venue=%s year=%s", venue_name, year)
            offset = 0
            page_size = 1000
            while True:
                payload = request_json(
                    session,
                    DBLP_PUBL_API,
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
                hits = extract_search_hits(payload)
                if not hits:
                    break

                for hit in hits:
                    paper = parse_paper_from_search_hit(hit, venue_name)
                    if not paper:
                        continue
                    if paper["dblp_url"] in seen_urls:
                        continue
                    seen_urls.add(paper["dblp_url"])
                    papers.append(paper)

                if len(hits) < page_size:
                    break
                offset += len(hits)

    logger.info("Fetched %s candidate DBLP records before title matching", len(papers))
    return papers


def choose_paper_url(ee_urls: List[str], doi: str) -> str:
    clean_urls = [clean_text(url) for url in ee_urls if clean_text(url)]
    for url in clean_urls:
        if "dblp.org" in url:
            continue
        if "doi.org" not in url:
            return url
    for url in clean_urls:
        if "dblp.org" not in url:
            return url
    if doi:
        return f"https://doi.org/{doi}"
    return NA


def fetch_paper_detail(
    dblp_url: str,
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
    paper_url = choose_paper_url(ee_urls, doi)

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


def select_best_title_candidate(
    candidates: Iterable[Dict[str, Any]],
    title: str,
    min_score: int = 80,
) -> Optional[Dict[str, Any]]:
    best_item: Optional[Dict[str, Any]] = None
    best_score = min_score
    for item in candidates:
        candidate_title = first_non_empty(
            item.get("title"),
            item.get("display_name"),
            item.get("name"),
        )
        if candidate_title == NA:
            continue
        score = title_similarity(title, candidate_title)
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
        return None

    payload = request_json(
        session,
        f"{CROSSREF_WORKS_API}/{quote(doi, safe='')}",
        request_cfg,
        logger,
    )
    message = payload.get("message") if payload else None
    source_cache["crossref_doi"] = message
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
        return None

    payload = request_json(
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
        title,
        min_score=78,
    )
    source_cache["crossref_title"] = best_item
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
        return None

    payload = request_json(
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
        return None

    payload = request_json(
        session,
        OPENALEX_WORKS_API,
        request_cfg,
        logger,
        params={"search": title, "per-page": 5},
    )
    results = payload.get("results", []) if payload else []
    best_item = select_best_title_candidate(results, title, min_score=80)
    source_cache["openalex_title"] = best_item
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
        return None

    payload = request_json(
        session,
        f"{SEMANTIC_SCHOLAR_PAPER_API}/DOI:{quote(doi, safe='')}",
        request_cfg,
        logger,
        params={"fields": "title,abstract,authors.name,authors.affiliations,venue,year,url,externalIds"},
    )
    source_cache["semantic_doi"] = payload
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
        return None

    payload = request_json(
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
    best_item = select_best_title_candidate(data, title, min_score=80)
    source_cache["semantic_title"] = best_item
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
        return None

    try:
        soup = BeautifulSoup(response.text, "xml")
        summary = soup.find("summary")
        abstract = clean_text(summary.get_text(" ", strip=True) if summary else "")
        source_cache["arxiv_abstract"] = abstract or None
        return source_cache["arxiv_abstract"]
    except Exception:
        source_cache["arxiv_abstract"] = None
        return None


def fetch_abstract(
    paper: Dict[str, Any],
    session: requests.Session,
    request_cfg: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, str]:
    doi = normalize_doi(paper.get("doi"))
    title = clean_text(paper.get("title"))
    paper_url = clean_text(paper.get("paper_url"))
    if not any([doi, title, paper_url]):
        return {
            "abstract": NA,
            "abstract_source": NA,
            "abstract_status": "missing_identifier",
        }

    try:
        crossref_doi = get_crossref_by_doi(paper, session, request_cfg, logger)
        if crossref_doi and crossref_doi.get("abstract"):
            return {
                "abstract": clean_crossref_abstract(crossref_doi["abstract"]),
                "abstract_source": "Crossref",
                "abstract_status": "success",
            }

        openalex_doi = get_openalex_by_doi(paper, session, request_cfg, logger)
        openalex_abstract = reconstruct_openalex_abstract(openalex_doi or {})
        if openalex_abstract != NA:
            return {
                "abstract": openalex_abstract,
                "abstract_source": "OpenAlex",
                "abstract_status": "success",
            }

        semantic_doi = get_semantic_scholar_by_doi(paper, session, request_cfg, logger)
        semantic_abstract = clean_text((semantic_doi or {}).get("abstract"))
        if semantic_abstract:
            return {
                "abstract": semantic_abstract,
                "abstract_source": "Semantic Scholar",
                "abstract_status": "success",
            }

        arxiv_abstract = fetch_arxiv_abstract(paper, session, request_cfg, logger)
        if arxiv_abstract:
            return {
                "abstract": arxiv_abstract,
                "abstract_source": "arXiv",
                "abstract_status": "success",
            }

        crossref_title = search_crossref_by_title(paper, session, request_cfg, logger)
        if crossref_title and crossref_title.get("abstract"):
            return {
                "abstract": clean_crossref_abstract(crossref_title["abstract"]),
                "abstract_source": "Crossref",
                "abstract_status": "success",
            }

        openalex_title = search_openalex_by_title(paper, session, request_cfg, logger)
        openalex_title_abstract = reconstruct_openalex_abstract(openalex_title or {})
        if openalex_title_abstract != NA:
            return {
                "abstract": openalex_title_abstract,
                "abstract_source": "OpenAlex",
                "abstract_status": "success",
            }

        semantic_title = search_semantic_scholar_by_title(paper, session, request_cfg, logger)
        semantic_title_abstract = clean_text((semantic_title or {}).get("abstract"))
        if semantic_title_abstract:
            return {
                "abstract": semantic_title_abstract,
                "abstract_source": "Semantic Scholar",
                "abstract_status": "success",
            }

        return {
            "abstract": NA,
            "abstract_source": NA,
            "abstract_status": "not_found",
        }
    except requests.RequestException:
        return {
            "abstract": NA,
            "abstract_source": NA,
            "abstract_status": "request_failed",
        }
    except Exception:
        logger.exception("Unexpected error while fetching abstract for %s", title)
        return {
            "abstract": NA,
            "abstract_source": NA,
            "abstract_status": "parse_failed",
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
    authors = paper.get("authors") or []
    if not authors:
        return {
            "authors": [],
            "affiliations": [NA],
            "affiliation_source": NA,
            "affiliation_mode": NA,
            "affiliation_status": "not_found",
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
            }

        return {
            "authors": authors,
            "affiliations": [NA],
            "affiliation_source": NA,
            "affiliation_mode": NA,
            "affiliation_status": "not_found",
        }
    except requests.RequestException:
        return {
            "authors": authors,
            "affiliations": [NA],
            "affiliation_source": NA,
            "affiliation_mode": NA,
            "affiliation_status": "request_failed",
        }
    except Exception:
        logger.exception("Unexpected error while fetching affiliations for %s", paper.get("title"))
        return {
            "authors": authors,
            "affiliations": [NA],
            "affiliation_source": NA,
            "affiliation_mode": NA,
            "affiliation_status": "parse_failed",
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
        record.get("paper_url"),
        f"https://doi.org/{doi}" if doi else "",
        record.get("dblp_url"),
    )


def build_llm_default_result(config: Dict[str, Any], llm_status: str) -> Dict[str, str]:
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
        return build_llm_default_result(config, "missing_input")

    if not has_abstract and not title_translation_enabled:
        return build_llm_default_result(config, "no_abstract")

    if client is None:
        return build_llm_default_result(config, "llm_unavailable")

    categories = config["classification"]["categories"]
    allow_new_category = bool(config["classification"]["allow_new_category"])
    openai_cfg = config["openai"]
    summary_language_name = "中文" if summary_language == "zh" else "英文"

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
    return build_llm_default_result(config, "failed")


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
    status = clean_text(record.get("abstract_status")).lower()
    if status == "success" and clean_text(record.get("abstract")) not in {"", NA}:
        return False
    if status in {"not_found", "missing_identifier"}:
        return False
    return True


def should_refresh_affiliations(record: Dict[str, Any]) -> bool:
    status = clean_text(record.get("affiliation_status")).lower()
    if status == "success":
        return False
    if status == "not_found":
        return False
    return True


def should_refresh_llm(record: Dict[str, Any], no_llm: bool, config: Dict[str, Any]) -> bool:
    status = clean_text(record.get("llm_status")).lower()
    if no_llm and status == "no_llm":
        return False
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
        if key not in merged or is_meaningful_value(value) or isinstance(value, bool):
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


def collect_resume_candidates(
    cache_index: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    venues = set(config["dblp"]["venues"])
    year_start = config["dblp"]["year_start"]
    year_end = config["dblp"]["year_end"]
    candidates: List[Dict[str, Any]] = []

    for record in collect_unique_cached_records(cache_index):
        source_venue = first_non_empty(record.get("source_venue"), record.get("venue"))
        if source_venue not in venues:
            continue

        year = safe_int(record.get("year"))
        if year is not None and not (year_start <= year <= year_end):
            continue

        title = clean_text(record.get("title"))
        if not title:
            continue

        record["matched"] = match_title(title, config["match_rules"])
        record["normalized_title"] = normalize_title(title)
        if record["matched"]:
            candidates.append(record)

    logger.info(
        "Resume-only mode loaded %s matched cached papers for venues=%s years=%s-%s",
        len(candidates),
        config["dblp"]["venues"],
        year_start,
        year_end,
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

    session = build_requests_session()
    cache_enabled = bool(config["cache"]["enabled"])
    cache_path = config["cache"]["path"]
    cache_index = load_cache(cache_path) if cache_enabled else {}
    client = None if args.no_llm else build_openai_client(config)

    if not args.no_llm and client is None:
        logger.warning(
            "LLM is enabled but OpenAI client could not be initialized. "
            "Papers will be processed without successful summarization."
        )

    if args.test_ai:
        logger.info("Testing AI API configuration using config: %s", args.config)
        return 0 if test_ai_configuration(config, client, logger) else 1

    logger.info(
        "Starting crawl for venues=%s years=%s-%s no_llm=%s resume_only=%s limit=%s cache_enabled=%s",
        config["dblp"]["venues"],
        config["dblp"]["year_start"],
        config["dblp"]["year_end"],
        args.no_llm,
        args.resume_only,
        args.limit,
        cache_enabled,
    )

    matched_candidates: List[Dict[str, Any]] = []
    if args.resume_only:
        if not cache_enabled:
            logger.error("--resume-only requires cache.enabled=true in the config.")
            return 1
        if not cache_index:
            logger.warning("Resume-only mode found no cache records to process.")
            return 0
        matched_candidates = collect_resume_candidates(cache_index, config, logger)
    else:
        papers = fetch_papers_from_dblp(
            venues=config["dblp"]["venues"],
            year_start=config["dblp"]["year_start"],
            year_end=config["dblp"]["year_end"],
            venue_overrides=config["dblp"].get("venue_stream_overrides", {}),
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
            append_cache_record(record, cache_path, cache_index, cache_enabled)
            if record["matched"]:
                matched_candidates.append(record)

    if args.limit is not None:
        matched_candidates = matched_candidates[: max(args.limit, 0)]

    logger.info("Matched %s papers after applying title rules", len(matched_candidates))

    processed_records: List[Dict[str, Any]] = []
    exportable_count = 0

    for candidate in tqdm(matched_candidates, desc="Processing matched papers", unit="paper"):
        cached = lookup_cached_record(cache_index, candidate) or {}
        record = merge_record(candidate, cached)
        current_title = record.get("title", NA)

        need_detail = (
            clean_text(record.get("detail_status")).lower() != "success"
            or not normalize_doi(record.get("doi"))
            or clean_text(record.get("paper_url")) in {"", NA}
            or not record.get("authors")
        )
        if record.get("completed") and record.get("skip_export") and not need_detail:
            logger.info("Skipping cached non-exportable record: %s", current_title)
            continue
        if (
            record.get("completed")
            and not record.get("skip_export")
            and not need_detail
            and not should_refresh_abstract(record)
            and not should_refresh_affiliations(record)
            and (args.no_llm or not should_refresh_llm(record, args.no_llm, config))
        ):
            processed_records.append(record)
            exportable_count += 1
            logger.info("Reusing completed cached paper: %s", current_title)
            continue

        logger.info("Processing paper: %s", current_title)

        if need_detail:
            detail = fetch_paper_detail(record.get("dblp_url", ""), session, config["request"], logger)
            record = merge_record(detail, record)
            append_cache_record(record, cache_path, cache_index, cache_enabled)

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
            append_cache_record(record, cache_path, cache_index, cache_enabled)
            continue

        if should_refresh_abstract(record):
            abstract_info = fetch_abstract(record, session, config["request"], logger)
            record = merge_record(abstract_info, record)
            append_cache_record(record, cache_path, cache_index, cache_enabled)
        logger.info(
            "Abstract status=%s source=%s title=%s",
            record.get("abstract_status", NA),
            record.get("abstract_source", NA),
            current_title,
        )

        if should_refresh_affiliations(record):
            affiliation_info = fetch_affiliations(record, session, config["request"], logger)
            record = merge_record(affiliation_info, record)
            append_cache_record(record, cache_path, cache_index, cache_enabled)
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
                    "llm_status": "no_llm",
                }
            )
            append_cache_record(record, cache_path, cache_index, cache_enabled)
        elif should_refresh_llm(record, args.no_llm, config):
            llm_info = summarize_and_classify(record, config, client, logger)
            record = merge_record(llm_info, record)
            append_cache_record(record, cache_path, cache_index, cache_enabled)
        logger.info("LLM status=%s title=%s", record.get("llm_status", NA), current_title)

        record["completed"] = True
        record["skip_export"] = False
        append_cache_record(record, cache_path, cache_index, cache_enabled)

        processed_records.append(record)
        exportable_count += 1
        logger.info("Current exportable paper count=%s", exportable_count)

    csv_count = export_csv(processed_records, config["output"]["csv_dir"], config, logger)
    logger.info("Done. CSV rows written=%s", csv_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
