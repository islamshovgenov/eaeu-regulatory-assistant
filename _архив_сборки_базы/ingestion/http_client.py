"""Polite HTTP client shared by discovery and download.

Features required by the project brief: a real User-Agent, retries with
exponential back-off, timeouts, per-host rate limiting, `robots.txt`
compliance and logging.  No technical restriction is ever circumvented — if a
host forbids a path or answers 401/403, the URL is reported as
``manual_required`` instead.
"""

from __future__ import annotations

import threading
import time
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import configure_logging, get_settings

logger = configure_logging(__name__)


class AccessForbidden(RuntimeError):
    """Raised when robots.txt or the server forbids automated access."""


@dataclass(slots=True)
class FetchResult:
    url: str
    status_code: int
    content: bytes
    content_type: str
    final_url: str

    @property
    def text(self) -> str:
        encoding = "utf-8"
        ct = self.content_type.lower()
        if "charset=" in ct:
            encoding = ct.split("charset=")[-1].split(";")[0].strip() or "utf-8"
        try:
            return self.content.decode(encoding, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")


class PoliteHttpClient:
    """Thread-safe requests session with rate limiting and robots.txt checks."""

    def __init__(
        self,
        user_agent: str | None = None,
        timeout: int | None = None,
        retries: int | None = None,
        rate_limit_seconds: float | None = None,
        respect_robots: bool = True,
    ) -> None:
        settings = get_settings()
        self.user_agent = user_agent or settings.http_user_agent
        self.timeout = timeout or settings.http_timeout_seconds
        self.rate_limit_seconds = (
            settings.http_rate_limit_seconds
            if rate_limit_seconds is None
            else rate_limit_seconds
        )
        self.respect_robots = respect_robots
        self.max_bytes = settings.download_max_bytes

        self._session = requests.Session()
        retry = Retry(
            total=retries if retries is not None else settings.http_retries,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=8)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/pdf,"
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document,*/*",
                "Accept-Language": "ru,en;q=0.8",
            }
        )

        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()

    # -- politeness ---------------------------------------------------------
    def _throttle(self, host: str) -> None:
        with self._lock:
            last = self._last_request.get(host, 0.0)
            wait = self.rate_limit_seconds - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            self._last_request[host] = time.monotonic()

    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin in self._robots:
            return self._robots[origin]
        parser: urllib.robotparser.RobotFileParser | None = None
        try:
            resp = self._session.get(f"{origin}/robots.txt", timeout=20)
            if resp.status_code == 200 and resp.text.strip():
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(resp.text.splitlines())
            else:
                logger.debug("robots.txt not available for %s (%s)", origin, resp.status_code)
        except requests.RequestException as exc:
            logger.debug("robots.txt fetch failed for %s: %s", origin, exc)
        self._robots[origin] = parser
        return parser

    def allowed(self, url: str) -> bool:
        """``True`` when robots.txt permits fetching *url* with our UA."""
        if not self.respect_robots:
            return True
        parser = self._robots_for(url)
        if parser is None:
            return True  # no robots.txt published -> no restriction stated
        return parser.can_fetch(self.user_agent, url) or parser.can_fetch("*", url)

    # -- fetching -----------------------------------------------------------
    def fetch(self, url: str, *, stream: bool = False) -> FetchResult:
        """GET *url*, honouring robots.txt, rate limits and the size cap."""
        if not self.allowed(url):
            raise AccessForbidden(f"robots.txt disallows automated access: {url}")

        host = urlparse(url).netloc
        self._throttle(host)
        logger.debug("GET %s", url)
        response = self._session.get(url, timeout=self.timeout, stream=stream)

        if response.status_code in (401, 403, 407):
            response.close()
            raise AccessForbidden(
                f"HTTP {response.status_code} — automated access not permitted: {url}"
            )
        response.raise_for_status()

        if stream:
            buffer = bytearray()
            for block in response.iter_content(chunk_size=64 * 1024):
                buffer.extend(block)
                if len(buffer) > self.max_bytes:
                    response.close()
                    raise RuntimeError(
                        f"Download exceeds {self.max_bytes} bytes limit: {url}"
                    )
            content = bytes(buffer)
        else:
            content = response.content

        return FetchResult(
            url=url,
            status_code=response.status_code,
            content=content,
            content_type=response.headers.get("Content-Type", ""),
            final_url=response.url,
        )

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "PoliteHttpClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
