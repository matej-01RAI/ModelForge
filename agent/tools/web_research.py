"""Web research tool - searches for ML papers, best practices, hyperparameters."""

import json
import re
import socket
import ipaddress
import urllib.parse
import urllib.request
from langchain_core.tools import tool


def _is_private_url(url: str) -> bool:
    """Check if a URL resolves to a private/internal IP address (SSRF protection)."""
    try:
        hostname = urllib.parse.urlparse(url).hostname
        if not hostname:
            return True
        if hostname in ("localhost", "0.0.0.0") or hostname.endswith(".local"):
            return True
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return True
    except (socket.gaierror, ValueError):
        return False
    return False


def _duckduckgo_search(query: str, max_results: int = 8) -> list:
    """Search DuckDuckGo HTML and parse results. Returns list of {title, url, snippet}."""
    encoded = urllib.parse.quote(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; MLModelBuildingAgent/1.0)",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    results = []
    # Parse result blocks — each result is in a <div class="result ...">
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html, re.DOTALL,
    )
    for raw_url, raw_title, raw_snippet in blocks[:max_results]:
        # DuckDuckGo wraps URLs in a redirect — extract the actual URL
        actual_url = raw_url
        if "uddg=" in raw_url:
            match = re.search(r'uddg=([^&]+)', raw_url)
            if match:
                actual_url = urllib.parse.unquote(match.group(1))

        title = re.sub(r'<[^>]+>', '', raw_title).strip()
        snippet = re.sub(r'<[^>]+>', '', raw_snippet).strip()
        if title and actual_url:
            results.append({"title": title, "url": actual_url, "snippet": snippet})

    return results


@tool
def search_web(query: str) -> str:
    """Search the web for ML research, best hyperparameters, model architectures,
    and techniques. Returns titles, URLs, and snippets from search results.

    Good queries:
    - "best hyperparameters random forest classification tabular data"
    - "LSTM vs Transformer time series forecasting benchmark"
    - "XGBoost vs neural network small dataset comparison"
    - "scikit-learn pipeline categorical encoding missing values"

    Args:
        query: Search query focused on ML techniques, papers, or benchmarks.
    """
    try:
        results = _duckduckgo_search(query, max_results=8)

        if not results:
            return (
                f"No results found for: {query}\n"
                "Try rephrasing or use fetch_url on a known documentation page."
            )

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**")
            lines.append(f"   {r['url']}")
            if r['snippet']:
                lines.append(f"   {r['snippet']}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"[ERROR] Web search failed: {e}\nYou can also use the terminal: curl or python requests."


@tool
def fetch_url(url: str) -> str:
    """Fetch the text content of a URL. Useful for reading documentation pages,
    GitHub READMEs, or paper abstracts.

    Args:
        url: The URL to fetch.
    """
    try:
        if _is_private_url(url):
            return "[ERROR] Cannot fetch internal/private network URLs."
        req = urllib.request.Request(url, headers={"User-Agent": "MLModelBuildingAgent/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text" not in content_type and "json" not in content_type:
                return f"[SKIPPED] Non-text content type: {content_type}"
            raw = resp.read(100_000).decode("utf-8", errors="replace")
            # Strip HTML tags roughly for readability
            text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 10000:
                text = text[:10000] + "\n... [TRUNCATED]"
            return text
    except Exception as e:
        return f"[ERROR] Failed to fetch {url}: {e}"
