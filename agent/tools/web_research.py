"""Web research tool - searches for ML papers, best practices, hyperparameters."""

import json
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
        # Block common internal hostnames
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


@tool
def search_web(query: str) -> str:
    """Search the web for ML research papers, best hyperparameters, model architectures,
    and techniques. Use this to find state-of-the-art approaches before building a model.

    Good queries:
    - "best hyperparameters for random forest classification tabular data"
    - "LSTM vs Transformer time series forecasting benchmark"
    - "XGBoost vs neural network small dataset comparison"
    - "learning rate schedule ResNet image classification"

    Args:
        query: Search query focused on ML techniques, papers, or benchmarks.
    """
    try:
        # Use DuckDuckGo instant answer API (no key required)
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "MLModelBuildingAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        results = []
        if data.get("AbstractText"):
            results.append(f"## Summary\n{data['AbstractText']}")
            if data.get("AbstractURL"):
                results.append(f"Source: {data['AbstractURL']}")

        for topic in data.get("RelatedTopics", [])[:8]:
            if isinstance(topic, dict) and topic.get("Text"):
                text = topic["Text"]
                url = topic.get("FirstURL", "")
                results.append(f"- {text}\n  {url}")
            elif isinstance(topic, dict) and topic.get("Topics"):
                for sub in topic["Topics"][:3]:
                    if sub.get("Text"):
                        results.append(f"- {sub['Text']}\n  {sub.get('FirstURL', '')}")

        if not results:
            return (
                f"No direct results found for: {query}\n"
                "Try rephrasing with more specific ML terminology, "
                "or use the terminal to run: pip install arxiv && python -c \"import arxiv; ...\""
            )

        return "\n\n".join(results)
    except Exception as e:
        return f"[ERROR] Web search failed: {e}\nYou can also try using the terminal to search with curl or Python."


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
            import re
            text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 10000:
                text = text[:10000] + "\n... [TRUNCATED]"
            return text
    except Exception as e:
        return f"[ERROR] Failed to fetch {url}: {e}"
