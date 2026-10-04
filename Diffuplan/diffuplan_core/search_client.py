"""Small HTTP client for the Search-R1 retriever service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class SearchClient:
    url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 3
    timeout: float = 60.0
    fail_on_error: bool = False

    def batch_search(self, queries: list[str], topk: Optional[int] = None) -> list[str]:
        if not queries:
            return []

        payload = {
            "queries": queries,
            "topk": int(topk or self.topk),
            "return_scores": True,
        }
        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            results = response.json().get("result", [])
            return [self._passages_to_text(item) for item in results]
        except Exception as exc:
            if self.fail_on_error:
                raise RuntimeError(f"Retriever request failed: {type(exc).__name__}: {exc}") from exc
            print(f"[SearchClient] retriever request failed: {type(exc).__name__}: {exc}", flush=True)
            return [""] * len(queries)

    def search(self, query: str, topk: Optional[int] = None) -> str:
        results = self.batch_search([query], topk=topk)
        return results[0] if results else ""

    @staticmethod
    def _passages_to_text(retrieval_result: Any) -> str:
        lines: list[str] = []
        for idx, doc_item in enumerate(retrieval_result or []):
            document = doc_item.get("document", {}) if isinstance(doc_item, dict) else {}
            contents = str(document.get("contents", ""))
            title, _, body = contents.partition("\n")
            if not body:
                body = title
                title = ""
            prefix = f"Doc {idx + 1}"
            if title:
                prefix += f"(Title: {title})"
            lines.append(f"{prefix} {body}".rstrip())
        return "\n".join(lines)
