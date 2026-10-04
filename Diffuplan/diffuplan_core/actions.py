"""Control-tag parsing for editable-plan search rollouts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Optional


ACTION_TAGS = ("think", "plan", "search", "update", "replan", "answer")
ACTION_CLOSE_TAGS = tuple(f"</{tag}>" for tag in ACTION_TAGS)
TERMINAL_ACTION_TAGS = ("search", "update", "replan", "answer")
TERMINAL_ACTION_CLOSE_TAGS = tuple(f"</{tag}>" for tag in TERMINAL_ACTION_TAGS)
_ACTION_RE = re.compile(
    r"<(?P<tag>think|plan|search|update|replan|answer)>(?P<content>.*?)</(?P=tag)>",
    re.DOTALL,
)


@dataclass(frozen=True)
class Action:
    """A complete model action emitted with XML-like control tags."""

    kind: str
    content: str
    raw: str


def postprocess_to_first_action(text: str) -> str:
    """Trim generated text at the first terminal action closing tag."""

    best_end: Optional[int] = None
    for close_tag in TERMINAL_ACTION_CLOSE_TAGS:
        idx = text.find(close_tag)
        if idx == -1:
            continue
        end = idx + len(close_tag)
        if best_end is None or end < best_end:
            best_end = end
    return text[:best_end] if best_end is not None else text


def iter_action_blocks(text: str) -> Iterator[Action]:
    """Yield all complete control-tag blocks in textual order."""

    for match in _ACTION_RE.finditer(text or ""):
        yield Action(
            kind=match.group("tag"),
            content=match.group("content").strip(),
            raw=match.group(0),
        )


def parse_first_action_block(text: str) -> Optional[Action]:
    """Return the first complete control-tag block, terminal or not."""

    return next(iter_action_blocks(text), None)


def parse_action(text: str) -> Optional[Action]:
    """Return the first complete terminal action in a generated segment."""

    match = None
    for candidate in _ACTION_RE.finditer(text or ""):
        if candidate.group("tag") in TERMINAL_ACTION_TAGS:
            match = candidate
            break
    if match is None:
        return None
    return Action(
        kind=match.group("tag"),
        content=match.group("content").strip(),
        raw=match.group(0),
    )


def parse_marker_list(content: str) -> list[str]:
    """Parse '#A_1,#A_2' or '#Q_3, #Q_4' into marker strings."""

    markers: list[str] = []
    for item in re.split(r"[,\s]+", content.strip()):
        item = item.strip()
        if not item:
            continue
        if re.fullmatch(r"#[AQ]_\d+", item):
            markers.append(item)
    return markers


def extract_answer(text: str) -> str:
    """Extract the first answer payload from a full output."""

    match = re.search(r"<answer>(.*?)</answer>", text or "", re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_latest_plan_inner(text: str) -> str:
    """Return the inner text of the latest complete <plan> span."""

    stack: list[int] = []
    latest: tuple[int, int] | None = None
    for match in re.finditer(r"</?plan>", text or ""):
        if match.group(0) == "<plan>":
            stack.append(match.end())
            continue
        if stack:
            latest = (stack.pop(), match.start())
    if latest is None:
        return ""
    start, end = latest
    return (text or "")[start:end]
