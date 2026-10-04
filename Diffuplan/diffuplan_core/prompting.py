"""Prompt templates for editable-plan LLaDA2 inference."""

from __future__ import annotations


SYSTEM_PROMPT = (
    "You are an editable-plan search assistant. Follow the controller protocol exactly."
)

USER_RULES = "\n".join(
    [
        "Answer by maintaining an editable plan.",
        "Use only these action tags: <plan>, <think>, <search>, <update>, <replan>, and <answer>.",
        "Use newline block formatting:",
        "<tag>",
        "content",
        "</tag>",
        "",
        "Start with <plan>. Write concrete subquestions as #Q_1:, #Q_2:, ... .",
        "Each plan step should be an atomic answerable question, not an action note.",
        "Later plan steps may use earlier placeholders such as #A_1, but never future ones.",
        "For comparison questions, plan separate evidence steps before the comparison step.",
        "",
        "After <plan> or <information>, the next action must be a non-empty <think> block.",
        "Use <think> to briefly state what evidence is needed or what the observation changed.",
        "Every <search>, <update>, <replan>, and <answer> block must be immediately preceded by a non-empty <think> block.",
        "Use <search> for concise retrieval queries. Never generate <information>; it is environment context.",
        "",
        "Use <update> when evidence resolves answer placeholders in the plan. Content is only markers such as #A_1 or #A_1,#A_2.",
        "Use update to make unresolved future steps concrete; do not rewrite solved prefix steps.",
        "Use <replan> when the unresolved suffix needs a new path. Content is only step markers such as #Q_3 or #Q_3,#Q_4.",
        "Replan should preserve solved prefix steps and replace the failed unresolved suffix.",
        "After <update> or <replan>, stop and let the controller edit the plan.",
        "",
        "During controller-forced plan denoising only, <|reserved_token_6|> means <expand>.",
        "<expand> is valid only inside the editable plan region and must not appear in ordinary action generation.",
        "",
        "Use <answer> only when observed evidence supports the final answer. Output the exact short answer span only.",
    ]
)


def build_prompt(question: str) -> str:
    return (
        f"<role>SYSTEM</role>{SYSTEM_PROMPT}<|role_end|>"
        f"<role>HUMAN</role>{USER_RULES}\n\nQuestion: {question}<|role_end|>"
        f"<role>ASSISTANT</role>"
    )


def build_information_observation(info_text: str) -> str:
    return (
        "<|role_end|><role>HUMAN</role>"
        f"<information>\n{info_text}\n</information>\n"
        "<|role_end|><role>ASSISTANT</role>"
    )


def build_controller_observation(text: str) -> str:
    return (
        "<|role_end|><role>HUMAN</role>"
        f"\n{text.strip()}\n"
        "<|role_end|><role>ASSISTANT</role>"
    )


def build_invalid_action_observation() -> str:
    return build_controller_observation(
        "My previous action is invalid. "
        "If I want to plan, I should put the plan between <plan> and </plan> before a terminal action. "
        "If I want to search, I should put the query between <search> and </search>. "
        "If I want to update the plan, I should put only markers such as #A_1 between <update> and </update>. "
        "If I want to replan, I should put only step markers such as #Q_3 between <replan> and </replan>. "
        "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
        "Let me try again."
    )


def build_answer_direct_observation() -> str:
    return build_controller_observation(
        "The action budget is exhausted. I should now give the final answer between <answer> and </answer>."
    )
