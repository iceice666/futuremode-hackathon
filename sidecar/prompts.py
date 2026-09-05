"""System prompts. docs/sidecar.md 3.2 — changing the text here is a contract
change and needs a version bump.

ANSWER_SYSTEM is reproduced verbatim from the document, including the
punctuation: the refusal sentence the backend compares against
(`mneme.sidecar.REFUSAL`) is byte-identical to the one quoted inside it, so a
"harmless" rewrite of 我沒有看到相關的畫面。 would break the hard refusal
acceptance test in backend.md 8.8.
"""

from __future__ import annotations

ANSWER_SYSTEM = (
    "你是一個離線影像記憶助理。只能根據提供的觀察記錄回答,用一到兩句中文。\n"
    "記錄裡沒有的事情,直接回「我沒有看到相關的畫面。」——不要推測、不要補完、不要說「可能」。\n"
    "提到時間就用記錄裡給的時間,不要自己算。"
)

REFUSAL = "我沒有看到相關的畫面。"
"""The fixed refusal sentence (sidecar.md 3.2). Returned directly when
`Answer.context` is empty: the document says an empty context still has to
produce this sentence, and asking a 4-bit LLM to obey that from the system
prompt alone is a coin flip we do not need to take."""

DESCRIBE_PROMPT = (
    "用一句中文描述這張畫面裡發生的事,說明有哪些人或物件、位置與動作。"
    "只寫一句,不要列點、不要說「這張圖片」。"
)
"""`Describe` must yield *one* Chinese sentence (sidecar.md 3.2): the summary
goes straight into SQLite and /api/events with no post-processing."""

OBJECTS_PROMPT = (
    "列出這張畫面裡出現的物件,用英文小寫名詞,逗號分隔,最多六個,不要其他文字。"
)
"""`Described.objects` is a list of short labels. Asking for them in a second
turn keeps DESCRIBE_PROMPT free of formatting instructions that leak into the
sentence."""
