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
    "記錄是這個空間的攝影機剛拍到的,已經照相關程度挑選過,就是「剛剛」發生的事。"
    "裡面的「一個人」「有人」就是在場的人,問「我」時直接依記錄回答。\n"
    "記錄裡沒有的事情,直接回「我沒有看到相關的畫面。」——不要推測、不要補完、不要說「可能」。\n"
    "提到時間就用記錄裡給的時間,不要自己算。"
)
"""The second line exists because the records are third-person and present
tense ("一個人在拿著水杯") while the questions the demo is built around are
first-person and past ("我的充電器最後放在哪"). Without it the model treats
resolving 一個人 to 我 as the speculation the third line forbids, and refuses on
a record that plainly answers the question."""

ANSWER_EXAMPLES = [
    {
        "role": "user",
        "content": "觀察記錄:\n2026-09-05 10:00(台北時間) 一個人拿著一支紅色的雨傘。\n\n問題:我剛剛有沒有拿雨傘",
    },
    {"role": "assistant", "content": "有,你剛剛拿著一支紅色的雨傘。"},
    {
        "role": "user",
        "content": "觀察記錄:\n2026-09-05 10:00(台北時間) 一個人拿著一支紅色的雨傘。\n\n問題:我剛剛有沒有戴帽子",
    },
    {"role": "assistant", "content": "我沒有看到相關的畫面。"},
]
"""Prepended to every `Answer` turn. One first-person 有沒有 question the record
covers, one it does not.

Measured on Qwen2.5-VL-3B over nine cases: "我 + 有沒有" refused every time on
the instructions alone, even when the record matched word for word, while the
same question in the third person ("有人拿著水杯嗎") answered. Three rewrites of
ANSWER_SYSTEM did not move it -- one made retrieval-grounded location questions
refuse as well. The pair below took it from 5/9 to 8/9 with all three refusal
cases still refusing, which is the half that backend.md 8.8 tests.

The refusal example is not decoration: showing only the answer direction
loosens the model into answering unwitnessed questions too."""

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
