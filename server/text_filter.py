"""大腦輸出 → TTS 之前的文字清理。

模型偶爾會混進簡體字、大陸用語、括號動作、emoji，用 prompt 壓不乾淨，
在送給 TTS 之前用規則統一修掉（確定性、幾乎零延遲）。
server/brain.py 的每一段輸出都會經過這裡；scripts/eval_brain.py 也用它模擬送進 TTS 的文字。
"""
from __future__ import annotations

import re
from functools import lru_cache

import opencc

_S2TW = opencc.OpenCC("s2tw")  # 只做字形轉換（台灣字形），不做詞彙轉換，避免誤改
_KEEP = {"台"}  # s2tw 會把「台」轉「臺」，台灣都正常使用

# 大陸用語 → 台灣用語（只放不會誤傷的）
TERMS = {
    "哦": "喔", "老板": "老闆", "這兒": "這裡", "那兒": "那裡", "哪兒": "哪裡",
    "視頻": "影片", "軟件": "軟體", "硬件": "硬體", "出租車": "計程車", "地鐵": "捷運",
    "咖喱": "咖哩", "土豆": "馬鈴薯", "西紅柿": "番茄", "酸奶": "優格", "網絡": "網路",
    "屏幕": "螢幕", "內存": "記憶體", "硬盤": "硬碟", "鼠標": "滑鼠", "短信": "簡訊",
    "打印": "列印", "默認": "預設", "信息": "訊息", "充電寶": "行動電源", "公交": "公車",
}

_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U00002B00-\U00002BFF️‍]"
)
_PAREN = re.compile(r"[（(][^）)]{0,30}[）)]")  # （偷偷笑）之類的動作描寫
_STAR = re.compile(r"\*{1,2}([^*]{0,30})\*{1,2}")  # *抱抱* / **粗體**
_BULLET = re.compile(r"(^|\n)\s*([-*•●]|\d+[\.\)、])\s*")
_SPACES = re.compile(r"[ \t　]{2,}")
_DUP_PUNCT = re.compile(r"，+\s*[，。！？!?]")  # 「啦，，」「喔，。」→ 保留最後一個標點


@lru_cache(maxsize=None)
def is_simplified_char(c: str) -> bool:
    """單獨轉換就會變的字才算簡體（这→這）。「只」「台」這種繁體也合法的字不算。"""
    return c not in _KEEP and _S2TW.convert(c) != c


def to_taiwan(text: str) -> str:
    # 整句轉換才有上下文（发→發/髮），但只採用「原本是簡體字」位置的結果，
    # 避免把已經正確的繁體改壞（例如「只是」被轉成「隻是」）
    if any(is_simplified_char(c) for c in text):
        conv = _S2TW.convert(text)
        if len(conv) != len(text):
            conv = "".join(_S2TW.convert(c) for c in text)
        text = "".join(cv if is_simplified_char(c) else c for c, cv in zip(text, conv))
    for k, v in TERMS.items():
        text = text.replace(k, v)
    return text


def for_speech(text: str) -> str:
    """清成可以直接唸的文字。"""
    text = to_taiwan(text)
    text = _EMOJI.sub("", text)
    text = _PAREN.sub("", text)
    text = _STAR.sub(r"\1", text)
    text = _BULLET.sub(r"\1", text)
    text = text.replace("\n", " ")
    # 「～」對 TTS 沒意義，而且 xiaozhi 切第一句只認半形 ~ 和逗號：轉成逗號，唸起來有停頓、也能更早送出第一句
    text = text.replace("～", "，").replace("~", "，")
    text = _DUP_PUNCT.sub(lambda m: m.group(0)[-1], text)
    text = _SPACES.sub(" ", text)
    return text.strip()


if __name__ == "__main__":
    for s in [
        "我在这儿陪你，随时听你说。",
        "不是 ChatGPT 哦～（偷偷笑）你喜歡我嗎？",
        "台北今天天气不错😊 *抱抱* 去看視頻吧",
    ]:
        print(s, "→", for_speech(s))
