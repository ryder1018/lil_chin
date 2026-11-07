"""把大腦串流吐出來的文字切成「可以先唸」的句子。

規則參考自 xiaozhi-esp32-server 的 TTS 切句（MIT），見 THIRD_PARTY_NOTICES.md：
  - 第一句遇到逗號就切 → 越快送進 TTS，機器人越快開口
  - 之後遇到句號、問號、驚嘆號、分號、冒號才切 → 句子完整，語調比較自然
  - 括號裡面不切（括號會被 text_filter 整段拿掉）
另外：太短的片段（只有「喔，」）先不送，和下一段合併，避免機器人只唸一個字就停頓。
"""
from __future__ import annotations

FIRST_CUT = set("，,、～~。！？!?；;：:\n")
LATER_CUT = set("。！？!?；;：:\n")
OPEN, CLOSE = set("(（「『"), set(")）」』")
MIN_FIRST_CHARS = 2  # 第一句至少幾個字才切（單獨一個「喔」「嗯」不切）


def _visible_len(s: str) -> int:
    return sum(1 for c in s if c.isalnum())


class Segmenter:
    def __init__(self) -> None:
        self.buf = ""
        self.first = True

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out = []
        while True:
            cut = self._find_cut()
            if cut < 0:
                break
            seg, self.buf = self.buf[: cut + 1], self.buf[cut + 1 :]
            out.append(seg)
            self.first = False
        return out

    def _find_cut(self) -> int:
        depth = 0
        marks = FIRST_CUT if self.first else LATER_CUT
        for i, c in enumerate(self.buf):
            if c in OPEN:
                depth += 1
            elif c in CLOSE:
                depth = max(0, depth - 1)
            elif c in marks and depth == 0:
                if c in ":：" and i > 0 and self.buf[i - 1].isdigit():
                    continue  # 時間「16:24」不切
                if self.first and _visible_len(self.buf[: i + 1]) < MIN_FIRST_CHARS:
                    continue
                # 標點連在一起（「！？」「。」」）一起帶走
                j = i
                while j + 1 < len(self.buf) and (self.buf[j + 1] in marks or self.buf[j + 1] in CLOSE):
                    j += 1
                if j + 1 == len(self.buf) and self.buf[j] not in "\n":
                    # 後面還沒來，可能還有標點；等下一段再決定（除非已經很長）
                    if len(self.buf) < 40:
                        return -1
                return j
        return -1

    def flush(self) -> list[str]:
        seg, self.buf = self.buf, ""
        return [seg] if seg.strip() else []
