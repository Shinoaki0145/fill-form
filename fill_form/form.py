"""Selected-field dictation; no LLM, TTS or server persistence."""

import json
import re
import time


FIELDS = {"full_name": "Họ tên", "phone": "Số điện thoại", "address": "Địa chỉ"}
DIGITS = dict(zip("không một hai ba bốn năm sáu bảy tám chín".split(), "0123456789"))
DIGITS.update({"mốt": "1", "tư": "4", "lăm": "5", "bẩy": "7"})


def normalize_value(field, text):
    if field not in FIELDS or not isinstance(text, str) or len(text) > 2000:
        raise ValueError("Trường hoặc nội dung không hợp lệ (tối đa 2000 ký tự)")
    text = " ".join(text.split())
    if field != "phone" or not text:
        return text
    tokens = re.sub(r"[.,;:()\-]", " ", text.lower()).split()
    result = "".join(DIGITS.get(token, token) for token in tokens)
    if not re.fullmatch(r"\+?[0-9]{1,15}", result):
        raise ValueError("Số điện thoại: đọc từng chữ số hoặc nhập số, tối đa 15 chữ số")
    return result


class FormSession:
    recover_asr_error = True

    def __init__(self, owner):
        self.origin = time.perf_counter()
        self.generation = 0
        self.owner = owner
        self.values = dict.fromkeys(FIELDS, "")
        self.revisions = dict.fromkeys(FIELDS, 0)
        self.selected = "full_name"
        self.targets = {}
        self.transcript = ""
        self.error = ""

    def speech_started(self):
        self.generation += 1
        self.targets[self.owner.runner.turn.turn_id] = self.owner.file_target or (
            self.selected, self.revisions[self.selected]
        )
        self.emit("speech_started")

    def emit(self, event, **data):
        if event == "asr_failed":
            self.error = "ASR lỗi; dữ liệu cũ được giữ. Hãy nói lại hoặc nhập tay."
        self.owner.event(dict(event=event, t=round(time.perf_counter() - self.origin, 3),
                              generation=self.generation, **data))

    def typed(self, text):
        self.submit(text)

    def submit(self, text, utterance_generation=None, turn_id=None):
        target = self.targets.pop(turn_id, None)
        field, revision = target or (self.selected, self.revisions[self.selected])
        self.transcript = text.strip()
        self.emit("transcript_final", text=self.transcript, turn_id=turn_id, field=field)
        if not self.transcript:
            return
        if revision != self.revisions[field]:
            self.error = f"Giữ bản sửa tay của {FIELDS[field]}; bỏ kết quả ASR cũ."
            return
        try:
            value = normalize_value(field, self.transcript)
            separator = "" if field == "phone" else " "
            joined = separator.join(x for x in (self.values[field], value) if x)
            self.values[field] = normalize_value(field, joined)
            self.error = ""
            self.emit("form_updated", field=field)
        except ValueError as exc:
            self.error = str(exc)

    def select(self, field):
        if field not in FIELDS:
            raise ValueError("Trường không hợp lệ")
        self.selected = field

    def edit(self, field, text):
        value = normalize_value(field, text)
        self.values[field] = value
        self.revisions[field] += 1
        self.error = ""

    def export(self):
        if self.owner.mic or self.owner.file_active or self.owner.runner.busy or self.owner.audio_queued:
            raise ValueError("Tắt mic và chờ file/nhận dạng xong trước khi xuất")
        if not self.values["full_name"] or not self.values["address"]:
            raise ValueError("Điền họ tên và địa chỉ trước khi xuất")
        # shortcut: validate phone format only; add country-specific rules for a real form.
        if not re.fullmatch(r"(?:0[0-9]{9}|\+84[0-9]{9})", self.values["phone"]):
            raise ValueError("Số điện thoại cần 10 chữ số bắt đầu bằng 0 hoặc +84 và 9 chữ số")
        return json.dumps(self.values, ensure_ascii=False, indent=2)

