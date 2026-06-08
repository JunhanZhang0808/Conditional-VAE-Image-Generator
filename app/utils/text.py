import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text: str) -> List[str]:
    text = (text or "").lower().strip()
    return TOKEN_PATTERN.findall(text)


class Vocab:
    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1, max_size: int = 5000):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token, freq in counter.most_common(max_size):
            if freq < min_freq:
                continue
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, text: str, max_len: int = 32) -> List[int]:
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokenize(text)]
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def to_dict(self):
        return {"token_to_id": self.token_to_id}

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(data["token_to_id"])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
