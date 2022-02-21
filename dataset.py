from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        _label2idx: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self._label2idx = _label2idx
        self._idx2label = {idx: intent for intent, idx in self._label2idx.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        # {'text': ..., 'intent': ..., 'id': ...}, i.g.
        # "text": "i need you to book me a flight from ft lauderdale to houston on southwest",
        # "intent": "book_flight",
        # "id": "train-0"
        return self.data[index]

    @property
    def num_classes(self) -> int:
        return len(self._label2idx)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        texts, intents, ids = [], [], []
        for sample in samples:
            texts.append(sample['text'])
            ids.append(sample['id'])
            intents.append(sample['intent'])
        return {'text': texts, 'intent': intents, 'id': ids}

    def label2idx(self, label: str):
        return self._label2idx[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
