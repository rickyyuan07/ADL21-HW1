import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader

from dataset import SeqSlotDataset
from model import SeqSlotClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqSlotDataset(data, vocab, tag2idx, args.max_len)
    # create DataLoader for test dataset
    test_loader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        collate_fn=dataset.collate_fn
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    device = args.device
    model = SeqSlotClassifier(
        model=args.model,
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=dataset.num_classes,
    )
    model = model.to(device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    # predict dataset
    prediction = []
    for i, data in enumerate(tqdm(test_loader)):
        inputs, tokens_len = data['tokens'], data['len']
        inputs = torch.LongTensor(inputs).to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 2) # get the index of the class with the highest probability
        preds = [' '.join(dataset.idxs2tags(pred)[:tokens_len[i]]) for i, pred in enumerate(preds)]
        ids = data['id']
        prediction += list(zip(ids, preds))

    # write prediction to file (args.pred_file)
    print(f'Writing to {args.pred_file}...')
    with open(args.pred_file, 'w') as f:
        print('id,tags', file=f)
        for id, pred in prediction:
            print(f'{id},{pred}', file=f)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json",
        # required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/best.pt",
        # required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.tags.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--model", type=str, default='GRU')
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
