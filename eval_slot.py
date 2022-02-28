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

    data = json.loads(args.eval_file.read_text())
    dataset = SeqSlotDataset(data, vocab, tag2idx, args.max_len)
    # create DataLoader for test dataset
    eval_loader = DataLoader(
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

    # load weights into model
    model.load_state_dict(torch.load(args.ckpt_path))
    # evaluation
    prediction, gt = [], []
    for i, data in enumerate(tqdm(eval_loader)):
        inputs, tokens_len = data['tokens'], data['len']
        inputs = torch.LongTensor(inputs).to(device)
        labels = data['tags']
        gt += labels

        outputs = model(inputs)
        _, preds = torch.max(outputs, 2) # get the index of the class with the highest probability
        preds = [dataset.idxs2tags(pred)[:tokens_len[i]] for i, pred in enumerate(preds)]
        prediction += preds
    
    joint_acc, tok_acc, total_len = 0, 0, 0
    for pred, label in list(zip(prediction, gt)):
        ll = len(pred)
        total_len += ll
        acc = sum([int(x == y) for x, y in zip(pred, label)])
        joint_acc += int(ll == acc)
        tok_acc += acc
    
    print(f"Joint Accuracy: {(joint_acc/len(prediction)):.3f} ({joint_acc}/{len(prediction)})")
    print(f"Token Accuracy: {(tok_acc/total_len):.3f} ({tok_acc}/{total_len})")
    from seqeval.metrics import classification_report
    from seqeval.scheme import IOB2
    rep = classification_report(y_true=gt, y_pred=prediction, scheme=IOB2, mode='strict')
    print("\nseqeval Classification Report")
    print(rep)

def parse_args() -> Namespace:
    parser = ArgumentParser()
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
    parser.add_argument(
        "--eval_file",
        type=Path,
        help="Path to the eval file.",
        default="./data/slot/eval.json",
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
