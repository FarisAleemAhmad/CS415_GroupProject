"""Train one setting (model x loss x seed) and evaluate on val and test.

Examples:
  python -m src.train --model textcnn --loss oll --seed 0 --limit 20000 --epochs 2
  python -m src.train --model textcnn --loss ce --glove data/glove.840B.300d.txt
  python -m src.train --model distilbert --loss mll --seed 0

Test is evaluated once, with the checkpoint that had the best validation Macro F1.
Results are written to results/logs/<model>_<loss>_s<seed>.json
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import load_split
from src.evaluation.metrics import compute_metrics
from src.losses.losses import get_loss


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_textcnn(args, train_df, val_df, test_df):
    from src.models.textcnn import TextCNN, build_vocab, encode, load_glove
    vocab = build_vocab(train_df["text"], max_size=args.vocab_size)
    if args.glove:
        emb, found = load_glove(args.glove, vocab, dim=300)
        print(f"GloVe covers {found}/{len(vocab)} vocab words")
    else:
        print("WARNING: no --glove given, using random frozen embeddings (smoke test only)")
        emb = torch.randn(len(vocab), 300) * 0.1
        emb[0] = 0

    def mk(df, shuffle):
        ids = torch.tensor([encode(t, vocab, args.max_len) for t in df["text"]])
        y = torch.tensor(df["label"].values)
        return DataLoader(TensorDataset(ids, y), batch_size=args.batch_size, shuffle=shuffle)

    model = TextCNN(emb)
    fwd = lambda m, b: m(b[0])
    return model, mk(train_df, True), mk(val_df, False), mk(test_df, False), fwd


def build_distilbert(args, train_df, val_df, test_df):
    from transformers import AutoTokenizer
    from src.models.distilbert import DistilBertRater
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def mk(df, shuffle):
        # no padding here: each batch is padded to its own longest review (dynamic padding)
        ids = tok(list(df["text"]), truncation=True, max_length=args.max_len)["input_ids"]
        items = list(zip(ids, df["label"].tolist()))

        def collate(batch):
            L = max(len(i) for i, _ in batch)
            x = torch.zeros(len(batch), L, dtype=torch.long)  # 0 = [PAD] in this vocab
            m = torch.zeros(len(batch), L, dtype=torch.long)
            for r, (i, _) in enumerate(batch):
                x[r, :len(i)] = torch.tensor(i); m[r, :len(i)] = 1
            return x, m, torch.tensor([y for _, y in batch])

        return DataLoader(items, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)

    model = DistilBertRater()
    fwd = lambda m, b: m(b[0], b[1])
    return model, mk(train_df, True), mk(val_df, False), mk(test_df, False), fwd


@torch.no_grad()
def predict(model, loader, fwd, dev):
    """Returns (y_true, y_pred, probs) as numpy arrays."""
    model.eval()
    ys, ps = [], []
    for b in loader:
        b = [x.to(dev) for x in b]
        ps.append(torch.softmax(fwd(model, b).float(), -1).cpu()); ys.append(b[-1].cpu())
    probs = torch.cat(ps).numpy()
    return torch.cat(ys).numpy(), probs.argmax(-1), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["textcnn", "distilbert"], required=True)
    ap.add_argument("--loss", choices=["ce", "oll", "mll"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--vocab_size", type=int, default=50000)
    ap.add_argument("--alpha", type=float, default=1.0, help="OLL distance exponent")
    ap.add_argument("--lam", type=float, default=0.5, help="MLL: lam*CE + (1-lam)*OLL")
    ap.add_argument("--glove", default=None)
    ap.add_argument("--limit", type=int, default=None, help="subsample the TRAIN set only")
    ap.add_argument("--eval_limit", type=int, default=None, help="subsample val/test (smoke tests only)")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results/runs", help="each run gets its own folder here")
    ap.add_argument("--force", action="store_true", help="re-run even if the run folder is already complete")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    # model-specific defaults
    d = {"textcnn": (64, 1e-3, 256), "distilbert": (32, 3e-5, 256)}[args.model]
    args.batch_size = args.batch_size or d[0]
    args.lr = args.lr or d[1]
    args.max_len = args.max_len or d[2]

    name = f"{args.model}_{args.loss}_s{args.seed}{args.tag}"
    run_dir = Path(args.out) / name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        if json.loads(metrics_path.read_text()).get("status") == "complete":
            print(f"SKIP {name}: already complete ({metrics_path})")
            return
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lim_eval = args.eval_limit
    tr = load_split("train", args.data, args.limit, args.seed)
    va = load_split("val", args.data, lim_eval, args.seed)
    te = load_split("test", args.data, lim_eval, args.seed)

    meta = {
        "name": name, "args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
    }

    def save_metrics(status, hist, extra=None):
        metrics_path.write_text(json.dumps({**meta, "status": status, "history": hist, **(extra or {})}, indent=2))

    build = build_textcnn if args.model == "textcnn" else build_distilbert
    model, tr_l, va_l, te_l, fwd = build(args, tr, va, te)
    model.to(dev)
    crit = get_loss(args.loss, 5, args.alpha, args.lam).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    use_amp = args.model == "distilbert" and dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best, hist = -1, []
    t0 = time.time()
    for ep in range(args.epochs):
        t_ep = time.time()
        model.train()
        tot = 0.0
        for b in tr_l:
            b = [x.to(dev) for x in b]
            opt.zero_grad()
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = fwd(model, b)
            loss = crit(logits.float(), b[-1])
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item()
        yv, pv, prv = predict(model, va_l, fwd, dev)
        m = compute_metrics(yv, pv)
        hist.append({"epoch": ep + 1, "epoch_time_s": time.time() - t_ep, "train_loss": tot / len(tr_l),
                     **{f"val_{k}": v for k, v in m.items()}})
        print(hist[-1], flush=True)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(model.state_dict(), run_dir / "model.pt")        # best checkpoint on disk
            np.savez_compressed(run_dir / "val_preds.npz", y_true=yv, y_pred=pv, probs=prv.astype("float16"))
            meta["best_epoch"] = ep + 1
        save_metrics("running", hist)                                   # survives a crash mid-run
    train_time = time.time() - t0

    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=dev))
    yt, pt, prt = predict(model, te_l, fwd, dev)
    test_m = compute_metrics(yt, pt)
    print("TEST", test_m, flush=True)
    np.savez_compressed(run_dir / "test_preds.npz", y_true=yt, y_pred=pt, probs=prt.astype("float16"))
    save_metrics("complete", hist, {
        "test": test_m, "train_time_s": train_time, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if dev.type == "cuda" else None})


if __name__ == "__main__":
    main()
