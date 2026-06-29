"""Run neural-symbolic AddMNIST experiments.

The task trains on pairs of MNIST digits where high-plus-high pairs are excluded.
This means sums 14-18 are never observed as training labels, making the
high-plus-high test split a compositional generalization probe.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
METHODS = ("direct_sum", "symbolic_sum", "anchor_symbolic", "anchor_posterior")


@dataclass(frozen=True)
class RunConfig:
    method: str
    budget: int
    seed: int
    epochs: int
    batch_size: int
    lr: float
    anchor_per_class: int
    anchor_weight: float
    entropy_weight: float
    eval_pairs: int
    device: str
    use_amp: bool


class PairDataset(Dataset):
    """Dataset of digit-image pairs and their sum labels."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, pairs: np.ndarray):
        self.images = images
        self.labels = labels
        self.left = torch.as_tensor(pairs[:, 0], dtype=torch.long)
        self.right = torch.as_tensor(pairs[:, 1], dtype=torch.long)
        self.sums = self.labels[self.left] + self.labels[self.right]

    def __len__(self) -> int:
        return int(self.left.numel())

    def __getitem__(self, idx: int):
        left_idx = self.left[idx]
        right_idx = self.right[idx]
        return (
            self.images[left_idx],
            self.images[right_idx],
            self.sums[idx],
            self.labels[left_idx],
            self.labels[right_idx],
        )


class DigitDataset(Dataset):
    """Single-digit images and labels for tiny anchor supervision."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, indices: np.ndarray):
        self.images = images
        self.labels = labels
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, idx: int):
        source_idx = self.indices[idx]
        return self.images[source_idx], self.labels[source_idx]


class DigitCNN(nn.Module):
    """Small CNN that predicts a digit distribution for one MNIST image."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DirectSumCNN(nn.Module):
    """Pure neural baseline that predicts the pair sum from two image channels."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 19),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_mnist_tensors(data_dir: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    dataset = load_from_disk(data_dir)
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split in ("train", "test"):
        images = []
        labels = []
        for row in dataset[split]:
            arr = np.asarray(row["image"], dtype=np.float32) / 255.0
            arr = (arr - MNIST_MEAN) / MNIST_STD
            images.append(arr[None, :, :])
            labels.append(int(row["label"]))
        tensors[split] = (
            torch.tensor(np.stack(images), dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )
    return tensors


def sample_pairs(labels: torch.Tensor, n_pairs: int, seed: int, mode: str) -> np.ndarray:
    """Sample pair indices for restricted or high-high splits."""

    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    n = len(labels_np)
    if mode == "high_high":
        pool = np.flatnonzero(labels_np >= 5)
        left = rng.choice(pool, size=n_pairs, replace=True)
        right = rng.choice(pool, size=n_pairs, replace=True)
        return np.stack([left, right], axis=1).astype(np.int64)
    if mode != "restricted":
        raise ValueError(f"Unknown pair sampling mode: {mode}")

    pairs = np.empty((n_pairs, 2), dtype=np.int64)
    filled = 0
    while filled < n_pairs:
        need = n_pairs - filled
        left = rng.integers(0, n, size=need * 2)
        right = rng.integers(0, n, size=need * 2)
        keep = ~((labels_np[left] >= 5) & (labels_np[right] >= 5))
        kept = np.stack([left[keep], right[keep]], axis=1)
        take = min(need, len(kept))
        pairs[filled : filled + take] = kept[:take]
        filled += take
    return pairs


def select_anchor_indices(labels: torch.Tensor, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    indices = []
    for digit in range(10):
        digit_idx = np.flatnonzero(labels_np == digit)
        indices.extend(rng.choice(digit_idx, size=per_class, replace=False).tolist())
    rng.shuffle(indices)
    return np.asarray(indices, dtype=np.int64)


def make_sum_mask(device: torch.device) -> torch.Tensor:
    mask = torch.full((19, 10, 10), float("-inf"), device=device)
    for a in range(10):
        for b in range(10):
            mask[a + b, a, b] = 0.0
    return mask


def symbolic_log_probs(
    left_logits: torch.Tensor, right_logits: torch.Tensor, sum_mask: torch.Tensor
) -> torch.Tensor:
    left_logp = torch.log_softmax(left_logits, dim=-1)
    right_logp = torch.log_softmax(right_logits, dim=-1)
    pair_logp = left_logp[:, :, None] + right_logp[:, None, :]
    return torch.logsumexp(pair_logp[:, None, :, :] + sum_mask[None, :, :, :], dim=(2, 3))


def posterior_entropy(
    left_logits: torch.Tensor, right_logits: torch.Tensor, sums: torch.Tensor, sum_mask: torch.Tensor
) -> torch.Tensor:
    left_logp = torch.log_softmax(left_logits, dim=-1)
    right_logp = torch.log_softmax(right_logits, dim=-1)
    pair_logp = left_logp[:, :, None] + right_logp[:, None, :]
    masked = pair_logp + sum_mask[sums]
    norm = torch.logsumexp(masked, dim=(1, 2), keepdim=True)
    posterior = torch.exp(masked - norm)
    entropy = -(posterior * torch.nan_to_num(masked - norm, neginf=0.0)).sum(dim=(1, 2))
    return entropy


def cycle_loader(loader: DataLoader) -> Iterable:
    while True:
        for batch in loader:
            yield batch


def train_direct(
    config: RunConfig,
    train_loader: DataLoader,
    device: torch.device,
) -> nn.Module:
    model = DirectSumCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
    loss_fn = nn.CrossEntropyLoss()
    for _epoch in range(config.epochs):
        model.train()
        for left, right, sums, _left_digit, _right_digit in train_loader:
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
            sums = sums.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=config.use_amp):
                logits = model(left, right)
                loss = loss_fn(logits, sums)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return model


def train_symbolic(
    config: RunConfig,
    train_loader: DataLoader,
    anchor_loader: DataLoader | None,
    device: torch.device,
) -> nn.Module:
    model = DigitCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
    sum_mask = make_sum_mask(device)
    anchor_iter = cycle_loader(anchor_loader) if anchor_loader is not None else None
    ce = nn.CrossEntropyLoss()

    for _epoch in range(config.epochs):
        model.train()
        for left, right, sums, _left_digit, _right_digit in train_loader:
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
            sums = sums.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=config.use_amp):
                left_logits = model(left)
                right_logits = model(right)
                sum_logp = symbolic_log_probs(left_logits, right_logits, sum_mask)
                loss = nn.functional.nll_loss(sum_logp, sums)
                if config.method == "anchor_posterior":
                    loss = loss + config.entropy_weight * posterior_entropy(
                        left_logits, right_logits, sums, sum_mask
                    ).mean()
                if anchor_iter is not None:
                    anchor_x, anchor_y = next(anchor_iter)
                    anchor_x = anchor_x.to(device, non_blocking=True)
                    anchor_y = anchor_y.to(device, non_blocking=True)
                    loss = loss + config.anchor_weight * ce(model(anchor_x), anchor_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return model


@torch.no_grad()
def evaluate_pair_model(
    model: nn.Module,
    method: str,
    loader: DataLoader,
    device: torch.device,
    split: str,
    run_id: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    sum_mask = make_sum_mask(device)
    all_true = []
    all_pred = []
    all_conf = []
    all_nll = []
    for left, right, sums, _left_digit, _right_digit in loader:
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        sums = sums.to(device, non_blocking=True)
        if method == "direct_sum":
            logits = model(left, right)
            probs = torch.softmax(logits, dim=-1)
            logp = torch.log_softmax(logits, dim=-1)
        else:
            left_logits = model(left)
            right_logits = model(right)
            logp = symbolic_log_probs(left_logits, right_logits, sum_mask)
            probs = torch.exp(logp)
        pred = probs.argmax(dim=-1)
        conf = probs.max(dim=-1).values
        nll = -logp.gather(1, sums[:, None]).squeeze(1)
        all_true.append(sums.cpu())
        all_pred.append(pred.cpu())
        all_conf.append(conf.cpu())
        all_nll.append(nll.cpu())

    true = torch.cat(all_true).numpy()
    pred = torch.cat(all_pred).numpy()
    conf = torch.cat(all_conf).numpy()
    nll = torch.cat(all_nll).numpy()
    correct = (true == pred).astype(np.int64)
    metrics = {
        "split": split,
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(true, pred, labels=list(range(19)), average="macro", zero_division=0)),
        "nll": float(nll.mean()),
        "mean_confidence": float(conf.mean()),
        "n_examples": int(len(true)),
    }
    predictions = pd.DataFrame(
        {
            "run_id": run_id,
            "split": split,
            "true_sum": true,
            "pred_sum": pred,
            "confidence": conf,
            "nll": nll,
            "correct": correct,
        }
    )
    return metrics, predictions


@torch.no_grad()
def evaluate_digit_model(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    dataset = DigitDataset(images, labels, np.arange(len(labels), dtype=np.int64))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    true_batches = []
    pred_batches = []
    conf_batches = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        true_batches.append(y)
        pred_batches.append(probs.argmax(dim=-1).cpu())
        conf_batches.append(probs.max(dim=-1).values.cpu())
    true = torch.cat(true_batches).numpy()
    pred = torch.cat(pred_batches).numpy()
    conf = torch.cat(conf_batches).numpy()
    return {
        "digit_accuracy": float((true == pred).mean()),
        "digit_macro_f1": float(f1_score(true, pred, labels=list(range(10)), average="macro", zero_division=0)),
        "digit_mean_confidence": float(conf.mean()),
        "digit_examples": int(len(true)),
    }


def build_loaders(
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    budget: int,
    seed: int,
    eval_pairs: int,
    batch_size: int,
    anchor_per_class: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    train_images, train_labels = tensors["train"]
    test_images, test_labels = tensors["test"]
    train_pairs = sample_pairs(train_labels, budget, seed + 101, mode="restricted")
    iid_pairs = sample_pairs(test_labels, eval_pairs, seed + 202, mode="restricted")
    ood_pairs = sample_pairs(test_labels, eval_pairs, seed + 303, mode="high_high")
    anchor_idx = select_anchor_indices(train_labels, anchor_per_class, seed + 404)

    kwargs = {"batch_size": batch_size, "pin_memory": device.type == "cuda", "num_workers": 2}
    train_loader = DataLoader(
        PairDataset(train_images, train_labels, train_pairs),
        shuffle=True,
        drop_last=False,
        **kwargs,
    )
    iid_loader = DataLoader(PairDataset(test_images, test_labels, iid_pairs), shuffle=False, **kwargs)
    ood_loader = DataLoader(PairDataset(test_images, test_labels, ood_pairs), shuffle=False, **kwargs)
    anchor_loader = DataLoader(
        DigitDataset(train_images, train_labels, anchor_idx),
        batch_size=min(batch_size, len(anchor_idx)),
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
        num_workers=2,
    )
    return train_loader, iid_loader, ood_loader, anchor_loader


def run_one(
    config: RunConfig,
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    output_dir: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    set_seed(config.seed)
    device = torch.device(config.device)
    train_loader, iid_loader, ood_loader, anchor_loader = build_loaders(
        tensors,
        budget=config.budget,
        seed=config.seed,
        eval_pairs=config.eval_pairs,
        batch_size=config.batch_size,
        anchor_per_class=config.anchor_per_class,
        device=device,
    )
    run_id = f"{config.method}_b{config.budget}_s{config.seed}"
    start = time.perf_counter()
    if config.method == "direct_sum":
        model = train_direct(config, train_loader, device)
    else:
        use_anchor = config.method in {"anchor_symbolic", "anchor_posterior"}
        model = train_symbolic(config, train_loader, anchor_loader if use_anchor else None, device)
    train_seconds = time.perf_counter() - start

    rows: list[dict[str, object]] = []
    prediction_frames = []
    for split, loader in (("iid_restricted", iid_loader), ("ood_high_high", ood_loader)):
        metrics, predictions = evaluate_pair_model(model, config.method, loader, device, split, run_id)
        row = asdict(config) | {"run_id": run_id, "train_seconds": train_seconds} | metrics
        rows.append(row)
        prediction_frames.append(predictions)

    if config.method != "direct_sum":
        test_images, test_labels = tensors["test"]
        digit_metrics = evaluate_digit_model(model, test_images, test_labels, config.batch_size, device)
        rows.append(
            asdict(config)
            | {
                "run_id": run_id,
                "train_seconds": train_seconds,
                "split": "digit_test",
                "accuracy": digit_metrics["digit_accuracy"],
                "macro_f1": digit_metrics["digit_macro_f1"],
                "nll": np.nan,
                "mean_confidence": digit_metrics["digit_mean_confidence"],
                "n_examples": digit_metrics["digit_examples"],
                **digit_metrics,
            }
        )

    checkpoint_path = output_dir / f"{run_id}.pt"
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, checkpoint_path)
    return rows, pd.concat(prediction_frames, ignore_index=True)


def summarize_dataset(tensors: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, object]:
    summary = {}
    for split, (_images, labels) in tensors.items():
        counts = torch.bincount(labels, minlength=10).tolist()
        summary[split] = {"n": int(len(labels)), "label_counts": counts}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/mnist")
    parser.add_argument("--output-dir", default="results/addmnist")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anchor-per-class", type=int, default=10)
    parser.add_argument("--anchor-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.03)
    parser.add_argument("--eval-pairs", type=int, default=4000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    use_amp = device.type == "cuda" and not args.no_amp

    tensors = load_mnist_tensors(args.data_dir)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": str(device),
        "use_amp": use_amp,
        "dataset": summarize_dataset(tensors),
        "note": "Training excludes high-plus-high digit pairs; OOD split contains only high-plus-high pairs.",
    }
    if torch.cuda.is_available():
        metadata["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    all_rows = []
    all_predictions = []
    for budget in args.budgets:
        for seed in args.seeds:
            for method in args.methods:
                config = RunConfig(
                    method=method,
                    budget=budget,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    anchor_per_class=args.anchor_per_class,
                    anchor_weight=args.anchor_weight,
                    entropy_weight=args.entropy_weight,
                    eval_pairs=args.eval_pairs,
                    device=str(device),
                    use_amp=use_amp,
                )
                print(f"Running {method} budget={budget} seed={seed}", flush=True)
                rows, predictions = run_one(config, tensors, output_dir)
                all_rows.extend(rows)
                all_predictions.append(predictions)
                pd.DataFrame(all_rows).to_csv(output_dir / "metrics_partial.csv", index=False)

    metrics = pd.DataFrame(all_rows)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    breakdown = (
        predictions.groupby(["run_id", "split", "true_sum"], as_index=False)
        .agg(accuracy=("correct", "mean"), n=("correct", "size"), confidence=("confidence", "mean"))
    )
    breakdown.to_csv(output_dir / "sum_breakdown.csv", index=False)
    print(f"Saved metrics to {output_dir / 'metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
