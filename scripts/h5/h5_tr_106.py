"""
Training script for a tabular binary classification dataset stored in
HDF5, under a configurable data loading strategy.

The HDF5 file is expected to contain exactly two datasets at its root:
"X", a two dimensional array of shape (num_samples, num_features) with
one row per event, and "y", a one dimensional array of shape
(num_samples,) with one integer class label per event. Both num_features
and num_classes are read from the file (and from the command line,
respectively) rather than hard coded, so the same script works for any
dataset that follows this two dataset layout.

Four loading strategies are available, selected with --strategy:

eager: the entire X and y arrays are read into host memory once, before
training starts. After that, reading a sample is a plain in memory index
lookup, with no further disk access.

lazy: only the file path is kept until a sample is actually requested.
Each call reads exactly that row from disk through h5py, and nothing is
cached, so the same row read again later is read from disk again rather
than served from memory.

stream: the dataset is iterated strictly in file order, in fixed size
contiguous blocks, never revisiting rows within an epoch and never
retaining a block once its rows have been produced. Its DataLoader pins
the prefetch queue at the shallowest depth PyTorch allows (see
STREAM_PREFETCH_FACTOR below), so that stream is a genuine
minimum-anticipation baseline rather than silently inheriting a deeper
queue by default.

burst: the same sequential iteration as stream, with the DataLoader
configured to prefetch several batches ahead per worker. This is not a
separate dataset implementation, it is the stream dataset combined with a
larger value of PyTorch's own prefetch_factor argument, since anticipated
reading is a property of how many batches are queued ahead of
consumption, not of how a single row is located and read. The only
difference between stream and burst is therefore queue depth, which is
exactly the variable this pair is meant to isolate.

Only one of the four needs a class that is not a direct instance of a
standard PyTorch dataset abstraction: lazy and eager are both ordinary
map style Dataset subclasses, and stream is an ordinary IterableDataset
subclass. Burst needs no new class at all.
"""

import argparse
import time

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info


# Prefetch depth used by the stream strategy. PyTorch's own default when
# prefetch_factor is left unset and num_workers is greater than zero is 2,
# which means an unconfigured stream loader is already queueing two
# batches ahead per worker. Leaving that implicit would make the stream
# versus burst comparison a contrast between depth 2 and depth N while
# appearing to be a contrast between no prefetching and prefetching, so
# the depth is stated explicitly here instead. 1 is the shallowest value
# PyTorch accepts, 0 is rejected.
STREAM_PREFETCH_FACTOR = 1


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

def format_duration(seconds):
    """Formats a duration in seconds as a short human readable string, or 'unknown' for infinite/negative values."""
    if seconds == float("inf") or seconds < 0:
        return "unknown"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "{}h{:02d}m".format(hours, minutes)
    return "{}m{:02d}s".format(minutes, secs)


def report_progress(epoch, num_epochs, batch_idx, total_batches, epoch_start):
    """
    Prints one status line: current batch, throughput, elapsed time, and
    an estimated time remaining for the epoch if total_batches is known.

    This prints a plain line rather than redrawing a carriage return
    progress bar on purpose: the intended way to run this script at
    campaign scale is as a batch job with its output redirected to a log
    file, not watched live in a terminal, and a redrawing progress bar
    either looks like garbage or spams one line per update once it is
    written to a file instead of a real terminal. A periodic plain line
    reads fine in both places.
    """
    elapsed = time.time() - epoch_start
    rate = batch_idx / elapsed if elapsed > 0 else 0.0
    if total_batches:
        remaining = (total_batches - batch_idx) / rate if rate > 0 else float("inf")
        print(
            "  epoch {}/{}  batch {}/{}  {:.1f} batches/s  elapsed {}  eta {}".format(
                epoch, num_epochs, batch_idx, total_batches, rate, format_duration(elapsed), format_duration(remaining)
            )
        )
    else:
        print(
            "  epoch {}/{}  batch {}  {:.1f} batches/s  elapsed {}".format(
                epoch, num_epochs, batch_idx, rate, format_duration(elapsed)
            )
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_model(name, num_features, num_classes):
    """
    Build one of three classifier architectures over the same input and
    output sizes, selected by name.

    M1 is a single linear layer, equivalent to logistic regression: the
    smallest possible model for this task, with no hidden layers and no
    activation function (a single linear map needs none).

    M2 is a small multilayer perceptron with two hidden layers (128 and
    64 units) and ReLU activations between them. This is the
    conventional shape for a tabular classifier of this input size: deep
    enough to learn non linear interactions between features, small
    enough to train quickly.

    M3 is a much larger multilayer perceptron: four repeated blocks of
    Linear(512) then GELU then Linear(256) then GELU, followed by a
    Linear(128), Linear(64), and final Linear(num_classes) head. This is
    deliberately oversized for a 17 to 20 feature tabular input, its
    purpose is to exercise a heavier amount of compute per batch than M1
    or M2, not to be a well tuned classifier.
    """
    if name == "M1":
        return nn.Linear(num_features, num_classes)
    if name == "M2":
        return nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    if name == "M3":
        layers = []
        in_dim = num_features
        for _ in range(4):
            layers.append(nn.Linear(in_dim, 512))
            layers.append(nn.GELU())
            layers.append(nn.Linear(512, 256))
            layers.append(nn.GELU())
            in_dim = 256
        layers.append(nn.Linear(256, 128))
        layers.append(nn.Linear(128, 64))
        layers.append(nn.Linear(64, num_classes))
        return nn.Sequential(*layers)
    raise ValueError("Unknown model name: {}, expected one of M1, M2, M3".format(name))


# ---------------------------------------------------------------------------
# Eager strategy
# ---------------------------------------------------------------------------

class EagerHDF5Dataset(Dataset):
    """
    Reads the full X and y datasets into memory once, in __init__, using
    h5py's whole array slicing (dataset[:]). From that point on this is
    an ordinary in memory map style Dataset: __getitem__ never touches
    the file again, it only indexes the two tensors already held in
    memory.

    No custom indexing scheme is needed here, materializing the whole
    array up front and letting PyTorch's default Dataset indexing take
    over afterwards is sufficient to implement eager loading correctly.
    """

    def __init__(self, file_path):
        with h5py.File(file_path, "r") as f:
            X = f["X"][:]
            y = f["y"][:]
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Lazy strategy
# ---------------------------------------------------------------------------

class LazyHDF5Dataset(Dataset):
    """
    Keeps only the file path and the dataset length after __init__. The
    underlying h5py.File handle is opened lazily, the first time
    __getitem__ actually needs it, and stored on the instance for reuse
    by later calls within the same process. Every call reads exactly the
    requested row through h5py, nothing is cached across calls: reading
    the same index twice means two separate reads from disk.

    The handle is opened lazily rather than in __init__ on purpose. When
    DataLoader is given num_workers greater than zero, PyTorch's default
    multiprocessing start method on Linux is fork, which means each
    worker process starts as a copy of the parent. An h5py.File handle
    opened in the parent before that fork is not safe to share across
    the resulting worker processes. Opening the handle on first use
    instead guarantees each worker process opens and owns its own
    handle, the first time that particular worker needs it.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self._file = None
        with h5py.File(file_path, "r") as f:
            self._length = f["y"].shape[0]

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        self._ensure_open()
        x = self._file["X"][idx]
        y = self._file["y"][idx]
        return torch.from_numpy(x), torch.tensor(int(y), dtype=torch.long)


# ---------------------------------------------------------------------------
# Stream strategy (also the basis for burst)
# ---------------------------------------------------------------------------

class StreamHDF5Dataset(IterableDataset):
    """
    Iterates the dataset strictly in file order, reading fixed size
    contiguous blocks of rows out of the HDF5 file with a single slicing
    call per block, then yielding the rows of that block one at a time.
    A block is never retained after its rows have been yielded, so a
    later epoch reads every row from disk again rather than reusing
    anything kept from the previous pass.

    block_rows controls how many rows are read per underlying HDF5 call.
    The default, "auto", reads the chunk shape that the file itself was
    written with (f["X"].chunks) and uses that row count, so each read
    lines up with one storage chunk instead of reading across a chunk
    boundary. If the file has no chunking, a fixed fallback is used
    instead, since there is no chunk shape to align to.

    When DataLoader uses more than one worker, IterableDataset does not
    split work across them on its own, each worker would otherwise
    iterate the entire dataset independently. To avoid that, the row
    range is split by hand in __iter__, using get_worker_info, into as
    many contiguous non overlapping spans as there are workers, so each
    worker streams a distinct part of the file.

    That hand split is also why num_batches below exists rather than
    relying on len(loader). __len__ here reports rows, and PyTorch turns
    that into a batch count by dividing by batch_size, which is only
    correct for a single span. With several workers each one batches its
    own span independently, so every span contributes its own trailing
    partial batch and the true batch count is higher than that division
    suggests.
    """

    FALLBACK_BLOCK_ROWS = 1024

    def __init__(self, file_path, block_rows="auto"):
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self._length = f["y"].shape[0]
            if block_rows == "auto":
                chunks = f["X"].chunks
                self.block_rows = chunks[0] if chunks is not None else self.FALLBACK_BLOCK_ROWS
            else:
                self.block_rows = int(block_rows)

    def __len__(self):
        return self._length

    def _span_for(self, worker_id, num_workers):
        """Contiguous [start, end) row range assigned to one worker."""
        per_worker = int(np.ceil(self._length / num_workers))
        start = worker_id * per_worker
        end = min(start + per_worker, self._length)
        return start, max(start, end)

    def _worker_span(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return 0, self._length
        return self._span_for(worker_info.id, worker_info.num_workers)

    def num_batches(self, batch_size, num_workers):
        """
        Exact number of batches one epoch produces for this batch_size
        and worker count, accounting for the per worker span split: each
        worker batches only its own rows, so each non empty span ends in
        its own partial batch.
        """
        if num_workers < 1:
            return int(np.ceil(self._length / batch_size))
        total = 0
        for worker_id in range(num_workers):
            start, end = self._span_for(worker_id, num_workers)
            span = end - start
            if span > 0:
                total += int(np.ceil(span / batch_size))
        return total

    def __iter__(self):
        start, end = self._worker_span()
        with h5py.File(self.file_path, "r") as f:
            X_ds = f["X"]
            y_ds = f["y"]
            for block_start in range(start, end, self.block_rows):
                block_end = min(block_start + self.block_rows, end)
                X_block = X_ds[block_start:block_end]
                y_block = y_ds[block_start:block_end].astype(np.int64)
                for i in range(X_block.shape[0]):
                    yield torch.from_numpy(X_block[i]), torch.tensor(int(y_block[i]), dtype=torch.long)


# ---------------------------------------------------------------------------
# DataLoader construction per strategy
# ---------------------------------------------------------------------------

def build_dataloader(strategy, file_path, batch_size, num_workers, block_rows, prefetch_factor, persistent_workers, shuffle):
    """
    persistent_workers, when True, keeps the worker processes alive
    across epochs instead of the default behavior of tearing them down
    and respawning them at the start of every epoch. This is only
    meaningful when num_workers is at least one, since with zero workers
    all loading happens in the main process and there is no separate
    worker process to keep alive, so that combination is rejected
    explicitly here rather than left for PyTorch to reject with a less
    specific error.

    prefetch_factor is only ever meaningful when there are worker
    processes to do the prefetching, and PyTorch rejects the argument
    outright when num_workers is zero, so both the stream and burst
    branches only pass it when workers are actually present.
    """
    if persistent_workers and num_workers < 1:
        raise ValueError(
            "persistent_workers requires num_workers of at least 1: "
            "there is no worker process to keep alive across epochs otherwise"
        )

    if strategy == "eager":
        dataset = EagerHDF5Dataset(file_path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    if strategy == "lazy":
        dataset = LazyHDF5Dataset(file_path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    if strategy == "stream":
        dataset = StreamHDF5Dataset(file_path, block_rows=block_rows)
        stream_kwargs = {}
        if num_workers > 0:
            stream_kwargs["prefetch_factor"] = STREAM_PREFETCH_FACTOR
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            **stream_kwargs
        )

    if strategy == "burst":
        if num_workers < 1:
            raise ValueError(
                "strategy burst requires num_workers of at least 1: "
                "prefetch_factor has no effect without worker processes to do the prefetching"
            )
        if prefetch_factor <= STREAM_PREFETCH_FACTOR:
            raise ValueError(
                "strategy burst needs a prefetch_factor greater than the stream baseline of {}: "
                "got {}, which would make burst identical to stream".format(STREAM_PREFETCH_FACTOR, prefetch_factor)
            )
        dataset = StreamHDF5Dataset(file_path, block_rows=block_rows)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    raise ValueError("Unknown strategy: {}, expected one of eager, lazy, stream, burst".format(strategy))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def count_batches(loader, dataset, batch_size, num_workers):
    """
    Batches one epoch will produce.

    For the map style strategies this is just len(loader). For stream
    and burst it has to be computed from the per worker span split, see
    StreamHDF5Dataset.num_batches: len(loader) would divide the row
    count by batch_size as if a single span covered the whole file, and
    undercount the trailing partial batch that every additional worker
    contributes.
    """
    if isinstance(dataset, StreamHDF5Dataset):
        return dataset.num_batches(batch_size, num_workers)
    return len(loader)


def train(args):
    device = torch.device(args.device)

    with h5py.File(args.data_path, "r") as f:
        num_features = f["X"].shape[1]

    loader = build_dataloader(
        args.strategy,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.block_rows,
        args.prefetch_factor,
        args.persistent_workers,
        args.shuffle,
    )

    model = build_model(args.model, num_features, args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    total_batches = count_batches(loader, loader.dataset, args.batch_size, args.num_workers)

    effective_prefetch = "n/a"
    if args.num_workers > 0:
        if args.strategy == "stream":
            effective_prefetch = STREAM_PREFETCH_FACTOR
        elif args.strategy == "burst":
            effective_prefetch = args.prefetch_factor
    print(
        "config  strategy {}  model {}  workers {}  batch_size {}  shuffle {}  "
        "prefetch_factor {}  persistent_workers {}  batches/epoch {}".format(
            args.strategy,
            args.model,
            args.num_workers,
            args.batch_size,
            args.shuffle if args.strategy in ("eager", "lazy") else False,
            effective_prefetch,
            args.persistent_workers,
            total_batches,
        )
    )

    model.train()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        running_loss = 0.0
        num_samples = 0
        num_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            # Weighted by batch size rather than averaged over batches:
            # with several workers each span ends in its own partial
            # batch, so batches are not all the same size and a plain
            # mean over batches would overweight the short ones.
            batch_samples = y_batch.size(0)
            running_loss += loss.item() * batch_samples
            num_samples += batch_samples
            num_batches += 1

            if num_batches % args.log_every == 0 or num_batches == total_batches:
                report_progress(epoch + 1, args.epochs, num_batches, total_batches, epoch_start)

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / max(1, num_samples)
        print(
            "epoch {}/{}  loss {:.4f}  batches {}  samples {}  time {:.2f}s".format(
                epoch + 1, args.epochs, avg_loss, num_batches, num_samples, epoch_time
            )
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a classifier over an HDF5 dataset under a configurable loading strategy."
    )
    parser.add_argument("--data-path", required=True, help="Path to an HDF5 file with X and y datasets.")
    parser.add_argument("--strategy", required=True, choices=["eager", "lazy", "stream", "burst"])
    parser.add_argument("--model", default="M2", choices=["M1", "M2", "M3"])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shuffle map-style eager/lazy datasets. Stream/burst remain sequential by construction.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--block-rows",
        default="auto",
        help="Rows read per contiguous call for the stream and burst strategies. "
        "auto aligns to the file's own HDF5 chunk shape.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches queued ahead per worker, used only by the burst strategy and ignored by every "
        "other one. The stream strategy is pinned at {} so that stream versus burst isolates queue "
        "depth alone.".format(STREAM_PREFETCH_FACTOR),
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep worker processes alive across epochs instead of respawning them every epoch. "
        "Requires num_workers of at least 1, and has no effect with a single epoch.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Print a progress line every this many batches, so a long run shows it is "
        "still advancing instead of going silent until the epoch ends.",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=None,
        help="Optional PyTorch intra-op thread limit. Leave unset to use the runtime default.",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=None,
        help="Optional PyTorch inter-op thread limit. Leave unset to use the runtime default.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    # Thread limits are optional in this version. If these arguments are
    # omitted, PyTorch keeps the thread settings selected by its runtime
    # environment instead of forcing the historical 4/1 configuration.
    if args.torch_num_threads is not None:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_interop_threads)

    print("torch intra-op threads:", torch.get_num_threads())
    print("torch inter-op threads:", torch.get_num_interop_threads())
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
