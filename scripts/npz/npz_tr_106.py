 
#!/usr/bin/env python3
"""
Training script for a tabular binary classification dataset stored as
NPZ, under a configurable data loading strategy.

An NPZ file is a zip archive with one member per array, each member
being a plain .npy file (magic bytes, then a small header describing
shape and dtype, then the raw array bytes). numpy.savez writes these
members uncompressed (zip's ZIP_STORED mode), while numpy's
savez_compressed writes them deflated instead. This script assumes an
uncompressed archive, X.npy and y.npy, written by plain numpy.savez, and
checks that assumption explicitly rather than silently misbehaving if it
does not hold.

That assumption matters because it is the one thing that makes real row
level random access possible here at all. Python's zipfile module can
seek within a stored (uncompressed) member's own byte stream directly,
without decompressing anything, because the bytes in the archive already
are the array's bytes. A deflated member has no such property, DEFLATE
is a sequential decoding scheme with no way to jump to an arbitrary
output position without having decoded everything before it, so a
compressed archive could still be read eagerly or streamed forward, but
never given genuine per row random access. numpy's own high level
np.load interface does not expose this distinction or this seeking
capability at all, indexing an NpzFile object always reads and returns
the entire named array, so lazy and stream below bypass np.load
entirely and use the zipfile module and numpy.lib.format directly.

Four loading strategies are available, selected with --strategy:

eager: np.load reads both arrays into memory once, in full, before
training starts. This works the same way regardless of whether the
archive happens to be stored or deflated, since eager was always going
to materialize everything anyway.

lazy: the byte offset where each array's row data begins is found once,
up front, by opening each member and parsing its .npy header. From then
on, each sample read seeks directly to that row's offset inside the
member and reads exactly that row, separately for the feature vector and
for the label, since X and y live in two separate members of the
archive rather than one interleaved record the way the BIN format uses.
Nothing is cached: reading the same row again later means seeking and
reading it again.

stream: the archive is read forward, strictly in order, in blocks of
whole rows, from both members in step with each other, with nothing
retained after a block has been produced.

burst: the same sequential iteration as stream, with the DataLoader
configured to prefetch several batches ahead per worker. As with the
other formats, this needs no dataset class of its own, it is the
stream dataset combined with PyTorch's own prefetch_factor argument.

As before, only two classes are not direct instances of a standard
PyTorch dataset abstraction: lazy and eager are both ordinary map style
Dataset subclasses, and stream is an ordinary IterableDataset subclass.
Burst needs no new class at all.
"""

import argparse
import time
import zipfile

import numpy as np
from numpy.lib import format as npy_format
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info


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
    deliberately oversized for a small tabular input, its purpose is to
    exercise a heavier amount of compute per batch than M1 or M2, not to
    be a well tuned classifier.
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
# Shared zip member handling
# ---------------------------------------------------------------------------

def npy_member_name(key):
    """numpy.savez names each member after its keyword argument, with a .npy suffix."""
    return key + ".npy"


def open_npy_member(file_path, key):
    """
    Opens a fresh zipfile.ZipFile for this single member and returns a
    stream positioned at the start of its raw array data, along with the
    dtype, shape, and that data start offset (measured within the
    member's own stream, not within the archive as a whole).

    A separate zipfile.ZipFile instance is opened per member on purpose,
    rather than opening two members from one shared ZipFile object.
    Interleaving seeks and reads across two member streams that share a
    single underlying ZipFile was tried and produced corrupted reads,
    the two streams do not keep their positions independent of each
    other in that configuration. Two entirely separate ZipFile instances
    against the same underlying path do not have this problem, each
    keeps its own file descriptor and its own read position.

    Raises a clear error if the member is not stored uncompressed, since
    everything about seeking directly to a row offset below depends on
    the on disk bytes already being the array's raw bytes.
    """
    member_name = npy_member_name(key)
    zip_file = zipfile.ZipFile(file_path)
    info = zip_file.getinfo(member_name)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(
            "member {} in {} is compressed (compress_type {}), row level random access requires "
            "an archive written with plain numpy.savez, not savez_compressed".format(
                member_name, file_path, info.compress_type
            )
        )

    stream = zip_file.open(member_name)
    version = npy_format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = npy_format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = npy_format.read_array_header_2_0(stream)
    else:
        raise ValueError("unsupported npy format version {} in member {}".format(version, member_name))

    if fortran_order:
        raise ValueError(
            "member {} is stored in fortran order, row level random access below assumes the "
            "ordinary C order numpy uses by default".format(member_name)
        )

    data_offset = stream.tell()
    return zip_file, stream, dtype, shape, data_offset


def peek_shapes(file_path):
    """
    Opens X.npy and y.npy just long enough to read their headers, to
    learn num_samples and num_features, then closes everything. Used
    wherever only the shape is needed and no lasting handle should be
    kept open, such as computing dataset length up front or determining
    the model's input size before any DataLoader worker exists.
    """
    zip_x, stream_x, dtype_x, shape_x, _ = open_npy_member(file_path, "X")
    stream_x.close()
    zip_x.close()

    zip_y, stream_y, dtype_y, shape_y, _ = open_npy_member(file_path, "y")
    stream_y.close()
    zip_y.close()

    num_samples, num_features = shape_x
    return num_samples, num_features, dtype_x, dtype_y


# ---------------------------------------------------------------------------
# Eager strategy
# ---------------------------------------------------------------------------

class EagerNPZDataset(Dataset):
    """
    Reads both arrays into memory once, in __init__, with plain
    np.load. Unlike lazy and stream below, this works whether the
    archive is stored or deflated, since eager was always going to read
    every byte of both arrays regardless, there is no random access
    happening here to lose by allowing compression. After __init__, this
    is an ordinary in memory map style Dataset: __getitem__ never
    touches the file again.
    """

    def __init__(self, file_path):
        with np.load(file_path) as npz:
            X = npz["X"]
            y = npz["y"]
            self.X = torch.from_numpy(X.astype(np.float32, copy=True))
            self.y = torch.from_numpy(y.astype(np.int64, copy=True))
        self.num_features = self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Lazy strategy
# ---------------------------------------------------------------------------

class LazyNPZDataset(Dataset):
    """
    Learns num_samples once in __init__ through peek_shapes, and keeps
    only the file path and that count, no zip handles yet. On each
    __getitem__ call, the two members (X.npy and y.npy) are opened
    lazily on first use in whichever process calls it, and every call
    seeks directly to the requested row's byte offset inside each member
    and reads exactly one row's worth of bytes. Nothing is cached:
    reading the same index twice means two separate seeks and reads.

    Handles are opened lazily rather than in __init__, and stored per
    process, for the same reason as the other formats' lazy datasets:
    PyTorch's default multiprocessing start method for DataLoader
    workers on Linux is fork, and file handles opened in the parent
    process before that fork are not safe to hand to the resulting
    worker processes. Opening on first use guarantees each worker opens
    and owns its own handles.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        num_samples, num_features, _, _ = peek_shapes(file_path)
        self._length = num_samples
        self.num_features = num_features
        self._x = None
        self._y = None

    def _ensure_open(self):
        if self._x is None:
            zip_x, stream_x, dtype_x, shape_x, offset_x = open_npy_member(self.file_path, "X")
            row_bytes_x = shape_x[1] * dtype_x.itemsize
            self._x = (zip_x, stream_x, dtype_x, offset_x, row_bytes_x)

        if self._y is None:
            zip_y, stream_y, dtype_y, shape_y, offset_y = open_npy_member(self.file_path, "y")
            self._y = (zip_y, stream_y, dtype_y, offset_y, dtype_y.itemsize)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        self._ensure_open()

        _, stream_x, dtype_x, offset_x, row_bytes_x = self._x
        stream_x.seek(offset_x + idx * row_bytes_x)
        raw_x = stream_x.read(row_bytes_x)
        features = np.frombuffer(raw_x, dtype=dtype_x)

        _, stream_y, dtype_y, offset_y, row_bytes_y = self._y
        stream_y.seek(offset_y + idx * row_bytes_y)
        raw_y = stream_y.read(row_bytes_y)
        label = int(np.frombuffer(raw_y, dtype=dtype_y)[0])

        return torch.from_numpy(features.copy()), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Stream strategy (also the basis for burst)
# ---------------------------------------------------------------------------

class StreamNPZDataset(IterableDataset):
    """
    Iterates both members forward in step, strictly in row order,
    reading fixed size contiguous blocks of rows out of X.npy and the
    matching block out of y.npy with one read call per block per member.
    A block is never retained after its rows have been yielded, so a
    later epoch reads every row from disk again.

    When DataLoader uses more than one worker, the row range is split by
    hand in __iter__, using get_worker_info, into as many contiguous non
    overlapping spans as there are workers, and each worker seeks its
    two member streams directly to the start of its own span before
    reading forward, the same seeking capability lazy relies on, just
    used once per worker instead of once per row.

    block_rows defaults to a fixed fallback rather than an auto detected
    value: the array data inside an npy member is a flat, unchunked
    run of bytes with no storage granularity of its own to align to,
    the same reasoning as the BIN and CSV versions of this dataset.
    """

    FALLBACK_BLOCK_ROWS = 1024

    def __init__(self, file_path, block_rows="auto"):
        self.file_path = file_path
        num_samples, num_features, _, _ = peek_shapes(file_path)
        self._length = num_samples
        self.num_features = num_features
        self.block_rows = self.FALLBACK_BLOCK_ROWS if block_rows == "auto" else int(block_rows)

    def __len__(self):
        return self._length

    def _worker_span(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return 0, self._length
        per_worker = int(np.ceil(self._length / worker_info.num_workers))
        start = worker_info.id * per_worker
        end = min(start + per_worker, self._length)
        return start, end

    def __iter__(self):
        start, end = self._worker_span()
        if start >= end:
            return

        zip_x, stream_x, dtype_x, shape_x, offset_x = open_npy_member(self.file_path, "X")
        zip_y, stream_y, dtype_y, shape_y, offset_y = open_npy_member(self.file_path, "y")
        row_bytes_x = shape_x[1] * dtype_x.itemsize
        row_bytes_y = dtype_y.itemsize

        try:
            stream_x.seek(offset_x + start * row_bytes_x)
            stream_y.seek(offset_y + start * row_bytes_y)

            remaining = end - start
            while remaining > 0:
                rows_to_read = min(self.block_rows, remaining)

                raw_x = stream_x.read(rows_to_read * row_bytes_x)
                block_x_np = np.frombuffer(raw_x, dtype=dtype_x).reshape(rows_to_read, shape_x[1])

                raw_y = stream_y.read(rows_to_read * row_bytes_y)
                block_y_np = np.frombuffer(raw_y, dtype=dtype_y)

                # Convert/copy once per I/O block, then yield tensor views row by row.
                # This matches eager's float32/int64 training dtypes while avoiding
                # one NumPy allocation and one scalar Tensor construction per sample.
                block_x = torch.from_numpy(block_x_np.astype(np.float32, copy=True))
                block_y = torch.from_numpy(block_y_np.astype(np.int64, copy=True))

                for i in range(rows_to_read):
                    yield block_x[i], block_y[i]

                remaining -= rows_to_read
        finally:
            stream_x.close()
            zip_x.close()
            stream_y.close()
            zip_y.close()


# ---------------------------------------------------------------------------
# DataLoader construction per strategy
# ---------------------------------------------------------------------------

def build_dataloader(strategy, file_path, batch_size, num_workers, block_rows, prefetch_factor, persistent_workers):
    """
    persistent_workers, when True, keeps the worker processes alive
    across epochs instead of the default behavior of tearing them down
    and respawning them at the start of every epoch. This is only
    meaningful when num_workers is at least one, so that combination is
    rejected explicitly here rather than left for PyTorch to reject with
    a less specific error.
    """
    if persistent_workers and num_workers < 1:
        raise ValueError(
            "persistent_workers requires num_workers of at least 1: "
            "there is no worker process to keep alive across epochs otherwise"
        )

    if strategy == "eager":
        dataset = EagerNPZDataset(file_path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    if strategy == "lazy":
        dataset = LazyNPZDataset(file_path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    if strategy == "stream":
        dataset = StreamNPZDataset(file_path, block_rows=block_rows)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    if strategy == "burst":
        if num_workers < 1:
            raise ValueError(
                "strategy burst requires num_workers of at least 1: "
                "prefetch_factor has no effect without worker processes to do the prefetching"
            )
        dataset = StreamNPZDataset(file_path, block_rows=block_rows)
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

def train(args):
    run_start = time.perf_counter()
    device = torch.device(args.device)

    loader_setup_start = time.perf_counter()
    loader = build_dataloader(
        args.strategy,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.block_rows,
        args.prefetch_factor,
        args.persistent_workers,
    )
    loader_setup_time = time.perf_counter() - loader_setup_start
    num_features = loader.dataset.num_features

    model = build_model(args.model, num_features, args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None

    model.train()
    training_start = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        running_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if num_batches % args.log_every == 0 or num_batches == total_batches:
                report_progress(epoch + 1, args.epochs, num_batches, total_batches, epoch_start)

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / max(1, num_batches)
        print(
            "epoch {}/{}  loss {:.4f}  batches {}  time {:.2f}s".format(
                epoch + 1, args.epochs, avg_loss, num_batches, epoch_time
            )
        )

    training_time = time.perf_counter() - training_start
    total_time = time.perf_counter() - run_start
    print(
        "timing  loader_setup {:.3f}s  training {:.3f}s  total {:.3f}s".format(
            loader_setup_time, training_time, total_time
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a classifier over an NPZ dataset under a configurable loading strategy."
    )
    parser.add_argument("--data-path", required=True, help="Path to an NPZ file with X and y arrays.")
    parser.add_argument("--strategy", required=True, choices=["eager", "lazy", "stream", "burst"])
    parser.add_argument("--model", default="M2", choices=["M1", "M2", "M3"])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--block-rows",
        default="auto",
        help="Rows read per contiguous call for the stream and burst strategies.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches queued ahead per worker, used only by the burst strategy.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep worker processes alive across epochs instead of respawning them every epoch. "
        "Requires num_workers of at least 1.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Print a progress line every this many batches, so a long run shows it is "
        "still advancing instead of going silent until the epoch ends.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
