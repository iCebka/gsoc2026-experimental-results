#!/usr/bin/env python3
"""
Training script for a tabular binary classification dataset stored as
ROOT (TTree or RNTuple), under a configurable data loading strategy.

RDataLoader, part of ROOT.Experimental.ML, is a complete batch producing
pipeline built on top of RDataFrame. This makes it structurally
different from every loading mechanism the other four format scripts
in this series wrap: it is not a torch.utils.data.Dataset meant to be
handed to torch.utils.data.DataLoader, iterating dl.as_torch() already
yields ready to use (X, y) batches directly. Concretely, this means
PyTorch's own num_workers, persistent_workers, and prefetch_factor have
no equivalent role here for three of the four strategies below: ROOT
manages its own reading and any internal concurrency itself, not
through PyTorch worker processes. Where those PyTorch DataLoader
arguments would normally go, ROOT instead has its own knobs, used below
where relevant.

Four loading strategies are available, selected with --strategy:

eager: RDataLoader is constructed with load_eager=True, which reads the
whole dataset into memory once, at construction time, before training
starts, then serves every subsequent batch from memory.

stream: RDataLoader is constructed with load_eager=False (the library's
own default) and shuffle explicitly disabled, so it reads the dataset
forward in fixed size batches, chunk by chunk from disk, never
retaining a chunk once it has been consumed. Confusingly, ROOT's own
documentation calls this mode "lazy loading", but by this project's
vocabulary it is what we call stream: it visits rows through sequential
chunked reads, not through access by arbitrary row index. Anywhere this
script or its comments say lazy, they mean the strategy defined below,
not ROOT's own use of that word.

burst: the same chunked, forward only reading as stream, with
batches_in_memory raised above its default (its role, per ROOT's
documentation, is primarily the size of an internal shuffle buffer, but
holding more batches in that buffer necessarily means more of them have
already been read off disk ahead of being consumed, which is the
buffering effect this strategy is meant to exercise). shuffle is kept
disabled here too, on purpose, specifically so that the only difference
between stream and burst is that buffer depth, not also whether rows
come out reordered.

lazy: true random access by row index, with nothing cached between
accesses. Since RDataLoader offers no such capability at all, this
strategy is built directly on ROOT::RNTupleReader::GetView, which
provides exactly this: given a field name, GetView returns a callable
that reads that one field's value at an arbitrary entry index, an
operation documented to touch only that field's own on disk column, not
the whole entry. A separate view is opened per feature column plus one
for the target column, and every __getitem__ call reads through all of
them fresh, at the requested index, nothing retained. RNTuple stores
each field as its own separate column, so unlike lazy on a row major
format such as BIN or HDF5, one row here is not one contiguous read, it
is one small read per field, this many small reads is treated as the
real access pattern lazy has here, not an inefficiency to hide.

As with the other four formats, only one class is not a direct instance
of a standard PyTorch dataset abstraction: lazy is an ordinary map style
Dataset subclass, everything else is a direct use of RDataLoader with
different constructor arguments, no dataset class of its own needed.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import ROOT

from ROOT.Experimental.ML import RDataLoader

PROGRAM_START = time.time()

# Column type strings treated as numeric feature candidates. This is the
# exact same set the dataset converter script uses to decide which ROOT
# columns become features when generating the other four formats from
# this same source file, duplicated here rather than imported so that
# this script stays self contained, on purpose at the cost of the two
# copies being able to drift apart if one is edited without the other.
NUMERIC_TYPES = {
    "float",
    "double",
    "Float_t",
    "Double_t",
    "int",
    "unsigned int",
    "long",
    "unsigned long",
    "long long",
    "unsigned long long",
    "std::int32_t",
    "std::uint32_t",
    "std::int64_t",
    "std::uint64_t",
}


def debug(msg):
    elapsed = time.time() - PROGRAM_START
    print(f"[DEBUG {time.strftime('%H:%M:%S')} +{elapsed:.3f}s] {msg}", flush=True)

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
# Column selection, shared by every strategy
# ---------------------------------------------------------------------------

def select_feature_columns(rdf, target_col):
    """
    Walks every column RDataFrame reports, keeping the target column
    unconditionally and keeping every other column whose declared type
    is in NUMERIC_TYPES. Returns the feature column names in the same
    order RDataFrame reports them, target column excluded.

    This mirrors the dataset converter script's own column selection
    exactly, so that a model trained here sees the same feature set as
    the same source file converted to HDF5, BIN, CSV or NPZ. If this
    selection logic and the converter's ever disagree, the formats would
    silently stop describing the same dataset, so keeping them in step
    matters more than it might look like from either copy alone.
    """
    columns = [str(c) for c in rdf.GetColumnNames()]

    if target_col not in columns:
        raise ValueError("target column {} not found, available columns: {}".format(target_col, columns))

    feature_names = []
    for c in columns:
        if c == target_col:
            continue
        try:
            ctype = str(rdf.GetColumnType(c))
        except Exception:
            ctype = "unknown"
        if ctype in NUMERIC_TYPES:
            feature_names.append(c)

    return feature_names


def column_types(rdf, columns):
    """Returns {column_name: root_type_string} for the given columns, as reported by RDataFrame itself."""
    return {c: str(rdf.GetColumnType(c)) for c in columns}


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
# Lazy strategy: true random row access via RNTupleReader.GetView
# ---------------------------------------------------------------------------

class LazyROOTDataset(Dataset):
    """
    Keeps only the file path, object name, column names and their ROOT
    type strings after __init__, no reader opened yet. On first use in
    whichever process calls __getitem__, an RNTupleReader is opened and
    one GetView is created per feature column plus one for the target
    column, each kept open afterwards. Every call requests the required fields through the views at the
    requested index. No application-level sample cache is maintained by
    this Dataset. ROOT and the operating system may still satisfy repeated
    requests from their own internal caches.

    The reader and its views are opened lazily rather than in __init__,
    for the same reason as every other format's lazy dataset in this
    series: PyTorch's default multiprocessing start method for
    DataLoader workers on Linux is fork, and a reader opened in the
    parent process before that fork is unlikely to be safe to hand to
    the resulting worker processes, ROOT's own documentation discusses
    sharing a single reader across threads it manages itself, not across
    an externally forked process. Opening on first use guarantees each
    worker opens and owns its own reader.

    GetView's template argument has to match the field's on disk type
    exactly, so the exact ROOT type string RDataFrame reports for each
    column (Float_t, Double_t, std::int32_t, and so on) is threaded
    through from __init__ rather than assumed to be uniform across
    columns, since a real dataset's columns are not guaranteed to share
    one underlying type.
    """

    def __init__(self, root_path, tree_name, target_col):
        self.root_path = root_path
        self.tree_name = tree_name
        self.target_col = target_col

        rdf = ROOT.RDataFrame(tree_name, root_path)
        self.feature_names = select_feature_columns(rdf, target_col)
        self.num_features = len(self.feature_names)
        types = column_types(rdf, self.feature_names + [target_col])
        self.feature_types = [types[c] for c in self.feature_names]
        self.target_type = types[target_col]

        # Count(), the ordinary RDataFrame way to learn a row count,
        # rather than guessing at an RNTupleReader specific entry count
        # method name that may not match this ROOT version.
        #self._length = int(rdf.Count().GetValue())

        # GetNEntries used instead to avoid adding more I/O noise:
        reader = ROOT.RNTupleReader.Open(
            tree_name,
            root_path,
        )

        self._length = int(
            reader.GetNEntries()
        )

        self._reader = None
        self._feature_views = None
        self._target_view = None

    def _ensure_open(self):
        if self._reader is None:
            self._reader = ROOT.RNTupleReader.Open(self.tree_name, self.root_path)
            self._feature_views = [
                self._reader.GetView[t](name) for t, name in zip(self.feature_types, self.feature_names)
            ]
            self._target_view = self._reader.GetView[self.target_type](self.target_col)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        self._ensure_open()
        features = [view(idx) for view in self._feature_views]
        label = int(self._target_view(idx))
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Eager, stream and burst strategies: all three are RDataLoader itself
# ---------------------------------------------------------------------------

def build_rdataloader(strategy, rdf, feature_names, target_col, batch_size, prefetch_factor, shuffle, seed):
    """
    Constructs the RDataLoader for eager, stream or burst, the three
    strategies that map directly onto capabilities RDataLoader already
    has, so none of them need a dataset class of their own.

    columns is passed explicitly as the feature columns plus the target
    column, rather than omitted, so the loader reads only the columns
    this script actually uses even if the source RDataFrame carries
    others.

    shuffle is set explicitly for every branch rather than left on the
    library default (which is True): stream and burst are both meant to
    visit rows strictly in file order, matching what this project calls
    stream and burst for every other format, so shuffling would silently
    contradict that. eager is left shuffled, matching what eager does in
    the other four format scripts (DataLoader(shuffle=True) there).

    prefetch_factor here maps onto RDataLoader's batches_in_memory
    argument, used only by burst, kept above the library's own default
    of 10 specifically to make burst's buffer depth larger than stream's
    default, the two are otherwise identical in every other argument so
    that buffer depth is the only difference between them.
    """

    columns = list(feature_names) + [target_col]

    if strategy == "eager":
        return RDataLoader(
            rdf,
            columns=columns,
            target=target_col,
            batch_size=batch_size,
            load_eager=True,
            shuffle=shuffle,
            set_seed=seed,
        )

    if strategy == "stream":
        return RDataLoader(
            rdf,
            columns=columns,
            target=target_col,
            batch_size=batch_size,
            load_eager=False,
            shuffle=shuffle,
            batches_in_memory=2,
            set_seed=seed,
        )

    if strategy == "burst":
        return RDataLoader(
            rdf,
            columns=columns,
            target=target_col,
            batch_size=batch_size,
            load_eager=False,
            shuffle=shuffle,
            batches_in_memory=prefetch_factor,
            set_seed=seed,
        )

    raise ValueError("build_rdataloader only handles eager, stream, burst, got: {}".format(strategy))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    debug("entrered train()")
    debug(
        "configuration: strategy={} shuffle={} batch_size={} " 
        "prefetch_factor={} epochs={}".format(args.strategy,args.shuffle,args.batch_size,args.prefetch_factor,args.epochs,)
    )

    debug("before torch.device()")
    device = torch.device(args.device)
    debug("after torch.device()")

    if args.root_threads > 0:
        # The general ROOT level parallelism control for RDataFrame
        # backed operations. Left off (0) by default rather than
        # enabled implicitly, since this script cannot verify here how
        # strongly it affects RDataLoader's own internal chunking versus
        # only the surrounding RDataFrame evaluation.
        debug(f"enabling ROOT implicit MT threads")
        ROOT.EnableImplicitMT(args.root_threads)
        debug(f"ROOT implicit MT enabled")

    if args.strategy != "lazy" and (args.num_workers != 0 or args.persistent_workers):
        raise ValueError(
            "num_workers and persistent_workers only apply to strategy lazy in this script: "
            "eager, stream and burst are all RDataLoader itself, which manages its own reading "
            "rather than through PyTorch DataLoader worker processes. Use --root-threads instead "
            "for ROOT level parallelism with those three strategies."
        )

    if args.persistent_workers and args.num_workers < 1:
        raise ValueError(
            "persistent_workers requires num_workers of at least 1: "
            "there is no worker process to keep alive across epochs otherwise"
        )

    debug("before ROOT.RDataFrame()")
    rdf = ROOT.RDataFrame(args.tree_name, args.data_path)
    debug("after ROOT.RDataFrame()")

    debug("before select_feature_columns()")
    feature_names = select_feature_columns(rdf, args.target_col)
    debug(f"after select_feature_columns(): {len(feature_names)} features")
    num_features = len(feature_names)

    debug("before model construction")
    model = build_model(args.model, num_features, args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    debug("after model construction")

    if args.strategy == "lazy":
        dataset = LazyROOTDataset(args.data_path, args.tree_name, args.target_col)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
        )
        total_batches = len(loader)

        model.train()
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

    else:
        debug(f"before build_rdataloader(strategy={args.strategy})")
        dl = build_rdataloader(
            args.strategy, rdf, feature_names, args.target_col, args.batch_size, args.prefetch_factor, args.shuffle, args.seed
        )
        debug("after build_rdataloader")
        total_batches = None # In a previous iteration, eta was calculated but it introduced I/O noise, so now it
                                             # is eliminated

        model.train()
        for epoch in range(args.epochs):
            epoch_start = time.time()
            running_loss = 0.0
            num_batches = 0

            for X_batch, y_batch in dl.as_torch():
                X_batch = X_batch.float().to(device)
                # RDataLoader with a single target column has been
                # observed to yield y with a trailing singleton
                # dimension, shape (batch, 1) rather than (batch,).
                # CrossEntropyLoss requires a strictly 1D target, so this
                # is normalized explicitly rather than assumed away.
                y_batch = y_batch.long().reshape(-1).to(device)

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a classifier over a ROOT dataset under a configurable loading strategy."
    )
    parser.add_argument("--data-path", required=True, help="Path to the ROOT file.")
    parser.add_argument("--tree-name", default="tree", help="TTree or RNTuple object name inside the file.")
    parser.add_argument("--target-col", default="Label")
    parser.add_argument("--strategy", required=True, choices=["eager", "lazy", "stream", "burst"])
    shuffle_group = parser.add_mutually_exclusive_group(required=True)
    shuffle_group.add_argument("--shuffle",dest="shuffle",action="store_true",help="Enable sample shuffling.",)
    shuffle_group.add_argument("--no-shuffle",dest="shuffle",action="store_false",help="Disable sample shuffling.",)

    parser.add_argument("--model", default="M2", choices=["M1", "M2", "M3"])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help=(
            "Maps onto RDataLoader's batches_in_memory, used only by the "
            "burst strategy. A value of 4 is used for consistency with the "
            "prefetch configuration of the other data-loading experiments."
        ),
    )
    parser.add_argument(
        "--root-threads",
        type=int,
        default=0,
        help="Passed to ROOT.EnableImplicitMT if greater than 0. ROOT level parallelism, "
        "unrelated to num_workers below.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Only meaningful for strategy lazy, which is the only strategy in this script "
        "that uses torch.utils.data.DataLoader at all.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Only meaningful for strategy lazy, same reason as --num-workers. "
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
