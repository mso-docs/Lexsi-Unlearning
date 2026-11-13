#!/usr/bin/env python3
"""scripts/run_pipeline.py with an explicit random seed, without touching run_pipeline.py or src/.

    LEXSI_SEED=43 python scripts/run_pipeline_seeded.py --config ... --stage ... --run-name ...

With LEXSI_SEED unset this is exactly run_pipeline.py. With it set, the seed replaces the hard-coded
defaults the pipeline uses: the HF Trainer seed (LoRA init, data order, dropout) via TrainingArguments,
the `shuffle(seed=42)` of the injection mix, and the torch/numpy/python RNGs before RMU / NPO
(RMU draws a random steering direction). The seed is printed at start so it lands in the run log.
"""
from __future__ import annotations

import os
import random
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def apply_seed(seed: int) -> None:
    import transformers

    TA = transformers.TrainingArguments
    _orig_init = TA.__init__

    def _init(self, *args, **kwargs):
        kwargs.setdefault("seed", seed)
        kwargs.setdefault("data_seed", seed)
        _orig_init(self, *args, **kwargs)

    TA.__init__ = _init

    import datasets

    DS = datasets.Dataset
    _orig_shuffle = DS.shuffle

    def _shuffle(self, *args, **kwargs):
        if kwargs.get("seed") == 42:
            kwargs["seed"] = seed
        return _orig_shuffle(self, *args, **kwargs)

    DS.shuffle = _shuffle

    from lexsi_unlearn import unlearn as _un

    for name in ("run_rmu", "run_npo"):
        fn = getattr(_un, name)

        def _wrap(*a, __fn=fn, **kw):
            _seed_all(seed)
            return __fn(*a, **kw)

        setattr(_un, name, _wrap)
    _seed_all(seed)


def main() -> None:
    raw = (os.environ.get("LEXSI_SEED") or "").strip()
    if raw:
        seed = int(raw)
        apply_seed(seed)
        print(f"[run_pipeline_seeded] LEXSI_SEED={seed} (Trainer seed, injection shuffle, RMU/NPO RNG)", flush=True)
    else:
        print("[run_pipeline_seeded] LEXSI_SEED unset: identical to run_pipeline.py (default seed 42)", flush=True)
    sys.argv[0] = str(ROOT / "scripts" / "run_pipeline.py")
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
