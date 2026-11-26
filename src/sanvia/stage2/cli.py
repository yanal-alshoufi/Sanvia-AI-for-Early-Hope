"""Command line entry point for the Stage 2 research benchmark."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

ACTIONS = (
    "audit",
    "prepare",
    "plan",
    "smoke-test",
    "train-sequential",
    "train-multi-dataset",
    "finetune-vindr",
    "linear-probe",
    "evaluate",
    "compare",
    "export",
    "run-all",
)


def smoke_config(output: Path) -> dict:
    return dict(
        mode="smoke",
        name="synthetic",
        synthetic=True,
        pretrained=False,
        output=str(output),
        seeds=[42],
        mlflow=False,
        dvc=False,
        preview_images=True,
        source_training=dict(batch_size=2, accumulation=1, warmup_updates=1, train_updates=1, eval_every=2, workers=0),
    )


def main(argv=None):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    from sanvia.stage2.pipeline import Benchmark, configuration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="SANVIA checkout containing configs/")
    parser.add_argument("--config", type=Path, help="YAML override file, merged with the repository defaults")
    parser.add_argument("--output", type=Path, help="Private persistent output root")
    parser.add_argument(
        "--no-resume", action="store_true", help="Reject partial runs. Completed artifacts are verified and reused."
    )
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args(argv)
    overrides = yaml.safe_load(args.config.read_text()) if args.config else {}
    action = args.action
    if action == "smoke-test":
        from sanvia.utils.config import deep_update

        overrides = deep_update(smoke_config(args.output or Path("outputs/stage2").resolve()), overrides)
        if overrides.get("mode") != "smoke" or overrides.get("synthetic") is not True:
            parser.error("smoke-test requires mode=smoke and synthetic=true")
        action = "run-all"
    if args.output:
        overrides["output"] = str(args.output.resolve())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        bench = Benchmark(args.repo, configuration(args.repo, overrides), args.device)
        if action == "plan":
            import json

            print(json.dumps(bench.plan(), indent=2))
        else:
            print(bench.execute(action, not args.no_resume))
    except FileNotFoundError as exc:
        parser.exit(
            2, f"Required artifact or authorized data path is missing: {exc}. Check the paths in --config and run prepare.\n"
        )
    except torch.OutOfMemoryError as exc:
        parser.exit(
            2,
            f"Insufficient accelerator memory. Start a separately fingerprinted experiment with a smaller source batch size and matched accumulation. {exc}\n",
        )


if __name__ == "__main__":
    main()
