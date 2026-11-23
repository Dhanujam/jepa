import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os

import torch
from torch.utils.data import DataLoader

from training.trainer import JEPATrainer
from training.ixi_dataset import IXIMRIDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--masking-strategy",
        choices=["random", "tissue_aware"],
        default="random",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")

    torch.manual_seed(42)

    trainer = JEPATrainer(
        masking_strategy=args.masking_strategy
    )

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location=trainer.device,
            weights_only=True,
        )

        trainer.context_encoder.load_state_dict(
            checkpoint["context_encoder"]
        )
        trainer.target_encoder.load_state_dict(
            checkpoint["target_encoder"]
        )
        trainer.predictor.load_state_dict(
            checkpoint["predictor"]
        )
        trainer.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        trainer.step = checkpoint["step"]

        print("Resumed from:", args.resume)
        print("Previous training step:", trainer.step)

    print("Device:", trainer.device)
    print("Masking strategy:", args.masking_strategy)

    if args.data_dir:
        dataset = IXIMRIDataset(args.data_dir)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        print("Mode: REAL MRI TRAINING")
        data_iterator = iter(loader)

        for step in range(args.steps):
            try:
                images = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                images = next(data_iterator)

            loss = trainer.train_step(
                images.to(trainer.device)
            )

            if not torch.isfinite(torch.tensor(loss)):
                raise RuntimeError("Non-finite loss detected")

            print(
                f"Step {step + 1}/{args.steps} | "
                f"Total step: {trainer.step} | "
                f"JEPA loss: {loss:.6f}"
            )

        default_name = "jepa_ixi_pretrained.pth"

    else:
        print("Mode: SYNTHETIC SMOKE TEST (not MRI training)")

        for step in range(args.steps):
            images = torch.randn(
                args.batch_size, 1, 224, 224
            )
            loss = trainer.train_step(
                images.to(trainer.device)
            )

            if not torch.isfinite(torch.tensor(loss)):
                raise RuntimeError("Non-finite loss detected")

            print(
                f"Step {step + 1}/{args.steps} | "
                f"Total step: {trainer.step} | "
                f"JEPA loss: {loss:.6f}"
            )

        default_name = "jepa_smoke_test.pth"

    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_path = args.checkpoint or os.path.join(
        "checkpoints", default_name
    )

    trainer.save_checkpoint(checkpoint_path)

    print("Checkpoint saved:", os.path.abspath(checkpoint_path))
    print("Training run completed.")


if __name__ == "__main__":
    main()
