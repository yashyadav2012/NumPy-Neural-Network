#!/usr/bin/env python3
"""Train a neural network to solve the XOR problem.

Usage:
    python train.py            # train and print predictions
    python train.py --plot     # also generate plots in figures/
"""

from __future__ import annotations

import argparse

import numpy as np

from src.model import NeuralNetwork
from src.utils import (
    plot_decision_boundary,
    plot_loss_curve,
    print_predictions,
    print_training_summary,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a NumPy neural network on the XOR problem.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save decision-boundary and loss-curve plots.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10_000,
        help="Number of training epochs (default: 10 000).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: set up data, train the model, and report results."""
    args = parse_args()

    # ---- XOR dataset ----
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

    # ---- Model ----
    model = NeuralNetwork(
        input_size=2,
        hidden_size=4,
        output_size=1,
        learning_rate=args.lr,
        seed=42,
    )

    print("=" * 55)
    print("  NumPy Neural Network - XOR from Scratch")
    print("=" * 55)
    print(f"  Architecture : 2 -> 4 -> 1")
    print(f"  Activation   : Sigmoid")
    print(f"  Loss         : MSE")
    print(f"  Epochs       : {args.epochs:,}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Seed         : 42")
    print("=" * 55 + "\n")

    # ---- Training ----
    losses = model.train(X, y, epochs=args.epochs)

    # ---- Results ----
    predictions = model.predict(X)
    print_predictions(X, y, predictions)
    print_training_summary(losses)

    # ---- Optional plots ----
    if args.plot:
        plot_decision_boundary(model, X, y)
        plot_loss_curve(losses)


if __name__ == "__main__":
    main()
