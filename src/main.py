import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

# print('PyTorch:', torch.__version__)
# print('CUDA available:', torch.cuda.is_available())
# print('CUDA version:', torch.version.cuda)
# print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')

from dataset import PlantVillageDataset
from evaluate import evaluate
from model import create_model
from train import train_one_epoch

from tqdm import tqdm
import time
from collections import Counter


def print_class_counts(name, dataset):
    counts = Counter(dataset.labels)
    print(f"{name} distribution:")
    for cls, count in sorted(counts.items()):
        print(f"  Class {cls}: {count}")
    # for cls, count in sorted(counts.items()):
    #     class_name = dataset.dataset.class_names[cls]
    #     print(f"  {class_name}: {count}")

def format_elapsed_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}hr {minutes}m"

def parse_args():
    parser = argparse.ArgumentParser(description="PlantVillage baseline experiment")

    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model name from config")
    parser.add_argument("--experiment", type=str, default=None, help="Override experiment ID")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--objective", type=str, default=None, help="Objective or hypothesis being tested in this experiment.",)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform, val_transform


def create_optimizer(model, config):
    optimizer_name = config["training"]["optimizer"].lower()
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0.0)

    if optimizer_name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer, config):
    scheduler_name = config["training"].get("scheduler", "none").lower()
    epochs = config["training"]["epochs"]

    if scheduler_name == "none":
        return None

    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)

    if scheduler_name == "step":
        step_size = config["training"].get("scheduler_step_size", 5)
        gamma = config["training"].get("scheduler_gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    experiment_start_time = time.perf_counter()

    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.model is not None:
        config["model"]["name"] = args.model

    if args.experiment is not None:
        config["experiment"]["id"] = args.experiment

    if args.objective is not None:
        config["experiment"]["objective"] = args.objective

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate

    if args.optimizer is not None:
        config["training"]["optimizer"] = args.optimizer

    if args.scheduler is not None:
        config["training"]["scheduler"] = args.scheduler
        

    if args.seed is not None:
        config["seed"] = args.seed


    seed = config["seed"]
    set_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Experiment setup
    experiment_id = config["experiment"]["id"]
    model_name = config["model"]["name"]

    experiment_dir = Path("experiments") / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(experiment_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Experiment: {experiment_id}")
    print(f"Model: {model_name}")
    objective = config["experiment"]["objective"]
    print(f"Objective: {objective}")

    # Transforms
    train_transform, val_transform, test_transform = create_transforms(
        config["data"]["image_size"]
    )

    # Dataset
    dataset = PlantVillageDataset(
        data_path=config["data"]["path"],
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        seed=seed,
        n_splits=config["data"]["n_splits"],
    )

    # Experiment settings
    num_epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)

    all_split_results = []

    # 10-fold experiment
    for split_number in range(dataset.n_splits):

        split_start_time = time.perf_counter()

        print(f"\n{'=' * 60}")
        print(f"Split {split_number + 1}/{dataset.n_splits}")
        print(f"{'=' * 60}")

        trainset = dataset.train[split_number]
        valset = dataset.val[split_number]
        testset = dataset.test[split_number]

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,)

        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,)

        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,)

        print_class_counts("Train", trainset)
        print_class_counts("Val", valset)
        print_class_counts("Test", testset)

        # Model
        model = create_model(
            num_classes=dataset.num_classes,
            pretrained=config["model"].get("pretrained", True),
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = create_optimizer(model,config,)

        scheduler = create_scheduler(optimizer,config,)

        # Epoch loop
        history = []
        best_val_f1 = -float("inf")
        best_epoch = 0

        checkpoint_dir = experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        for epoch in tqdm(range(1, num_epochs + 1), desc=f"Split {split_number + 1}/{dataset.n_splits}"):

            train_loss, train_accuracy = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                class_names=dataset.class_names,
            )

            if scheduler is not None:
                scheduler.step()

            epoch_results = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_precision": val_metrics["weighted_precision"],
                "val_weighted_recall": val_metrics["weighted_recall"],
                "val_weighted_f1": val_metrics["weighted_f1"],
            }

            history.append(epoch_results)

            tqdm.write(
                f"Epoch {epoch:02d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f}"
            )

            # Save best checkpoint based on validation macro-F1
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                print(f"✅ Saved new best model at epoch {epoch} (val_loss={best_val_f1:.4f})")

                torch.save(
                    model.state_dict(),
                    checkpoint_dir / f"split_{split_number}_best.pt",
                )

        # Final test evaluation
        best_model_path = checkpoint_dir / f"split_{split_number}_best.pt"
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Track Time taken for each split
        split_elapsed = time.perf_counter() - split_start_time
        split_time = format_elapsed_time(split_elapsed)
        print(f"Split {split_number + 1} completed in {split_time}")

        test_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            class_names=dataset.class_names,
        )

        split_results = {
            "split": split_number,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "test_metrics": test_metrics,
            "history": history, # Has each epoch result
            "split_elapsed_seconds": split_elapsed,
            "split_elapsed_time": split_time,
        }

        all_split_results.append(split_results)

        save_json(
            split_results,
            experiment_dir / f"split_{split_number}_results.json",
        )

        print(
            f"Best Val F1: {best_val_f1:.4f} | "
            f"Test F1: {test_metrics['macro_f1']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f}"
        )

    # Calculate aggregate statistics across 10 splits
    test_metrics_to_average = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]

    aggregate_test_metrics = {}

    for metric in test_metrics_to_average:
        values = [
            split["test_metrics"][metric]
            for split in all_split_results
        ]

        aggregate_test_metrics[f"{metric}_mean"] = float(np.mean(values))
        aggregate_test_metrics[f"{metric}_std"] = float(np.std(values))

    # Time taken for whole experiment
    experiment_elapsed = time.perf_counter() - experiment_start_time
    experiment_time = format_elapsed_time(experiment_elapsed)
    print(f"\nExperiment completed in {experiment_time}")

    # Save complete experiment
    experiment_results = {
        "experiment_id": experiment_id,
        "objective": objective,
        "model": model_name,
        "seed": seed,
        "num_splits": dataset.n_splits,
        "experiment_elapsed_seconds": experiment_elapsed,
        "experiment_elapsed_time": experiment_time,
        "aggregate_test_metrics": aggregate_test_metrics,
        "splits": all_split_results,
    }

    save_json(
        experiment_results,
        experiment_dir / "results.json",
    )

    print(f"\nExperiment complete: {experiment_dir}")


if __name__ == "__main__":
    main()