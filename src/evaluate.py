import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluate", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            # important because F1 should generally be calculated over the entire validation/test set, not by calculating F1 for each batch and averaging the batch F1s.
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_samples = len(all_labels)
    average_loss = total_loss / total_samples

    accuracy = accuracy_score(all_labels, all_predictions)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        average=None, # → metrics for each class individually
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        average="macro", # → average across classes; each class has equal weight
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        average="weighted", # → average across classes weighted by each class's number of samples
        zero_division=0,
    )

    per_class = {}

    # From average=None,
    for i, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": int(support[i]),
        }

    metrics = {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }

    return metrics