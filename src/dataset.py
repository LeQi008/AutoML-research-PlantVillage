from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset


class PlantVillageDataset(Dataset):
    def __init__(
        self,
        data_path="data/archive/PlantVillage",
        train_transform=None,
        val_transform=None,
        test_transform=None,
        seed=42,
        n_splits=10,
    ):
        self.data_path = Path(data_path)
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.seed = seed
        self.n_splits = n_splits

        self.image_paths = []
        self.labels = []
        self.class_names = []

        self._load_data()
        self._create_folds()

    def _load_data(self):
        class_directories = sorted([d for d in self.data_path.iterdir() if d.is_dir()])

        self.class_names = [d.name for d in class_directories]
        class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for class_directory in class_directories:
            class_idx = class_to_idx[class_directory.name]

            for image_path in sorted(class_directory.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in valid_extensions:
                    self.image_paths.append(image_path)
                    self.labels.append(class_idx)

        print(f"Loaded {len(self.image_paths)} images across {self.num_classes} classes.")

    def _create_folds(self):
        labels = np.array(self.labels)

        splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        self.folds = [fold_indices for _, fold_indices in splitter.split(self.image_paths, labels)]

        self.train = []
        self.val = []
        self.test = []

        for split_number in range(self.n_splits):
            test_fold = split_number
            val_fold = (split_number + 1) % self.n_splits

            train_folds = [
                i for i in range(self.n_splits)
                if i not in {test_fold, val_fold}
            ]

            train_indices = np.concatenate([self.folds[i] for i in train_folds])
            val_indices = self.folds[val_fold]
            test_indices = self.folds[test_fold]

            # Convert splits into dataset objects
            self.train.append(
                PlantVillageSplit(self, train_indices, self.train_transform)
            )
            self.val.append(
                PlantVillageSplit(self, val_indices, self.val_transform)
            )
            self.test.append(
                PlantVillageSplit(self, test_indices, self.test_transform)
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = self.labels[index]

        if self.train_transform is not None:
            image = self.train_transform(image)

        return image, label


class PlantVillageSplit(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image_path = self.dataset.image_paths[original_idx]
        label = self.dataset.labels[original_idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @property
    def labels(self):
        return [self.dataset.labels[i] for i in self.indices]