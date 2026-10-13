import os
import re
import random

from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def get_patient_id(file_name):
    match = re.search(
        r"case[_-]?(\d+)",
        file_name,
        re.IGNORECASE
    )

    if match is None:
        raise ValueError(
            f"Cannot find patient ID in file name: {file_name}"
        )

    return match.group(1)


def get_mask_name(image_name):
    if image_name.startswith("case_"):
        return image_name.replace(
            "case_",
            "seg_",
            1
        )

    return image_name


def find_data_folders(data_root):
    image_folders = [
        os.path.join(
            data_root,
            "keras_png_slices_train"
        ),
        os.path.join(
            data_root,
            "keras_png_slices_validate"
        ),
        os.path.join(
            data_root,
            "keras_png_slices_test"
        )
    ]

    mask_folders = [
        os.path.join(
            data_root,
            "keras_png_slices_seg_train"
        ),
        os.path.join(
            data_root,
            "keras_png_slices_seg_validate"
        ),
        os.path.join(
            data_root,
            "keras_png_slices_seg_test"
        )
    ]

    return image_folders, mask_folders


def collect_pairs(
    image_folders,
    mask_folders
):
    pairs = []

    for image_folder, mask_folder in zip(
        image_folders,
        mask_folders
    ):
        if not os.path.isdir(image_folder):
            continue

        if not os.path.isdir(mask_folder):
            continue

        image_names = sorted([
            name
            for name in os.listdir(image_folder)
            if name.lower().endswith(".png")
        ])

        for image_name in image_names:
            mask_name = get_mask_name(
                image_name
            )

            image_path = os.path.join(
                image_folder,
                image_name
            )

            mask_path = os.path.join(
                mask_folder,
                mask_name
            )

            if not os.path.exists(mask_path):
                print(
                    "Missing mask:",
                    mask_path
                )
                continue

            pairs.append({
                "image": image_path,
                "mask": mask_path,
                "patient": get_patient_id(
                    image_name
                )
            })

    if len(pairs) == 0:
        raise RuntimeError(
            "No MRI-mask pairs were found."
        )

    return pairs


def split_by_patient(
    pairs,
    train_ratio=0.70,
    val_ratio=0.15,
    seed=42
):
    patients = sorted(
        set(
            pair["patient"]
            for pair in pairs
        )
    )

    random_generator = random.Random(seed)

    random_generator.shuffle(
        patients
    )

    total_patients = len(patients)

    train_end = int(
        total_patients
        * train_ratio
    )

    val_end = train_end + int(
        total_patients
        * val_ratio
    )

    train_patients = set(
        patients[:train_end]
    )

    val_patients = set(
        patients[train_end:val_end]
    )

    test_patients = set(
        patients[val_end:]
    )

    train_pairs = [
        pair
        for pair in pairs
        if pair["patient"] in train_patients
    ]

    val_pairs = [
        pair
        for pair in pairs
        if pair["patient"] in val_patients
    ]

    test_pairs = [
        pair
        for pair in pairs
        if pair["patient"] in test_patients
    ]

    assert train_patients.isdisjoint(
        val_patients
    )

    assert train_patients.isdisjoint(
        test_patients
    )

    assert val_patients.isdisjoint(
        test_patients
    )

    print(
        "Patients:",
        len(patients)
    )

    print(
        "Train patients:",
        len(train_patients)
    )

    print(
        "Validation patients:",
        len(val_patients)
    )

    print(
        "Test patients:",
        len(test_patients)
    )

    return (
        train_pairs,
        val_pairs,
        test_pairs
    )


class OASISDataset(Dataset):
    def __init__(
        self,
        pairs,
        image_size=128
    ):
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]

        image = Image.open(
            pair["image"]
        ).convert("L")

        mask = Image.open(
            pair["mask"]
        ).convert("L")

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR
        )

        # Nearest-neighbour resize avoids creating new mask labels.
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST
        )

        image = TF.to_tensor(
            image
        )

        mask = TF.pil_to_tensor(
            mask
        ).squeeze(0)

        # Original labels are 0, 85, 170 and 255.
        mask = (
            mask // 85
        ).long()

        return image, mask


def create_dataloaders(
    data_root,
    image_size=128,
    batch_size=16,
    num_workers=0,
    seed=42
):
    image_folders, mask_folders = find_data_folders(
        data_root
    )

    pairs = collect_pairs(
        image_folders,
        mask_folders
    )

    (
        train_pairs,
        val_pairs,
        test_pairs
    ) = split_by_patient(
        pairs,
        seed=seed
    )

    train_dataset = OASISDataset(
        train_pairs,
        image_size
    )

    val_dataset = OASISDataset(
        val_pairs,
        image_size
    )

    test_dataset = OASISDataset(
        test_pairs,
        image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(
        "Train slices:",
        len(train_dataset)
    )

    print(
        "Validation slices:",
        len(val_dataset)
    )

    print(
        "Test slices:",
        len(test_dataset)
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


if __name__ == "__main__":
    data_root = r"C:\Users\XUANYU LI\PycharmProjects\PythonProject7\keras_png_slices_data"

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(
        data_root,
        batch_size=8
    )

    images, masks = next(
        iter(train_loader)
    )

    print(
        "Images:",
        images.shape
    )

    print(
        "Masks:",
        masks.shape
    )

    print(
        "Mask labels:",
        torch.unique(masks)
    )