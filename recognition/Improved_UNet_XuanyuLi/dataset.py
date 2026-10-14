import os

import nibabel as nib
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


NUM_CLASSES = 6


class HipMRIDataset(Dataset):

    def __init__(
        self,
        image_folder,
        mask_folder
    ):
        self.image_folder = image_folder
        self.mask_folder = mask_folder

        self.image_files = sorted([
            file
            for file in os.listdir(image_folder)
            if file.endswith(".nii.gz")
        ])

        self.samples = []

        for image_name in self.image_files:

            mask_name = image_name.replace(
                "case_",
                "seg_",
                1
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
                raise FileNotFoundError(
                    f"Missing mask for {image_name}"
                )

            self.samples.append(
                (
                    image_path,
                    mask_path,
                    image_name
                )
            )


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):

        (
            image_path,
            mask_path,
            image_name
        ) = self.samples[index]

        image = nib.load(
            image_path
        ).get_fdata(
            caching="unchanged"
        )

        mask = nib.load(
            mask_path
        ).get_fdata(
            caching="unchanged"
        )

        # Some NIfTI files may contain an extra dimension.
        if image.ndim == 3:
            image = image[:, :, 0]

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        image = image.astype(
            np.float32
        )

        mask = mask.astype(
            np.int64
        )

        # Z-score normalisation.
        image = (
            image - image.mean()
        ) / (
            image.std() + 1e-8
        )

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            mask
        )

        return (
            image,
            mask,
            image_name
        )


def create_dataloaders(
    data_root,
    batch_size=8,
    num_workers=0
):

    train_dataset = HipMRIDataset(
        os.path.join(
            data_root,
            "keras_slices_train"
        ),
        os.path.join(
            data_root,
            "keras_slices_seg_train"
        )
    )

    validation_dataset = HipMRIDataset(
        os.path.join(
            data_root,
            "keras_slices_validate"
        ),
        os.path.join(
            data_root,
            "keras_slices_seg_validate"
        )
    )

    test_dataset = HipMRIDataset(
        os.path.join(
            data_root,
            "keras_slices_test"
        ),
        os.path.join(
            data_root,
            "keras_slices_seg_test"
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
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

    return (
        train_loader,
        validation_loader,
        test_loader
    )


if __name__ == "__main__":

    DATA_ROOT = (
        "/home/groups/comp3710/"
        "HipMRI_Study_open/"
        "keras_slices_data"
    )

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders(
        DATA_ROOT,
        batch_size=8,
        num_workers=0
    )

    print(
        "Train slices:",
        len(train_loader.dataset)
    )

    print(
        "Validation slices:",
        len(validation_loader.dataset)
    )

    print(
        "Test slices:",
        len(test_loader.dataset)
    )

    images, masks, names = next(
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

    print(
        "Image mean:",
        images.mean().item()
    )

    print(
        "Image std:",
        images.std().item()
    )

    print(
        "Example file:",
        names[0]
    )