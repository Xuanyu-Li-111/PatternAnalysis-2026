import argparse
import time

import torch
import matplotlib.pyplot as plt

from dataset import create_dataloaders

from modules import (
    BasicUNet,
    ImprovedUNet
)

from train import calculate_metrics


NUM_CLASSES = 4


def build_model(model_name):
    if model_name == "baseline":
        return BasicUNet(
            in_channels=1,
            num_classes=NUM_CLASSES
        )

    if model_name == "improved":
        return ImprovedUNet(
            in_channels=1,
            num_classes=NUM_CLASSES
        )

    raise ValueError(
        "Unknown model name."
    )


def dice_for_single_sample(
    prediction,
    target,
    smooth=1e-6
):
    scores = []

    for class_id in range(NUM_CLASSES):
        pred_class = (
            prediction == class_id
        )

        target_class = (
            target == class_id
        )

        intersection = (
            pred_class
            & target_class
        ).sum().float()

        pred_count = (
            pred_class
        ).sum().float()

        target_count = (
            target_class
        ).sum().float()

        dice = (
            2 * intersection
            + smooth
        ) / (
            pred_count
            + target_count
            + smooth
        )

        scores.append(
            dice.item()
        )

    return scores


def evaluate_test_set(
    model,
    loader,
    device
):
    model.eval()

    all_cases = []

    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, masks in loader:
            images_device = images.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            logits = model(
                images_device
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            total_time += (
                time.perf_counter()
                - start_time
            )

            predictions = torch.argmax(
                logits,
                dim=1
            ).cpu()

            total_images += images.size(0)

            for i in range(
                images.size(0)
            ):
                dice_scores = dice_for_single_sample(
                    predictions[i],
                    masks[i]
                )

                mean_dice = sum(dice_scores[1:]) / len(dice_scores[1:])

                all_cases.append({
                    "image": images[i].clone(),
                    "mask": masks[i].clone(),
                    "prediction": predictions[i].clone(),
                    "dice": dice_scores,
                    "mean_dice": mean_dice
                })

    average_time = (
        total_time
        / total_images
    )

    return (
        all_cases,
        average_time
    )


def save_case(
    case,
    rank,
    model_name
):
    image = case["image"]
    ground_truth = case["mask"]
    prediction = case["prediction"]

    error_map = (
        prediction != ground_truth
    ).float()

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    axes[0].imshow(
        image[0],
        cmap="gray"
    )

    axes[0].set_title(
        "MRI"
    )

    axes[1].imshow(
        ground_truth,
        cmap="viridis",
        vmin=0,
        vmax=NUM_CLASSES - 1
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[2].imshow(
        prediction,
        cmap="viridis",
        vmin=0,
        vmax=NUM_CLASSES - 1
    )

    axes[2].set_title(
        "Prediction"
    )

    axes[3].imshow(
        error_map,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[3].set_title(
        "Error Map"
    )

    for axis in axes:
        axis.axis("off")

    title_text = (
        f"{model_name} failure case {rank} "
        f"- mean Dice: {case['mean_dice']:.4f}"
    )

    figure.suptitle(
        title_text
    )

    plt.tight_layout()

    file_name = (
        f"{model_name}_failure_case_{rank}.png"
    )

    plt.savefig(
        file_name,
        dpi=150
    )

    plt.close()

    print(
        f"Saved {file_name}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--model",
        choices=[
            "baseline",
            "improved"
        ],
        default="improved"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--num-failures",
        type=int,
        default=5
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device
    )

    (
        _,
        _,
        test_loader
    ) = create_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=8,
        num_workers=0
    )

    model = build_model(
        args.model
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        all_cases,
        average_time
    ) = evaluate_test_set(
        model,
        test_loader,
        device
    )

    all_cases.sort(
        key=lambda case: case["mean_dice"]
    )

    worst_cases = all_cases[
        :args.num_failures
    ]

    print(
        "\nWorst test cases"
    )

    for rank, case in enumerate(
        worst_cases,
        start=1
    ):
        print(
            f"\nCase {rank}"
        )

        print(
            f"Mean Dice: "
            f"{case['mean_dice']:.4f}"
        )

        print(
            "Per-class Dice:",
            [
                round(score, 4)
                for score in case["dice"]
            ]
        )

        save_case(
            case,
            rank,
            args.model
        )

    print(
        "\nAverage inference time per image:",
        f"{average_time * 1000:.3f} ms"
    )


if __name__ == "__main__":
    main()