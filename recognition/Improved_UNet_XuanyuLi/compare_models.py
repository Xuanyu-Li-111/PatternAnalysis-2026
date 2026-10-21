import argparse

import torch
import matplotlib.pyplot as plt

from dataset import create_dataloaders
from modules import BasicUNet, ImprovedUNet


NUM_CLASSES = 6


def load_model(
    model,
    checkpoint_path,
    device
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


def dice_for_sample(
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


def compare_models(
    baseline,
    improved,
    loader,
    device
):
    results = []

    with torch.no_grad():
        for images, masks in loader:
            images_device = images.to(device)

            baseline_logits = baseline(
                images_device
            )

            improved_logits = improved(
                images_device
            )

            baseline_predictions = torch.argmax(
                baseline_logits,
                dim=1
            ).cpu()

            improved_predictions = torch.argmax(
                improved_logits,
                dim=1
            ).cpu()

            for i in range(
                images.size(0)
            ):
                baseline_dice = dice_for_sample(
                    baseline_predictions[i],
                    masks[i]
                )

                improved_dice = dice_for_sample(
                    improved_predictions[i],
                    masks[i]
                )

                baseline_foreground = (
                    sum(baseline_dice[1:])
                    / (NUM_CLASSES - 1)
                )

                improved_foreground = (
                    sum(improved_dice[1:])
                    / (NUM_CLASSES - 1)
                )

                difference = (
                    improved_foreground
                    - baseline_foreground
                )

                results.append({
                    "image":
                        images[i].clone(),

                    "mask":
                        masks[i].clone(),

                    "baseline_prediction":
                        baseline_predictions[i].clone(),

                    "improved_prediction":
                        improved_predictions[i].clone(),

                    "baseline_dice":
                        baseline_dice,

                    "improved_dice":
                        improved_dice,

                    "baseline_mean":
                        baseline_foreground,

                    "improved_mean":
                        improved_foreground,

                    "difference":
                        difference
                })

    return results


def save_comparison(
    result,
    file_name,
    title
):
    image = result["image"]
    mask = result["mask"]

    baseline_prediction = (
        result["baseline_prediction"]
    )

    improved_prediction = (
        result["improved_prediction"]
    )

    baseline_error = (
        baseline_prediction != mask
    ).float()

    improved_error = (
        improved_prediction != mask
    ).float()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8)
    )

    axes[0, 0].imshow(
        image[0],
        cmap="gray"
    )

    axes[0, 0].set_title(
        "MRI"
    )

    axes[0, 1].imshow(
        mask,
        cmap="viridis",
        vmin=0,
        vmax=NUM_CLASSES - 1
    )

    axes[0, 1].set_title(
        "Ground Truth"
    )

    axes[0, 2].axis("off")

    axes[1, 0].imshow(
        baseline_prediction,
        cmap="viridis",
        vmin=0,
        vmax=NUM_CLASSES - 1
    )

    axes[1, 0].set_title(
        f"Basic U-Net\n"
        f"Dice={result['baseline_mean']:.4f}"
    )

    axes[1, 1].imshow(
        improved_prediction,
        cmap="viridis",
        vmin=0,
        vmax=NUM_CLASSES - 1
    )

    axes[1, 1].set_title(
        f"Improved U-Net\n"
        f"Dice={result['improved_mean']:.4f}"
    )

    difference_map = (
        improved_error
        - baseline_error
    )

    axes[1, 2].imshow(
        difference_map,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    axes[1, 2].set_title(
        "Error Difference"
    )

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle(
        title
    )

    plt.tight_layout()

    plt.savefig(
        file_name,
        dpi=150
    )

    plt.close()


def print_result(
    name,
    result
):
    print(
        f"\n{name}"
    )

    print(
        "Basic foreground Dice:",
        f"{result['baseline_mean']:.4f}"
    )

    print(
        "Improved foreground Dice:",
        f"{result['improved_mean']:.4f}"
    )

    print(
        "Difference:",
        f"{result['difference']:+.4f}"
    )

    print(
        "Basic per-class:",
        [
            round(value, 4)
            for value
            in result["baseline_dice"]
        ]
    )

    print(
        "Improved per-class:",
        [
            round(value, 4)
            for value
            in result["improved_dice"]
        ]
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        required=True
    )

    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--improved-checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--num-cases",
        type=int,
        default=3
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
        image_size=128,
        batch_size=8,
        num_workers=0
    )

    baseline = load_model(
        BasicUNet(),
        args.baseline_checkpoint,
        device
    )

    improved = load_model(
        ImprovedUNet(),
        args.improved_checkpoint,
        device
    )

    results = compare_models(
        baseline,
        improved,
        test_loader,
        device
    )

    results.sort(
        key=lambda result:
            result["difference"]
    )

    worse_cases = results[
        :args.num_cases
    ]

    better_cases = results[
        -args.num_cases:
    ][::-1]

    print(
        "\nCases where Improved U-Net is worse"
    )

    for index, result in enumerate(
        worse_cases,
        start=1
    ):
        print_result(
            f"Worse case {index}",
            result
        )

        save_comparison(
            result,
            f"paired_worse_{index}.png",
            (
                f"Improved U-Net worse case {index} "
                f"(delta={result['difference']:+.4f})"
            )
        )

    print(
        "\nCases where Improved U-Net is better"
    )

    for index, result in enumerate(
        better_cases,
        start=1
    ):
        print_result(
            f"Better case {index}",
            result
        )

        save_comparison(
            result,
            f"paired_better_{index}.png",
            (
                f"Improved U-Net better case {index} "
                f"(delta={result['difference']:+.4f})"
            )
        )

    improved_count = sum(
        result["difference"] > 0
        for result in results
    )

    worse_count = sum(
        result["difference"] < 0
        for result in results
    )

    same_count = (
        len(results)
        - improved_count
        - worse_count
    )

    average_difference = (
        sum(
            result["difference"]
            for result in results
        )
        / len(results)
    )

    print(
        "\nOverall paired comparison"
    )

    print(
        "Test slices:",
        len(results)
    )

    print(
        "Improved better:",
        improved_count
    )

    print(
        "Improved worse:",
        worse_count
    )

    print(
        "Approximately equal:",
        same_count
    )

    print(
        "Average foreground Dice difference:",
        f"{average_difference:+.6f}"
    )


if __name__ == "__main__":
    main()