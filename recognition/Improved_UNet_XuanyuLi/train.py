import argparse
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from dataset import create_dataloaders

from modules import (
    BasicUNet,
    ImprovedUNet,
    count_parameters
)


NUM_CLASSES = 4


def dice_loss(
    logits,
    targets,
    smooth=1e-6
):
    probabilities = torch.softmax(
        logits,
        dim=1
    )

    targets_one_hot = F.one_hot(
        targets,
        num_classes=NUM_CLASSES
    ).permute(
        0,
        3,
        1,
        2
    ).float()

    intersection = (
        probabilities
        * targets_one_hot
    ).sum(
        dim=(0, 2, 3)
    )

    total = (
        probabilities.sum(
            dim=(0, 2, 3)
        )
        + targets_one_hot.sum(
            dim=(0, 2, 3)
        )
    )

    dice = (
        2 * intersection
        + smooth
    ) / (
        total
        + smooth
    )

    return 1 - dice.mean()


def calculate_loss(
    logits,
    targets
):
    cross_entropy = F.cross_entropy(
        logits,
        targets
    )

    dice = dice_loss(
        logits,
        targets
    )

    return cross_entropy + dice


def calculate_metrics(
    predictions,
    targets,
    smooth=1e-6
):
    dice_scores = []
    iou_scores = []

    for class_id in range(NUM_CLASSES):
        predicted_class = (
            predictions == class_id
        )

        target_class = (
            targets == class_id
        )

        intersection = (
            predicted_class
            & target_class
        ).sum().float()

        predicted_pixels = (
            predicted_class
        ).sum().float()

        target_pixels = (
            target_class
        ).sum().float()

        union = (
            predicted_class
            | target_class
        ).sum().float()

        dice = (
            2 * intersection
            + smooth
        ) / (
            predicted_pixels
            + target_pixels
            + smooth
        )

        iou = (
            intersection
            + smooth
        ) / (
            union
            + smooth
        )

        dice_scores.append(
            dice.item()
        )

        iou_scores.append(
            iou.item()
        )

    return dice_scores, iou_scores


def evaluate(
    model,
    loader,
    device
):
    model.eval()

    total_loss = 0.0

    predictions_list = []
    targets_list = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(
                device
            )

            masks = masks.to(
                device
            )

            logits = model(images)

            loss = calculate_loss(
                logits,
                masks
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            predictions_list.append(
                predictions.cpu()
            )

            targets_list.append(
                masks.cpu()
            )

    predictions = torch.cat(
        predictions_list,
        dim=0
    )

    targets = torch.cat(
        targets_list,
        dim=0
    )

    dice_scores, iou_scores = calculate_metrics(
        predictions,
        targets
    )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    return (
        average_loss,
        dice_scores,
        iou_scores
    )


def measure_inference_time(
    model,
    loader,
    device
):
    model.eval()

    images, _ = next(
        iter(loader)
    )

    images = images.to(device)

    with torch.no_grad():
        for _ in range(3):
            model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = (
        time.perf_counter()
        - start_time
    )

    return (
        total_time
        / images.size(0)
    )


def save_training_plots(history):
    epoch_numbers = range(
        1,
        len(history["train_loss"]) + 1
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epoch_numbers,
        history["train_loss"],
        label="Training loss"
    )

    plt.plot(
        epoch_numbers,
        history["val_loss"],
        label="Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "training_loss.png",
        dpi=150
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        epoch_numbers,
        history["val_dice"]
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice")
    plt.title("Validation Dice")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "validation_dice.png",
        dpi=150
    )

    plt.close()


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


def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs,
    learning_rate,
    checkpoint_path
):
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_dice = 0.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": []
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        start_time = time.perf_counter()

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = calculate_loss(
                logits,
                masks
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        (
            val_loss,
            val_dice,
            val_iou
        ) = evaluate(
            model,
            val_loader,
            device
        )

        mean_val_dice = (
            sum(val_dice)
            / len(val_dice)
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_dice"].append(
            mean_val_dice
        )

        epoch_time = (
            time.perf_counter()
            - start_time
        )

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        print(
            f"Training loss: {train_loss:.4f}"
        )

        print(
            f"Validation loss: {val_loss:.4f}"
        )

        print(
            "Validation Dice:",
            [
                round(value, 4)
                for value in val_dice
            ]
        )

        print(
            "Validation IoU:",
            [
                round(value, 4)
                for value in val_iou
            ]
        )

        print(
            f"Mean validation Dice: "
            f"{mean_val_dice:.4f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_time:.1f} seconds"
        )

        if mean_val_dice > best_dice:
            best_dice = mean_val_dice

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "best_val_dice":
                        best_dice
                },
                checkpoint_path
            )

            print(
                "Saved best model."
            )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        test_loss,
        test_dice,
        test_iou
    ) = evaluate(
        model,
        test_loader,
        device
    )

    latency = measure_inference_time(
        model,
        test_loader,
        device
    )

    print(
        "\nTest results"
    )

    print(
        f"Test loss: {test_loss:.4f}"
    )

    for class_id in range(NUM_CLASSES):
        print(
            f"Class {class_id}: "
            f"Dice={test_dice[class_id]:.4f}, "
            f"IoU={test_iou[class_id]:.4f}"
        )

    print(
        "Mean Dice:",
        round(
            sum(test_dice)
            / NUM_CLASSES,
            4
        )
    )

    print(
        "Trainable parameters:",
        f"{count_parameters(model):,}"
    )

    print(
        "Inference time per image:",
        f"{latency * 1000:.3f} ms"
    )

    if device.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        print(
            "Peak GPU memory:",
            f"{peak_memory:.2f} MB"
        )

    save_training_plots(
        history
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
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
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
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
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = build_model(
        args.model
    ).to(device)

    checkpoint_path = (
        f"best_{args.model}_unet.pth"
    )

    print(
        "Trainable parameters:",
        f"{count_parameters(model):,}"
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_path=checkpoint_path
    )


if __name__ == "__main__":
    main()