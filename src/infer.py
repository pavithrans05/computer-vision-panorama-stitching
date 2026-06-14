import os
import json
import torch
import cv2
import numpy as np

from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

from model import GCPModel


# ==========================================================
# CONFIGURATION
# ==========================================================

TEST_DIR = "../test_dataset"

# Use best REG=5 loss model
WEIGHTS_PATH = "../weights/best_loss_reg5.pth"

OUTPUT_JSON_PATH = "../predictions.json"

TARGET_SIZE = (512, 512)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Running inference on device:", device)


# ==========================================================
# LABEL MAPPING
# ==========================================================

idx_to_shape = {
    0: "Cross",
    1: "Square",
    2: "L-Shape"
}


# ==========================================================
# TRANSFORMS
# ==========================================================

# IMPORTANT:
# This matches your current dataset.py
# (image / 255.0 without ImageNet normalization)

base_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor()
])


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():

    model = GCPModel().to(device)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {WEIGHTS_PATH}"
        )

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location=device
    )

    # New checkpoint dictionary format
    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            f"Loaded checkpoint "
            f"(epoch {checkpoint['epoch'] + 1})"
        )

        if "f1" in checkpoint:
            print(
                f"Checkpoint F1: "
                f"{checkpoint['f1']:.4f}"
            )

    # Backward compatibility
    else:

        model.load_state_dict(checkpoint)

        print("Loaded legacy state_dict")

    model.eval()

    return model


# ==========================================================
# FIND TEST IMAGES
# ==========================================================

def get_test_files():

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    test_files = []

    for root, _, files in os.walk(TEST_DIR):

        for file in files:

            if file.lower().endswith(
                valid_extensions
            ):

                full_path = os.path.join(
                    root,
                    file
                )

                rel_path = os.path.relpath(
                    full_path,
                    TEST_DIR
                ).replace("\\", "/")

                test_files.append(
                    (
                        full_path,
                        rel_path
                    )
                )

    # deterministic order
    test_files.sort(
        key=lambda x: x[1]
    )

    return test_files


# ==========================================================
# SINGLE IMAGE INFERENCE
# ==========================================================

def predict_image(
    model,
    image_path
):

    img_pil = Image.open(
        image_path
    ).convert("RGB")

    orig_w, orig_h = img_pil.size

    img_tensor = base_transform(
        img_pil
    )

    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        coords_pred, logits = model(
            img_tensor
        )

    norm_x, norm_y = (
        coords_pred[0]
        .cpu()
        .numpy()
    )

    # Safety clamp
    norm_x = np.clip(
        norm_x,
        0.0,
        1.0
    )

    norm_y = np.clip(
        norm_y,
        0.0,
        1.0
    )

    # Convert back to original image coordinates
    x = float(norm_x * orig_w)
    y = float(norm_y * orig_h)

    pred_class = torch.argmax(
        logits,
        dim=1
    ).item()

    shape = idx_to_shape[
        pred_class
    ]

    return x, y, shape


# ==========================================================
# MAIN
# ==========================================================

def main():

    model = load_model()

    test_files = get_test_files()

    print(
        f"Found "
        f"{len(test_files)} "
        f"test images."
    )

    predictions = {}

    for full_path, rel_path in tqdm(
        test_files,
        desc="Inference"
    ):

        try:

            x, y, shape = predict_image(
                model,
                full_path
            )

            predictions[rel_path] = {
                "mark": {
                    "x": x,
                    "y": y
                },
                "verified_shape": shape
            }

        except Exception as e:

            print(
                f"Error processing "
                f"{rel_path}: {e}"
            )

    with open(
        OUTPUT_JSON_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            predictions,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nSaved predictions to:"
    )

    print(
        OUTPUT_JSON_PATH
    )

    print(
        f"Total predictions: "
        f"{len(predictions)}"
    )


if __name__ == "__main__":
    main()