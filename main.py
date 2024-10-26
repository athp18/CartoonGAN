import cv2
from PIL import Image
import argparse
import os
from inference import CycleGANInference
from models import SAVE_PATH
import torch

os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"


def main(
    checkpoint_path=None,
    input_path=None,
    output_path=None,
    mode="image",
    direction="AtoB",
    source_id=0,
):
    # pipeline initialization
    if torch.cuda.is_available():
        pipeline = CycleGANInference(checkpoint_path=checkpoint_path, device="cuda")
    else:
        pipeline = CycleGANInference(checkpoint_path=checkpoint_path)

    if mode == "image":
        # validate image
        assert os.path.exists(
            input_path
        ), f"Input image path does not exist: {input_path}"

        # Load and translate single image
        input_img = Image.open(input_path).convert("RGB")
        pipeline.save_translation(input_img, output_path, direction=direction)
        print(f"Translated image saved to: {output_path}")

    elif mode == "camera":
        # Start live inference
        pipeline.live_inference(
            direction=direction, source="camera", source_id=source_id
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN Inference")

    # required args
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "camera"],
        required=True,
        help='Inference mode: "image" for single image translation or "camera" for live webcam inference',
    )

    parser.add_argument(
        "--direction",
        type=str,
        choices=["AtoB", "BtoA"],
        default="AtoB",
        help="Translation direction (default: AtoB)",
    )

    # optional args
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to the model checkpoint"
    )

    # image mode args
    parser.add_argument(
        "--input_path", type=str, help="Path to input image (required for image mode)"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save translated image (required for image mode)",
    )

    # Arguments for camera mode
    parser.add_argument(
        "--source_id",
        type=int,
        default=0,
        help="Camera device ID for live inference (default: 0)",
    )

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == "image":
        if args.input_path is None or args.output_path is None:
            parser.error("--input_path and --output_path are required for image mode")

    main(
        checkpoint_path=args.checkpoint_path,
        input_path=args.input_path,
        output_path=args.output_path,
        mode=args.mode,
        direction=args.direction,
        source_id=args.source_id,
    )
