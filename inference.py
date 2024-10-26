import torch
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
from models import Generator, SAVE_PATH


class CycleGANInference:
    """
    A pipeline for live inference using the trained CycleGAN model.

    This class handles:
        - Loading trained generators from checkpoints
        - Preprocessing input images
        - Performing image-to-image translation
        - Postprocessing and saving/displaying output images
        - Live video stream processing

    Example Usage:
        pipeline = CycleGANInference(checkpoint_path='path_to_checkpoint.pth', device='cuda')
        output_image = pipeline.translate_image(input_image)
        pipeline.live_inference(direction='AtoB', source='camera', source_id=0)

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        device (str): Device to perform inference on ('cuda' or 'cpu').
        image_size (tuple, optional): Desired input image size. Defaults to (256, 256).
    """

    def __init__(self, checkpoint_path=None, device="cuda", image_size=(256, 256)):
        assert os.path.exists(
            os.path.abspath(checkpoint_path)
        ), "The provided checkpoint path does not exist"

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self.inverse_transform = transforms.Compose(
            [
                transforms.Normalize(mean=(0, 0, 0), std=(2, 2, 2)),
                transforms.Normalize(mean=(-0.5, -0.5, -0.5), std=(1, 1, 1)),
                transforms.ToPILImage(),
            ]
        )
        self.checkpoint_path = (
            os.path.join(SAVE_PATH, "best_model.pth")
            if checkpoint_path is None
            else checkpoint_path
        )
        self.generator_AB, self.generator_BA = self.load_models(checkpoint_path)

    def load_models(self, checkpoint_path):
        """
        Load the generator models from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            tuple: (generator_AB, generator_BA) loaded models.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Initialize generators
        generator_AB = Generator().to(self.device)
        generator_BA = Generator().to(self.device)

        # Load state dicts
        generator_AB.load_state_dict(checkpoint["G_AB_state_dict"])
        generator_BA.load_state_dict(checkpoint["G_BA_state_dict"])

        # Set to evaluation mode
        generator_AB.eval()
        generator_BA.eval()

        print(f"Models loaded successfully from {checkpoint_path}")

        return generator_AB, generator_BA

    def preprocess(self, image):
        """
        Preprocess the input image for the generator.

        Args:
            image (PIL.Image or np.ndarray): Input image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL.Image or a numpy.ndarray.")

        return self.transform(image).unsqueeze(0).to(self.device)

    def postprocess(self, tensor):
        """
        Postprocess the output tensor to an image.

        Args:
            tensor (torch.Tensor): Output tensor from the generator.

        Returns:
            PIL.Image: Postprocessed image.
        """
        tensor = tensor.squeeze(0).cpu()
        image = self.inverse_transform(tensor)
        return image

    def translate_image(self, image, direction="AtoB"):
        """
        Translate an image from one domain to another.

        Args:
            image (PIL.Image or np.ndarray): Input image.
            direction (str, optional): 'AtoB' or 'BtoA'. Defaults to 'AtoB'.

        Returns:
            PIL.Image: Translated image.
        """
        preprocessed = self.preprocess(image)
        with torch.no_grad():
            if direction == "AtoB":
                fake_B = self.generator_AB(preprocessed)
                output_image = self.postprocess(fake_B)
            elif direction == "BtoA":
                fake_A = self.generator_BA(preprocessed)
                output_image = self.postprocess(fake_A)
            else:
                raise ValueError("direction must be 'AtoB' or 'BtoA'")
        return output_image

    def live_inference(self, direction="AtoB", source="camera", source_id=0):
        """
        Perform live inference from a camera or video source.

        Args:
            direction (str, optional): 'AtoB' or 'BtoA'. Defaults to 'AtoB'.
            source (str, optional): 'camera' or path to video file. Defaults to 'camera'.
            source_id (int, optional): Camera device ID. Defaults to 0.

        This function captures live video from the specified source, performs translation,
        and displays the result in real-time.
        """
        if source == "camera":
            cap = cv2.VideoCapture(source_id)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("Error: Unable to open video source.")
            return

        print("Starting live inference. Press 'q' to quit.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Translate image
                translated_image = self.translate_image(frame_rgb, direction=direction)

                # Convert translated image to BGR for OpenCV
                translated_bgr = cv2.cvtColor(
                    np.array(translated_image), cv2.COLOR_RGB2BGR
                )

                # Concatenate original and translated images
                combined = np.hstack((frame, translated_bgr))

                # Display the combined image
                cv2.imshow(
                    "Live Inference - Original (Left) | Translated (Right)", combined
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Exiting live inference.")
                    break
        except KeyboardInterrupt:
            print("Live inference interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Video capture released and windows closed.")

    def save_translation(self, input_image, output_path, direction="AtoB"):
        """
        Translate an image and save the output.

        Args:
            input_image (PIL.Image or np.ndarray): Input image.
            output_path (str): Path to save the translated image.
            direction (str, optional): 'AtoB' or 'BtoA'. Defaults to 'AtoB'.
        """
        translated_image = self.translate_image(input_image, direction=direction)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        translated_image.save(output_path)
        print(f"Translated image saved to {output_path}")
