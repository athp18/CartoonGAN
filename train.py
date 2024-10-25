import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import itertools
from tqdm import tqdm
from models import *
from optimizations import *
from images import *
from utils import *


def train(resume_from=None):
    # Setup device and reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Training hyperparameters
    config = {
        "batch_size": 4,
        "num_workers": 2,  # Adjusted for Colab
        "num_epochs": 200,
        "save_frequency": 5,  # Save every N epochs
        "sample_frequency": 500,  # Save samples every N batches
    }

    # Setup directories
    os.makedirs(SAVE_PATH, exist_ok=True)
    samples_dir = os.path.join(SAVE_PATH, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Create data transforms
    transform = transforms.Compose(
        [
            transforms.Resize((286, 286)),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    # Create dataloader
    try:
        dataset = ImageDataset(root="data1", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return

    # Initialize model
    if resume_from:
        print(f"Resuming training from checkpoint: {resume_from}")
        model = CycleGAN(device, checkpoint_path=os.path.join(SAVE_PATH, resume_from))
    else:
        print("Starting fresh training")
        model = CycleGAN(device)

    # Training metrics tracking
    best_loss = model.best_loss
    start_epoch = model.start_epoch
    training_history = {
        "generator_losses": [],
        "discriminator_losses": [],
        "identity_losses": [],
        "gan_losses": [],
        "cycle_losses": [],
    }

    print(f"\nStarting training from epoch {start_epoch + 1}")
    print(f"Total epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Training device: {device}")
    print("-" * 50)

    try:
        for epoch in range(start_epoch, config["num_epochs"]):
            epoch_metrics = {
                "loss_G": 0.0,
                "loss_D_A": 0.0,
                "loss_D_B": 0.0,
                "loss_identity": 0.0,
                "loss_GAN": 0.0,
                "loss_cycle": 0.0,
            }

            # Create progress bar
            with tqdm(
                dataloader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}'
            ) as pbar:
                for i, batch in enumerate(pbar):
                    try:
                        # Move data to device
                        real_A = batch["A"].to(device)
                        real_B = batch["B"].to(device)

                        # Training step
                        losses = model.train_step(real_A, real_B)

                        # Update metrics
                        for key in epoch_metrics:
                            epoch_metrics[key] += losses[key]

                        # Update progress bar
                        pbar.set_postfix(
                            {
                                "G": f"{losses['loss_G']:.4f}",
                                "D_A": f"{losses['loss_D_A']:.4f}",
                                "D_B": f"{losses['loss_D_B']:.4f}",
                            }
                        )

                        # Save sample images
                        if i % config["sample_frequency"] == 0:
                            with torch.no_grad():
                                fake_A, fake_B = model.generate_images(real_A, real_B)
                                save_comparison_grid(
                                    real_A,
                                    fake_B,
                                    real_B,
                                    fake_A,
                                    os.path.join(
                                        samples_dir,
                                        f"samples_epoch_{epoch+1}_batch_{i}.png",
                                    ),
                                )

                        # Periodic checkpoint saving
                        if i > 0 and i % 1000 == 0:
                            checkpoint_name = (
                                f"periodic_checkpoint_epoch_{epoch+1}_batch_{i}.pth"
                            )
                            model.save_checkpoint(
                                epoch + 1, epoch_metrics, checkpoint_name
                            )

                    except Exception as e:
                        print(f"\nError during batch processing: {e}")
                        # Save emergency checkpoint
                        emergency_name = (
                            f"emergency_checkpoint_epoch_{epoch+1}_batch_{i}.pth"
                        )
                        model.save_checkpoint(epoch + 1, epoch_metrics, emergency_name)
                        raise e

            # Calculate average epoch metrics
            num_batches = len(dataloader)
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

            # Update training history
            training_history["generator_losses"].append(epoch_metrics["loss_G"])
            training_history["discriminator_losses"].append(
                (epoch_metrics["loss_D_A"] + epoch_metrics["loss_D_B"]) / 2
            )
            training_history["identity_losses"].append(epoch_metrics["loss_identity"])
            training_history["gan_losses"].append(epoch_metrics["loss_GAN"])
            training_history["cycle_losses"].append(epoch_metrics["loss_cycle"])

            # Update learning rates
            model.scheduler_G.step()
            model.scheduler_D_A.step()
            model.scheduler_D_B.step()

            # Save best model
            total_loss = sum(epoch_metrics.values())
            if total_loss < best_loss:
                best_loss = total_loss
                model.best_loss = best_loss
                model.save_checkpoint(epoch + 1, epoch_metrics, "best_model.pth")

            # Regular epoch checkpoint
            if (epoch + 1) % config["save_frequency"] == 0:
                model.save_checkpoint(
                    epoch + 1, epoch_metrics, f"checkpoint_epoch_{epoch+1}.pth"
                )

            # Save training history
            history_path = os.path.join(SAVE_PATH, "training_history.pt")
            torch.save(training_history, history_path)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{config['num_epochs']} Summary:")
            print(f"Generator Loss: {epoch_metrics['loss_G']:.4f}")
            print(f"Discriminator A Loss: {epoch_metrics['loss_D_A']:.4f}")
            print(f"Discriminator B Loss: {epoch_metrics['loss_D_B']:.4f}")
            print(f"Identity Loss: {epoch_metrics['loss_identity']:.4f}")
            print(f"GAN Loss: {epoch_metrics['loss_GAN']:.4f}")
            print(f"Cycle Loss: {epoch_metrics['loss_cycle']:.4f}")
            print(f"Best Total Loss: {best_loss:.4f}")
            print("-" * 50)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save interrupt checkpoint
        model.save_checkpoint(epoch + 1, epoch_metrics, "interrupt_checkpoint.pth")
        print("Interrupt checkpoint saved")

    except Exception as e:
        print(f"\nError during training: {e}")
        # Save emergency checkpoint
        model.save_checkpoint(epoch + 1, epoch_metrics, "emergency_checkpoint.pth")
        raise e

    finally:
        # Save final model state
        model.save_checkpoint(epoch + 1, epoch_metrics, "final_model.pth")
        print("\nTraining completed")
        print(f"Final model saved to {SAVE_PATH}")


if __name__ == "main":
    # If you want to resume training, set this to your checkpoint filename
    # Example: resume_from = 'checkpoint_epoch_50.pth'
    # Set to None for fresh training
    resume_from = None

    try:
        print("Initializing CycleGAN training...")
        print(
            f"{'Starting fresh training' if resume_from is None else f'Resuming from {resume_from}'}"
        )
        print(f"Saving models to: {SAVE_PATH}")
        print("-" * 50)

        train(resume_from=resume_from)

    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Print the full traceback for debugging
        import traceback

        traceback.print_exc()
