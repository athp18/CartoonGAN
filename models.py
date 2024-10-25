import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import os
import itertools
from optimizations import ReplayBuffer

SAVE_PATH = os.path.join(os.getcwd(), "cycle_gan_models")
os.makedirs(SAVE_PATH, exist_ok=True)


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """Generator using pretrained ResNet50 as encoder."""

    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()

        # we use Resnet50 pretrained for faster inference
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            *list(resnet.children())[:-3]
        )  # Include up to Layer3

        # Freeze early layers for faster training
        for param in list(self.encoder.parameters())[:5]:  # Freeze layers up to Layer1
            param.requires_grad = False

        # Decoder with unsampling
        self.decoder = nn.Sequential(
            ResidualBlock(1024),
            ResidualBlock(1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, output_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


class Discriminator(nn.Module):
    """Memory-efficient PatchGAN discriminator using MobileNetV2 features."""

    def __init__(self, input_channels=3):
        super().__init__()

        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Use fewer layers to maintain spatial dimensions
        self.features = nn.Sequential(*list(mobilenet.features)[:7])

        # Freeze early layers
        for param in list(self.features.parameters())[:-2]:
            param.requires_grad = False

        # Add transition layers to handle channel dimensions
        self.transition = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        features = self.transition(features)
        return self.classifier(features)


class CycleGAN:
    """Optimized CycleGAN implementation."""

    def __init__(self, device, checkpoint_path=None):
        self.device = device

        # Initialize models
        self.G_AB = Generator().to(self.device)
        self.G_BA = Generator().to(self.device)
        self.D_A = Discriminator().to(self.device)
        self.D_B = Discriminator().to(self.device)

        # use an amp scaler
        if device == torch.device("cuda"):
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = torch.amp.GradScaler()

        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # optimizer
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=0.0002,
            betas=(0.5, 0.999),
        )
        self.optimizer_D_A = torch.optim.Adam(
            self.D_A.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D_B = torch.optim.Adam(
            self.D_B.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        # Learning rate schedulers
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_G, T_0=50, T_mult=2
        )
        self.scheduler_D_A = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_D_A, T_0=50, T_mult=2
        )
        self.scheduler_D_B = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_D_B, T_0=50, T_mult=2
        )

        self.fake_A_buffer = ReplayBuffer(max_size=50)
        self.fake_B_buffer = ReplayBuffer(max_size=50)

        # Training state
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.current_step = 0

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def save_checkpoint(self, epoch, losses, checkpoint_name):
        """Save complete training state to Google Drive."""
        checkpoint_path = os.path.join(SAVE_PATH, checkpoint_name)
        checkpoint = {
            "epoch": epoch,
            "step": self.current_step,
            "G_AB_state_dict": self.G_AB.state_dict(),
            "G_BA_state_dict": self.G_BA.state_dict(),
            "D_A_state_dict": self.D_A.state_dict(),
            "D_B_state_dict": self.D_B.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_A_state_dict": self.optimizer_D_A.state_dict(),
            "optimizer_D_B_state_dict": self.optimizer_D_B.state_dict(),
            "scheduler_G_state_dict": self.scheduler_G.state_dict(),
            "scheduler_D_A_state_dict": self.scheduler_D_A.state_dict(),
            "scheduler_D_B_state_dict": self.scheduler_D_B.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "losses": losses,
            "best_loss": self.best_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to Google Drive: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load complete training state."""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model states
            self.G_AB.load_state_dict(checkpoint["G_AB_state_dict"])
            self.G_BA.load_state_dict(checkpoint["G_BA_state_dict"])
            self.D_A.load_state_dict(checkpoint["D_A_state_dict"])
            self.D_B.load_state_dict(checkpoint["D_B_state_dict"])

            # Load optimizer states
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
            self.optimizer_D_A.load_state_dict(checkpoint["optimizer_D_A_state_dict"])
            self.optimizer_D_B.load_state_dict(checkpoint["optimizer_D_B_state_dict"])

            # Load scheduler states
            self.scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])
            self.scheduler_D_A.load_state_dict(checkpoint["scheduler_D_A_state_dict"])
            self.scheduler_D_B.load_state_dict(checkpoint["scheduler_D_B_state_dict"])

            # Load scaler state
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            # Load training state
            self.start_epoch = checkpoint["epoch"]
            self.current_step = checkpoint.get("step", 0)
            self.best_loss = checkpoint["best_loss"]

            print(f"Successfully loaded checkpoint:")
            print(f"Epoch: {self.start_epoch}")
            print(f"Step: {self.current_step}")
            print(f"Best Loss: {self.best_loss:.4f}")
            return True

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def train_step(self, real_A, real_B):
        """Perform a single training step."""
        # Set models to training mode
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()

        # Increment step counter
        self.current_step += 1

        # Train Generators
        self.optimizer_G.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            # Identity loss
            same_B = self.G_AB(real_B)
            loss_identity_B = self.criterion_identity(same_B, real_B) * 5.0

            same_A = self.G_BA(real_A)
            loss_identity_A = self.criterion_identity(same_A, real_A) * 5.0

            # GAN loss
            fake_B = self.G_AB(real_A)
            pred_fake_B = self.D_B(fake_B)
            target_real = torch.ones_like(pred_fake_B).to(self.device)
            loss_GAN_AB = self.criterion_GAN(pred_fake_B, target_real)

            fake_A = self.G_BA(real_B)
            pred_fake_A = self.D_A(fake_A)
            loss_GAN_BA = self.criterion_GAN(pred_fake_A, target_real)

            # Cycle loss
            recovered_A = self.G_BA(fake_B)
            loss_cycle_A = self.criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = self.G_AB(fake_A)
            loss_cycle_B = self.criterion_cycle(recovered_B, real_B) * 10.0

            # Total generator loss
            loss_G = (
                loss_identity_A
                + loss_identity_B
                + loss_GAN_AB
                + loss_GAN_BA
                + loss_cycle_A
                + loss_cycle_B
            )

        # Generator backward pass
        self.scaler.scale(loss_G).backward()
        self.scaler.unscale_(self.optimizer_G)
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            max_norm=1.0,
        )
        self.scaler.step(self.optimizer_G)

        # Train Discriminator A
        self.optimizer_D_A.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda"):
            pred_real = self.D_A(real_A)
            loss_D_real = self.criterion_GAN(
                pred_real, torch.ones_like(pred_real).to(self.device)
            )

            fake_A = self.fake_A_buffer.push_and_pop(fake_A)
            pred_fake = self.D_A(fake_A.detach())
            loss_D_fake = self.criterion_GAN(
                pred_fake, torch.zeros_like(pred_fake).to(self.device)
            )

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

        self.scaler.scale(loss_D_A).backward()
        self.scaler.unscale_(self.optimizer_D_A)
        torch.nn.utils.clip_grad_norm_(self.D_A.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer_D_A)

        # Train Discriminator B
        self.optimizer_D_B.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda"):
            pred_real = self.D_B(real_B)
            loss_D_real = self.criterion_GAN(
                pred_real, torch.ones_like(pred_real).to(self.device)
            )

            fake_B = self.fake_B_buffer.push_and_pop(fake_B)
            pred_fake = self.D_B(fake_B.detach())
            loss_D_fake = self.criterion_GAN(
                pred_fake, torch.zeros_like(pred_fake).to(self.device)
            )

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5

        self.scaler.scale(loss_D_B).backward()
        self.scaler.unscale_(self.optimizer_D_B)
        torch.nn.utils.clip_grad_norm_(self.D_B.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer_D_B)

        self.scaler.update()

        return {
            "loss_G": loss_G.item(),
            "loss_D_A": loss_D_A.item(),
            "loss_D_B": loss_D_B.item(),
            "loss_identity": (loss_identity_A + loss_identity_B).item(),
            "loss_GAN": (loss_GAN_AB + loss_GAN_BA).item(),
            "loss_cycle": (loss_cycle_A + loss_cycle_B).item(),
        }

    def generate_images(self, real_A, real_B):
        """Generate fake images for visualization."""
        self.G_AB.eval()
        self.G_BA.eval()

        with torch.no_grad():
            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)

        self.G_AB.train()
        self.G_BA.train()

        return fake_A, fake_B
