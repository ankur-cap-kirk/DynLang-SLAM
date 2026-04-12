"""Autoencoder for compressing CLIP features (768-dim -> 16-dim).

Trained online during SLAM to compress high-dimensional CLIP features into
compact latent codes that can be stored per-Gaussian and rendered via splatting.
Follows the LangSplat approach but adapted for online/incremental training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageAutoencoder(nn.Module):
    """MLP autoencoder: CLIP 768-dim -> 16-dim latent -> 768-dim reconstruction.

    Architecture:
        Encoder: 768 -> 256 -> 128 -> 16
        Decoder: 16 -> 128 -> 256 -> 768

    Loss: L1 reconstruction + cosine distance (preserves semantic similarity).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        latent_dim: int = 16,
        lr: float = 0.001,
        device: str = "cuda",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        # Encoder: 768 -> 256 -> 128 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, latent_dim),
        ).to(device)

        # Decoder: 16 -> 128 -> 256 -> 768
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        ).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Pre-allocated tensor buffer (avoids repeated torch.stack on deque)
        self._buffer_maxlen = 100000
        self._buffer = torch.zeros(self._buffer_maxlen, input_dim)  # CPU
        self._buffer_ptr = 0  # write pointer (circular)
        self._buffer_count = 0  # actual number of valid entries
        self._is_frozen = False
        self._train_steps = 0

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Encode CLIP features to latent space.

        Args:
            features: (..., 768) CLIP features

        Returns:
            (..., 16) latent codes
        """
        return self.encoder(features)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent codes back to CLIP space.

        Args:
            latent: (..., 16) latent codes

        Returns:
            (..., 768) reconstructed features
        """
        return self.decoder(latent)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full encode-decode pass.

        Returns:
            (latent, reconstructed) tuple
        """
        latent = self.encode(features)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    def compute_loss(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute autoencoder loss: L1 + cosine distance.

        Args:
            features: (N, 768) normalized CLIP features

        Returns:
            (loss, loss_dict) tuple
        """
        latent, reconstructed = self.forward(features)

        # L1 reconstruction loss
        loss_l1 = F.l1_loss(reconstructed, features)

        # Cosine distance loss (1 - cosine_similarity)
        cos_sim = F.cosine_similarity(reconstructed, features, dim=-1)
        loss_cos = (1.0 - cos_sim).mean()

        # Combined loss (equal weight, following LangSplat)
        loss = loss_l1 + loss_cos

        loss_dict = {
            "l1": loss_l1.item(),
            "cosine": loss_cos.item(),
            "total": loss.item(),
        }
        return loss, loss_dict

    def add_features(self, features: torch.Tensor) -> None:
        """Add CLIP features to the training buffer.

        Args:
            features: (N, 768) CLIP features to store for training
        """
        feats_cpu = features.detach().cpu()
        n = feats_cpu.shape[0]

        # Write into pre-allocated circular buffer (no allocation)
        for i in range(n):
            self._buffer[self._buffer_ptr] = feats_cpu[i]
            self._buffer_ptr = (self._buffer_ptr + 1) % self._buffer_maxlen
            self._buffer_count = min(self._buffer_count + 1, self._buffer_maxlen)

    def train_step(self, batch_size: int = 512, num_steps: int = 10) -> dict:
        """Run training steps on buffered features.

        Args:
            batch_size: samples per training step
            num_steps: number of gradient steps

        Returns:
            dict with average loss info
        """
        if self._is_frozen or self._buffer_count < batch_size:
            return {"total": 0.0, "l1": 0.0, "cosine": 0.0, "steps": 0}

        self.train()
        total_loss = {"l1": 0.0, "cosine": 0.0, "total": 0.0}

        # Slice the valid portion of the pre-allocated buffer (no copy)
        buffer_tensor = self._buffer[:self._buffer_count]

        for _ in range(num_steps):
            # Random sample from buffer
            indices = torch.randint(0, len(buffer_tensor), (batch_size,))
            batch = buffer_tensor[indices].to(self.device)

            self.optimizer.zero_grad()
            loss, loss_dict = self.compute_loss(batch)
            loss.backward()
            self.optimizer.step()

            for k in total_loss:
                total_loss[k] += loss_dict[k]

            self._train_steps += 1

        self.eval()

        avg_loss = {k: v / num_steps for k, v in total_loss.items()}
        avg_loss["steps"] = self._train_steps
        avg_loss["buffer_size"] = self._buffer_count
        return avg_loss

    def freeze(self) -> None:
        """Freeze the autoencoder (stop training, inference only)."""
        self._is_frozen = True
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze the autoencoder (resume training)."""
        self._is_frozen = False
        for param in self.parameters():
            param.requires_grad_(True)

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    @property
    def buffer_size(self) -> int:
        return self._buffer_count
