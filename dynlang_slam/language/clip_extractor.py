"""CLIP feature extraction for DynLang-SLAM.

Extracts 768-dim language features from images and text using OpenCLIP ViT-L/14.
Supports per-mask feature extraction for LangSplat-style language fields.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import open_clip


class CLIPExtractor:
    """Extracts CLIP features from images, masked regions, and text."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        self.device = device
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        except Exception as e:
            # HuggingFace download may fail (401/gated repo) — fall back to
            # downloading OpenAI's original weights from their public CDN
            print(f"open_clip HF download failed ({e}), trying OpenAI CDN...")
            self.model, _, self.preprocess = self._load_from_openai_cdn(model_name)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.feat_dim = 768  # ViT-L/14 output dimension

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """Encode a full image to a 768-dim CLIP feature.

        Args:
            image: (H, W, 3) uint8 numpy array [0, 255] or float [0, 1]

        Returns:
            (768,) normalized feature vector on device
        """
        pil_img = self._to_pil(image)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img_tensor)
        return F.normalize(feat.squeeze(0).float(), dim=-1)

    @torch.no_grad()
    def encode_masks(
        self,
        image: np.ndarray,
        masks: list[dict],
    ) -> list[torch.Tensor]:
        """Encode masked image regions to CLIP features.

        For each mask, crops the image to the mask's bounding box, zeros out
        pixels outside the mask, and encodes through CLIP.

        Args:
            image: (H, W, 3) uint8 numpy array [0, 255]
            masks: list of dicts with 'segmentation' (H, W bool) and 'bbox' [x, y, w, h]

        Returns:
            list of (768,) normalized feature vectors
        """
        if len(masks) == 0:
            return []

        # Batch process for efficiency
        batch = []
        for mask_info in masks:
            seg = mask_info["segmentation"]  # (H, W) bool
            bbox = mask_info["bbox"]  # [x, y, w, h]
            x, y, w, h = [int(v) for v in bbox]

            # Crop to bounding box
            crop = image[y:y+h, x:x+w].copy()

            # Fill non-mask pixels with image mean (avoids black-fill CLIP bias)
            mask_crop = seg[y:y+h, x:x+w]
            mean_color = image.mean(axis=(0, 1)).astype(np.uint8)
            crop[~mask_crop] = mean_color

            pil_crop = Image.fromarray(crop)
            img_tensor = self.preprocess(pil_crop)
            batch.append(img_tensor)

        # Batch encode
        batch_tensor = torch.stack(batch).to(self.device)
        feats = self.model.encode_image(batch_tensor)
        feats = F.normalize(feats.float(), dim=-1)

        return [feats[i] for i in range(len(feats))]

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a text query to a 768-dim CLIP feature.

        Args:
            text: text string (e.g., "a chair")

        Returns:
            (768,) normalized feature vector on device
        """
        tokens = self.tokenizer([text]).to(self.device)
        feat = self.model.encode_text(tokens)
        return F.normalize(feat.squeeze(0).float(), dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode multiple text queries.

        Args:
            texts: list of text strings

        Returns:
            (N, 768) normalized feature matrix on device
        """
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1)

    @staticmethod
    def _load_from_openai_cdn(model_name: str):
        """Download CLIP weights from OpenAI's public CDN (no auth needed)."""
        import os
        import urllib.request

        _OPENAI_URLS = {
            "ViT-L-14": "https://openaipublic.azureedge.net/clip/models/"
                        "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            "ViT-B-32": "https://openaipublic.azureedge.net/clip/models/"
                        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58ber7bc90/ViT-B-32.pt",
            "ViT-B-16": "https://openaipublic.azureedge.net/clip/models/"
                        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        }
        url = _OPENAI_URLS.get(model_name)
        if url is None:
            raise ValueError(f"No OpenAI CDN URL for {model_name}. Available: {list(_OPENAI_URLS.keys())}")

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "clip")
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.join(cache_dir, f"{model_name}.pt")

        if not os.path.exists(filename):
            print(f"  Downloading {model_name} from OpenAI CDN...")
            urllib.request.urlretrieve(url, filename)
            print(f"  Saved to {filename}")

        # open_clip can load from a local .pt file path directly
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=filename
        )
        return model, None, preprocess

    def _to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image)
