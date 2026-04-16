"""Test the language pipeline on a single Replica frame.

Runs SAM mask generation at 3 scales, CLIP feature extraction per mask,
and a text query similarity test. Saves visualizations to results/language_test/.
"""

import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
from PIL import Image

# ---- Config ----
FRAME_PATH = os.path.join(PROJECT_ROOT, "data", "Replica", "room0", "results", "frame000100.jpg")
SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")
SAM_MODEL = "sam2.1_hiera_t"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "language_test")
TEXT_QUERIES = ["chair", "table", "floor", "wall", "plant", "sofa"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DynLang-SLAM Language Pipeline Test")
print("=" * 60)

# ---- Load image ----
print(f"\nLoading image: {FRAME_PATH}")
image = np.array(Image.open(FRAME_PATH).convert("RGB"))
H, W = image.shape[:2]
print(f"  Image size: {W}x{H}")
Image.fromarray(image).save(os.path.join(OUTPUT_DIR, "input.jpg"))

# ---- Step 1: CLIP extractor ----
print("\n--- Step 1: Loading CLIP ViT-L/14 ---")
t0 = time.time()
from dynlang_slam.language import CLIPExtractor
clip = CLIPExtractor(model_name="ViT-L-14", pretrained="openai", device="cuda")
print(f"  CLIP loaded in {time.time()-t0:.1f}s (feat_dim={clip.feat_dim})")

# Test global image encoding
t0 = time.time()
global_feat = clip.encode_image(image)
print(f"  Global image encode: {time.time()-t0:.3f}s -> shape {global_feat.shape}")

# Test text encoding
t0 = time.time()
text_feats = clip.encode_texts(TEXT_QUERIES)
print(f"  Text encode ({len(TEXT_QUERIES)} queries): {time.time()-t0:.3f}s -> shape {text_feats.shape}")

# ---- Step 2: SAM mask generation ----
print("\n--- Step 2: Loading SAM2.1 Hiera-T ---")
t0 = time.time()
from dynlang_slam.language import SAMExtractor
sam = SAMExtractor(
    model_type=SAM_MODEL,
    checkpoint_path=SAM_CHECKPOINT,
    device="cuda",
)
print(f"  SAM2 loaded in {time.time()-t0:.1f}s")

print("\nGenerating masks at 3 scales...")
for scale_name in ["whole", "part", "subpart"]:
    t0 = time.time()
    masks = sam.generate_masks_single_scale(image, scale=scale_name)
    dt = time.time() - t0
    print(f"  {scale_name}: {len(masks)} masks in {dt:.1f}s")

    # Visualize
    vis = SAMExtractor.visualize_masks(image, masks)
    Image.fromarray(vis).save(os.path.join(OUTPUT_DIR, f"masks_{scale_name}.jpg"))

# ---- Step 3: Full feature map pipeline ----
print("\n--- Step 3: Building pixel-aligned feature maps ---")
from dynlang_slam.language import FeatureMapBuilder
builder = FeatureMapBuilder(clip, sam, device="cuda")

t0 = time.time()
feature_maps = builder.build_feature_maps(image)
total_time = time.time() - t0
print(f"  Full pipeline: {total_time:.1f}s")
for scale_name, fmap in feature_maps.items():
    print(f"    {scale_name}: shape {fmap.shape}, norm range [{fmap.norm(dim=-1).min():.3f}, {fmap.norm(dim=-1).max():.3f}]")

# ---- Step 4: Text query heatmaps (raw cosine + relevancy) ----
print("\n--- Step 4: Text query heatmaps ---")

# Average all 3 scales for richer features (memory-efficient: accumulate then free)
import torch
avg_feat_map = feature_maps["whole"].clone()
avg_feat_map += feature_maps["part"]
avg_feat_map += feature_maps["subpart"]
avg_feat_map /= 3.0
# Free individual maps to save GPU memory
del feature_maps
torch.cuda.empty_cache()
avg_feat_map = torch.nn.functional.normalize(avg_feat_map, dim=-1)

for query in TEXT_QUERIES:
    query_feat = clip.encode_text(query)

    # Raw cosine similarity
    sim_map = FeatureMapBuilder.compute_similarity(avg_feat_map, query_feat)
    sim_np = sim_map.cpu().numpy()

    # Relevancy scoring (LangSplat contrastive)
    rel_map = FeatureMapBuilder.compute_relevancy(avg_feat_map, query_feat, clip)
    rel_np = rel_map.cpu().numpy()

    # Save raw cosine heatmap
    heatmap_raw = FeatureMapBuilder.similarity_to_heatmap(sim_map)
    blend_raw = (image.astype(np.float32) * 0.4 + heatmap_raw.astype(np.float32) * 0.6)
    Image.fromarray(blend_raw.clip(0, 255).astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR, f"query_{query}_raw.jpg"))

    # Save relevancy heatmap (sharper)
    heatmap_rel = FeatureMapBuilder.similarity_to_heatmap(rel_map)
    blend_rel = (image.astype(np.float32) * 0.4 + heatmap_rel.astype(np.float32) * 0.6)
    Image.fromarray(blend_rel.clip(0, 255).astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR, f"query_{query}_relevancy.jpg"))

    print(f"  '{query}': raw=[{sim_np.min():.3f}, {sim_np.max():.3f}]  "
          f"relevancy=[{rel_np.min():.3f}, {rel_np.max():.3f}]")

print(f"\n{'='*60}")
print(f"Results saved to {OUTPUT_DIR}/")
print(f"Total pipeline time per frame: {total_time:.1f}s")
print(f"{'='*60}")
