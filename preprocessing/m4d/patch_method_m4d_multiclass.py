"""
[File] patch_method_m4d_multiclass.py
[Purpose] Adaptive Image Patching for M4D Multi-class Dataset.
[Description]
    - Specialized for M4D dataset with 5 classes: 
      0: Sea Surface, 1: Oil Spill, 2: Look-alike, 3: Ship, 4: Land
    - Adaptive Strategy:
      - Patches containing "Oil Spill" (Class 1) -> Dense Stride (Overlap).
      - Patches NOT containing "Oil Spill" -> Sparse Stride (No Overlap).
    - Supports Train/Val/Test splitting.
[Author] Assistant (GitHub Copilot)
"""
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import pandas as pd
from datetime import datetime
import concurrent.futures 

# PIL settings for large images
Image.MAX_IMAGE_PIXELS = None

# --- Configuration ---
# CPU Cores (None = all available)
NUM_WORKERS = None

# Paths
SOURCE_BASE_DIR = "/home/yuan/Yuan/OIL_Project_12_7/dataset/M4D" 
# Example output path - User can change this
OUTPUT_BASE_DIR = "/home/yuan/Yuan/OIL_Project_12_7/dataset/M4D_Patched_Adaptive"

# The M4D dataset structure based on pad_validation_m4d_v2.py is:
# dataset/M4D/train/images
# dataset/M4D/train/labels_1D
CATEGORIES = ["."] 
SPLITS = ["train", "test"] 

# Patch Settings
# M4D images are often around 1250x650. 
# We use small patches (e.g. 512) for training to increase sample count.
PATCH_SIZE = 512

# Overlap Settings
# 1. Train: Oil (Class 1) gets overlap. Others do not.
OVERLAP_TRAIN_OIL = 256  # stride = 256
OVERLAP_TRAIN_OTHER = 0  # stride = 512 (No overlap)

# 2. Test: Standard sliding window for reconstruction
OVERLAP_TEST = 256       

RANDOM_SEED = 42

# Only for background (Class 0 only patches), drop some to balance?
# Set to 1.0 to keep all.
BACKGROUND_KEEP_RATIO = 1.0 

# Output directory structure control
SEPARATE_OUTPUT = False # For multiclass, usually we don't separate pos/neg folders like binary.

# M4D Class Definitions
# 0: Sea, 1: Oil, 2: Look-alike, 3: Ship, 4: Land
OIL_CLASS_ID = 1

# ===================================================================
# === Core Functions
# ===================================================================

def check_contains_oil(mask_segment_pil, oil_class_id):
    """Check if the mask patch contains the specific Oil class ID."""
    segment_array = np.array(mask_segment_pil)
    # Check if any pixel matches the oil class ID
    return np.any(segment_array == oil_class_id)

def slice_image_standard_generator(image_pil, mask_pil, patch_size, overlap):
    """
    Standard generator for Test set (Uniform sliding window).
    """
    image_width, image_height = image_pil.size
    pad_color_image = (0, 0, 0) # Black padding for RGB
    pad_value_mask = 0          # Class 0 (Sea) padding for Mask

    stride = patch_size - overlap
    
    # Ensure stride is at least 1
    stride = max(1, stride)
    
    # Calculate grid
    x_steps = (image_width - patch_size) // stride + 1 if image_width > patch_size else 1
    y_steps = (image_height - patch_size) // stride + 1 if image_height > patch_size else 1
    
    # If image is smaller than patch, we just crop once (0,0) with padding handled below?
    # Actually loops handle coordinate generation.
    
    # Use standard loop that ensures coverage
    for y in range(0, image_height, stride):
        for x in range(0, image_width, stride):
            # Adjust if we want strict sliding window or allow padding at edges
            # Here we follow the logic: Crop -> Pad if necessary
            
            # Simple check to avoid creating tiny slivers if not desired, 
            # but usually we want full coverage.
            
            x_end = min(x + patch_size, image_width)
            y_end = min(y + patch_size, image_height)
            
            crop_box = (x, y, x_end, y_end)
            patch_img = image_pil.crop(crop_box)
            patch_mask = mask_pil.crop(crop_box)
            
            # Pad if smaller than desired patch size
            if patch_img.size != (patch_size, patch_size):
                padded_img = Image.new(image_pil.mode, (patch_size, patch_size), pad_color_image)
                padded_seg = Image.new(mask_pil.mode, (patch_size, patch_size), pad_value_mask)
                padded_img.paste(patch_img, (0, 0))
                padded_seg.paste(patch_mask, (0, 0))
                patch_img = padded_img
                patch_mask = padded_seg
            
            has_oil = check_contains_oil(patch_mask, OIL_CLASS_ID)
            
            yield patch_img, patch_mask, x, y, has_oil

def slice_image_adaptive_generator_m4d(image_pil, mask_pil, patch_size, overlap_oil, oil_class_id):
    """
    Adaptive generator for Train set.
    Priority 1: Oil Regions -> Dense Overlap
    Priority 2: Non-Oil Regions -> No Overlap (Sparse)
    """
    image_width, image_height = image_pil.size
    pad_color_image = (0, 0, 0)
    pad_value_mask = 0 
    
    # Keep track of covered areas to avoid duplicates if specific logic requires,
    # but simplest approach is 2 passes like the binary script.
    
    # --- Pass 1: Oil Dense Scan ---
    stride_oil = patch_size - overlap_oil
    stride_oil = max(1, stride_oil)
    
    for y in range(0, image_height, stride_oil):
        for x in range(0, image_width, stride_oil):
            x_end = min(x + patch_size, image_width)
            y_end = min(y + patch_size, image_height)
            crop_box = (x, y, x_end, y_end)
            
            # Check mask BEFORE checking duplicate logic? 
            # Actually, standard approach: Crop mask, check content.
            # But this is expensive to crop everything twice. 
            # Let's just do it; robustness first.
            
            patch_mask = mask_pil.crop(crop_box)
            
            if check_contains_oil(patch_mask, oil_class_id):
                patch_img = image_pil.crop(crop_box)
                
                # Pad
                if patch_img.size != (patch_size, patch_size):
                    padded_img = Image.new(image_pil.mode, (patch_size, patch_size), pad_color_image)
                    padded_seg = Image.new(mask_pil.mode, (patch_size, patch_size), pad_value_mask)
                    padded_img.paste(patch_img, (0, 0))
                    padded_seg.paste(patch_mask, (0, 0))
                    patch_img = padded_img
                    patch_mask = padded_seg
                
                yield patch_img, patch_mask, x, y, True

    # --- Pass 2: Non-Oil Sparse Scan (Background / Other Classes) ---
    stride_bg = patch_size # No overlap
    
    for y in range(0, image_height, stride_bg):
        for x in range(0, image_width, stride_bg):
            x_end = min(x + patch_size, image_width)
            y_end = min(y + patch_size, image_height)
            crop_box = (x, y, x_end, y_end)
            
            patch_mask = mask_pil.crop(crop_box)
            
            # Only yield if it does NOT contain oil (because oil is covered in Pass 1)
            # Note: This might miss some identical crops if grids align perfectly, 
            # but usually stride_oil != stride_bg, so grid won't align perfectly.
            # To be strictly non-redundant, we would need a mask map of "visited pixels".
            # For simplicity and data augmentation, slight redundancy is acceptable.
            # However, if we want to strictly avoid "Oil" in this pass:
            
            if not check_contains_oil(patch_mask, oil_class_id):
                patch_img = image_pil.crop(crop_box)
                
                 # Pad
                if patch_img.size != (patch_size, patch_size):
                    padded_img = Image.new(image_pil.mode, (patch_size, patch_size), pad_color_image)
                    padded_seg = Image.new(mask_pil.mode, (patch_size, patch_size), pad_value_mask)
                    padded_img.paste(patch_img, (0, 0))
                    padded_seg.paste(patch_mask, (0, 0))
                    patch_img = padded_img
                    patch_mask = padded_seg
                
                yield patch_img, patch_mask, x, y, False

def count_class_components(mask_pil, class_id):
    """
    Count the number of connected components for a specific class ID.
    Only connected pixels of 'class_id' are counted.
    """
    mask_np = np.array(mask_pil)
    # Create binary mask for this class
    binary_mask = (mask_np == class_id).astype(np.uint8) * 255
    
    # Use cv2.connectedComponents (or findContours)
    # Binary mask must be uint8.
    num_labels, labels_im = cv2.connectedComponents(binary_mask, connectivity=8)
    
    # num_labels includes the background as label 0.
    # So if there are components of this class, the count is num_labels - 1.
    # If binary_mask is all black (0 of this class), num_labels is 1 (only bg).
    return max(0, num_labels - 1)

def process_single_image(job_args):
    (
        image_filename, source_img_dir, source_mask_dir, main_output_path,
        category, split, patch_size, overlap_oil, effective_bg_keep_ratio
    ) = job_args

    # Setup output paths
    output_img_dir = os.path.join(main_output_path, "images", split)
    output_label_dir = os.path.join(main_output_path, "labels", split)
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    patch_count_oil = 0
    patch_count_others = 0
    
    # Initialize component counts for classes 0-4
    # 0: Sea, 1: Oil, 2: Look-alike, 3: Ship, 4: Land
    class_component_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    base_filename = os.path.splitext(image_filename)[0]
    image_path = os.path.join(source_img_dir, image_filename)
    
    mask_path = os.path.join(source_mask_dir, base_filename + ".png")
    if not os.path.exists(mask_path):
         mask_path = os.path.join(source_mask_dir, base_filename + ".tif")
         if not os.path.exists(mask_path):
             return 0, 0, class_component_counts
        
    try:
        with Image.open(image_path) as img_pil_raw, Image.open(mask_path) as mask_pil_raw:
            img_pil = img_pil_raw.convert("RGB")
            # Multiclass mask: Values 0,1,2,3,4. 
            mask_pil = mask_pil_raw.convert("L") 
            
            # --- [Task 1] Count Original Components for ALL classes ---
            for cid in range(5):
                class_component_counts[cid] = count_class_components(mask_pil, cid)

            # Select Generator
            if split == 'test':
                generated_patches = slice_image_standard_generator(
                    img_pil, mask_pil, patch_size, OVERLAP_TEST
                )
            else:
                generated_patches = slice_image_adaptive_generator_m4d(
                    img_pil, mask_pil, patch_size, overlap_oil, OIL_CLASS_ID
                )
            
            for patch_img, patch_mask, x, y, has_oil in generated_patches:
                
                if split != 'test' and not has_oil:
                    if random.random() > effective_bg_keep_ratio:
                        continue
                
                patch_base_name = f"{base_filename}_x{x}_y{y}"
                
                save_img_path = os.path.join(output_img_dir, f"{patch_base_name}.png") 
                save_lbl_path = os.path.join(output_label_dir, f"{patch_base_name}.png")
                
                patch_img.save(save_img_path)
                patch_mask.save(save_lbl_path) 
                
                if has_oil:
                    patch_count_oil += 1
                else:
                    patch_count_others += 1
                    
    except Exception as e:
        print(f"\n  [ERROR] Processing file '{image_filename}': {e}")
        import traceback
        traceback.print_exc()

    return patch_count_oil, patch_count_others, class_component_counts

def main():
    random.seed(RANDOM_SEED)
    
    # [v1.5 Update] M4D structure is: M4D/{split}/images and M4D/{split}/labels_1D
    # The pad_validation_m4d_v2.py generates standardized output at:
    # M4D_Padded_Multiclass/images/train, M4D_Padded_Multiclass/labels/train
    # If using Raw M4D, we need to adapt paths.
    # Assuming we want to run patching on the RAW M4D (1250x650) directly which has {split}/images structure.
    
    output_dir_name = f"M4D_P{PATCH_SIZE}_TrainOilOv{OVERLAP_TRAIN_OIL}_TestOv{OVERLAP_TEST}"
    if BACKGROUND_KEEP_RATIO < 1.0:
        output_dir_name += f"_BG{int(BACKGROUND_KEEP_RATIO*100)}"
        
    main_output_path = os.path.join(OUTPUT_BASE_DIR, output_dir_name)
    
    print(f"Output Directory: {main_output_path}")
    print(f"Mode: Adaptive M4D Multiclass")
    print(f"  - Target 'Oil' Class ID: {OIL_CLASS_ID}")
    print(f"  - Train Oil Overlap    : {OVERLAP_TRAIN_OIL}")
    print(f"  - Train Other Overlap  : {OVERLAP_TRAIN_OTHER} (None)")
    print(f"  - Test Overlap         : {OVERLAP_TEST}")
    
    num_workers = NUM_WORKERS if NUM_WORKERS is not None else (os.cpu_count() or 4)
    print(f"Using {num_workers} CPU workers.")

    current_run_stats = []
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # M4D Structure iteration
    
    for category in CATEGORIES:
        # If category is ".", it means we look directly inside SOURCE_BASE_DIR
        
        category_path_part = "" if category == "." else category
        
        for split in SPLITS:
            print(f"\n--- Processing Split: '{split}' ---")
            
            # [Fix] Logic based on pad_validation_m4d_v2.py
            # The structure is dataset/M4D/{split}/images/
            # The labels are dataset/M4D/{split}/labels_1D/
            
            base_split_dir = os.path.join(SOURCE_BASE_DIR, category_path_part, split)
            source_img_dir = os.path.join(base_split_dir, "images")
            source_mask_dir = os.path.join(base_split_dir, "labels_1D")
            
            if not os.path.isdir(source_img_dir):
                # Fallback check: maybe it IS in standardized structure (images/{split})?
                # Check standard structure: dataset/M4D/images/{split}
                alt_source_img_dir = os.path.join(SOURCE_BASE_DIR, category_path_part, "images", split)
                if os.path.isdir(alt_source_img_dir):
                    source_img_dir = alt_source_img_dir
                    source_mask_dir = os.path.join(SOURCE_BASE_DIR, category_path_part, "labels", split)
                    # Also try labels_1D if labels not found
                    if not os.path.isdir(source_mask_dir):
                         source_mask_dir = os.path.join(SOURCE_BASE_DIR, category_path_part, "labels_1D", split)
                else:
                    print(f"  [Skip] Directory not found: {source_img_dir} or {alt_source_img_dir}")
                    continue

            # Double check mask dir
            if not os.path.isdir(source_mask_dir):
                 print(f"  [Warning] Mask directory not found: {source_mask_dir}")
            
            image_files = sorted([f for f in os.listdir(source_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
            source_image_count = len(image_files)
            print(f"  Found {source_image_count} images at {source_img_dir}")
            print(f"  Looking for labels at {source_mask_dir}")
            
            if source_image_count == 0: continue

            # Determine parameters
            # For Test, we use Standard Generator which uses OVERLAP_TEST global constant logic inside process_single_image
            # Just pass OVERLAP_TRAIN_OIL as the overlap arguments, 
            #   but inside process_single_image we handle split=='test' separately anyway.
            
            current_overlap_oil = OVERLAP_TRAIN_OIL
            # However, to be cleaner, let's pass the relevant overlap.
            
            effective_bg_keep_ratio = BACKGROUND_KEEP_RATIO
            if split == 'test': effective_bg_keep_ratio = 1.0

            jobs = []
            for image_filename in image_files:
                jobs.append((
                    image_filename, source_img_dir, source_mask_dir, main_output_path,
                    category, split, PATCH_SIZE, current_overlap_oil, effective_bg_keep_ratio
                ))

            # Run parallel
            total_oil, total_others = 0, 0
            # Initialize totals for all classes
            total_class_components = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(executor.map(process_single_image, jobs), total=len(jobs)))
            
            for o, b, class_counts in results:
                total_oil += o
                total_others += b
                # Aggregate class counts
                if isinstance(class_counts, dict):
                    for cid in total_class_components:
                        total_class_components[cid] += class_counts.get(cid, 0)

            print(f"  Result: Oil-containing patches: {total_oil}, Others: {total_others}")
            print(f"  Original Components found: {total_class_components}")
            
            # Prepare log entry
            log_entry = {
                "RunTimestamp": run_timestamp,
                "Split": split,
                "SourceImages": source_image_count,
                "PatchesWithOil": total_oil, 
                "PatchesOthers": total_others,
                "TotalPatches": total_oil + total_others
            }
            # Add component counts to log
            for cid in total_class_components:
                # 0: Sea, 1: Oil, 2: Look-alike, 3: Ship, 4: Land
                class_name_map = {0: "Sea", 1: "Oil", 2: "LookAlike", 3: "Ship", 4: "Land"}
                log_entry[f"Orig_Comp_{class_name_map[cid]}"] = total_class_components[cid]
            
            current_run_stats.append(log_entry)

    if current_run_stats:
        excel_log_path = os.path.join(OUTPUT_BASE_DIR, "m4d_patch_log.xlsx")
        new_df = pd.DataFrame(current_run_stats)
        if os.path.exists(excel_log_path):
            try:
                pd.concat([pd.read_excel(excel_log_path), new_df], ignore_index=True).to_excel(excel_log_path, index=False)
            except:
                new_df.to_excel(excel_log_path, index=False)
        else:
            new_df.to_excel(excel_log_path, index=False)
            
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()
