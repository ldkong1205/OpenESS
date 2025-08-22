import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.segmentation import slic
from concurrent.futures import ThreadPoolExecutor, as_completed


def compute_slic(img_path, num_segments):
    """
    Compute SLIC superpixels for a single image.
    Args:
        img_path (str): Path to the input image.
        num_segments (int): Number of segments for SLIC.
    Returns:
        np.ndarray: Segmentation map (uint8).
    """
    im = Image.open(img_path)
    im_np = np.array(im)
    segments_slic = slic(
        im_np, n_segments=num_segments, compactness=6, sigma=3.0, start_label=0
    ).astype(np.uint8)
    return segments_slic


def process_one(img_path, out_path, num_segments):
    """
    Process a single image: run SLIC and save the result.
    Args:
        img_path (str): Path to the input image.
        out_path (str): Path to save the segmentation result.
        num_segments (int): Number of SLIC segments.
    """
    seg = compute_slic(img_path, num_segments)
    im = Image.fromarray(seg)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--model", type=str, default="minkunet",
        help="Specify the target model: minkunet or voxelnet"
    )
    parser.add_argument(
        "--num_segments", type=int, default=100,
        help="Number of segments for SLIC"
    )
    parser.add_argument(
        "--dataset", type=str, default="DSEC",
        help="Dataset name under 'data/'"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of threads for parallel processing"
    )
    args = parser.parse_args()
    assert args.model in ["minkunet", "voxelnet"]

    data_root = "data"
    dataset_path = os.path.join(data_root, args.dataset)

    tasks = []
    sequence_paths = [
        os.path.join(dataset_path, seq)
        for seq in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, seq))
    ]

    for seq_path in sequence_paths:
        seq_img_path = os.path.join(seq_path, "images_aligned", "left")
        seq_spp_path = os.path.join(seq_path, "sp_slic_rgb", "left")
        if not os.path.isdir(seq_img_path):
            continue

        img_list = [
            fn for fn in os.listdir(seq_img_path)
            if fn.lower().endswith(".png")
        ]
        img_list.sort()

        for img_name in img_list:
            img_path = os.path.join(seq_img_path, img_name)
            out_name = img_name.replace(".png", f"_slic_{args.num_segments}.png")
            out_path = os.path.join(seq_spp_path, out_name)
            if not os.path.exists(out_path):
                tasks.append((img_path, out_path, args.num_segments))

    if len(tasks) == 0:
        print("No images to process. All outputs may already exist.")
        exit(0)

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one, *t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing (threads)"):
            try:
                _ = fut.result()
            except Exception as e:
                errors.append(e)

    if errors:
        print(f"\nCompleted with {len(errors)} failures. Example error: {errors[0]}")
    else:
        print("\nAll tasks completed successfully.")
