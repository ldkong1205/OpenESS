import os
import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import multiprocessing as mp

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def run_worker(gpu_id, tasks, sam_checkpoint, model_type, skip_exist=True, verbose=False):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    itr = tasks if verbose else tqdm(tasks, desc=f"GPU{gpu_id}", ncols=80)
    for img_path, sp_img_save_path in itr:
        try:
            if skip_exist and os.path.exists(sp_img_save_path):
                continue

            image = cv2.imread(img_path)
            if image is None:
                if verbose:
                    print(f"[GPU{gpu_id}] Warning: fail to read {img_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masks = mask_generator.generate(image)
            sorted_anns = sorted(masks, key=lambda x: x["area"], reverse=True)

            segments_sam = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for idx, ann in enumerate(sorted_anns):
                m = ann["segmentation"]
                segments_sam[m] = idx

            Image.fromarray(segments_sam).save(sp_img_save_path)

        except Exception as e:
            print(f"[GPU{gpu_id}] Error on {img_path}: {e}")


def split_chunks(lst, n):
    if n <= 1:
        return [lst]
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def collect_tasks(root_folder):

    all_tasks = []
    for seq in os.listdir(root_folder):
        seq_path = os.path.join(root_folder, seq)
        img_dir = os.path.join(seq_path, "images_aligned")
        if not os.path.isdir(img_dir):
            continue
        sp_root = os.path.join(seq_path, "superpixels_sam")
        os.makedirs(sp_root, exist_ok=True)

        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            sp_img_save_path = os.path.join(sp_root, img_name)
            all_tasks.append((img_path, sp_img_save_path))
    return all_tasks


def parse_option():
    parser = argparse.ArgumentParser("SAM multi-GPU inference", add_help=True)
    parser.add_argument("-r", "--root_folder", default="data/DDD17",
                        help="root folder of dataset")
    parser.add_argument("-p", "--sam_checkpoint", type=str,
                        default="pretrained_checkpoints/sam_vit_h_4b8939.pth",
                        help="path of pretrained model")
    parser.add_argument("--model_type", type=str, default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"], help="SAM model type")
    parser.add_argument("--devices", type=str, default="", 
                        help='GPU list, for example "0,1,2". None for all GPUs are available.')
    parser.add_argument("--skip_exist", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_option()

    if args.devices.strip():
        gpu_list = [int(x) for x in args.devices.split(",")]
    else:
        num = torch.cuda.device_count()
        if num == 0:
            raise RuntimeError("No CUDA device available.")
        gpu_list = list(range(num))

    tasks = collect_tasks(args.root_folder)
    if len(tasks) == 0:
        print("No images found. Check your root_folder structure.")
        exit(0)

    chunks = split_chunks(tasks, len(gpu_list))

    print(f"Total images: {len(tasks)} | GPUs: {gpu_list} | "
          f"avg per GPU: ~{len(tasks)//max(1,len(gpu_list))}")

    ctx = mp.get_context("spawn")
    procs = []
    for rank, gpu_id in enumerate(gpu_list):
        p = ctx.Process(
            target=run_worker,
            args=(gpu_id, chunks[rank], args.sam_checkpoint, args.model_type, args.skip_exist, args.verbose),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("Done.")