import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import argparse
from PIL import Image
from tqdm import tqdm


def compute_sam(img_path, sp_img_save_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    segments_sam = np.zeros((image.shape[0], image.shape[1]))
    for id, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        segments_sam[m] = id

    im = Image.fromarray(segments_sam.astype(np.uint8))
    im.save(
        sp_img_save_path
    )

def parse_option():
    parser = argparse.ArgumentParser('SAM', add_help=False)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='data/DDD17')
    parser.add_argument('-p', '--sam_checkpoint', help='path of pretrained model', type=str,
                        default='pretrained_checkpoints/sam_vit_h_4b8939.pth')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_option()
    sam_checkpoint = args.sam_checkpoint
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    for seq in tqdm(os.listdir(args.root_folder)):
        seq_path = os.path.join(args.root_folder, seq)
        sp_root = os.path.join(seq_path, "superpixels_sam")
        os.makedirs(sp_root, exist_ok=True)
        for img_name in tqdm(os.listdir(os.path.join(seq_path, "images_aligned"))):
            img_path = os.path.join(seq_path, "images_aligned", img_name)
            sp_img_save_path = os.path.join(sp_root, img_name)
            compute_sam(img_path, sp_img_save_path)