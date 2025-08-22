import glob
import random

import os
from os.path import join, exists, dirname, basename

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as f
import torchvision.transforms.functional as TF

from datasets.extract_data_tools.example_loader_ddd17 import load_files_in_directory, extract_events_from_memmap
import datasets.data_util as data_util
from PIL import Image


def get_split(dirs, split):
    return {
        "train": [dirs[0], dirs[2], dirs[3], dirs[4], dirs[5]],
        "valid": [dirs[1]]
    }[split]


def unzip_segmentation_masks(dirs):
    for d in dirs:
        assert exists(join(d, "segmentation_masks.zip"))
        if not exists(join(d, "segmentation_masks")):
            print("Unzipping segmentation mask in %s" % d)
            os.system("unzip %s -d %s" % (join(d, "segmentation_masks"), d))


class DDD17Events(Dataset):
    def __init__(
        self,
        root,
        split="train",
        event_representation='voxel_grid',
        nr_events_data=5,
        delta_t_per_data=50,
        nr_bins_per_data=5,
        require_paired_data=False,
        separate_pol=False,
        normalize_event=False,
        augmentation=False,
        fixed_duration=False,
        nr_events_per_data=32000,
        resize=True,
        random_crop=False,
        config_option:str = '',
        pl_sources:str = '',
        superpixel_sources:str = '',
        skip_ratio:int = 1,
        if_sam_distillation:bool = False,
    ):
        data_dirs = sorted(glob.glob(join(root, "dir*")))
        assert len(data_dirs) > 0
        assert split in ["train", "valid", "test"]

        self.split = split
        self.augmentation = augmentation
        self.fixed_duration = fixed_duration
        self.nr_events_per_data = nr_events_per_data

        self.nr_events_data = nr_events_data
        self.delta_t_per_data = delta_t_per_data

        if self.fixed_duration:
            self.t_interval = nr_events_data * delta_t_per_data
        else:
            self.t_interval = -1
            self.nr_events = self.nr_events_data * self.nr_events_per_data
        assert self.t_interval in [10, 50, 250, -1]

        self.nr_temporal_bins = nr_bins_per_data
        self.require_paired_data = require_paired_data
        self.event_representation = event_representation
        self.shape = [260, 346]
        self.resize = resize
        self.shape_resize = [260, 352]
        self.random_crop = random_crop
        self.shape_crop = [120, 216]
        self.separate_pol = separate_pol
        self.normalize_event = normalize_event

        self.dirs = get_split(data_dirs, split)

        self.skip_ratio = skip_ratio
        if self.skip_ratio == 1:
            self.files = []
            for d in self.dirs:
                label_files = glob.glob(join(d, "segmentation_masks", "*.png"))
                orignal_length = len(label_files)
                self.files += label_files
                print("Seq '{}': '{}' data loaded.".format(
                    d, orignal_length
                ))

        else:
            self.files = []
            for d in self.dirs:
                label_files = glob.glob(join(d, "segmentation_masks", "*.png"))
                orignal_length = len(label_files)
                new_length = orignal_length // self.skip_ratio
                label_files = label_files[:new_length+1]
                self.files += label_files
                print("Seq '{}': '{}' of '{}' data loaded with skipping ratio '{}' .".format(
                    d, len(label_files), orignal_length, self.skip_ratio
                ))

        self.img_timestamp_event_idx = {}
        self.event_data = {}
        self.event_dirs = self.dirs
        for d in self.event_dirs:
            img_timestamp_event_idx, t_events, xyp_events, _ = load_files_in_directory(d, self.t_interval)
            self.img_timestamp_event_idx[d] = img_timestamp_event_idx
            self.event_data[d] = [t_events, xyp_events]
            
        
        self.config_option = config_option
        self.pl_sources = pl_sources
        self.superpixel_sources = superpixel_sources
        self.if_sam_distillation = if_sam_distillation

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        segmentation_mask_file = self.files[idx]
        segmentation_mask = cv2.imread(segmentation_mask_file, 0)
        
        if self.resize:
            segmentation_mask = cv2.resize(
                segmentation_mask, (self.shape_resize[1], self.shape_resize[0] - 60),
                interpolation=cv2.INTER_NEAREST,
            )
        label = np.array(segmentation_mask)
        label_tensor = torch.from_numpy(label).long()

        if self.config_option == 'recon2voxel' or self.config_option == 'frame2voxel':
            directory = dirname(dirname(segmentation_mask_file))
            img_idx = int(basename(segmentation_mask_file).split("_")[-1].split(".")[0]) - 1
            img_timestamp_event_idx = self.img_timestamp_event_idx[directory]
            t_events, xyp_events = self.event_data[directory]

            events = extract_events_from_memmap(
                t_events, xyp_events,
                img_idx, img_timestamp_event_idx,
                self.fixed_duration,
                self.nr_events,
            )

            t_ns = events[:, 2]
            delta_t_ns = int((t_ns[-1] - t_ns[0]) / self.nr_events_data)
            nr_events_loaded = events.shape[0]
            nr_events_temp = nr_events_loaded // self.nr_events_data

            id_end = 0
            event_tensor = None
            for i in range(self.nr_events_data):
                id_start = id_end
                if self.fixed_duration:
                    id_end = np.searchsorted(t_ns, t_ns[0] + (i + 1) * delta_t_ns)
                else:
                    id_end += nr_events_temp

                if id_end > nr_events_loaded:
                    id_end = nr_events_loaded

                event_representation = data_util.generate_input_representation(
                    events[id_start:id_end],
                    self.event_representation,
                    self.shape,
                    nr_temporal_bins=self.nr_temporal_bins,
                    separate_pol=self.separate_pol,
                )
                event_representation = torch.from_numpy(event_representation)

                if self.normalize_event:
                    event_representation = data_util.normalize_voxel_grid(event_representation)

                if self.resize:
                    event_representation_resize = f.interpolate(
                        event_representation.unsqueeze(0),
                        size=(self.shape_resize[0], self.shape_resize[1]),
                        mode='bilinear', align_corners=True,
                    )
                    event_representation = event_representation_resize.squeeze(0)

                if event_tensor is None:
                    event_tensor = event_representation
                else:
                    event_tensor = torch.cat([event_tensor, event_representation], dim=0)

            event_tensor = event_tensor[:, :-60, :]
            

        file_path = segmentation_mask_file

        if self.config_option == 'frame2voxel' or self.config_option == 'frame2recon':
            frame_path = file_path.replace('segmentation_masks', 'images_aligned')

            a = frame_path.split('segmentation_')
            if frame_path.split('/')[-3] == 'dir0' or frame_path.split('/')[-3] == 'dir1':
                frame_path = a[0] + a[1]
                frame_path = frame_path.replace(frame_path.split('/')[-1], 'img_' + frame_path.split('/')[-1])
            else:
                frame_path = a[0] + '00' + a[1]
            
            frame = np.array(Image.open(frame_path))
            frame = [frame / 255]
            frame = torch.tensor(np.array(frame, dtype=np.float32).transpose(0, 3, 1, 2))
            frame = torch.squeeze(frame, dim=0)

        if self.config_option == 'recon2voxel' or self.config_option == 'frame2recon':
            recon_path = file_path.replace('segmentation_masks', 'reconstructions')
            recon = np.array(Image.open(recon_path))
            recon = [recon / 255]
            recon = torch.tensor(np.array(recon, dtype=np.float32).transpose(0, 3, 1, 2))
            recon = torch.squeeze(recon, dim=0)

        if self.split == 'train':
            pl_path = file_path.replace('segmentation_masks', self.pl_sources)

            a = pl_path.split('segmentation_')
            if pl_path.split('/')[-3] == 'dir0' or pl_path.split('/')[-3] == 'dir1':
                pl_path = a[0] + a[1]
                pl_path = pl_path.replace(pl_path.split('/')[-1], 'segmentation_' + pl_path.split('/')[-1])
            else:
                pl_path = a[0] + '00' + a[1]
            pl = np.array(Image.open(pl_path))
            if self.resize:
                pl = cv2.resize(
                    pl, (self.shape_resize[1], self.shape_resize[0] - 60),
                    interpolation=cv2.INTER_NEAREST,
                )
            pl = torch.tensor(pl).squeeze(0).long()

        else:
            pl = torch.ones_like(label_tensor)

        if len(self.superpixel_sources) > 1:
            
            if self.superpixel_sources == 'sp_slic_rgb':
                sp_path = file_path.replace('segmentation_masks', self.superpixel_sources)
            if self.superpixel_sources == 'sp_sam_rgb':
                sp_path = file_path.replace('segmentation_masks', 'superpixels_sam')

            a = sp_path.split('segmentation_')
            if sp_path.split('/')[-3] == 'dir0' or sp_path.split('/')[-3] == 'dir1':
                sp_path = a[0] + a[1]
                sp_path = sp_path.replace(sp_path.split('/')[-1], 'img_' + sp_path.split('/')[-1])
            else:
                sp_path = a[0] + '00' + a[1]
            if self.superpixel_sources == 'sp_slic_rgb':
                sp_path = sp_path.replace('.png', '_slic_25.png')
            superpixel = np.array(Image.open(sp_path))
            if self.resize:
                superpixel = cv2.resize(
                    superpixel, (self.shape_resize[1], self.shape_resize[0] - 60),
                    interpolation=cv2.INTER_NEAREST,
                )
            superpixel = torch.tensor(superpixel).long()
        else:
            superpixel = torch.ones_like(label_tensor)


        if self.config_option == 'recon2voxel':
            if self.augmentation:
                if random.random() >= 0.5:
                    event_tensor = torch.flip(event_tensor, [2])
                    label_tensor = torch.flip(label_tensor, [1])
                    recon = torch.flip(recon, [2])
                    pl = torch.flip(pl, [1])
                    superpixel = torch.flip(superpixel, [1])

                if random.random() >= 0.5:
                    brightness_factor = random.uniform(0.8, 1.2)
                    recon = TF.adjust_brightness(recon, brightness_factor)

                if random.random() >= 0.5:
                    contrast_factor = random.uniform(0.8, 1.2)
                    recon = TF.adjust_contrast(recon, contrast_factor)

                if random.random() >= 0.5:
                    noise = torch.randn(recon.size()) * 0.05
                    recon = recon + noise

            return event_tensor, label_tensor, recon, pl, superpixel, file_path


        elif self.config_option == 'frame2voxel':
            if self.augmentation:
                if random.random() >= 0.5:
                    event_tensor = torch.flip(event_tensor, [2])
                    label_tensor = torch.flip(label_tensor, [1])
                    frame = torch.flip(frame, [2])
                    pl = torch.flip(pl, [1])
                    superpixel = torch.flip(superpixel, [1])

                if random.random() >= 0.5:
                    brightness_factor = random.uniform(0.8, 1.2)
                    frame = TF.adjust_brightness(frame, brightness_factor)
                
                if random.random() >= 0.5:
                    contrast_factor = random.uniform(0.8, 1.2)
                    frame = TF.adjust_contrast(frame, contrast_factor)
                
                if random.random() >= 0.5:
                    noise = torch.randn(frame.size()) * 0.05
                    frame = frame + noise

            return event_tensor, label_tensor, frame, pl, superpixel, file_path


        elif self.config_option == 'frame2recon':
            if self.augmentation:
                if random.random() >= 0.5:
                    recon = torch.flip(recon, [2])
                    label_tensor = torch.flip(label_tensor, [1])
                    frame = torch.flip(frame, [2])
                    pl = torch.flip(pl, [1])
                    superpixel = torch.flip(superpixel, [1])
                
                if random.random() >= 0.5:
                    brightness_factor = random.uniform(0.8, 1.2)
                    recon = TF.adjust_brightness(recon, brightness_factor)
                    brightness_factor = random.uniform(0.8, 1.2)
                    frame = TF.adjust_brightness(frame, brightness_factor)

                if random.random() >= 0.5:
                    contrast_factor = random.uniform(0.8, 1.2)
                    recon = TF.adjust_contrast(recon, contrast_factor)
                    contrast_factor = random.uniform(0.8, 1.2)
                    frame = TF.adjust_contrast(frame, contrast_factor)

                if random.random() >= 0.5:
                    noise = torch.randn(recon.size()) * 0.05
                    recon = recon + noise
                    noise = torch.randn(frame.size()) * 0.05
                    frame = frame + noise

            return frame, label_tensor, recon, pl, superpixel, file_path