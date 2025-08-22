import os
import argparse
import time
import logging
import yaml
import shutil

import numpy as np
import torch

from e2vid.options.inference_options import set_inference_options


class Settings:
    def __init__(self, settings_yaml, generate_log=True):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.load(stream, yaml.Loader)

            # --- hardware ---
            hardware = settings['hardware']
            gpu_device = hardware['gpu_device']

            self.gpu_device = torch.device("cpu") if gpu_device == "cpu" else torch.device("cuda:" + str(gpu_device))

            self.num_cpu_workers = hardware['num_cpu_workers']
            if self.num_cpu_workers < 0:
                self.num_cpu_workers = os.cpu_count()

            self.path_to_model = 'e2vid/pretrained/E2VID_lightweight.pth.tar'

            # --- Model ---
            model = settings['model']
            self.model_name = model['model_name']
            self.skip_connect_encoder = model['skip_connect_encoder']  # True
            self.skip_connect_task = model['skip_connect_task']  # True
            self.skip_connect_task_type = model['skip_connect_task_type']  # 'concat'
            self.data_augmentation_train = model['data_augmentation_train']  # True
            self.train_on_event_labels = model['train_on_event_labels']  # False
            self.unfrozen_e2vid = model['unfrozen_e2vid']

            # --- E2VID Config ---
            parser = argparse.ArgumentParser(description='E2VID.')
            parser.add_argument('-c', '--path_to_model', default=self.path_to_model, type=str,
                                help='path to model weights')
            set_inference_options(parser)
            args, unknown = parser.parse_known_args()
            self.e2vid_config = args


            # --- dataset sensor b ---
            dataset = settings['dataset']
            self.dataset_name_b = dataset['name_b']
            self.sensor_b_name = self.dataset_name_b.split('_')[-1]  # 'events'
            self.split_train_b = 'train'
            self.event_representation_b = None
            self.nr_events_window_b = None
            self.nr_temporal_bins_b = None
            self.separate_pol_b = False
            self.normalize_event_b = False
            self.require_paired_data_train_b = False
            self.require_paired_data_val_b = False
            self.input_channels_b_paired = None
            self.read_two_imgs_b = None
            self.extension_dataset_path_b = None

            if self.dataset_name_b in ['EventScape_recurrent_events', 'DSEC_events', 'DDD17_events', 'E2VIDDriving_events']:
                if self.dataset_name_b == 'DSEC_events':
                    dataset_specs = dataset['DSEC_events']
                    self.delta_t_per_data_b = dataset_specs['delta_t_per_data']
                    self.semseg_label_train_b = False
                    self.semseg_label_val_b = True
                elif self.dataset_name_b == 'E2VIDDriving_events':
                    dataset_specs = dataset['E2VIDDriving_events']
                    self.semseg_label_train_b = False
                    self.semseg_label_val_b = False
                else:
                    if self.dataset_name_b == 'DDD17_events':
                        dataset_specs = dataset['DDD17_events']
                        self.split_train_b = dataset_specs['split_train']
                        self.delta_t_per_data_b = dataset_specs['delta_t_per_data']
                    else:
                        dataset_specs = dataset['eventscape_events']
                        self.nr_events_files_b = dataset_specs['nr_events_files_per_data']
                    self.semseg_label_train_b = True
                    self.semseg_label_val_b = True
                
                self.fixed_duration_b = dataset_specs['fixed_duration']  # False
                self.nr_events_data_b = dataset_specs['nr_events_data']  # 20
                self.event_representation_b = dataset_specs['event_representation']  # 'voxel_grid'
                self.nr_events_window_b = dataset_specs['nr_events_window']  # 100000
                self.nr_temporal_bins_b = dataset_specs['nr_temporal_bins']  # 5

                if self.event_representation_b == 'voxel_grid':
                    self.separate_pol_b = dataset_specs['separate_pol']  # False
                    self.input_channels_b = dataset_specs['nr_temporal_bins']  # 5
                    if self.separate_pol_b:  # False
                        self.input_channels_b = dataset_specs['nr_temporal_bins'] * 2
                elif self.event_representation_b == 'ev_segnet':
                    self.input_channels_b = 6
                else:
                    self.input_channels_b = 2
                
                self.normalize_event_b = dataset_specs['normalize_event']  # False
                self.require_paired_data_train_b = dataset_specs['require_paired_data_train']  # False
                self.require_paired_data_val_b = dataset_specs['require_paired_data_val']  # True
                if self.require_paired_data_train_b or self.require_paired_data_val_b:
                    self.input_channels_b_paired = 3
            else:
                raise ValueError("Specified Dataset Sensor B: %s is not implemented" % self.dataset_name_b)

            if 'EventScape' in self.dataset_name_b:
                self.towns_b = dataset_specs['towns']
            self.img_size_b = dataset_specs['shape']
            self.dataset_path_b = dataset_specs['dataset_path']
            assert os.path.isdir(self.dataset_path_b)


            # --- Task ---
            task = settings['task']
            self.semseg_num_classes = task['semseg_num_classes']  # 6 for DDD17, 11 for DSEC
            if self.semseg_num_classes == 6:
                self.semseg_ignore_label = 255
                self.semseg_class_names = [
                    'flat', 'background', 'object', 'vegetation', 'human', 'vehicle',
                ]
                self.semseg_color_map = np.zeros((self.semseg_num_classes, 3), dtype=np.uint8)
                self.semseg_color_map[0] = [128, 64,128]
                self.semseg_color_map[1] = [70, 70, 70]
                self.semseg_color_map[2] = [220,220,  0]
                self.semseg_color_map[3] = [107,142, 35]
                self.semseg_color_map[4] = [220, 20, 60]
                self.semseg_color_map[5] = [  0,  0,142]

            elif self.semseg_num_classes == 11:
                self.semseg_ignore_label = 255
                self.semseg_class_names = [
                    'background', 'building', 'fence', 'person', 'pole', 'road',
                    'sidewalk', 'vegetation', 'car', 'wall', 'traffic sign',
                ]
                self.semseg_color_map = np.zeros((self.semseg_num_classes, 3), dtype=np.uint8)
                self.semseg_color_map[0] = [0, 150, 255]    # background
                self.semseg_color_map[1] = [118, 118, 118]  # building
                self.semseg_color_map[2] = [214, 220, 229]  # fence
                self.semseg_color_map[3] = [4, 50, 255]     # person
                self.semseg_color_map[4] = [190, 153, 153]  # pole
                self.semseg_color_map[5] = [155, 55, 255]   # road
                self.semseg_color_map[6] = [102, 102, 156]  # sidewalk
                self.semseg_color_map[7] = [0, 176, 80]     # vegetation
                self.semseg_color_map[8] = [250, 188, 1]    # car
                self.semseg_color_map[9] = [152, 251, 152]  # wall
                self.semseg_color_map[10] = [255, 0, 0]     # traffic-sign

            elif self.semseg_num_classes == 19:
                self.semseg_ignore_label = 255
                self.semseg_class_names = [
                    'road', 'sidewalk', 'building', 'wall', 'fence',
                    'pole', 'traffic light', 'traffic sign',
                    'vegetation', 'terrain', 'sky',
                    'person', 'rider',
                    'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
                ]
                self.semseg_color_map = np.zeros((self.semseg_num_classes, 3), dtype=np.uint8)
                self.semseg_color_map[0] = [0, 0, 0]
                self.semseg_color_map[1] = [70, 70, 70]
                self.semseg_color_map[2] = [190, 153, 153]
                self.semseg_color_map[3] = [220, 20, 60]
                self.semseg_color_map[4] = [153, 153, 153]
                self.semseg_color_map[5] = [128, 64, 128]
                self.semseg_color_map[6] = [244, 35, 232]
                self.semseg_color_map[7] = [107, 142, 35]
                self.semseg_color_map[8] = [0, 0, 142]
                self.semseg_color_map[9] = [102, 102, 156]
                self.semseg_color_map[10] = [220, 220, 0]
            
            # --- checkpoint ---
            checkpoint = settings['checkpoint']
            self.save_checkpoint = checkpoint['save_checkpoint']
            self.resume_training = checkpoint['resume_training']
            assert isinstance(self.resume_training, bool)
            self.resume_ckpt_file = checkpoint['resume_file']

            # --- directories ---
            directories = settings['dir']
            log_dir = directories['log']

            # --- logs ---
            if generate_log:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                self.timestr = timestr
                log_dir = os.path.join(log_dir, timestr)
                os.makedirs(log_dir)
                settings_copy_filepath = os.path.join(log_dir, os.path.split(settings_yaml)[-1])
                shutil.copyfile(settings_yaml, settings_copy_filepath)
                # logger
                logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, 'running.log'))
                self.logger = logging.getLogger()
                # checkpoints
                self.ckpt_dir = os.path.join(log_dir, 'checkpoints')
                os.mkdir(self.ckpt_dir)
                # visualization
                self.vis_dir = os.path.join(log_dir, 'visualization')
                os.mkdir(self.vis_dir)
            else:
                self.ckpt_dir = os.path.join(log_dir, 'checkpoints')
                self.vis_dir = os.path.join(log_dir, 'visualization')

            # --- optimization ---
            optimization = settings['optim']
            self.batch_size_b = int(optimization['batch_size_b'])
            self.lr_voxel = float(optimization['lr_voxel'])
            self.lr_recon = float(optimization['lr_recon'])
            self.lr_frame = float(optimization['lr_frame'])
            self.lr_decay = float(optimization['lr_decay'])
            self.num_epochs = int(optimization['num_epochs'])
            self.val_epoch_step = int(optimization['val_epoch_step'])
            self.weight_task_loss = float(optimization['weight_task_loss'])
            self.task_loss = optimization['task_loss']


            # --- clip model ---
            clip_config = settings['clip']
            self.config_option = clip_config['config_option']
            self.skip_ratio = clip_config['skip_ratio']

            self.text_embeddings_path = clip_config['text_embeddings_path']
            self.maskclip_checkpoint = clip_config['maskclip_checkpoint']
            self.visual_projs_path = clip_config['visual_projs_path']
            self.output_stride = int(clip_config['output_stride'])
            self.pretrained_backbone = clip_config['pre_trained_backbone']

            # supervised only
            self.if_supervised_only = clip_config['if_supervised_only']

            # pretraining
            if_pretraining = clip_config.get('if_pretraining', None)
            if if_pretraining is not None:
                self.if_pretraining = if_pretraining
                self.image_weights = clip_config['image_weights']
                self.if_spatial_contrastive = clip_config['if_spatial_contrastive']
                self.superpixel_sources = clip_config['superpixel_sources']
                self.superpixel_size = clip_config['superpixel_size']
                self.if_dense_clip_supervision = clip_config['if_dense_clip_supervision']
                self.pl_sources = clip_config['pl_sources']
                self.if_sam_distillation = clip_config['if_sam_distillation']

            # finetuning
            if_finetuning = clip_config.get('if_finetuning', None)
            if if_finetuning is not None:
                self.if_finetuning = if_finetuning
                self.load_pretrained_weights = clip_config['load_pretrained_weights']
                self.pretrained_file = clip_config['pretrained_file']
                self.if_switchable_train = clip_config['if_switchable_train']
            self.frozen_backbone = clip_config.get('frozen_backbone', False)

            # linear probing
            self.if_linear_probing = clip_config.get('if_linear_probing', False)

            self.use_amp = clip_config.get('use_amp', False)