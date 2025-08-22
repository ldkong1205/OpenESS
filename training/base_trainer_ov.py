"""
Adapted from: https://github.com/uzh-rpg/rpg_ev-transfer
"""
from __future__ import division

import torch
from tqdm import tqdm

tqdm.monitor_interval = 0

import time

import numpy as np

from utils.saver import CheckpointSaver
from datasets.wrapper_dataloader import WrapperDataset
import utils.viz_utils as viz_utils


class BaseTrainer(object):
    """BaseTrainer class to be inherited"""

    def __init__(self, settings):
        self.settings = settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.do_val_training_epoch = True

        self.init_fn()
        self.createDataLoaders()

        self.models_dict = {k: v.to(self.device) for k, v in self.models_dict.items()}

        load_optimizer = False
        if self.settings.resume_training:
            load_optimizer = False

            self.saver = CheckpointSaver(save_dir=settings.ckpt_dir)
            self.checkpoint = self.saver.load_checkpoint(
                self.models_dict, self.optimizers_dict,
                checkpoint_file=self.settings.resume_ckpt_file,
                load_optimizer=load_optimizer,
            )
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['step_count']

        else:
            self.saver = CheckpointSaver(save_dir=settings.ckpt_dir)
            if self.settings.load_pretrained_weights:
                self.saver.load_pretrained_weights(
                    self.models_dict, self.models_dict.keys(),
                    self.settings.pretrained_file,
                    self.settings.frozen_backbone
                )
                for name, param in self.models_dict["model_recon"].named_parameters():
                    if param.requires_grad:
                        print(f"Trainable: {name}")
                self.settings.logger.info('Pretrained checkpoints loaded from {}'.format(self.settings.pretrained_file))
            self.epoch_count = 0
            self.step_count = 0
            self.checkpoint = None

        self.epoch = self.epoch_count
        self.validation_embeddings = []
        self.val_confusion_matrix = np.zeros([len(self.object_classes), len(self.object_classes)])

        optimizer_epoch_count = self.epoch_count if load_optimizer else 0
        self.lr_schedulers = {}
        total_steps = self.settings.num_epochs * len(self.train_loader_sensor_b)
        print("Total steps: {}.".format(total_steps))

        self.lr_schedulers = {
            k: torch.optim.lr_scheduler.CosineAnnealingLR(
                v, T_max=total_steps)
                for k, v in self.optimizers_dict.items()
            }
        print(self.lr_schedulers)
        self.settings.logger.info(self.lr_schedulers)

    def init_fn(self):
        """Model is constructed in child class"""
        pass

    def getDataloader(self, dataset_name):
        """Returns the dataset loader specified in the settings file"""
        if dataset_name == 'DSEC_events':
            from datasets.DSEC_events_loader import DSECEvents
            return DSECEvents
        elif dataset_name == 'DDD17_events':
            from datasets.ddd17_events_loader import DDD17Events
            return DDD17Events


    def createDSECDataset(
            self,
            dataset_name:str,
            dsec_dir:str,
            batch_size:int,
            nr_events_data:int,
            delta_t_per_data:int,
            nr_events_window:int,
            augmentation:bool,
            event_representation:str,
            nr_bins_per_data:int,
            require_paired_data_train:bool,
            require_paired_data_val:bool,
            separate_pol:bool,
            normalize_event:bool,
            semseg_num_classes:int,
            fixed_duration:bool,
            config_option:str,
            pl_sources:str,
            superpixel_sources:str,
            skip_ratio:int,
            if_sam_distillation:bool,
        ):
        """
        Creates the validation and the training data based on the provided paths and parameters.
        """
        dataset_builder = self.getDataloader(dataset_name)

        train_dataset = dataset_builder(
            dsec_dir=dsec_dir,
            nr_events_data=nr_events_data,
            delta_t_per_data=delta_t_per_data,
            nr_events_window=nr_events_window,
            augmentation=augmentation,
            mode='train',
            event_representation=event_representation,
            nr_bins_per_data=nr_bins_per_data,
            require_paired_data=require_paired_data_train,
            separate_pol=separate_pol,
            normalize_event=normalize_event,
            semseg_num_classes=semseg_num_classes,
            fixed_duration=fixed_duration,
            config_option=config_option,
            pl_sources=pl_sources,
            superpixel_sources=superpixel_sources,
            skip_ratio=skip_ratio,
            if_sam_distillation=if_sam_distillation,
        )
        val_dataset = dataset_builder(
            dsec_dir=dsec_dir,
            nr_events_data=nr_events_data,
            delta_t_per_data=delta_t_per_data,
            nr_events_window=nr_events_window,
            augmentation=False,
            mode='val',
            event_representation=event_representation,
            nr_bins_per_data=nr_bins_per_data,
            require_paired_data=require_paired_data_val,
            separate_pol=separate_pol,
            normalize_event=normalize_event,
            semseg_num_classes=semseg_num_classes,
            fixed_duration=fixed_duration,
            config_option=config_option,
            pl_sources=pl_sources,
            superpixel_sources='',
            skip_ratio=2,
            if_sam_distillation=False,
        )

        self.object_classes = []

        dataset_loader = torch.utils.data.DataLoader

        train_loader_sensor = dataset_loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=self.settings.num_cpu_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )
        val_loader_sensor = dataset_loader(
            val_dataset,
            batch_size=batch_size,
            num_workers=self.settings.num_cpu_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )
        print('DSEC num of batches: {}, {}.'.format(len(train_loader_sensor), len(val_loader_sensor)))
        self.settings.logger.info('DSEC num of batches: {}, {}.'.format(len(train_loader_sensor), len(val_loader_sensor)))

        return train_loader_sensor, val_loader_sensor

    def createDDD17EventsDataset(
            self,
            dataset_name: str,
            root: str,
            split_train,
            batch_size: int,
            nr_events_data,
            delta_t_per_data,
            nr_events_per_data,
            augmentation,
            event_representation,
            nr_bins_per_data,
            require_paired_data_train,
            require_paired_data_val,
            separate_pol,
            normalize_event,
            fixed_duration,
            config_option:str,
            pl_sources:str,
            superpixel_sources:str,
            skip_ratio:int,
            if_sam_distillation:bool,
        ):
        """
        Creates the validation and the training data based on the provided paths and parameters.
        """
        dataset_builder = self.getDataloader(dataset_name)

        train_dataset = dataset_builder(
            root=root,
            split=split_train,
            event_representation=event_representation,
            nr_events_data=nr_events_data,
            delta_t_per_data=delta_t_per_data,
            nr_bins_per_data=nr_bins_per_data,
            require_paired_data=require_paired_data_train,
            separate_pol=separate_pol,
            normalize_event=normalize_event,
            augmentation=augmentation,
            fixed_duration=fixed_duration,
            nr_events_per_data=nr_events_per_data,
            config_option=config_option,
            pl_sources=pl_sources,
            superpixel_sources=superpixel_sources,
            skip_ratio=skip_ratio,
            if_sam_distillation=if_sam_distillation,
        )
        val_dataset = dataset_builder(
            root=root,
            split='valid',
            event_representation=event_representation,
            nr_events_data=nr_events_data,
            delta_t_per_data=delta_t_per_data,
            nr_bins_per_data=nr_bins_per_data,
            require_paired_data=require_paired_data_val,
            separate_pol=separate_pol,
            normalize_event=normalize_event,
            augmentation=False,
            fixed_duration=fixed_duration,
            nr_events_per_data=nr_events_per_data,
            config_option=config_option,
            pl_sources=pl_sources,
            superpixel_sources='',
            skip_ratio=1,
            if_sam_distillation=False,
        )

        self.object_classes = []

        dataset_loader = torch.utils.data.DataLoader

        train_loader_sensor = dataset_loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=self.settings.num_cpu_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )
        val_loader_sensor = dataset_loader(
            val_dataset,
            batch_size=batch_size,
            num_workers=self.settings.num_cpu_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )
        self.settings.logger.info('DDD17Events num of batches: {}, {}'.format(len(train_loader_sensor), len(val_loader_sensor)))

        return train_loader_sensor, val_loader_sensor

    def combineDataloaders(self):
        """Combines two dataloader to one dataloader."""
        self.train_loader = WrapperDataset(
            self.train_loader_sensor_a, self.train_loader_sensor_b, self.device,
        )

    def createDataLoaders(self):

        if self.settings.dataset_name_b == 'DSEC_events':
            out = self.createDSECDataset(
                self.settings.dataset_name_b,
                self.settings.dataset_path_b,
                self.settings.batch_size_b,
                self.settings.nr_events_data_b,
                self.settings.delta_t_per_data_b,
                self.settings.nr_events_window_b,
                self.settings.data_augmentation_train,
                self.settings.event_representation_b,
                self.settings.nr_temporal_bins_b,
                self.settings.require_paired_data_train_b,
                self.settings.require_paired_data_val_b,
                self.settings.separate_pol_b,
                self.settings.normalize_event_b,
                self.settings.semseg_num_classes,
                self.settings.fixed_duration_b,
                self.settings.config_option,
                self.settings.pl_sources,
                self.settings.superpixel_sources,
                self.settings.skip_ratio,
                self.settings.if_sam_distillation,
            )

        elif self.settings.dataset_name_b == 'DDD17_events':
            out = self.createDDD17EventsDataset(
                self.settings.dataset_name_b,
                self.settings.dataset_path_b,
                self.settings.split_train_b,
                self.settings.batch_size_b,
                self.settings.nr_events_data_b,
                self.settings.delta_t_per_data_b,
                self.settings.nr_events_window_b,
                self.settings.data_augmentation_train,
                self.settings.event_representation_b,
                self.settings.nr_temporal_bins_b,
                self.settings.require_paired_data_train_b,
                self.settings.require_paired_data_val_b,
                self.settings.separate_pol_b,
                self.settings.normalize_event_b,
                self.settings.fixed_duration_b,
                self.settings.config_option,
                self.settings.pl_sources,
                self.settings.superpixel_sources,
                self.settings.skip_ratio,
                self.settings.if_sam_distillation,
            )

        else:
            out = self.createDataset(
                self.settings.dataset_name_b,
                self.settings.dataset_path_b,
                self.settings.towns_b,
                self.settings.img_size_b,
                self.settings.batch_size_b,
                self.settings.nr_events_data_b,
                self.settings.nr_events_files_b,
                self.settings.nr_events_window_b,
                self.settings.data_augmentation_train,
                self.settings.event_representation_b,
                self.settings.nr_temporal_bins_b,
                self.settings.require_paired_data_train_b,
                self.settings.require_paired_data_val_b,
                self.settings.read_two_imgs_b,
                self.settings.separate_pol_b,
                self.settings.normalize_event_b,
                self.settings.semseg_num_classes,
            )
        
        self.train_loader_sensor_b, self.val_loader_sensor_b = out


    def training(self):
        val_epoch_step = self.settings.val_epoch_step

        for _ in tqdm(
                range(self.epoch_count, self.settings.num_epochs),
                total=self.settings.num_epochs,
                initial=self.epoch_count,
            ):

            self.trainEpoch()

            if (self.epoch_count % val_epoch_step) == 0:
                self.valEpochs()

                if self.settings.save_checkpoint:
                    self.saver.save_checkpoint_model_single(self.models_dict, self.epoch_count, self.step_count)
                    tqdm.write('Checkpoint saved')
            for opt in self.optimizers_dict:
                self.lr_schedulers[opt].step()
            
            self.epoch_count += 1    
        
    def pretraining(self):
        val_epoch_step = self.settings.val_epoch_step

        for _ in tqdm(
                range(self.epoch_count, self.settings.num_epochs),
                total=self.settings.num_epochs,
                initial=self.epoch_count,
            ):

            self.trainEpoch()

            if (self.epoch_count % val_epoch_step) == 0:
                if self.settings.save_checkpoint:
                    self.saver.save_checkpoint_model(self.models_dict, self.epoch_count, self.step_count)
                    tqdm.write('Checkpoint saved')

            for opt in self.optimizers_dict:
                self.lr_schedulers[opt].step()
            
            self.epoch_count += 1 


    def valEpochs(self):
        self.resetValidationStatistics()

        with torch.no_grad():
            for model in self.models_dict:
                self.models_dict[model].eval()

            self.valEpoch(self.val_loader_sensor_b, 'sensor_b')

            self.resetValidationStatistics()

        self.pbar.close()


    def valEpoch(self, data_loader, sensor_name):
        val_dataset_length = data_loader.__len__()

        self.pbar = tqdm(total=val_dataset_length, unit='Batch', unit_scale=True)
        tqdm.write("Validation on " + sensor_name)

        cumulative_losses = {}

        for i_batch, sample_batched in enumerate(data_loader):
            self.valBatchStep(
                sample_batched,
                sensor_name,
                i_batch,
                cumulative_losses,
                val_dataset_length,
            )
            self.pbar.update(1)
        
        metrics = self.metrics_semseg_b.get_metrics_summary()
        metric_miou = metrics['miou']
        cumulative_losses['semseg_sensor_b_miou'] = metric_miou

        metric_acc = metrics['acc']
        cumulative_losses['semseg_sensor_b_acc'] = metric_acc
        
        self.settings.logger.info('')
        for k, v in metrics.items():
            if k != 'cm':
                self.settings.logger.info(" '{}': '{:.2f}'%.".format(k, v.item()))
        self.settings.logger.info('')
        self.pbar.close()

        self.saveValStatistics('val', sensor_name)
    

    def valBatchStep(self, sample_batched, sensor, i_batch, cumulative_losses, val_dataset_length):
        nr_reconstr_vis = 3
        vis_step_size = max(val_dataset_length // nr_reconstr_vis, 1)
        vis_reconstr_idx = i_batch // vis_step_size if (i_batch % vis_step_size) == vis_step_size - 1 else -1  # -1

        if len(sample_batched) > 2:
            file_path = sample_batched[-1]
            sample_batched = sample_batched[:-3]

        if type(sample_batched[0]) is list:
            sample_batched = [[tensor.to(self.device) for tensor in sensor_batch] for sensor_batch in sample_batched]
        else:
            sample_batched = [tensor.to(self.device) for tensor in sample_batched]

        out = self.val_step(sample_batched, sensor, i_batch, vis_reconstr_idx, file_path)

        for k, v in out[0].items():
            if k in cumulative_losses:
                cumulative_losses[k] += v
            else:
                cumulative_losses[k] = v
    

    def visualize_epoch(self):
        viz_ratio = 0.5
        return self.step_count % int(viz_ratio * self.train_loader.__len__()) == 0


    def addValidationMatrix(self, sensor):
        self.val_confusion_matrix \
            = self.val_confusion_matrix / (np.sum(self.val_confusion_matrix, axis=-1, keepdims=True) + 1e-9)

        plot_confusion_matrix = viz_utils.visualizeConfusionMatrix(self.val_confusion_matrix)

        tag = 'val/Confusion_Matrix_' + sensor
        tag.replace('sensor_a', self.settings.sensor_a_name).replace('sensor_b', self.settings.sensor_b_name)
        self.val_confusion_matrix = np.zeros([len(self.object_classes), len(self.object_classes)])


    def saveEmbeddingSpace(self):
        accumulated_features = None
        accumulated_labels = []

        for feature_data, label_list in self.validation_embeddings:
            if accumulated_features is None:
                accumulated_features = feature_data
            else:
                accumulated_features = np.concatenate([
                    accumulated_features, feature_data,
                ], axis=0)
            accumulated_labels += label_list

        self.validation_embeddings = []

    def summaries(self, losses, mode="train"):
        return

    def train_summaries(self, losses):
        nr_steps_avg = 50

        for key, value in losses.items():
            if key in self.train_statistics:
                self.train_statistics[key][0] += value
                self.train_statistics[key][1] += 1
            else:
                self.train_statistics[key] = [value, 1]

        if self.step_count % nr_steps_avg == (nr_steps_avg - 1):
            for key, _ in self.train_statistics.items():
                losses[key] = (self.train_statistics[key][0]) / self.train_statistics[key][1]
            
            self.train_statistics = {}
            self.summaries(losses, mode="train")

    def val_summaries(self, statistics, total_nr_steps):
        for k, v in statistics.items():
            tag = k.replace('sensor_a', self.settings.sensor_a_name).replace('sensor_b', self.settings.sensor_b_name)

    def img_summaries(self, tag, img, step=None):
        tag = tag.replace('sensor_a', self.settings.sensor_a_name).replace('sensor_b', self.settings.sensor_b_name)
        self.summary_writer.add_image(tag, img, step)
        self.summary_writer.flush()

    def addDifferenceLatentSpace(self, latent_repr, sensor, vis_reconstr_idx):
        """Saves the latent space representation of sensor_a and computes the difference with sensor b"""
        if sensor == "sensor_a":
            self.val_latent_space.append(latent_repr)

        elif sensor == "sensor_b":
            nrow = latent_repr.shape[0]
            pca_sensor_a = self.computePCA(self.val_latent_space[vis_reconstr_idx])

            rgb_grid = viz_utils.createRGBGrid([pca_sensor_a], nrow)
            self.img_summaries(
                'val_sensors_latent/pca_latent_space_sensor_a_' + str(vis_reconstr_idx),
                rgb_grid, self.epoch_count,
            )
            difference_map_a_b = torch.abs(
                self.val_latent_space[vis_reconstr_idx] - latent_repr
            ).sum(dim=1, keepdim=True)

            rgb_grid = viz_utils.createRGBGrid([difference_map_a_b], nrow)
            self.img_summaries(
                'val_sensors_latent/difference_paired_data_' + str(vis_reconstr_idx),
                rgb_grid, self.epoch_count,
            )

    def computePCA(self, tensor):
        n_batches, n_channels, height, width = tensor.shape
        tensor = tensor.transpose(1, 3)

        _, _, V = torch.pca_lowrank(tensor.reshape([-1, n_channels]), q=3)
        pca_tensor = torch.matmul(tensor, V)

        return pca_tensor.transpose(1, 3)

    def get_lr(self):
        return next(iter(self.lr_schedulers.values())).get_last_lr()[0]

    def resetValidationStatistics(self):
        """If wanted, needs to be implement in a child class"""
        pass

    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def val_step(self, input_batch, sensor, i_batch, vis_reconstr_idx, file_path):
        raise NotImplementedError('You need to provide a _train_step method')

    def val_train_stats_step(self, input_batch, sensor, i_batch, cumulative_losses):
        raise NotImplementedError('You need to provide a val_train_acc_step method')

    def saveValStatistics(self, mode, sensor):
        """If wanted, needs to be implement in a child class"""
        pass

    def test(self):
        pass