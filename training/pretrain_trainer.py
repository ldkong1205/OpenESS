import torch
import torchvision
import torch.nn.functional as f

import math
from tqdm import tqdm

from PIL import Image

from utils import radam
import utils.viz_utils as viz_utils

from models.style_networks import SemSegE2VID
import training.base_trainer_ov
from evaluation.metrics import MetricsSemseg
from utils.loss_functions import TaskLoss, symJSDivLoss, NCELoss
from utils.viz_utils import plot_confusion_matrix

from e2vid.utils.loading_utils import load_model
from e2vid.image_reconstructor import ImageReconstructor

from models import (
    Preprocessing,
    maskClipFeatureExtractor,
)
from models.deeplabv3 import deeplabv3_resnet50
from models.image_model import DilationFeatureExtractor

import os
from torchvision.utils import save_image

from torch.cuda.amp import autocast, GradScaler


label_colors_6classes = {
    0: (155, 55, 255),   # flat
    1: (0, 150, 255),    # background
    2: (255, 0, 0),      # object
    3: (0, 176, 80),     # vegetation
    4: (4, 50, 255),     # human
    5: (250, 188, 1),    # vehicle
}

label_colors_11classes = {
    0: (0, 150, 255),    # background
    1: (118, 118, 118),  # building
    2: (214, 220, 229),  # fence
    3: (4, 50, 255),     # person
    4: (190, 153, 153),  # pole
    5: (155, 55, 255),   # road
    6: (102, 102, 156),  # sidewalk
    7: (0, 176, 80),     # vegetation
    8: (250, 188, 1),    # car
    9: (152, 251, 152),  # wall
    10: (255, 0, 0),     # traffic-sign
}

label_colors_19classes = {
    0: (155, 55, 255),   # road
    1: (102, 102, 156),  # sidewalk
    2: (118, 118, 118),  # building
    3: (152, 251, 152),  # wall
    4: (214, 220, 229),  # fence
    5: (190, 153, 153),  # pole
    6: (115, 254, 255),  # traffic-light
    7: (255, 0, 0),      # traffic-sign
    8: (0, 176, 80),     # vegetation
    9: (152, 82, 0),     # terrain
    10: (0, 150, 255),   # sky
    11: (4, 50, 255),    # person
    12: (255, 47, 147),  # rider
    13: (250, 188, 1),   # car
    14: (0, 145, 147),   # truck
    15: (147, 144, 0),   # bus
    16: (255, 147, 0),   # train
    17: (212, 252, 122), # motorcycle
    18: (255, 64, 255),  # bicycle
}


class OpenESSPretrainModel(training.base_trainer_ov.BaseTrainer):
    def __init__(self, settings, train=True):
        self.is_training = train
        super(OpenESSPretrainModel, self).__init__(settings)
        self.do_val_training_epoch = False

    def init_fn(self):
        self.buildModels()
        self.createOptimizerDict()

        self.task_loss = TaskLoss(
            losses=self.settings.task_loss, gamma=2.0, num_classes=self.settings.semseg_num_classes,
            ignore_index=self.settings.semseg_ignore_label, reduction='mean',
        )
        self.l1_loss = torch.nn.L1Loss()
        self.kl_loss = symJSDivLoss()
        self.nce_loss = NCELoss(temperature=0.07)

        self.train_statistics = {}

        self.metrics_semseg_b = MetricsSemseg(
            self.settings.semseg_num_classes,
            self.settings.semseg_ignore_label,
            self.settings.semseg_class_names,
        )
    
    def buildModels(self):

        self.models_dict = dict()

        if self.settings.config_option == 'recon2voxel':
            self.front_end_sensor_b, _ = load_model(self.settings.path_to_model)
            if not self.settings.unfrozen_e2vid:
                for param in self.front_end_sensor_b.parameters():
                    param.requires_grad = False
                self.front_end_sensor_b.eval()

            self.input_height = math.ceil(self.settings.img_size_b[0] / 8.0) * 8
            self.input_width  = math.ceil(self.settings.img_size_b[1] / 8.0) * 8

            self.reconstructor = ImageReconstructor(
                self.front_end_sensor_b,
                self.input_height, self.input_width,
                self.settings.nr_temporal_bins_b,
                self.settings.gpu_device,
                self.settings.e2vid_config,
            )

            self.models_dict["front_sensor_b"] = self.front_end_sensor_b
            self.settings.logger.info("Loading E2VID model ...")
            self.settings.logger.info("Model loaded. Model parameters: '{:.3f} M.\n".format(get_n_params(self.front_end_sensor_b)/1e6))

            self.task_backend = SemSegE2VID(
                input_c=256,
                output_c=self.settings.semseg_num_classes,
                skip_connect=self.settings.skip_connect_task,
                skip_type=self.settings.skip_connect_task_type,
                text_embeddings_path=self.settings.text_embeddings_path,
            )
            self.models_dict["back_end"] = self.task_backend
            self.settings.logger.info("Loading Task Backend model ...")
            self.settings.logger.info("Model loaded. Model parameters: '{:.3f} M.\n".format(get_n_params(self.task_backend)/1e6))

            self.model_recon = DilationFeatureExtractor(
                image_weights=self.settings.image_weights
            )
            self.models_dict["model_recon"] = self.model_recon
            self.settings.logger.info("Loading Segmentation (Reconstruction)  model ...")
            self.settings.logger.info("Model loaded. Model parameters: '{:.3f} M.\n".format(get_n_params(self.model_recon)/1e6))


        elif self.settings.config_option == 'frame2voxel':
            self.front_end_sensor_b, _ = load_model(self.settings.path_to_model)
            if not self.settings.unfrozen_e2vid:
                for param in self.front_end_sensor_b.parameters():
                    param.requires_grad = False
                self.front_end_sensor_b.eval()

            self.input_height = math.ceil(self.settings.img_size_b[0] / 8.0) * 8
            self.input_width  = math.ceil(self.settings.img_size_b[1] / 8.0) * 8

            self.reconstructor = ImageReconstructor(
                self.front_end_sensor_b,
                self.input_height, self.input_width,
                self.settings.nr_temporal_bins_b,
                self.settings.gpu_device,
                self.settings.e2vid_config,
            )

            self.models_dict["front_sensor_b"] = self.front_end_sensor_b
            self.settings.logger.info("Loading E2VID model ...")
            self.settings.logger.info("Model loaded. Model parameters: '{:.3f} M.\n".format(get_n_params(self.front_end_sensor_b)/1e6))

            self.task_backend = SemSegE2VID(
                input_c=256,
                output_c=self.settings.semseg_num_classes,
                skip_connect=self.settings.skip_connect_task,
                skip_type=self.settings.skip_connect_task_type,
                text_embeddings_path=self.settings.text_embeddings_path,
            )
            self.models_dict["back_end"] = self.task_backend
            self.settings.logger.info("Loading Task Backend model ...")
            self.settings.logger.info("Model loaded. Model parameters: '{:.3f} M.\n".format(get_n_params(self.task_backend)/1e6))

            self.model_frame = DilationFeatureExtractor(
                image_weights=self.settings.image_weights
            )
            self.models_dict["model_frame"] = self.model_frame
            self.settings.logger.info("Loading Segmentation (Frame)  model ...")
            self.settings.logger.info("Model loaded. Model parameters: '{:.3f} M.\n".format(get_n_params(self.model_frame)/1e6))

        elif self.settings.config_option == 'frame2recon':
            self.model_recon = deeplabv3_resnet50(
                num_classes=self.settings.semseg_num_classes,
                text_embeddings_path=self.settings.text_embeddings_path,
                output_stride=self.settings.output_stride,
                pretrained_backbone=self.settings.pretrained_backbone,
            )
            self.models_dict["model_recon"] = self.model_recon
            self.settings.logger.info("Loading Segmentation (Reconstruction)  model ...")
            self.settings.logger.info("Model loaded. Model parameters: '{:.3f} M.\n".format(get_n_params(self.model_recon)/1e6))

            self.model_frame = DilationFeatureExtractor(
                image_weights=self.settings.image_weights
            )
            self.models_dict["model_frame"] = self.model_frame
            self.settings.logger.info("Loading Segmentation (Frame)  model ...")
            self.settings.logger.info("Model loaded. Model parameters: '{:.3f} M.\n".format(get_n_params(self.model_frame)/1e6))


    def createOptimizerDict(self):
        """Creates the dictionary containing the optimizer for the the specified subnetworks"""
        if not self.is_training:
            self.optimizers_dict = {}
            return

        if self.settings.config_option == 'recon2voxel':
            params_voxel = filter(lambda p: p.requires_grad, self.task_backend.parameters())
            params_recon = filter(lambda p: p.requires_grad, self.model_recon.parameters())
            if not self.settings.unfrozen_e2vid:
                params_e2vid = filter(lambda p: p.requires_grad, self.front_end_sensor_b.parameters())
                params_voxel = list(params_e2vid) + list(params_voxel)
            optimizer_voxel = torch.optim.AdamW(params_voxel, lr=self.settings.lr_voxel)
            optimizer_recon = torch.optim.AdamW(params_recon, lr=self.settings.lr_recon)

            self.optimizers_dict = {
                'optimizer_voxel': optimizer_voxel,
                'optimizer_recon': optimizer_recon,
            }

        elif self.settings.config_option == 'frame2voxel':
            params_voxel = filter(lambda p: p.requires_grad, self.task_backend.parameters())
            params_frame = filter(lambda p: p.requires_grad, self.model_frame.parameters())
            if not self.settings.unfrozen_e2vid:
                params_e2vid = filter(lambda p: p.requires_grad, self.front_end_sensor_b.parameters())
                params_voxel = list(params_e2vid) + list(params_voxel)
            optimizer_voxel = torch.optim.AdamW(params_voxel, lr=self.settings.lr_voxel)
            optimizer_frame = torch.optim.AdamW(params_frame, lr=self.settings.lr_frame)

            self.optimizers_dict = {
                'optimizer_voxel': optimizer_voxel,
                'optimizer_frame': optimizer_frame,
            }

        elif self.settings.config_option == 'frame2recon':
            params_recon = filter(lambda p: p.requires_grad, self.model_recon.parameters())
            params_frame = filter(lambda p: p.requires_grad, self.model_frame.parameters())

            optimizer_recon = torch.optim.AdamW(params_recon, lr=self.settings.lr_recon)
            optimizer_frame = torch.optim.AdamW(params_frame, lr=self.settings.lr_frame)

            self.optimizers_dict = {
                'optimizer_recon': optimizer_recon,
                'optimizer_frame': optimizer_frame,
            }

        elif self.settings.config_option == 'recon_only':
            params_recon = filter(lambda p: p.requires_grad, self.model_recon.parameters())

            optimizer_recon = radam.RAdam(params_recon, lr=self.settings.lr_recon, weight_decay=0., betas=(0., 0.999))

            self.optimizers_dict = {
                'optimizer_recon': optimizer_recon
            }

        print(self.optimizers_dict)
        self.settings.logger.info(self.optimizers_dict)

        if self.settings.use_amp:
            self.scaler = GradScaler()
            print("Using AMP training ...")
            self.settings.logger.info("Using AMP training ...")
        else:
            self.scaler = None

    def trainEpoch(self):
        self.pbar = tqdm(total=self.train_loader_sensor_b.__len__(), unit='Batch', unit_scale=True)

        num_batch_train = self.train_loader_sensor_b.__len__()

        for model in self.models_dict:
            self.models_dict[model].train()

        for i_batch, sample_batched in enumerate(self.train_loader_sensor_b):
            out = self.train_step(sample_batched)

            self.train_summaries(out[0])

            if self.settings.if_spatial_contrastive:
                contrastive_loss_spatial = out[0]['contrastive_nce_loss']
            else:
                contrastive_loss_spatial = 0

            if self.settings.if_dense_clip_supervision:
                dense_clip_supervision_loss = out[0]['dense_clip_loss']
            else:
                dense_clip_supervision_loss = 0

            if self.settings.if_sam_distillation:
                sam_distillation_loss = out[0]['sam_distillation_loss']
            else:
                sam_distillation_loss = 0

            if i_batch % 20 == 0:
                print(
                    'epoch: [{0}][{1}/{2}], ' 'loss (contrastive): {loss_contrastive:.5f}, ' 'loss (dense_clip): {loss_dense_clip:.5f}, ' 'loss (sam_distillation): {loss_sam_distillation:.5f}, '.format(
                        self.epoch_count, i_batch, num_batch_train,
                        loss_contrastive=contrastive_loss_spatial, loss_dense_clip=dense_clip_supervision_loss, loss_sam_distillation=sam_distillation_loss,
                    )
                )
                self.settings.logger.info(
                    'epoch: [{0}][{1}/{2}], ' 'loss (contrastive): {loss_contrastive:.5f}, ' 'loss (dense_clip): {loss_dense_clip:.5f}, ' 'loss (sam_distillation): {loss_sam_distillation:.5f}, '.format(
                        self.epoch_count, i_batch, num_batch_train,
                        loss_contrastive=contrastive_loss_spatial, loss_dense_clip=dense_clip_supervision_loss, loss_sam_distillation=sam_distillation_loss,
                    )
                )
            
            self.step_count += 1
            self.pbar.set_postfix(TrainLoss='{:.2f}'.format(out[-1].data.cpu().numpy()))
            self.pbar.update(1)
        
        self.pbar.close()

    def train_step(self, input_batch):
        
        optimizers_list = []

        if self.settings.config_option == 'recon2voxel':
            optimizers_list.append('optimizer_voxel')
            optimizers_list.append('optimizer_recon')
        elif self.settings.config_option == 'frame2voxel':
            optimizers_list.append('optimizer_voxel')
            optimizers_list.append('optimizer_frame')
        elif self.settings.config_option == 'frame2recon':
            optimizers_list.append('optimizer_recon')
            optimizers_list.append('optimizer_frame')
        else:
            raise NotImplementedError

        for key_word in optimizers_list:
            optimizer_key_word = self.optimizers_dict[key_word]
            optimizer_key_word.zero_grad()

        with autocast(enabled=self.settings.use_amp):
            d_final_loss, d_losses, d_outputs = self.task_train_step(input_batch)

        if self.settings.use_amp:
            self.scaler.scale(d_final_loss).backward()

            for key_word in optimizers_list:
                optimizer_key_word = self.optimizers_dict[key_word]
                self.scaler.step(optimizer_key_word)
                self.scaler.update()
        
        else:
            d_final_loss.backward()
            for key_word in optimizers_list:
                optimizer_key_word = self.optimizers_dict[key_word]
                optimizer_key_word.step()

        return d_losses, d_outputs, d_final_loss


    def task_train_step(self, batch):

        losses = {}
        outputs = {}
        t_loss = 0.

        for model in self.models_dict:
            self.models_dict[model].train()
            if model in ['front_sensor_b']:
                if not self.settings.unfrozen_e2vid:
                    self.models_dict[model].eval()
            elif model in ['model_clip']:
                self.models_dict[model].eval()


        if self.settings.config_option == 'recon2voxel':
            event = batch[0].to(self.device)
            recon = batch[2].to(self.device)
            pl = batch[3].to(self.device)
            if self.settings.if_spatial_contrastive:
                superpixels = batch[4].to(self.device)

            feat_recon = self.models_dict['model_recon'](recon)
            self.reconstructor.last_states_for_each_channel = {'grayscale': None}

            for i in range(self.settings.nr_events_data_b):
                event_tensor = event[
                    :, i * self.settings.input_channels_b:(i + 1) * self.settings.input_channels_b, :, :
                ]
                _, _, latent_real = self.reconstructor.update_reconstruction(event_tensor)
                
            loss_dense_clip_supervision, _, feat_voxel = self.trainTaskStepPretrain('sensor_b', latent_real, pl, losses)

            if self.settings.if_spatial_contrastive:
                superpixels = (
                    torch.arange(
                    0, feat_voxel.shape[0] * self.settings.superpixel_size, self.settings.superpixel_size, device=feat_voxel.device,
                )[:, None, None] + superpixels)

                superpixels_I = superpixels.flatten()
                total_pixels = superpixels_I.shape[0]
                idx_I = torch.arange(total_pixels, device=superpixels.device)

                with torch.no_grad():
                    one_hot_I = torch.sparse_coo_tensor(torch.stack((
                        superpixels_I, idx_I), dim=0),
                        torch.ones(total_pixels, device=superpixels.device),
                    )

                k = one_hot_I @ feat_voxel.permute(0, 2, 3, 1).flatten(0, 2)
                k = k / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

                q = one_hot_I @ feat_recon.permute(0, 2, 3, 1).flatten(0, 2)
                q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

                loss_contrastive_nce = self.nce_loss(k, q)
                losses['contrastive_' + 'nce' + '_loss'] = loss_contrastive_nce.detach()
                t_loss += loss_contrastive_nce
            
            if self.settings.if_dense_clip_supervision:
                t_loss += loss_dense_clip_supervision


        elif self.settings.config_option == 'frame2voxel':
            event = batch[0].to(self.device)
            frame = batch[2].to(self.device)
            pl = batch[3].to(self.device)
            if self.settings.if_spatial_contrastive:
                superpixels = batch[4].to(self.device)

            feat_frame = self.models_dict['model_frame'](frame)
            self.reconstructor.last_states_for_each_channel = {'grayscale': None}

            for i in range(self.settings.nr_events_data_b):
                event_tensor = event[
                    :, i * self.settings.input_channels_b:(i + 1) * self.settings.input_channels_b, :, :
                ]
                _, _, latent_real = self.reconstructor.update_reconstruction(event_tensor)
                
            loss_dense_clip_supervision, _, feat_voxel = self.trainTaskStepPretrain('sensor_b', latent_real, pl, losses)

            if self.settings.if_spatial_contrastive:
                superpixels = (
                    torch.arange(
                    0, feat_voxel.shape[0] * self.settings.superpixel_size, self.settings.superpixel_size, device=feat_voxel.device,
                )[:, None, None] + superpixels)

                superpixels_I = superpixels.flatten()
                total_pixels = superpixels_I.shape[0]
                idx_I = torch.arange(total_pixels, device=superpixels.device)

                with torch.no_grad():
                    one_hot_I = torch.sparse_coo_tensor(torch.stack((
                        superpixels_I, idx_I), dim=0),
                        torch.ones(total_pixels, device=superpixels.device),
                    )

                k = one_hot_I @ feat_voxel.permute(0, 2, 3, 1).flatten(0, 2)
                k = k / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

                q = one_hot_I @ feat_frame.permute(0, 2, 3, 1).flatten(0, 2)
                q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

                loss_contrastive_nce = self.nce_loss(k, q)
                losses['contrastive_' + 'nce' + '_loss'] = loss_contrastive_nce.detach()
                t_loss += loss_contrastive_nce

            if self.settings.if_dense_clip_supervision:
                t_loss += loss_dense_clip_supervision


        elif self.settings.config_option == 'frame2recon':
            frame = batch[0].to(self.device)
            recon = batch[2].to(self.device)
            pl = batch[3].to(self.device)
            if self.settings.if_spatial_contrastive:
                superpixels = batch[4].to(self.device)
            if self.settings.if_sam_distillation:
                feat_sam = batch[5].to(self.device)

            feat_frame = self.models_dict['model_frame'](frame)
            logits_recon, feat_recon = self.models_dict['model_recon'](recon)

            if self.settings.if_spatial_contrastive:
                superpixels = (torch.arange(
                    0, feat_recon.shape[0] * self.settings.superpixel_size, self.settings.superpixel_size, device=feat_recon.device,
                    )[:, None, None] + superpixels)

                superpixels_I = superpixels.flatten()
                total_pixels = superpixels_I.shape[0]
                idx_I = torch.arange(total_pixels, device=superpixels.device)

                with torch.no_grad():
                    one_hot_I = torch.sparse_coo_tensor(torch.stack((
                        superpixels_I, idx_I), dim=0),
                        torch.ones(total_pixels, device=superpixels.device),
                    )

                k = one_hot_I @ feat_recon.permute(0, 2, 3, 1).flatten(0, 2)
                k = k / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

                q = one_hot_I @ feat_frame.permute(0, 2, 3, 1).flatten(0, 2)
                q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

                loss_contrastive_nce = self.nce_loss(k, q)
                losses['contrastive_' + 'nce' + '_loss'] = loss_contrastive_nce.detach()
                t_loss += loss_contrastive_nce

            if self.settings.if_dense_clip_supervision:
                if self.settings.if_switchable_train and self.epoch_count >= 5:
                    pl = logits_recon.argmax(dim=1)
                loss_dense_clip_supervision = self.task_loss(logits_recon, pl) * self.settings.weight_task_loss
                losses['dense_clip' + '_loss'] = loss_dense_clip_supervision.detach()
                t_loss += loss_dense_clip_supervision

            if self.settings.if_sam_distillation:
                original_H, original_W = feat_recon.shape[2:]

                max_dim = max(original_H, original_W)
                resized_feature = f.interpolate(feat_sam, size=(max_dim, max_dim), mode='bilinear', align_corners=False)

                feat_sam = resized_feature[:, :, :original_H, :]

                loss_sam_distillation = torch.mean(1 - f.cosine_similarity(feat_sam, feat_recon, dim=1))
                losses['sam_distillation' + '_loss'] = loss_sam_distillation.detach()
                t_loss += loss_sam_distillation

        else:
            raise NotImplementedError

        return t_loss, losses, outputs


    def trainTaskStep(self, sensor_name, content_features, labels, losses):
        for key in content_features.keys():
            content_features[key] = content_features[key].detach()
        
        task_backend = self.models_dict["back_end"]

        pred, feat_head = task_backend(content_features)

        loss_pred = self.task_loss(pred[1], labels) * self.settings.weight_task_loss
        losses['semseg_' + sensor_name + '_loss'] = loss_pred.detach()

        return loss_pred, pred, feat_head

    def trainTaskStepPretrain(self, sensor_name, content_features, pl, losses):
        for key in content_features.keys():
            content_features[key] = content_features[key].detach()
        task_backend = self.models_dict["back_end"]
        pred, feat_head = task_backend(content_features)

        if self.settings.if_switchable_train and self.epoch_count >= 5:
            pl = pred[1].argmax(dim=1)
        
        loss_dense_clip_supervision = self.task_loss(pred[1], pl) * self.settings.weight_task_loss
        losses['dense_clip' + '_loss'] = loss_dense_clip_supervision.detach()

        return loss_dense_clip_supervision, pred, feat_head

    def visTaskStep(self, data, pred, labels, img_fake):
        pred = pred[1]
        pred_lbl = pred.argmax(dim=1)

        semseg = viz_utils.prepare_semseg(
            pred_lbl,
            self.settings.semseg_color_map,
            self.settings.semseg_ignore_label,
        )
        semseg_gt = viz_utils.prepare_semseg(
            labels,
            self.settings.semseg_color_map,
            self.settings.semseg_ignore_label,
        )

        nrow = 4
        viz_tensors = torch.cat((
            viz_utils.createRGBImage(data[:nrow], separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(self.device),
            viz_utils.createRGBImage(semseg[:nrow].to(self.device)),
            viz_utils.createRGBImage(semseg_gt[:nrow].to(self.device)),
            viz_utils.createRGBImage(img_fake[:nrow]),
        ), dim=0)

        rgb_grid = torchvision.utils.make_grid(viz_tensors, nrow=nrow)
        self.img_summaries('train/semseg_event', rgb_grid, self.step_count)

    def visualizeSensorB(self, data, content_first_sensor, labels, img_fake, paired_data, vis_reconstr_idx, sensor):
        nrow = 4
        vis_tensors = [
            viz_utils.createRGBImage(data[:nrow], separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(self.device)
        ]

        task_backend = self.models_dict["back_end"]
        pred = task_backend(content_first_sensor)
        pred = pred[1]
        pred_lbl = pred.argmax(dim=1)

        labels = f.interpolate(labels.float().unsqueeze(1), size=(self.input_height, self.input_width), mode='nearest').squeeze(1).long()

        semseg = viz_utils.prepare_semseg(
            pred_lbl,
            self.settings.semseg_color_map,
            self.settings.semseg_ignore_label,
        )
        vis_tensors.append(viz_utils.createRGBImage(semseg[:nrow].to(self.device)))

        semseg_gt = viz_utils.prepare_semseg(
            labels,
            self.settings.semseg_color_map,
            self.settings.semseg_ignore_label,
        )
        vis_tensors.append(viz_utils.createRGBImage(semseg_gt[:nrow]).to(self.device))
        vis_tensors.append(viz_utils.createRGBImage(img_fake[:nrow]).to(self.device))

        rgb_grid = torchvision.utils.make_grid(torch.cat(vis_tensors, dim=0), nrow=nrow)
        self.img_summaries(
            'val_' + sensor + '/reconst_input_' + sensor + '_' + str(vis_reconstr_idx),
            rgb_grid, self.epoch_count,
        )


    def val_step(self, batch, sensor, i_batch, vis_reconstr_idx, file_path):
        """Calculates the performance measurements based on the input"""
        losses = {}

        gt = batch[1]

        if self.settings.config_option == 'recon2voxel' or self.settings.config_option == 'frame2voxel':
            event = batch[0]

            self.reconstructor.last_states_for_each_channel = {'grayscale': None}
            for i in range(self.settings.nr_events_data_b):
                event_tensor = event[
                    :, i * self.settings.input_channels_b:(i + 1) * self.settings.input_channels_b, :, :
                ]
                _, _, content_first_sensor = self.reconstructor.update_reconstruction(event_tensor)

            task_backend = self.models_dict["back_end"]
            pred, _ = task_backend(content_first_sensor)
            pred = pred[1]

        elif self.settings.config_option == 'frame2recon':
            recon = batch[2]

            pred, _ = self.models_dict['model_recon'](recon)

        pred_lbl = pred.argmax(dim=1)  # [bs, 440, 640]
        loss_pred = self.task_loss(pred, target=gt)
        losses['semseg_' + sensor + '_loss'] = loss_pred.detach()
        self.metrics_semseg_b.update_batch(pred_lbl, gt)
        
        return losses, None

    def resetValidationStatistics(self):
        self.metrics_semseg_b.reset()


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp