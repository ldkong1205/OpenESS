# Getting Started
- [Getting Started](#getting-started)
  - [Pretraining](#pretraining)
    - [Configuration Options](#configuration-options)
    - [Training Notes](#training-notes)
  - [Linear Probing](#linear-probing)
    - [Configuration Options](#configuration-options-1)
    - [Training Notes](#training-notes-1)
  - [Finetuning](#finetuning)
    - [Configuration Options](#configuration-options-2)
    - [Training Notes](#training-notes-2)


## Pretraining
Configuration files (`.yaml`) for pretraining are provided in the `config/pretrain` directory. We include settings for two datasets, **DSEC** and **DDD17**, where superpixels (constructed with either SAM or SLIC) and event representations (via reconstruction or voxelization) are specified.  

⚠️ **Important**: Please ensure that the dataset directory structure strictly follows our required format; otherwise, the training pipeline may fail.

---

### Configuration Options

The main settings related to our paper can be adjusted directly in the YAML files. Below are the configurable options and their valid ranges:

- **Log directory**: `dir -> log`  
- **Event representation**: `clip -> config_option`  
  Options: `{'recon2voxel', 'frame2voxel', 'frame2recon'}`
- **Superpixel source**: `clip -> superpixel_sources`  
  Options: `{'sp_sam_rgb', 'sp_slic_rgb'}`
- **Superpixel size**: `clip -> superpixel_size`  
- **Image branch pretrained model**: `clip -> image_weights`  
  Options: `{'moco_v1', 'moco_v2', 'swav', 'deepcluster_v2', 'dino'}`
- **Pseudo-label generation model**: `clip -> pl_sources`  
  Options: `{'pl_fcclip_rgb', 'pl_maskclip_rgb'}`

---

### Training Notes

- Currently, **only single-GPU training is supported**.  
- To launch pretraining, run the following command with your selected configuration file:

```shell
python train.py --settings_file ${SELECTED_CONFIG_PATH}
```

## Linear Probing
Linear probing is used for a **lightweight evaluation of pretrained weights**, where only a small set of linear layers are trainable while the backbone remains frozen. This provides a quick validation of the representation quality learned during pretraining.  

Configuration files (`.yaml`) for linear probing are provided in the `config/linear_probe` directory. Similar to **Pretraining**, we include settings for both **DSEC** and **DDD17** datasets, with superpixels (constructed using SAM or SLIC) and event representations (via reconstruction or voxelization).  

---

### Configuration Options

The main settings related to our paper can be adjusted directly in the YAML files. Key configurable parameters include:

- **Log directory**: `dir -> log`  
- **Event representation**: `clip -> config_option`  
  Options: `{'recon2voxel', 'frame2voxel', 'frame2recon'}`
- **Superpixel source**: `clip -> superpixel_sources`  
  Options: `{'sp_sam_rgb', 'sp_slic_rgb'}`
- **Pretrained weights**: `clip -> pre_trained_backbone`  
  (path to the corresponding pretrained log directory)

---

### Training Notes

- Currently, **only single-GPU training is supported**.  
- To run linear probing, use the following command:

```shell
python train.py --settings_file ${SELECTED_CONFIG_PATH}
```

## Finetuning
In the **Finetuning stage**, we evaluate the performance of the model under different levels of supervision, i.e., varying the proportion of ground-truth labels used for training.  

Configuration files (`.yaml`) for finetuning are provided in the `config/finetunes` directory. Similar to **Pretraining**, we include settings for both **DSEC** and **DDD17** datasets, with superpixels (constructed using SAM or SLIC) and event representations (via reconstruction or voxelization).  

### Configuration Options

The main settings related to our paper can be adjusted directly in the YAML files. Key configurable parameters include:

- **Log directory**: `dir -> log`  
- **Event representation**: `clip -> config_option`  
  Options: `{'recon2voxel', 'frame2voxel', 'frame2recon'}`
- **Superpixel source**: `clip -> superpixel_sources`  
  Options: `{'sp_sam_rgb', 'sp_slic_rgb'}`
- **Pretrained weights**: `clip -> pretrained_file`  
  (path to the corresponding pretrained log directory)
- **Label ratio**: `clip -> skip_ratio`  
  Options: `{'100', '20', '10', '5', '1'}`  
  Mapping: `{ '1': 100%, '5': 20%, '10': 10%, '20': 5%, '100': 1% }`  
  (representing the percentage of ground-truth labels used)
- **Training epochs**: `clip -> num_epochs`  
  Typically, fewer labels require more training epochs.

---

### Training Notes

- Currently, **only single-GPU training is supported**.  
- To launch finetuning, run the following command:

```shell
python train.py --settings_file ${SELECTED_CONFIG_PATH}
```
