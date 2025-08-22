from pathlib import Path

from DSEC.dataset.provider import DatasetProvider


def DSECEvents(
        dsec_dir: str,
        nr_events_data:int = 1,
        delta_t_per_data:int = 50,
        nr_events_window:int = -1,
        augmentation:bool = False,
        mode:str = 'train',
        task:str = 'segmentation',
        event_representation:str = 'voxel_grid',
        nr_bins_per_data:int = 5,
        require_paired_data:bool = False,
        separate_pol:bool = False,
        normalize_event:bool = False,
        semseg_num_classes:int = 11,
        fixed_duration:bool = False,
        resize:bool = False,
        config_option:str = '',
        pl_sources:str = '',
        superpixel_sources:str = '',
        skip_ratio:int = 1,
        if_sam_distillation:bool = False,
    ):
    """
    Creates an iterator over the EventScape dataset.

    :param root: path to dataset root
    :param height: height of dataset image
    :param width: width of dataset image
    :param nr_events_window: number of events summed in the sliding histogram
    :param augmentation: flip, shift and random window start for training
    :param mode: 'train', 'test' or 'val'
    """
    dsec_dir = Path(dsec_dir)
    assert dsec_dir.is_dir()

    dataset_provider = DatasetProvider(
        dsec_dir,
        mode,
        event_representation=event_representation,
        nr_events_data=nr_events_data,
        delta_t_per_data=delta_t_per_data,
        nr_events_window=nr_events_window,
        nr_bins_per_data=nr_bins_per_data,
        require_paired_data=require_paired_data,
        normalize_event=normalize_event,
        separate_pol=separate_pol,
        semseg_num_classes=semseg_num_classes,
        augmentation=augmentation,
        fixed_duration=fixed_duration,
        resize=resize,
        config_option=config_option,
        pl_sources=pl_sources,
        superpixel_sources=superpixel_sources,
        skip_ratio=skip_ratio,
        if_sam_distillation=if_sam_distillation,
    )
    
    if mode == 'train':
        train_dataset = dataset_provider.get_train_dataset()
        return train_dataset
    else:
        val_dataset = dataset_provider.get_val_dataset()
        return val_dataset