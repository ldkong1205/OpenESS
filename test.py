import argparse

from config.settings import Settings
from training.openess_trainer import OpenESSModel
from training.sup_only_trainer import SupOnlyModel
from training.pretrain_trainer import OpenESSPretrainModel
from training.finetune_trainer import OpenESSFineTuneModel
from training.linear_probe_trainer import OpenESSLinearProbeModel

import numpy as np
import torch
import random
import os

seed_value = 1205
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument(
        '--settings_file', help='Path to settings yaml', default='config/finetunes/DSEC/sam/frame2recon_fcclip_sam_1.yaml')

    args = parser.parse_args()

    settings_filepath = args.settings_file
    settings = Settings(settings_filepath, generate_log=True)

    if settings.if_supervised_only:
        trainer = SupOnlyModel(settings=settings)
        return
    elif settings.if_pretraining:
        trainer = OpenESSPretrainModel(settings=settings)
        return
    elif settings.if_finetuning:
        trainer = OpenESSFineTuneModel(settings=settings)
        trainer.valEpochs()
    elif settings.if_linear_probing:
        trainer = OpenESSLinearProbeModel(settings=settings)
        trainer.valEpochs()
    else:
        return

if __name__ == "__main__":
    main()