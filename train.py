#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import torch
import warnings


from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager
from src.utils.reproducibility import set_reproducible_seed

from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")

import os
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Simple single-node init for Colab
    cfg.DIST_INIT_PATH = "env://"

    # Setup output dir: OUTPUT_DIR / DATA.NAME / FEATURE / lr_wd / seed_<n>
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}"
    )
    run_name = "run1" if cfg.SEED is None else f"seed_{int(cfg.SEED)}"
    output_path = os.path.join(output_dir, output_folder, run_name)

    PathManager.mkdirs(output_path)
    cfg.OUTPUT_DIR = output_path

    cfg.freeze()
    return cfg



def get_loaders(cfg, logger):
    if cfg.DATA.USE_TRAINVAL:
        logger.info("Loading combined train+validation data...")
        train_loader = data_loader.construct_trainval_loader(cfg)
    else:
        logger.info("Loading training data with an independent validation split...")
        train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    # not really needed for vtab
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader


def train(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here

    # Seed before dataset, DataLoader, and model construction.
    applied_seed = set_reproducible_seed(cfg.SEED, deterministic=True)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)
    logger.info(f"Reproducibility seed: {applied_seed}")

    if not train_loader:
        raise RuntimeError("No training loader was constructed")
    return trainer.train_classifier(train_loader, val_loader, test_loader)


def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
