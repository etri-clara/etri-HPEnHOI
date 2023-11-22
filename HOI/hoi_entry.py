# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import logging

from utils.arguments import load_opt_command

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args=None):
    '''
    [Main function for the entry point]
    1. Set environment variables for distributed training.
    2. Load the config file and set up the trainer.
    '''


    opt, cmdline_args = load_opt_command(args)

    command = cmdline_args.command

    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir

    # update_opt(opt, command)
    world_size = 1
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

    if opt['TRAINER'] == 'hdecoder':
        from trainer import HDecoder_Trainer as Trainer
    else:
        assert False, "The trainer type: {} is not defined!".format(opt['TRAINER'])

    if opt["WANDB"]:
        import wandb
        wdb = wandb
        wdb.init(
            config=opt,
            project="X-Decoder_HOI_230817",
            name=f'MODEL({opt["MODEL"]["NAME"] + "_" + opt["MODEL"]["TYPE"]})_EPOCHS({opt["SOLVER"]["MAX_NUM_EPOCHS"]})_SCHEDULER({opt["SOLVER"]["LR_SCHEDULER_NAME"]}_{opt["SOLVER"]["LR_STEP"]})_LOG_EVERY({opt["LOG_EVERY"]})',
        )
        opt["WANDB"] = wdb

    trainer = Trainer(opt)
    os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'

    if command == "train":
        trainer.train()
    elif command == "evaluate":
        trainer.eval()
    else:
        raise ValueError(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
    sys.exit(0)
