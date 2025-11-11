###################
# Running program #
###################

import os
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import argparse
import torch
import numpy as np

from lightning_module import run_training, evaluate
from utils import load_data, load_data_lazy, load_data_lazy_nordics, is_nordics
from evaluation import calculate_scores, calculate_scores_simple

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="era5_temp_precip_data", help="choose a data set to use")
    parser.add_argument("--model", default="flow")
    parser.add_argument("--steps", default=5, type=int)
    parser.add_argument("--bias_flow", default="no")
    parser.add_argument("--model_id", default="test")
    parser.add_argument("--data_path", default = "./data/")
    parser.add_argument("--model_path", default = "./model_checkpoints/")
    parser.add_argument("--upsampling_factor", default=4, type=int)
    parser.add_argument("--constraints", default="none", help="none, back_projection, bilateral, additive") 
    parser.add_argument("--ode_order", default=1, type=int, help="1st or 2nd order")
    parser.add_argument("--dim_channels", default=1, type=int, help="number of variables")
    parser.add_argument("--number_channels", default=128, type=int)
    parser.add_argument("--number_residual_blocks", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, help="learning rate", type=float)
    parser.add_argument("--batch_size", default=32, type=int) 
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--test_val_train", default="val")
    parser.add_argument("--training_evalonly", default="training")
    parser.add_argument("--use_validation", default="yes")
    parser.add_argument("--sharded_data", default="no") # for ERA5 (always for Nordics)

    parser.add_argument("--nordics", default="False")
    parser.add_argument("--nordics_start", default=2000, type=int)
    parser.add_argument("--nordics_end", default=2020, type=int)
    parser.add_argument("--nordics_month", default=0, type=int)
    parser.add_argument("--nordics_patch_eval", default="False")

    parser.add_argument("--nordics_constants", default="False")
    parser.add_argument("--nordics_time", default="False")

    parser.add_argument("--gpus", default=1, type=int,
                        help='number of GPUs per node')
    parser.add_argument("--nodes", default=1, type=int,
                        help='number of nodes')
    parser.add_argument("--cpus", default="9", type=int,
                        help="used to control num_workers in DataLoader")
    
    return parser.parse_args()

def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.data_path+'/prediction'):
        os.makedirs(args.data_path+'/prediction')
    

    if is_nordics(args): 
    # NGCD + E-OBS data
    # TODO: diffusion support (loss, eval, ...)
        if args.training_evalonly == "sample_test":
            print("Save sample...")
            data = load_data_lazy_nordics(args)
            batch = next(iter(data["train"]))
            torch.save(batch, "batch.pt")
            print("Closing...")
            return
    
        if args.training_evalonly == "metrics":
                # metrics+report for existing prediction
                calculate_scores_simple(args)
        else:
            data = load_data_lazy_nordics(args)
            if args.training_evalonly == "training" or args.training_evalonly == "resume": 
                batch = next(iter(data["train"]))
                xb = batch["inputs"]       
                yb = batch["targets"]
            elif args.training_evalonly == "eval" or args.training_evalonly == "metrics": 
                batch = next(iter(data["val"]))
                xb = batch["inputs"]       
                yb = batch["targets"]
            print("nordic data batch shape")
            print(xb.shape, yb.shape, xb.dtype, type(xb))

            if args.training_evalonly == "training":
                # run training
                run_training(args, data)
            elif args.training_evalonly == "resume":
                # resume training
                run_training(args, data, resume=True) 
            else:       
                # run evaluation
                evaluate(args, data) 

    else: 
    # ERA5 data
        if args.training_evalonly == "metrics":
            #metrics+report for existing prediction
            calculate_scores_simple(args)
        else:
            print("loading data...", flush=True)
            # data = load_data(args)
            if args.sharded_data == "yes":
                data = load_data_lazy(args)
                xb, yb = next(iter(data["train"]))
                print("sharded_data batch shape")
                print(xb.shape, yb.shape)
            else:
                data = load_data(args)
                xb, yb, *rest = next(iter(data["train"]))
                print("data batch shape")
                print(xb.shape, yb.shape)

            if args.training_evalonly == "training":
                #run training
                run_training(args, data) 

            elif args.training_evalonly == "resume":
                #resume training
                run_training(args, data, resume=True) 

            else:       
                #run evaluation
                evaluate(args, data) 
        
if __name__ == '__main__':
    args = add_arguments()
    main(args)