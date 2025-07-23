###################
# Running program #
###################

import argparse

from lightning_module import run_training, evaluate
from utils import load_data
from evaluation import calculate_scores, calculate_scores_simple
import os

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="era5_temp_precip_data", help="choose a data set to use")
    parser.add_argument("--model", default="flow")
    parser.add_argument("--bias_flow", default="no")
    parser.add_argument("--model_id", default="test")
    parser.add_argument("--data_path", default = "./data/")
    parser.add_argument("--model_path", default = "./model_checkpoints/")
    parser.add_argument("--upsampling_factor", default=4, type=int)
    parser.add_argument("--constraints", default="none", help="none, back_projection, bilateral, additive") 
    parser.add_argument("--ode_order", default=1, type=int, help="1st or 2nd order")
    parser.add_argument("--dim_channels", default=2, type=int, help="number of variables")
    parser.add_argument("--number_channels", default=128, type=int)
    parser.add_argument("--number_residual_blocks", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, help="learning rate", type=float)
    parser.add_argument("--batch_size", default=32, type=int) 
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--test_val_train", default="val")
    parser.add_argument("--training_evalonly", default="training")
    parser.add_argument("--use_validation", default="yes")

    parser.add_argument("--gpus", default=1, type=int,
                        help='number of GPUs per node')
    parser.add_argument("--nodes", default=1, type=int,
                        help='number of nodes')
    
    return parser.parse_args()

def main(args):
    if not os.path.exists('./model_checkpoints'):
        os.makedirs('./model_checkpoints')
    if not os.path.exists('./data/prediction'):
        os.makedirs('./data/prediction')

    if args.training_evalonly == "metrics":
        #metrics+report for existing prediction
        calculate_scores_simple(args)
    else:
        print("loading data...", flush=True)
        data = load_data(args)
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