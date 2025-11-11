######################
# Model evaluation   #
# Calculating scores #   
# Saving outputs     #
######################

# modified from constrained-downscaling/training.py

import os
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import csv
import numpy as np
from torch.utils.data import TensorDataset
from torchmetrics.functional import multiscale_structural_similarity_index_measure, structural_similarity_index_measure
from torchmetrics.regression import MeanAbsolutePercentageError
from skimage import transform
import sys

from utils import process_for_training, process_for_eval, recon_patches_nordics, run_diffusion_model, run_flow_matching, is_diffusion, use_constraints, month_indices, is_nordics

device = 'cuda'
        
def evaluate_model(data, args, model):
    
    full_pred = torch.zeros(data["val_shape_out"]) 

    with tqdm(data["val"], unit="batch", file=sys.stdout) as tepoch:     
        for i, batchdata in enumerate(tepoch): 
            if use_constraints(args): # ERA5 only
                inputs, targets, inputs_val_original = batchdata
                inputs, targets, inputs_original = process_for_training(inputs, targets, inputs_val_original)
            else:
                if is_nordics(args):
                    inputs = batchdata["inputs"]       
                    targets = batchdata["targets"]
                    labels = batchdata.get("labels", None)
                    aux = batchdata.get("aux", None)
                else:
                    inputs, targets = batchdata
                inputs_original = None
                if args.nordics_constants == "True": 
                    inputs, targets, labels, aux = process_for_training(inputs, targets, labels=labels, aux=aux)
                else:
                    inputs, targets = process_for_training(inputs, targets)
            
            if is_nordics(args):
                inputs = torch.nan_to_num(inputs, nan=0.0) # remove nans before inputting to model
            
            if is_diffusion(args):
                outputs = run_diffusion_model(inputs, model, args, device, inputs_original)         
            else:
                if args.nordics_constants == "True":
                    outputs = run_flow_matching(inputs, model, args, device, inputs_original, labels=labels, aux=aux)  
                else:
                    outputs = run_flow_matching(inputs, model, args, device, inputs_original)  

            outputs, targets = process_for_eval(outputs, targets, args, inputs) 
                
            full_pred[i*args.batch_size:i*args.batch_size+outputs.shape[0],...] = outputs.detach().cpu()
            
    if is_nordics(args) and args.nordics_patch_eval != "True":
        full_pred = recon_patches_nordics(full_pred, args)

    if is_nordics(args):
        if args.nordics_patch_eval == "True":
            if args.nordics_start == args.nordics_end and args.nordics_month > 0:
                torch.save(full_pred, f"{args.data_path}prediction/patches_{args.dataset}_{args.model_id}_{args.test_val_train}_{args.constraints}_steps={args.steps}_ode={args.ode_order}_year={args.nordics_start}_month={args.nordics_month}.pt")
            else:
                torch.save(full_pred, f"{args.data_path}prediction/patches_{args.dataset}_{args.model_id}_{args.test_val_train}_{args.constraints}_steps={args.steps}_ode={args.ode_order}_years={args.nordics_start}-{args.nordics_end}.pt")
        else:
            torch.save(full_pred, f"{args.data_path}prediction/{args.dataset}_{args.model_id}_{args.test_val_train}_{args.constraints}_steps={args.steps}_ode={args.ode_order}_years={args.nordics_start}-{args.nordics_end}.pt")
    else:
        torch.save(full_pred, f"{args.data_path}prediction/{args.dataset}_{args.model_id}_{args.test_val_train}_{args.constraints}_steps={args.steps}_ode={args.ode_order}.pt")
    calculate_scores_simple(args)


# TODO: add nordics, update paths
def calculate_scores(args):
    input_val = torch.load(args.data_path+args.dataset+'/'+ args.test_val_train+'/input_'+ args.test_val_train+'.pt')
    target_val = torch.load(args.data_path+args.dataset+'/'+ args.test_val_train+'/target_'+ args.test_val_train+'.pt')

    # for /era5_temp_precip_data/, if dim_channels = 1: temperature, if dim_channels = 2: temperature, precipitation
    input_val = input_val[:, 0:args.dim_channels, ...]
    target_val = target_val[:, 0:args.dim_channels, ...]

    val_data = TensorDataset(input_val, target_val)
    pred = np.zeros(target_val.shape)
    max_val = target_val.max()
    min_val = target_val.min()
    mse = 0
    mae = 0
    mape = 0
    ssim = 0
    mean_bias = 0
    mean_abs_bias = 0
    mass_violation = 0
    ms_ssim = 0
    corr = 0
    neg_mean = 0
    neg_num = 0
    
    l2_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()

    mean_abs_percentage_error = MeanAbsolutePercentageError()
    
    pred = torch.load(args.data_path+'prediction/'+args.dataset+'_'+args.model_id+ '_' + args.test_val_train+'_'+args.constraints+'.pt')

    pred = pred.detach().cpu().numpy() 

    for j in range(args.dim_channels):
        for i,(lr, hr) in enumerate(tqdm(val_data, file=sys.stdout)):
            im = lr.numpy()
            mse += l2_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item()
            mae += l1_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item()
            mape += mean_abs_percentage_error(torch.Tensor(pred[i,j,...]), hr[j,...]).item()
            mean_bias += torch.mean( hr[j,...]-torch.Tensor(pred[i,j,...]))
            mean_abs_bias += torch.abs(torch.mean( hr[j,...]-torch.Tensor(pred[i,j,...])))
            corr += pearsonr(torch.Tensor(pred[i,j,...]).flatten(),  hr[j,...].flatten())

            ms_ssim += multiscale_structural_similarity_index_measure(torch.Tensor(pred[i,j,...])[None, None, ...], # [H, W] to 4D Tensor [B, C, H, W] for function
                                                                    hr[j,...][None, None, ...], # [H, W] to 4D Tensor [B, C, H, W] for function
                                                                    data_range=max_val-min_val, kernel_size=11, 
                                                                    betas=(0.2856, 0.3001, 0.2363))
            ssim += structural_similarity_index_measure(torch.Tensor(pred[i,j,...])[None, None, ...], # [H, W] to 4D Tensor [B, C, H, W] for function
                                                        hr[j,...][None, None, ...], # [H, W] to 4D Tensor [B, C, H, W] for function
                                                        data_range=max_val-min_val, kernel_size=11)
            
            neg_num += np.sum(pred[i,j,...] < 0)
            neg_mean += np.sum(pred[pred < 0])/(pred.shape[-1]*pred.shape[-1])

            mass_violation += np.mean( np.abs(transform.downscale_local_mean(pred[i,j,...], (args.upsampling_factor,args.upsampling_factor)) - im[j,...]))

        mse *= 1/input_val.shape[0]
        mae *= 1/input_val.shape[0]
        mape *= 1/input_val.shape[0]
        ssim *= 1/input_val.shape[0]
        mean_bias *= 1/input_val.shape[0]
        mean_abs_bias *= 1/input_val.shape[0]
        corr *= 1/input_val.shape[0]
        ms_ssim *= 1/input_val.shape[0]
        neg_mean *= 1/input_val.shape[0]
        mass_violation *= 1/input_val.shape[0]
        psnr = calculate_pnsr(mse, target_val.max() )   
        rmse = torch.sqrt(torch.Tensor([mse])).numpy()[0]
        ssim = float(ssim.numpy())
        ms_ssim =float( ms_ssim.numpy())
        psnr = psnr.numpy()
        corr = float(corr.numpy())
        mean_bias = float(mean_bias.numpy())
        mean_abs_bias = float(mean_abs_bias.numpy())
        scores = {'MSE':mse, 'RMSE':rmse, 'PSNR': psnr[0], 'MAE':mae, 'MAPE': mape, 'SSIM':ssim,  'MS SSIM': ms_ssim, 'Pearson corr': corr, 'Mean bias': mean_bias, 'Mean abs bias': mean_abs_bias, 'Mass_violation': mass_violation, 'neg mean': neg_mean, 'neg num': neg_num}
        print(scores)
        create_report(scores, args, j)

# faster alternative with less metrics
def calculate_scores_simple(args):

    # load prediction
    if is_nordics(args):
        if args.nordics_patch_eval == "True":
            if args.nordics_start == args.nordics_end and args.nordics_month > 0:
                pred = torch.load(f"{args.data_path}prediction/patches_{args.dataset}_{args.model_id}_{args.test_val_train}_{args.constraints}_steps={args.steps}_ode={args.ode_order}_year={args.nordics_start}_month={args.nordics_month}.pt")
            else:
                pred = torch.load(f"{args.data_path}prediction/patches_{args.dataset}_{args.model_id}_{args.test_val_train}_{args.constraints}_steps={args.steps}_ode={args.ode_order}_years={args.nordics_start}-{args.nordics_end}.pt")
        else:
            pred = torch.load(f"{args.data_path}prediction/{args.dataset}_{args.model_id}_{args.test_val_train}_{args.constraints}_steps={args.steps}_ode={args.ode_order}_years={args.nordics_start}-{args.nordics_end}.pt")
    else:
        pred = torch.load(f"{args.data_path}prediction/{args.dataset}_{args.model_id}_{args.test_val_train}_{args.constraints}_steps={args.steps}_ode={args.ode_order}.pt")
    
    # load target
    if args.nordics == "True":
        target_val = np.load(f"{args.data_path}{args.dataset}/eobs/eobs_patches_{args.nordics_start}.npy") # TODO: support multiple years
        target_val = torch.from_numpy(target_val.copy()).to(pred.dtype)
        if args.nordics_time == "True":
            target_val = target_val[2:, :]

        if args.nordics_start == args.nordics_end and args.nordics_month > 0:
            # only evaluate a single month
            month_start, month_end = month_indices(args)
            target_val = target_val[month_start:month_end, :]

        if args.nordics_patch_eval != "True":
            target_val = recon_patches_nordics(target_val, args)
            # TODO: save target patches (if not existing)

    else:
        target_val = torch.load(args.data_path+args.dataset+'/'+ args.test_val_train+'/target_'+ args.test_val_train+'.pt')

    # if dim_channels = 1: temperature, if dim_channels = 2: temperature, precipitation
    target_val = target_val[:, 0:args.dim_channels, ...]
    
    l2_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()

    mean_abs_percentage_error = MeanAbsolutePercentageError()
    
    if is_nordics(args):
        # remove nan pixels for mape
        target_flat = target_val.flatten()
        pred_flat = pred.flatten()
        mask = torch.isfinite(target_flat) & torch.isfinite(pred_flat)

        # remove nans before calulating scores
        pred = torch.nan_to_num(pred, nan=0.0)
        target_val = torch.nan_to_num(target_val, nan=0.0)

    for j in tqdm(range(args.dim_channels), file=sys.stdout):
        mae = l1_crit(pred[:,j,...], target_val[:, j,...]).item()
        mse = l2_crit(pred[:,j,...], target_val[:, j,...]).item()
        if is_nordics(args):
            mape = mean_abs_percentage_error(pred_flat[mask], target_flat[mask]).item() # TODO: multiple C
        else:
            mape = mean_abs_percentage_error(pred[:,j,...], target_val[:, j,...]).item()
        mean_bias = torch.mean( target_val[:,j,...]-pred[:,j,...])
        mean_abs_bias = torch.abs(mean_bias)

        psnr = calculate_pnsr(mse, target_val.max()) 
        psnr = psnr.numpy()  
        rmse = torch.sqrt(torch.Tensor([mse])).numpy()[0]
        mean_bias = float(mean_bias.numpy())
        mean_abs_bias = float(mean_abs_bias.numpy())
        scores = {'MSE':mse, 'RMSE':rmse, 'PSNR': psnr[0], 'MAE':mae, 'MAPE': mape, 'Mean bias': mean_bias, 'Mean abs bias': mean_abs_bias}
        print(scores)
        create_report(scores, args, j)


def calculate_pnsr(mse, max_val):
    return 20 * torch.log10(max_val / torch.sqrt(torch.Tensor([mse])))
                                            
def create_report(scores, args, j):
    args_dict = args_to_dict(args)
    # combine scorees and args dict
    args_scores_dict = args_dict | scores
    # save dict
    save_dict(args_scores_dict, args, j)
    
def args_to_dict(args):
    return vars(args)
                                            
def save_dict(dictionary, args, j):
    if is_nordics(args):
        if args.nordics_start == args.nordics_end and args.nordics_month > 0:
            file = f"{args.data_path}{args.model_id}_{args.dataset}_{args.constraints}_var_{j+1}_steps={args.steps}_ode={args.ode_order}_year={args.nordics_start}_month={args.nordics_month}.csv"
        else:
            file = f"{args.data_path}{args.model_id}_{args.dataset}_{args.constraints}_var_{j+1}_steps={args.steps}_ode={args.ode_order}_years={args.nordics_start}-{args.nordics_end}.csv"
    else:
        file = f"{args.data_path}{args.model_id}_{args.dataset}_{args.constraints}_var_{j+1}_steps={args.steps}_ode={args.ode_order}.csv"    

    w = csv.writer(open(file, 'w'))         
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])

def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val





