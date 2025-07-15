######################
# Model evaluation   #
# Calculating scores #   
# Saving outputs     #
######################

# modified from constrained-downscaling/training.py

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

from utils import process_for_training, process_for_eval, run_diffusion_model, run_flow_matching, is_diffusion, use_constraints

device = 'cuda'
        
def evaluate_model(data, args, model):
    
    full_pred = torch.zeros(data["val_shape_out"]) 

    with tqdm(data["val"], unit="batch") as tepoch:     
        for i, batchdata in enumerate(tepoch): 
            if use_constraints(args):
                inputs, targets, inputs_val_original = batchdata
                inputs, targets, inputs_original = process_for_training(inputs, targets, inputs_val_original)
            else:
                inputs, targets = batchdata
                inputs_original = None
                inputs, targets = process_for_training(inputs, targets)
            
            if is_diffusion(args):
                outputs = run_diffusion_model(inputs, model, args, device, inputs_original)         
            else:
                outputs = run_flow_matching(inputs, model, args, device, inputs_original)  

            outputs, targets = process_for_eval(outputs, targets, args, inputs) 
                
            full_pred[i*args.batch_size:i*args.batch_size+outputs.shape[0],...] = outputs.detach().cpu()

    torch.save(full_pred, args.data_path+'prediction/'+args.dataset+'_'+args.model_id+ '_' + args.test_val_train+'_'+args.constraints+'.pt')
    calculate_scores_simple(args)

def calculate_scores(args):
    input_val = torch.load(args.data_path+args.dataset+'/'+ args.test_val_train+'/input_'+ args.test_val_train+'.pt')
    target_val = torch.load(args.data_path+args.dataset+'/'+ args.test_val_train+'/target_'+ args.test_val_train+'.pt')

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
        for i,(lr, hr) in enumerate(tqdm(val_data)):
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
    target_val = torch.load(args.data_path+args.dataset+'/'+ args.test_val_train+'/target_'+ args.test_val_train+'.pt')
    if target_val.ndim == 5:
        target_val = target_val.squeeze(2) # 5D to 4D

    pred = np.zeros(target_val.shape)
    
    l2_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()

    mean_abs_percentage_error = MeanAbsolutePercentageError()
    
    pred = torch.load(args.data_path+'prediction/'+args.dataset+'_'+args.model_id+ '_' + args.test_val_train+'_'+args.constraints+'.pt')

    for j in tqdm(range(args.dim_channels)):
        mae = l1_crit(pred[:,j,...], target_val[:, j,...]).item()
        mse = l2_crit(pred[:,j,...], target_val[:, j,...]).item()
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
    w = csv.writer(open(args.data_path+args.model_id+'_'+args.dataset+'_'+args.constraints+'_var_'+str(j+1)+'.csv', 'w'))      
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





