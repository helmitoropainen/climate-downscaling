######################################
# Data loading and processing        #
# Model training loss and inference  #
######################################

import os
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
# import kornia as K
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
import psutil
import json
import sys

from sharded_dataset import ShardedDataset, ShardedDatasetNordics

device = 'cuda'

# modified from constrained-downscaling/utils.py
# func load_data() for ERA5 data (memory-heavy for multi-gpu)
def load_data(args):
    print("load_data()", flush=True)

    print("total memory:", psutil.virtual_memory().total / 1e9, "GB", flush=True)
    print("starting load_data()...", flush=True)
    print("initial memory usage:", psutil.virtual_memory().used / 1e9, "GB", flush=True)
    print("available memory:", psutil.virtual_memory().available / 1e9, "GB", flush=True)

    input_train = torch.load(args.data_path+args.dataset+'/train/input_train.pt')
    target_train = torch.load(args.data_path+args.dataset+'/train/target_train.pt')
    if load_val(args):
        if args.test_val_train == 'test':
            input_val = torch.load(args.data_path+args.dataset+'/test/input_test.pt')
            target_val = torch.load(args.data_path+args.dataset+'/test/target_test.pt')
        elif args.test_val_train == 'val':
            input_val = torch.load(args.data_path+args.dataset+'/val/input_val.pt')
            target_val = torch.load(args.data_path+args.dataset+'/val/target_val.pt')
        elif args.test_val_train == 'train':
            input_val = input_train
            target_val = target_train

    print("data loaded...", flush=True)
    print("memory usage:", psutil.virtual_memory().used / 1e9, "GB", flush=True)
    print("available memory:", psutil.virtual_memory().available / 1e9, "GB", flush=True)

    # if dim_channels = 1: temperature, if dim_channels = 2: temperature, precipitation
    input_train = input_train[:, 0:args.dim_channels, ...]
    target_train = target_train[:, 0:args.dim_channels, ...]
    if load_val(args):
        input_val = input_val[:, 0:args.dim_channels, ...]
        target_val = target_val[:, 0:args.dim_channels, ...]

    # for /era5_temp_precip_data/, if dim_channels = 1: temperature, if dim_channels = 2: temperature, precipitation
    input_train = input_train[:, 0:args.dim_channels, ...]
    target_train = target_train[:, 0:args.dim_channels, ...]
    input_val = input_val[:, 0:args.dim_channels, ...]
    target_val = target_val[:, 0:args.dim_channels, ...]

    input_train_orig = None 
    if load_val(args): input_val_orig = None

    # define dimesions
    global train_shape_in , train_shape_out, val_shape_in, val_shape_out
    train_shape_in = input_train.shape
    train_shape_out = target_train.shape
    if load_val(args): 
        val_shape_in = input_val.shape
        val_shape_out = target_val.shape

    # precipitatoin from [m] -> [mm] and p = log(100*p+1)
    if (args.dim_channels > 1):
        input_train[:, 1, ...] = preprocess_precip(input_train[:, 1, ...])
        target_train[:, 1, ...] = preprocess_precip(target_train[:, 1, ...])

        if load_val(args):
            input_val[:, 1, ...] = preprocess_precip(input_val[:, 1, ...])
            target_val[:, 1, ...] = preprocess_precip(target_val[:, 1, ...])

    # mean, std, min, max
    global mean, std, mean_res, std_res
    global max_val, min_val, max_val_res, min_val_res
    mean = torch.zeros((args.dim_channels,1))
    std = torch.zeros((args.dim_channels,1))
    mean_res = torch.zeros((args.dim_channels,1))
    std_res = torch.zeros((args.dim_channels,1))
    max_val = torch.zeros((args.dim_channels,1))
    min_val = torch.zeros((args.dim_channels,1))
    max_val_res = torch.zeros((args.dim_channels,1))
    min_val_res = torch.zeros((args.dim_channels,1))

    for i in range(args.dim_channels):
        mean[i] = target_train[:,i,...].mean()
        std[i] = target_train[:,i,...].std()
        max_val[i] = target_train[:,i,...].max() # raw HR data
        min_val[i] = target_train[:,i,...].min() # raw HR data

    print("########### raw target ###########")
    print("mean:", mean, "std:", std)
    print("min:", min_val, "max:", max_val)

    input_train_resized = torch.zeros_like(target_train)
    if load_val(args): input_val_resized = torch.zeros_like(target_val)
    
    # linear interpolation for LR data to match HR dimensions
    for i in range(args.dim_channels):
        input_train_resized[:,i,...] = interp_transform_to_data(input_train[:,i,...], (train_shape_out[-2], train_shape_out[-1])) 
        if load_val(args): input_val_resized[:,i,...] = interp_transform_to_data(input_val[:,i,...], (val_shape_out[-2], val_shape_out[-1]))

    if is_diffusion(args):
        # use residuals for diffusion
        target_train_res = target_train - input_train_resized
        if load_val(args): target_val_res = target_val - input_val_resized

        for i in range(args.dim_channels):
            mean_res[i] = target_train_res[:,i,...].mean()
            std_res[i] = target_train_res[:,i,...].std()
            max_val_res[i] = target_train_res[:,i,...].max() # residuals
            min_val_res[i] = target_train_res[:,i,...].min() # residuals 

        print("########### residual ###########")
        print("mean:", mean_res, "std:", std_res)
        print("min:", min_val_res, "max:", max_val_res)

    # if model is used with physical constraints, original LR data is needed
    if use_constraints(args):
        input_train_orig = input_train
        if load_val(args): input_val_orig = input_val

    input_train = input_train_resized
    if load_val(args): input_val = input_val_resized

    if is_diffusion(args):
        # residuals as diffusion targets
        target_train = target_train_res
        if load_val(args): 
            target_val = target_val_res
            del target_val_res
    
        del target_train_res

    # normalise data
    input_train = normalise(args, input_train, min_val, max_val)
    if load_val(args): input_val = normalise(args, input_val, min_val, max_val)

    if is_diffusion(args):
        if load_val(args): target_val = normalise(args, target_val, min_val_res, max_val_res) # diffusion target
        target_train = normalise(args, target_train, min_val_res, max_val_res) # diffusion target
    else:
        target_train = normalise(args, target_train, min_val, max_val) # flow target
        if load_val(args): target_val = normalise(args, target_val, min_val, max_val) # flow target

    if use_constraints(args):
        input_train_orig = normalise(args, input_train_orig, min_val, max_val)
        if load_val(args): input_val_orig = normalise(args, input_val_orig, min_val, max_val)

    # save processedd data
    numworkers=min(args.cpus - 1, 8)
    if use_constraints(args):
        train_data = TensorDataset(input_train,  target_train, input_train_orig)
        if load_val(args): val_data = TensorDataset(input_val, target_val, input_val_orig)
        train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=numworkers, pin_memory=True)
        if load_val(args): val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=numworkers, pin_memory=True)
    else:
        train_data = TensorDataset(input_train,  target_train)
        if load_val(args): val_data = TensorDataset(input_val, target_val)
        train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=numworkers, pin_memory=True)
        if load_val(args): val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=numworkers, pin_memory=True)

    print("data saved...")
    print("memory usage:", psutil.virtual_memory().used / 1e9, "GB")
    print("available memory:", psutil.virtual_memory().available / 1e9, "GB", flush=True)

    if load_val(args):
        return {"train": train,
                "val": val,
                "train_shape_in": train_shape_in,
                "train_shape_out": train_shape_out,
                "val_shape_in": val_shape_in,
                "val_shape_out": val_shape_out}
    else:
         return {"train": train,
                "train_shape_in": train_shape_in,
                "train_shape_out": train_shape_out
                }

# func load_data_lazy() for ERA5 data (supports multi-gpu training)
def load_data_lazy(args):

    print("total memory:", psutil.virtual_memory().total / 1e9, "GB", flush=True)
    print("starting load_data_lazy()...", flush=True)
    print("initial memory usage:", psutil.virtual_memory().used / 1e9, "GB", flush=True)
    print("available memory:", psutil.virtual_memory().available / 1e9, "GB", flush=True)

    n_chunks = 25 # ERA5 train data separated to 25 files
    input_files = [f'{args.data_path}{args.dataset}/train_chunks/input_train_chunk_{i}.npy' for i in range(n_chunks)]
    target_files = [f'{args.data_path}{args.dataset}/train_chunks/target_train_chunk_{i}.npy' for i in range(n_chunks)]
    
    numworkers=min(args.cpus - 1, 8)
    train_dataset = ShardedDataset(input_files, target_files, dim_channels=args.dim_channels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=numworkers, pin_memory=True)

    train_shape_out = torch.Size([0, args.dim_channels, 128, 128]) # TODO: hardcoded...
    train_shape_in = torch.Size([0, args.dim_channels, 32, 32])

    print("data saved...")
    print("memory usage:", psutil.virtual_memory().used / 1e9, "GB")
    print("available memory:", psutil.virtual_memory().available / 1e9, "GB", flush=True)
    
    return {"train": train_loader,
            "train_shape_in": train_shape_in,
            "train_shape_out": train_shape_out
            }

# func load_data_lazy_nordics() process NGCD/E-OBS data, supports multi-gpu
def load_data_lazy_nordics(args):

    # TODO: add validation option

    PATCH_SIZE = 128 # size of Nordic data
    dtype=torch.float32 # precision of Nordic data

    MIN_MAX_FILE = f"{args.data_path}ngcd_minmax.json"
    with open(MIN_MAX_FILE) as f: d = json.load(f)
    global max_val, min_val
    min_val = torch.tensor([d['min']], dtype=dtype)
    max_val = torch.tensor([d['max']], dtype=dtype)

    if args.nordics_constants == "True":
        IDX_KEPT_FILE = f"{args.data_path}{args.dataset}/logs/idx_keep_2020.npy"
        LAND_COVER_FILE = f"{args.data_path}land_cover_2020.pt"
        Z_VALUE_FILE = f"{args.data_path}z_values.pt"
        idx_kept = np.load(IDX_KEPT_FILE)
        land_cover = torch.load(LAND_COVER_FILE)
        z_values = torch.load(Z_VALUE_FILE)
        constants = {"idx_kept" : idx_kept,
                     "land_cover" : land_cover, 
                     "z_values" : z_values}
    else:
        constants = None

    if args.nordics_time == "True":
        time = True
    else:
        time= False

    print("total memory:", psutil.virtual_memory().total / 1e9, "GB", flush=True)
    print("starting load_data_lazy_nordics()...", flush=True)
    print("initial memory usage:", psutil.virtual_memory().used / 1e9, "GB", flush=True)
    print("available memory:", psutil.virtual_memory().available / 1e9, "GB", flush=True)

    target_files = [
        f'{args.data_path}{args.dataset}/ngcd/ngcd_patches_{year}.npy'
        for year in range(args.nordics_start, args.nordics_end + 1)
    ]

    input_files = [
        f'{args.data_path}{args.dataset}/eobs/eobs_patches_{year}.npy'
        for year in range(args.nordics_start, args.nordics_end + 1)
    ]

    if args.training_evalonly in ["training", "resume", "finetune", "sample_test"]:
        numworkers=min(args.cpus - 1, 8)
        train_dataset = ShardedDatasetNordics(input_files, target_files, args.dim_channels, min_val, max_val, constants, time=time, dataset=args.dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=numworkers, pin_memory=True)
        
        total_patches = len(train_dataset)
        print("# total patches:", total_patches)

        train_shape_out = torch.Size([total_patches, args.dim_channels, PATCH_SIZE, PATCH_SIZE])
        train_shape_in = torch.Size([total_patches, args.dim_channels, PATCH_SIZE, PATCH_SIZE])

        print("data saved...")
        print("memory usage:", psutil.virtual_memory().used / 1e9, "GB")
        print("available memory:", psutil.virtual_memory().available / 1e9, "GB", flush=True)
        
        return {"train": train_loader,
                "train_shape_in": train_shape_in,
                "train_shape_out": train_shape_out
                }
    
    else: 
        numworkers=min(args.cpus - 1, 8)
        val_dataset = ShardedDatasetNordics(input_files, target_files, args.dim_channels, min_val, max_val, constants, time=time, dataset=args.dataset)

        if args.nordics_start == args.nordics_end and args.nordics_month > 0:
            # only evaluate a single month
            month_start, month_end = month_indices(args)
            month_ds_indices = [global_idx for global_idx, (file_idx, local_idx) in enumerate(val_dataset.index_map)
                                if month_start <= local_idx < month_end]

            val_dataset = Subset(val_dataset, month_ds_indices)

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=numworkers, pin_memory=True)

        total_patches = len(val_dataset)
        print("# total patches:", total_patches)

        val_shape_out = torch.Size([total_patches, args.dim_channels, PATCH_SIZE, PATCH_SIZE])
        val_shape_in = torch.Size([total_patches, args.dim_channels, PATCH_SIZE, PATCH_SIZE])

        print("data saved...")
        print("memory usage:", psutil.virtual_memory().used / 1e9, "GB")
        print("available memory:", psutil.virtual_memory().available / 1e9, "GB", flush=True)
        
        return {"val": val_loader,
                "val_shape_in": val_shape_in,
                "val_shape_out": val_shape_out
                }

def month_indices(args):
    # reduce yearly data to one month

    DATES_FILE = f"{args.data_path}{args.dataset}/logs/dates_{args.nordics_start}.json"
    DAY_PTR_FILE = f"{args.data_path}{args.dataset}/logs/day_ptr_{args.nordics_start}.npy"
    with open(DATES_FILE) as f: dates = json.load(f)
    day_ptr = np.load(DAY_PTR_FILE)

    months = [int(date.split("-")[1]) for date in dates]
    first = months.index(args.nordics_month)
    last = max(loc for loc, val in enumerate(months) if val == args.nordics_month)
    month_range = (day_ptr[first], day_ptr[last+1])

    return month_range

# Diffusion model loss
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, model, images, conditional_img=None, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1])
        rnd_normal = rnd_normal.type_as(images) 
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data)**2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma

        D_yn = model(y + n, sigma, conditional_img, labels, augment_labels=augment_labels)
        
        loss = weight * ((D_yn - y) ** 2)

        return loss
    
def get_loss_diffusion(model, targets, inputs):
    edm_loss_fn = EDMLoss()
    with torch.cuda.amp.autocast():
        edm_loss = edm_loss_fn(model=model, images=targets, conditional_img=inputs)
        edm_loss = torch.mean(edm_loss)
    return edm_loss   

# Flow matching loss
class FlowLoss:
    def __call__(self, model, x0, x1, bias=False, nordics=False, labels=None, aux=None):
        
        # settings for Nordic data
        penalty = True
        alpha = 0.5

        def downsample_data(fine, N=10):
            coarse = fine.unfold(2, N, N).unfold(3, N, N)
            coarse = coarse.mean(dim=(-2, -1))
            return coarse
        
        def downsample_include_nan(x, N=10):
            # x: (B, C, H, W) with NaNs
            mask = torch.isfinite(x).float()
            x_filled = torch.nan_to_num(x, nan=0.0)
            size = (int(x.shape[-2] / N), int(x.shape[-1] / N))

            sum_pool  = F.interpolate(x_filled, size=size, mode="area") * (x.shape[2] * x.shape[3]) / (size[0] * size[1])
            count_pool = F.interpolate(mask, size=size, mode="area") * (x.shape[2] * x.shape[3]) / (size[0] * size[1])

            out = sum_pool / torch.clamp(count_pool, min=1.0)
            out[count_pool == 0] = float("nan")  # if a cell had no valid input
            return out
        
        def pad_image(x, pad_px=6, val=float("nan")): # (128 x 128) -> (140 x 140) for 10x downsampling
            pad = (pad_px,) * 4
            x_pad = F.pad(x, pad, "constant", val)
            return x_pad

        if nordics:
            # TODO: add bias option?

            t = torch.rand(x0.shape[0], 1, 1, 1) # uniform distribution, shape (B, 1, 1, 1)
            t = t.type_as(x0)
            dx_t = x1 - x0 # residual to be predicted by model

            loss_fn = torch.nn.MSELoss()

            nan_mask = torch.isfinite(x0) & torch.isfinite(x1)
            x0_nansafe = torch.nan_to_num(x0, nan=0.0)
            x1_nansafe = torch.nan_to_num(x1, nan=0.0)

            x_t = (1 - t) * x0_nansafe + t * x1_nansafe # linear combination of x0 and x1, non nans into model

            use_constants = not (labels is None and aux is None)

            if use_constants:
                output_res = model(x_t, t, condition_img=aux, class_labels=labels)
            else:
                output_res = model(x_t, t)

            output_hr = x0+output_res

            reg_loss = loss_fn(output_res[nan_mask], dx_t[nan_mask])
            
            if penalty:
                output_hr_nan = output_hr.clone()
                output_hr_nan[~nan_mask] = float("nan")
                output_lr = downsample_include_nan(pad_image(output_hr_nan))
                input_lr = downsample_include_nan(pad_image(x0))

                nan_mask_coarse = torch.isfinite(output_lr) & torch.isfinite(input_lr)

                lr_loss = loss_fn(output_lr[nan_mask_coarse], input_lr[nan_mask_coarse]) # calculate loss for non-nan areas

                loss = reg_loss + alpha * lr_loss

            else:
                loss = reg_loss

        else:
            # regular Flow loss for ERA5 data

            t = torch.rand(x0.shape[0], 1, 1, 1) # uniform distribution, shape (B, 1, 1, 1)
            t = t.type_as(x0)
            dx_t = x1 - x0 # residual to be predicted by model

            loss_fn = torch.nn.MSELoss()

            if bias == "yes":
                t_bias = torch.sin( (t-1) * torch.pi / 2) + 1 # bias to t=0
                x_t = (1 - t_bias) * x0 + t_bias * x1 # linear combination of x0 and x1
                loss = loss_fn(model(x_t, t_bias), dx_t)
            else:
                x_t = (1 - t) * x0 + t * x1 # linear combination of x0 and x1
                loss = loss_fn(model(x_t, t), dx_t) 

        return loss                                           

def get_loss_flow(model, targets, inputs, bias=False, nordics=False, labels=None, aux=None):
    flow_loss_fn = FlowLoss()
    with torch.cuda.amp.autocast():
        flow_loss = flow_loss_fn(model=model, x0=inputs, x1=targets, bias=bias, nordics=nordics, labels=labels, aux=aux)
    return flow_loss

def process_for_training(inputs, targets, inputs_original=None, aux=None, labels=None): 
    inputs = inputs.to(device)            
    targets = targets.to(device)
    if inputs_original != None: 
        inputs_original = inputs_original.to(device)
        return inputs, targets, inputs_original
    if not (aux is None and labels is None):
        aux = aux.to(device)
        labels = labels.to(device)
        return inputs, targets, labels, aux
    else:
        return inputs, targets

def process_for_eval(outputs, targets, args, inputs=None): 
    outputs = outputs.to(device)

    if is_diffusion(args):
        outputs = de_normalise(args, outputs, min_val_res.to(device), max_val_res.to(device))
        targets = de_normalise(args, targets, min_val_res.to(device), max_val_res.to(device))
        inputs = de_normalise(args, inputs, min_val.to(device), max_val.to(device))

        # with diffusion, transform residuals to HR image (HR = LR+res)
        outputs = outputs + inputs
        targets = targets + inputs

    else:
        outputs = de_normalise(args, outputs, min_val.to(device), max_val.to(device))
        targets = de_normalise(args, targets, min_val.to(device), max_val.to(device))

    # precipitation from log scale to [m] 
    if (args.dim_channels > 1):
        outputs[:, 1, ...] = postprocess_precip(outputs[:, 1, ...])
        targets[:, 1, ...] = postprocess_precip(targets[:, 1, ...])

    return outputs, targets

# TODO: support 2 previous timesteps (nordics_time = True)
# combine patched data before evaluation
def recon_patches_nordics(patches, args): # TODO: support for sub-year evaluation

    print("### processing patched data for evaluation...")

    # masks for nans and padding
    MASK_FILE = f"{args.data_path}masks.npz"
    mask_out = np.load(MASK_FILE)
    row_mask = mask_out["row"]
    col_mask = mask_out["col"]
    nan_mask = mask_out["nan"]

    # reconstruct patches into full images
    recon_patches = []
    # loop years selected for evaluation
    for year in range(args.nordics_start, args.nordics_end + 1):
        print(f"processing year {year}...")

        META_FILE = f"{args.data_path}{args.dataset}/logs/meta_{year}.json"
        with open(META_FILE) as f: d = json.load(f)
        num_patches = d['num_patches']
        img_shape = d['padded_shape']
        patch_size = d['patch_size']
        stride = d['stride']

        idx_kept = np.load(f"{args.data_path}{args.dataset}/logs/idx_keep_{year}.npy")
        day_ptr = np.load(f"{args.data_path}{args.dataset}/logs/day_ptr_{year}.npy")

        start_idx_y = day_ptr[0] # 1st of year
        end_idx_y = day_ptr[-1] # last of year
        patches_year = patches[start_idx_y:end_idx_y, :] 

        # loop days in year, reconstruct each day
        for day in range(len(day_ptr) - 1):
            start_idx_d = day_ptr[day]
            end_idx_d = day_ptr[day + 1]

            daily_patches = patches_year[start_idx_d:end_idx_d]
            idx_kept_daily = idx_kept[start_idx_d:end_idx_d]

            # join patches
            recon_patches_daily = combine_patches(daily_patches, num_patches, img_shape, idx_kept_daily, patch_size, stride)

            # mask
            recon_patches_daily = recon_patches_daily[:, :, :nan_mask.shape[0], :nan_mask.shape[1]]
            recon_patches_daily[0, 0, nan_mask] = float("nan")

            recon_patches.append(recon_patches_daily[0])
        
        print(f"year {year} processed!")

    recon_patches = torch.stack(recon_patches, dim=0)

    return recon_patches

def is_diffusion(args): # check for diffusion model (other option is flow matching)
    if args.model == 'diffusion':
        return True
    else: 
        # flow
        return False

def is_nordics(args): # if patched Nordic data is used (other option ERA5)
    if args.nordics == 'True':
        return True
    else: 
        # ERA5
        return False
    
def use_constraints(args): # check whether constraints are used
    if args.constraints != 'none':
        return True
    else: 
        # no constraints
        return False
    
def load_val(args): # check whether constraints are used
    if args.use_validation != 'yes':
        # no validation data
        return False
    else: 
        # validation data
        return True

def interp_transform_to_data(coarse, fine_shape):
    interp_transform = torchvision.transforms.Resize(fine_shape,
                                                     interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                     antialias=True)
    interp_coarse = interp_transform(coarse)
    return interp_coarse

def normalise(args, data, min, max):
    for i in range(args.dim_channels):
        # min-max normalisation
        data[:, i, ...] = (data[:, i, ...] - min[i]) / (max[i] - min[i])
        # [0, 1] -> [-1, 1]
        data[:, i, ...] = data[:, i, ...] * 2 - 1
    return data

def de_normalise(args, data, min, max):
    for i in range(args.dim_channels):
        # [-1, 1] -> [0, 1]
        data[:, i, ...] = (data[:, i, ...] + 1) / 2
        # min-max de-normalisation
        data[:, i, ...] = data[:, i, ...] * (max[i] - min[i]) + min[i]
    return data 

def preprocess_precip(tensor):
    return torch.log(tensor * 1000 + 1)

def postprocess_precip(tensor):
    return (torch.exp(tensor) - 1) / 1000

def fill_nans(patches, num_patches, image_shape, idx_kept, patch_size):
    _, C, _, _ = image_shape
    nH, nW = num_patches
    tot_num_patches = nH * nW
    full_patches = torch.zeros([tot_num_patches, C, patch_size, patch_size]).type_as(patches)
    full_patches[:] = float("nan")
    full_patches[idx_kept, :] = patches

    return full_patches

def combine_patches(patches, num_patches, img_shape, idx_kept, patch_size, stride):
    """
    Combine overlapping patches back into full images using mean overlap.

    Args:
        patches: tensor (B * n_patches, C, patch_size, patch_size)
        num_patches: tulpe (nH, nW)
        img_shape: tuple (B, C, H, W)
        patch_size: int
        stride: int

    Returns:
        Reconstructed full-size image tensor (B, C, H, W)
    """
    
    patches = fill_nans(patches, num_patches, img_shape, idx_kept, patch_size)

    B, C, H, W = img_shape
    nH, nW = num_patches

    device = patches.device
    output = torch.zeros((B, C, H, W), device=device)
    norm_map = torch.zeros_like(output)

    patch_idx = 0
    for b in range(B):
        for i in range(nH):
            for j in range(nW):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + patch_size
                w_end = w_start + patch_size

                output[b, :, h_start:h_end, w_start:w_end] += patches[patch_idx]
                norm_map[b, :, h_start:h_end, w_start:w_end] += 1
                patch_idx += 1

    return (output / norm_map)

# modified from ClimateDiffuse Inference.py sample_model_EDS
@torch.no_grad()
def run_diffusion_model(inputs, model, args, device, inputs_original=None,
                        sigma_min=0.002, sigma_max=80, rho=7, S_churn=40,
                        S_min=0, S_max=float('inf'), S_noise=1):
    
    if args.ode_order == 1:
        ode = "Euler"
    elif args.ode_order == 2:
        ode = "Heun"
    
    num_steps = args.steps

    inputs = inputs.to(device)

    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)

    init_noise = torch.randn((inputs.shape[0], args.dim_channels, inputs.shape[-2],
                              inputs.shape[-1]),
                              dtype=torch.float64, device=device)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64,
                                device=init_noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([model.round_sigma(t_steps),
                         torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = init_noise.to(torch.float64) * t_steps[0]

    for i, (t_cur, t_next) in tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])), total=num_steps, desc="Diffusion steps", file=sys.stdout):

        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = model.round_sigma(t_cur + gamma * t_cur) 
        noise = torch.randn_like(x_cur)
        noise_added = (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * noise
        x_hat = (x_cur + noise_added)

        # ODE step 1
        denoised = model(x_hat, t_hat, inputs).to(torch.float64)
        t_term = (t_next - t_hat) / t_hat
        x_next = (1 + t_term) * x_hat - t_term * denoised

        # ODE step 2
        if ode == "Heun":
            d_cur = (x_hat - denoised) / t_hat 
            # Apply 2nd order correction.
            if i < num_steps - 1:
                # heun
                denoised = model(x_next, t_next, inputs).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    predicted = x_next

    # physical constraints
    if use_constraints(args):
        predicted = diffusion_constraint(predicted, inputs, inputs_original, args)

    return predicted

@torch.no_grad()
def run_flow_matching(inputs, model, args, device, inputs_original=None, labels=None, aux=None):

    if args.ode_order == 1:
        ode = "Euler"
    elif args.ode_order == 2:
        ode = "Midpoint"

    num_steps = args.steps

    time_steps = torch.linspace(0, 1.0, num_steps + 1, device=device)

    def step(x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1, 1, 1) # shape (B, 1, 1, 1)
        
        if not (labels is None and aux is None):
            if ode == "Euler":
                x_next = x_t + (t_end - t_start) * model(x_t, t_start, condition_img=aux, class_labels=labels) 

            elif ode == "Midpoint":
                x_next =  x_t + (t_end - t_start) * model(
                    x_t + model(x_t, t_start, condition_img=aux, class_labels=labels) * (t_end - t_start) / 2,
                    t_start + (t_end - t_start) / 2, condition_img=aux, class_labels=labels)
        else:
            if ode == "Euler":
                x_next = x_t + (t_end - t_start) * model(x_t, t_start) 

            elif ode == "Midpoint":
                x_next =  x_t + (t_end - t_start) * model(
                    x_t + model(x_t, t_start) * (t_end - t_start) / 2,
                    t_start + (t_end - t_start) / 2)

        return x_next
    
    x = inputs
    # for i in tqdm(range(num_steps), desc="Flow steps", file=sys.stdout):
    for i in range(num_steps):
        x = step(x, time_steps[i], time_steps[i + 1])

    predicted = x

    # physical constraints
    if use_constraints(args):
        predicted = flow_constraint(predicted, inputs, inputs_original, args)

    return predicted

def diffusion_constraint(predicted, inputs, inputs_original, args):

    inputs_con = inputs.clone()
   
    downsample = torchvision.transforms.Resize((val_shape_in[-2], val_shape_in[-1]),
                                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                               antialias=True)
    upsample = torchvision.transforms.Resize((val_shape_out[-2], val_shape_out[-1]),
                                             interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                             antialias=True)
    
    constraint = args.constraints # back projection, additive constraint, bilateral filter

    if constraint == "back_projection":
        predicted = de_normalise(args, predicted, min_val_res.to(device), max_val_res.to(device))
        inputs_original = de_normalise(args, inputs_original, min_val.to(device), max_val.to(device))
        inputs_con = de_normalise(args, inputs_con, min_val.to(device), max_val.to(device))

        cur_hr = inputs_con + predicted # (upsampled) LR image + current residual estimate
        cur_hr_downsample = downsample(cur_hr)
        diff = cur_hr_downsample - inputs_original # (downsampled) HR estimate - ground truth LR
        diff_upsample = upsample(diff)
        lam = 50
        cur_hr = cur_hr + 1/lam * diff_upsample # back projection: add small fraction of diff to current estimate
        predicted = cur_hr - inputs_con # residual = estimated HR - LR

        predicted = normalise(args, predicted, min_val_res.to(device), max_val_res.to(device))   

    elif constraint == "bilateral":
        predicted = de_normalise(args, predicted, min_val_res.to(device), max_val_res.to(device))
        inputs_con = de_normalise(args, inputs_con, min_val.to(device), max_val.to(device))

        cur_hr = inputs_con + predicted # (upsampled) LR image + current residual estimate
        filtered_cur = K.filters.joint_bilateral_blur(cur_hr, inputs_con, (3, 3), 0.1, (0.5, 0.5)) # bilateral smoothing
        predicted = filtered_cur - inputs_con # residual = estimated HR - LR
        
        predicted = normalise(args, predicted, min_val_res.to(device), max_val_res.to(device))       

    elif constraint == "additive":
        predicted = de_normalise(args, predicted, min_val_res.to(device), max_val_res.to(device))
        inputs_original = de_normalise(args, inputs_original, min_val.to(device), max_val.to(device))
        inputs_con = de_normalise(args, inputs_con, min_val.to(device), max_val.to(device))

        cur_hr = inputs_con + predicted # (upsampled) LR image + current residual estimate
        cur_hr_downsample = downsample(cur_hr)
        # physical constraint using estimated HR and input LR
        # modified from constrained-downscaling/models.py
        cur_hr = cur_hr + torch.kron(inputs_original-cur_hr_downsample, 
                                     torch.ones((args.upsampling_factor, args.upsampling_factor)).type_as(predicted)) 

        predicted = normalise(args, predicted, min_val_res.to(device), max_val_res.to(device))       
    
    return predicted

def flow_constraint(predicted, inputs, inputs_original, args):
    constraint = args.constraints

    downsample = torchvision.transforms.Resize((val_shape_in[-2], val_shape_in[-1]),
                                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                               antialias=True)
    upsample = torchvision.transforms.Resize((val_shape_out[-2], val_shape_out[-1]),
                                             interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                             antialias=True)
    
    constraint = args.constraints # back projection, additive constraint, bilateral filter

    if constraint == "back_projection":
        predicted_downsample = downsample(predicted)
        diff = predicted_downsample - inputs_original # (downsampled) HR estimate - ground truth LR
        diff_upsample = upsample(diff)
        lam = 50
        predicted = predicted + 1/lam * diff_upsample # back projection: add small fraction of diff to current estimate
            
    elif constraint == "bilateral":
        predicted = K.filters.joint_bilateral_blur(predicted, inputs, (3, 3), 0.1, (0.5, 0.5)) # bilateral smoothing
    
    elif constraint == "additive":
        predicted_downsample = downsample(predicted)
        # physical constraint using estimated HR and input LR
        # modified from constrained-downscaling/models.py
        predicted = predicted + torch.kron(inputs_original-predicted_downsample, 
                                            torch.ones((args.upsampling_factor, args.upsampling_factor)).type_as(predicted))

    return predicted
    
        