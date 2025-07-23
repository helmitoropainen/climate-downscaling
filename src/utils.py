######################################
# Data loading and processing        #
# Model training loss and inference  #
######################################

import torch
import torch.nn as nn
import torchvision
import numpy as np
import kornia as K
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = 'cuda'

# modified from constrained-downscaling/utils.py
def load_data(args):

    input_train = torch.load(args.data_path+args.dataset+'/train/input_train.pt')
    target_train = torch.load(args.data_path+args.dataset+'/train/target_train.pt')
    if args.test_val_train == 'test':
        input_val = torch.load(args.data_path+args.dataset+'/test/input_test.pt')
        target_val = torch.load(args.data_path+args.dataset+'/test/target_test.pt')
    elif args.test_val_train == 'val':
        input_val = torch.load(args.data_path+args.dataset+'/val/input_val.pt')
        target_val = torch.load(args.data_path+args.dataset+'/val/target_val.pt')
    elif args.test_val_train == 'train':
        input_val = input_train
        target_val = target_train

    # for /era5_temp_precip_data/, if dim_channels = 1: temperature, if dim_channels = 2: temperature, precipitation
    input_train = input_train[:, 0:args.dim_channels, ...]
    target_train = target_train[:, 0:args.dim_channels, ...]
    input_val = input_val[:, 0:args.dim_channels, ...]
    target_val = target_val[:, 0:args.dim_channels, ...]

    input_train_orig = None 
    input_val_orig = None

    # define dimesions
    global train_shape_in , train_shape_out, val_shape_in, val_shape_out
    train_shape_in = input_train.shape
    train_shape_out = target_train.shape
    val_shape_in = input_val.shape
    val_shape_out = target_val.shape

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
    input_val_resized = torch.zeros_like(target_val)
    
    # linear interpolation for LR data to match HR dimensions
    for i in range(args.dim_channels):
        input_train_resized[:,i,...] = interp_transform_to_data(input_train[:,i,...], (train_shape_out[-2], train_shape_out[-1])) 
        input_val_resized[:,i,...] = interp_transform_to_data(input_val[:,i,...], (val_shape_out[-2], val_shape_out[-1]))

    if is_diffusion(args):
        # use residuals for diffusion
        target_train_res = target_train - input_train_resized
        target_val_res = target_val - input_val_resized

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
        input_val_orig = input_val

    input_train = input_train_resized
    input_val = input_val_resized

    del input_train_resized, input_val_resized

    if is_diffusion(args):
        # residuals as diffusion targets
        target_train = target_train_res
        target_val = target_val_res
    
        del target_train_res, target_val_res

    # normalise data
    input_train = normalise(args, input_train, min_val, max_val)
    input_val = normalise(args, input_val, min_val, max_val)

    if is_diffusion(args):
        target_val = normalise(args, target_val, min_val_res, max_val_res) # diffusion target
        target_train = normalise(args, target_train, min_val_res, max_val_res) # diffusion target
    else:
        target_train = normalise(args, target_train, min_val, max_val) # flow target
        target_val = normalise(args, target_val, min_val, max_val) # flow target

    if use_constraints(args):
        input_train_orig = normalise(args, input_train_orig, min_val, max_val)
        input_val_orig = normalise(args, input_val_orig, min_val, max_val)

    # save processedd data
    if use_constraints(args):
        train_data = TensorDataset(input_train,  target_train, input_train_orig)
        val_data = TensorDataset(input_val, target_val, input_val_orig)
        train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    else:
        train_data = TensorDataset(input_train,  target_train)
        val_data = TensorDataset(input_val, target_val)
        train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    return {"train": train,
            "val": val,
            "train_shape_in": train_shape_in,
            "train_shape_out": train_shape_out,
            "val_shape_in": val_shape_in,
            "val_shape_out": val_shape_out}

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

class FlowLoss:
    def __call__(self, model, x0, x1, bias):
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

def get_loss_flow(model, targets, inputs, bias):
    flow_loss_fn = FlowLoss()
    with torch.cuda.amp.autocast():
        flow_loss = flow_loss_fn(model=model, x0=inputs, x1=targets, bias=bias)
    return flow_loss

def process_for_training(inputs, targets, inputs_original=None): 
    inputs = inputs.to(device)            
    targets = targets.to(device)
    if inputs_original != None: 
        inputs_original = inputs_original.to(device)
        return inputs, targets, inputs_original
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

    return outputs, targets

def is_diffusion(args): # check for diffusion model (other option is flow matching)
    if args.model == 'diffusion':
        return True
    else: 
        # flow
        return False
    
def use_constraints(args): # check whether constraints are used
    if args.constraints != 'none':
        return True
    else: 
        # no constraints
        return False

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

# modified from ClimateDiffuse Inference.py sample_model_EDS
@torch.no_grad()
def run_diffusion_model(inputs, model, args, device, inputs_original=None, num_steps=50,
                        sigma_min=0.002, sigma_max=80, rho=7, S_churn=40,
                        S_min=0, S_max=float('inf'), S_noise=1):
    
    if args.ode_order == 1:
        ode = "Euler"
    elif args.ode_order == 2:
        ode = "Heun"

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

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):

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
def run_flow_matching(inputs, model, args, device, inputs_original=None, num_steps=5):

    if args.ode_order == 1:
        ode = "Euler"
    elif args.ode_order == 2:
        ode = "Midpoint"

    time_steps = torch.linspace(0, 1.0, num_steps + 1, device=device)

    def step(x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1, 1, 1) # shape (B, 1, 1, 1)
        
        if ode == "Euler":
            x_next = x_t + (t_end - t_start) * model(x_t, t_start) 

        elif ode == "Midpoint":
            x_next =  x_t + (t_end - t_start) * model(
                x_t + model(x_t, t_start) * (t_end - t_start) / 2,
                t_start + (t_end - t_start) / 2)

        return x_next
    
    x = inputs
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
    
        