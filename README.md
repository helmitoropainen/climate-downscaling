# Climate dowscaling

4x Climate downscaling using Diffusion Models or Flow Matching

This repo is modified from 
https://github.com/RolnickLab/constrained-downscaling 
and 
https://github.com/robbiewatt1/ClimateDiffuse

## Setup

Clone the repo and install the requirements with conda
```sh
$ git clone https://github.com/helmitoropainen/climate-downscaling
$ cd climate-downscaling
$ conda env create -f requirements.yml
$ conda activate climate-downscaling
```

## Get the data

A dataset for this repo is available at: https://drive.google.com/file/d/11308-R6nhsLhVq0aCXVNZssKyiaRCOs-/view?usp=drive_link. 

Download with wget:
```sh
$ wget --no-check-certificate 'https://drive.usercontent.google.com/download?id=11308-R6nhsLhVq0aCXVNZssKyiaRCOs-&export=download&authuser=0&confirm=t&uuid=2b22cf25-c51f-432f-bf7f-1beb19d89e80&at=AN8xHoqzzFwrogFVqAgGhmAlrorC:1752567480691' -O era5_temp_precip_data.zip
```

unzip
```sh
$ mkdir data/era5_temp_precip_data/
$ unzip -o era5_temp_precip_data.zip -d data/era5_temp_precip_data/
$ rm era5_temp_precip_data.zip 
```

The ERA5 data (0.25°) is downloaded from https://github.com/google-research/weatherbench2 and contains hourly measurements for 2-meter temperature [°C] in the 1st channel and total precipitaion [m] in the 2nd channel. The training set covers years 1959–2013, the validation set years 2014–2018, and the test set years 2019–2023 (measurements for 2023 end in October). Only 30 random hourly samples are included for each month. The dataset covers the Europe region (33–65° N, 0–32°), resulting in 128x128 images and a corresponding 32x32 low resolution image created by taking means of 4x4 patches.

## Model training
Example setups:

Run Flow Matching:
```sh
srun python ./src/main.py --dataset era5_temp_precip_data --model flow --model_id flow_model
```
Run Diffusion Model:
```sh
srun python ./src/main.py --dataset era5_temp_precip_data --model diffusion --model_id diffusion_model
```
Run Flow Matching with 4 GPUs:
```sh
srun python ./src/main.py --dataset era5_temp_precip_data --model flow --model_id flow_model_4GPU --gpus 4
```
## Model inference
Example sampling with the Flow Matching model:
```sh
srun python ./src/main.py --dataset era5_temp_precip_data --model flow --model_id flow_model --test_val_train test --training_evalonly evalonly
```

Example sampling with the Flow Matching model and back_projection constraint:
```sh
srun python ./src/main.py --dataset era5_temp_precip_data --model flow --model_id flow_model --test_val_train test --training_evalonly evalonly --constraint back_projection
```