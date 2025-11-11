import os
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

from torch.utils.data import Dataset
import torchvision
import torch
import numpy as np

class ShardedDataset(Dataset):
    def __init__(self, input_file_list, target_file_list, dim_channels, dtype=torch.float32):
        self.input_file_list = input_file_list
        self.target_file_list = target_file_list
        self.dtype = dtype
        self.min = torch.tensor([-45.3113, 0.0000], dtype=dtype)
        self.max = torch.tensor([4.9545e+01, 8.1480], dtype=dtype) 
        self.dim_channels = dim_channels

        # Build index: global_idx -> (file_id, local_idx)
        self.index_map = []
        for file_id, f in enumerate(input_file_list): # input and target indices match
            n_samples = self._count_samples_in_file(f)
            self.index_map.extend([(file_id, i) for i in range(n_samples)])
    
    def _count_samples_in_file(self, file_path):
        # read metadata (length) from file
        return np.load(file_path, mmap_mode='r').shape[0]
    
    def __getitem__(self, idx):
        file_id, local_idx = self.index_map[idx]
        # load from .npy with lazy loading
        input_data_raw = np.load(self.input_file_list[file_id], mmap_mode='r')[local_idx] # [C, H, W]
        target_data = np.load(self.target_file_list[file_id], mmap_mode='r')[local_idx] # [C, H, W]
        
        # convert to tensor
        input_data_raw = torch.from_numpy(input_data_raw.copy()).to(self.dtype)
        target_data = torch.from_numpy(target_data.copy()).to(self.dtype) 

        # if dim_channels = 1: temperature, if dim_channels = 2: temperature, precipitation
        input_data_raw = input_data_raw[0:self.dim_channels, ...]
        target_data = target_data[0:self.dim_channels, ...]

        # linear interpolation for LR data to match HR dimensions
        input_data = self.interp_transform_to_data(input_data_raw, (target_data.shape[-2], target_data.shape[-1])) 

        if (self.dim_channels > 1):
            input_data[1,...] = self.preprocess_precip(input_data[1,...])
            target_data[1,...] = self.preprocess_precip(target_data[1,...])

        input_data = self.normalise(input_data, self.min, self.max, self.dim_channels)
        target_data = self.normalise(target_data, self.min, self.max, self.dim_channels)
 
        return input_data, target_data
    
    def normalise(self, data, min, max, C):
        for i in range(C):
            # min-max normalisation
            data[i, ...] = (data[i, ...] - min[i]) / (max[i] - min[i])
            # [0, 1] -> [-1, 1]
            data[i, ...] = data[i, ...] * 2 - 1

        return data
    
    def preprocess_precip(self, data):
        return torch.log(data * 1000 + 1)

    def interp_transform_to_data(self, coarse, fine_shape):
        coarse = coarse.unsqueeze(0) # requires [B, C, H, W]
        interp_transform = torchvision.transforms.Resize(fine_shape,
                                                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                        antialias=True)
        interp_coarse = interp_transform(coarse)
        interp_coarse = interp_coarse.squeeze(0) # back to [C, H, W]
        return interp_coarse

    def __len__(self):
        return len(self.index_map)


class ShardedDatasetNordics(Dataset):
    def __init__(self, input_file_list, target_file_list, dim_channels, min_val, max_val, constants=None, time=False, dataset=None, dtype=torch.float32):
        self.input_file_list = input_file_list
        self.target_file_list = target_file_list
        self.dtype = dtype
        self.min = min_val
        self.max = max_val
        self.dim_channels = dim_channels
        
        self.constants = constants is not None
        self.time = time

        # TODO: from file
        if dataset == "yearly_shards": version = 1
        elif dataset == "yearly_shatds_v2": version = 2
        
        if version == 1: # PATCH: 128, STRIDE: 120
            self.n_patches = 117 
            self.nrows, self.ncols = 16, 12
        elif version == 2: # PATCH: 128, STRIDE: 100
            self.n_patches = 174 
            self.nrows, self.ncols = 19, 14 
        else:
            print("check dataset")

        if time:
            self.min = self.min.expand(3)
            self.max = self.max.expand(3)

        if self.constants: 
            self.idx_kept = constants["idx_kept"]
            self.land_cover_patches = constants["land_cover"]
            self.z_value_patches = constants["z_values"]

        # Build index: global_idx -> (file_id, local_idx)
        self.index_map = []
        for file_id, f in enumerate(input_file_list): # input and target indices match
            n_samples = self._count_samples_in_file(f)
            if self.time:
                n_samples -= 2*self.n_patches # skip first 2 samples (TODO: take from previous file?)
            self.index_map.extend([(file_id, i) for i in range(n_samples)])
    
    def _count_samples_in_file(self, file_path):
        # read metadata (length) from file
        return np.load(file_path, mmap_mode='r').shape[0]
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError
        
        file_id, local_idx = self.index_map[idx]

        # load from .npy with lazy loading
        if not self.time:
            input_data = np.load(self.input_file_list[file_id], mmap_mode='r')[local_idx] # [C, H, W]
            target_data = np.load(self.target_file_list[file_id], mmap_mode='r')[local_idx] # [C, H, W]
        else:
            input_data1 = np.load(self.input_file_list[file_id], mmap_mode='r')[local_idx] # [C, H, W]
            input_data2 = np.load(self.input_file_list[file_id], mmap_mode='r')[local_idx+self.n_patches] # [C, H, W]
            input_data3 = np.load(self.input_file_list[file_id], mmap_mode='r')[local_idx+2*self.n_patches] # [C, H, W]
            input_data = np.concat([input_data1, input_data2, input_data3], axis=0)           # [3C, H, W]

            target_data = np.load(self.target_file_list[file_id], mmap_mode='r')[local_idx+2*self.n_patches] # [C, H, W]

        # convert to tensor
        input_data = torch.from_numpy(input_data.copy()).to(self.dtype)
        target_data = torch.from_numpy(target_data.copy()).to(self.dtype) 

        # normalise
        target_data = self.normalise(target_data, self.min, self.max, self.dim_channels)

        if not self.time:
            input_data = self.normalise(input_data, self.min, self.max, self.dim_channels)
        else:
            input_data = self.normalise(input_data, self.min, self.max, self.dim_channels*3)
            input_data1 = input_data[0:self.dim_channels]
            input_data2 = input_data[self.dim_channels:2*self.dim_channels]
            input_data3 = input_data[2*self.dim_channels:3*self.dim_channels]

            input_data1 = torch.nan_to_num(input_data1)
            input_data2 = torch.nan_to_num(input_data2)
            input_data = input_data3

        if self.constants:
            # constant variables
            patch = self.idx_kept[local_idx]            
            row = (patch // self.ncols) / (self.nrows-1) * 2 - 1 
            col = (patch % self.ncols) / (self.ncols-1) * 2 - 1
            day_idx = local_idx // self.n_patches
            if self.time:
                day_idx -= 2
            doy = (day_idx/365) * 2 - 1
            labels = torch.tensor([row, col, doy], dtype=self.dtype)

            patch_idx = local_idx % self.n_patches 
            land_cover = self.land_cover_patches[patch_idx, :]
            z_values = self.z_value_patches[patch_idx, :]
            aux = torch.cat([land_cover, z_values], dim=0)
            if self.time:
                aux = torch.cat([input_data2, input_data1, aux], dim=0)

            sample = {"inputs": input_data,
                      "targets": target_data,
                      "labels": labels,     # patch information, doy
                      "aux": aux            # previous timesteps, land-cover patch (-1: water, 1: land, 0: nan), z-values (0, 1)
                      }
        else:
            sample = {"inputs": input_data,
                     "targets": target_data}
 
        # return input_data, target_data 
        return sample
    
    def normalise(self, data, min, max, C):
        for i in range(C):
            # min-max normalisation
            data[i, ...] = (data[i, ...] - min[i]) / (max[i] - min[i])
            # [0, 1] -> [-1, 1]
            data[i, ...] = data[i, ...] * 2 - 1

        return data

    def __len__(self):
        return len(self.index_map)
