import xarray as xr
import torch
import numpy as np
import pandas as pd
import tqdm


# fetch data from weatherbench2
wb_ds_og = xr.open_zarr(
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
    )
print("data downloaded")

# crop for europe
lats = wb_ds_og['latitude'].values
lons = wb_ds_og['longitude'].values
print(len(lats), len(lons))
ERA5_centre = [16, 49] # 35° – 65° N, 0° – 32° E

lat_idx_center = np.argmin(np.abs(lats - ERA5_centre[1]))
lon_idx_center = np.argmin(np.abs(lons - ERA5_centre[0]))
print(lat_idx_center, lon_idx_center)

lat_start_idx = lat_idx_center - 64
lon_start_idx = lon_idx_center - 64

print(lat_start_idx, lon_start_idx)

lon_slice = lons[lon_start_idx:lon_start_idx + 128]

lat_slice = lats[lat_start_idx:lat_start_idx + 128]

print(len(lat_slice), len(lon_slice))

eu_ds =  wb_ds_og[["2m_temperature", "total_precipitation"]].sel(latitude=lat_slice, longitude=lon_slice)
print("eu data selected")

# sample random times
time_index = eu_ds.time.to_index()

df = pd.DataFrame({'time': time_index})
df['year'] = df.time.dt.year
df['month'] = df.time.dt.month

grouped = df.groupby(['year', 'month'])

samples = grouped.sample(n=30, replace=False)

samples_sorted = samples.sort_values('time')

sampled_times = samples_sorted['time'].values

samples_sorted['reduced_index'] = np.arange(len(sampled_times))

# save train/val/test indices when all years: 1959-2023 (65) (55/5/5)
train_samples = samples_sorted[(samples_sorted["time"].dt.year >= 1959) & (samples_sorted["time"].dt.year <= 2013)]['reduced_index'].values
val_samples = samples_sorted[(samples_sorted["time"].dt.year >= 2014) & (samples_sorted["time"].dt.year <= 2018)]['reduced_index'].values
test_samples = samples_sorted[(samples_sorted["time"].dt.year >= 2019) & (samples_sorted["time"].dt.year <= 2023)]['reduced_index'].values

eu_ds_reduced = eu_ds.sel(time=sampled_times)
print("data reduced by 1/24")

# convert to tensor and save
temp_sample_eu_reduced = eu_ds_reduced["2m_temperature"][:]
precip_sample_eu_reduced = eu_ds_reduced["total_precipitation"][:]

eu_temp_chunks = temp_sample_eu_reduced.chunk({"time": 64})
eu_precip_chunks = precip_sample_eu_reduced.chunk({"time": 64})

n_samples = eu_temp_chunks.shape[0]
tensor_all = torch.empty((n_samples, 2, 128, 128), dtype=torch.float32)

batch_size = 256

for i in tqdm.trange(0, n_samples, batch_size):
    temp_batch = eu_temp_chunks.isel(time=slice(i, i + batch_size)).compute().values
    precip_batch = eu_precip_chunks.isel(time=slice(i, i + batch_size)).compute().values
    temp_tensor = torch.from_numpy(temp_batch).float()
    temp_tensor -= 273.15 # K to °C
    precip_tensor = torch.from_numpy(precip_batch).float()
    tensor = torch.stack([temp_tensor, precip_tensor], dim=1)
    tensor_all[i:i + tensor.shape[0]] = tensor

target_train = tensor_all[train_samples, :]
target_val = tensor_all[val_samples, :]
target_test = tensor_all[test_samples, :]

print("data split to train/val/test tensors")

torch.save(target_train, ".../data/era5_temp_precip_wb_data/train/target_train.pt")
torch.save(target_val, ".../data/era5_temp_precip_wb_data/val/target_val.pt")
torch.save(target_test, ".../data/era5_temp_precip_wb_data/test/target_test.pt")
print("data saved")