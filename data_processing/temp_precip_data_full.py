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

# eu_ds =  wb_ds_og[["2m_temperature", "total_precipitation"]].sel(latitude=lat_slice, longitude=lon_slice)
eu_ds =  wb_ds_og[["2m_temperature", "total_precipitation"]].isel(time=[1000, 1001, 1002, 1003, 1004, 1006, 1007, 500000, 500001, 500002, 500003, 560000, 560001, 560002]).sel(latitude=lat_slice, longitude=lon_slice) # small test
print("eu data selected")
print(eu_ds)

# time indices to pandas
time_index = eu_ds.time.to_index()

df = pd.DataFrame({'time': time_index})
df['year'] = df.time.dt.year
df['month'] = df.time.dt.month

# save train/val/test indices when all years: 1959-2023 (65) (55/5/5)
train_samples = df[(df["time"].dt.year >= 1959) & (df["time"].dt.year <= 2013)].index
val_samples = df[(df["time"].dt.year >= 2014) & (df["time"].dt.year <= 2018)].index
test_samples = df[(df["time"].dt.year >= 2019) & (df["time"].dt.year <= 2023)].index

n_train = len(train_samples)
n_val = len(val_samples)
n_test = len(test_samples)

# convert to tensor and save
temp_sample_eu = eu_ds["2m_temperature"][:]
precip_sample_eu = eu_ds["total_precipitation"][:]

batch_size = 256

eu_temp_chunks = temp_sample_eu.chunk({"time": 64})
eu_precip_chunks = precip_sample_eu.chunk({"time": 64})

print(type(eu_temp_chunks.data))


# test data
print("processing test data...")
n_test = len(test_samples)
print(n_test, "samples")
tensor_test = torch.empty((n_test, 2, 128, 128), dtype=torch.float32)

for i in tqdm.trange(n_train+n_val, n_val+n_train+n_test, batch_size):
    batch_end_i = min(i + batch_size, n_train+n_val+n_test)
    temp_batch = eu_temp_chunks.isel(time=slice(i, batch_end_i)).compute().values
    precip_batch = eu_precip_chunks.isel(time=slice(i, batch_end_i)).compute().values
    temp_tensor = torch.from_numpy(temp_batch).float()
    temp_tensor -= 273.15 # K to °C
    precip_tensor = torch.from_numpy(precip_batch).float()
    tensor = torch.stack([temp_tensor, precip_tensor], dim=1)
    tensor_test[i-(n_train+n_val):i-(n_train+n_val) + tensor.shape[0]] = tensor

print("test data processed")
print(tensor_test.shape)
# torch.save(tensor_test, "/scratch/project_2012243/data/era5_temp_precip_wb_full_data/test/target_test.pt")
torch.save(tensor_test, "./data/era5_temp_precip_wb_full_data/test/target_test.pt")
print("test data saved")
del tensor_test


# val data
print("processing val data...")
print(n_val, "samples")
tensor_val = torch.empty((n_val, 2, 128, 128), dtype=torch.float32)

for i in tqdm.trange(n_train, n_train+n_val, batch_size):
    batch_end_i = min(i + batch_size, n_train+n_val)
    temp_batch = eu_temp_chunks.isel(time=slice(i, batch_end_i)).compute().values
    precip_batch = eu_precip_chunks.isel(time=slice(i, batch_end_i)).compute().values
    temp_tensor = torch.from_numpy(temp_batch).float()
    temp_tensor -= 273.15 # K to °C
    precip_tensor = torch.from_numpy(precip_batch).float()
    tensor = torch.stack([temp_tensor, precip_tensor], dim=1)
    tensor_val[i - n_train : i - n_train + tensor.shape[0]] = tensor

print("val data processed")
print(tensor_val.shape)
# torch.save(tensor_val, "/scratch/project_2012243/data/era5_temp_precip_wb_full_data/val/target_val.pt")
torch.save(tensor_val, "./data/era5_temp_precip_wb_full_data/val/target_val.pt")
print("val data saved")
del tensor_val


# train data
print("processing train data...")
print(n_train, "samples")
tensor_train = torch.empty((n_train, 2, 128, 128), dtype=torch.float32)

for i in tqdm.trange(0, n_train, batch_size):
    batch_end_i = min(i + batch_size, n_train)
    temp_batch = eu_temp_chunks.isel(time=slice(i, batch_end_i)).compute().values
    precip_batch = eu_precip_chunks.isel(time=slice(i, batch_end_i)).compute().values
    print("temp_batch type")
    print(type(temp_batch))
    temp_tensor = torch.from_numpy(temp_batch).float()
    temp_tensor -= 273.15 # K to °C
    precip_tensor = torch.from_numpy(precip_batch).float()
    tensor = torch.stack([temp_tensor, precip_tensor], dim=1)
    tensor_train[i:i + tensor.shape[0]] = tensor


print("train data processed")
print(tensor_train.shape)
torch.save(tensor_train, "./data/era5_temp_precip_wb_full_data/train/target_train.pt")
print("train data saved")
del tensor_train

print("all data saved")