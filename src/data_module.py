import torch
import torchvision
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import psutil

class ClimateDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.min = torch.tensor([[-45.3113], [0.0000]])  
        #self.max = torch.tensor([[4.9545e+01], [3.4553e-02]]) 
        self.max = torch.tensor([[4.9545e+01], [8.1480]]) 

    def setup(self, stage=None):
        rank = self.trainer.local_rank if hasattr(self, "trainer") else 0
        print(f"[rank{rank}] loading chunk_{rank}.pt ...")

        target_path = f"{self.args.data_path+self.args.dataset}/train/target_train_chunk_{rank}.pt"
        input_path = f"{self.args.data_path+self.args.dataset}/train/input_train_chunk_{rank}.pt"

        target_tensor = torch.load(target_path)[:1000]
        input_tensor_raw = torch.load(input_path)[:1000]

        print("data loaded...", flush=True)
        print("memory usage:", psutil.virtual_memory().used / 1e9, "GB", flush=True)
        print("available memory:", psutil.virtual_memory().available / 1e9, "GB", flush=True)

        input_tensor = torch.zeros_like(target_tensor)
    
        # linear interpolation for LR data to match HR dimensions
        input_tensor = self.interp_transform_to_data(input_tensor_raw, (128, 128)) 

        print("input data resized...", flush=True)
        print("memory usage:", psutil.virtual_memory().used / 1e9, "GB", flush=True)
        print("available memory:", psutil.virtual_memory().available / 1e9, "GB", flush=True)
        
        input_tensor[:,1,...] = self.transform_precip(input_tensor[:,1,...])
        target_tensor[:,1,...] = self.transform_precip(target_tensor[:,1,...])

        input_tensor = self.normalise(input_tensor)
        target_tensor = self.normalise(target_tensor)

        print("data normalised...", flush=True)
        print("memory usage:", psutil.virtual_memory().used / 1e9, "GB", flush=True)
        print("available memory:", psutil.virtual_memory().available / 1e9, "GB", flush=True)

        self.train_dataset = TensorDataset(input_tensor, target_tensor)

        print("dataset created...", flush=True)
        print("memory usage:", psutil.virtual_memory().used / 1e9, "GB", flush=True)
        print("available memory:", psutil.virtual_memory().available / 1e9, "GB", flush=True)

    def normalise(self, data):
        min = self.min.view(1, -1, 1, 1)
        max = self.max.view(1, -1, 1, 1)
        data = (data - min) / (max - min)
        data = data * 2 - 1
        return data
    
    def transform_precip(self, data):
        return torch.log(data * 1000 + 1)

    def interp_transform_to_data(self, coarse, fine_shape):
        interp_transform = torchvision.transforms.Resize(fine_shape,
                                                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                        antialias=True)
        interp_coarse = interp_transform(coarse)
        return interp_coarse

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8)
