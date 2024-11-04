import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DisplacementDataset(Dataset):
    def __init__(self, u, v, roi):
        self.Disp_u = self.to_tensor(u)
        self.Disp_v = self.to_tensor(v)
        self.mask = roi
        # Get indices where the mask is greater than 0, returning (y, x) coordinates
        self.XY_roi = np.column_stack(np.where(self.mask > 0))
        self.XY_roi = torch.tensor(self.XY_roi)
        
    
    def __len__(self):
        N, _ = self.XY_roi.shape
        return N
    
    def __getitem__(self, idx):
        # Get the (y, x) coordinates corresponding to the index
        y, x = self.XY_roi[idx, :]
        # Extract the u and v values based on the (y, x) coordinates
        u_sample = self.Disp_u[y, x]
        v_sample = self.Disp_v[y, x]
        return u_sample, v_sample, (y, x)

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.tensor(array, dtype=torch.float32)
        elif isinstance(array, (int, float)):
            return torch.tensor([array], dtype=torch.float32)
        else:
            raise TypeError("Unsupported type for to_tensor")
        
    def data_collect(self, device):
        return 0
        
def collate_fn_D(batch):
    # Separate the u, v samples and (y, x) coordinates from the batch and convert to numpy arrays
    u_batch = torch.tensor([item[0] for item in batch]).to(device)
    v_batch = torch.tensor([item[1] for item in batch]).to(device)
    xy_batch = torch.tensor([item[2] for item in batch]).to(device)
    return u_batch, v_batch, xy_batch



class StrainDataset(Dataset):
    def __init__(self, dudx, dudy, dvdx, dvdy, roi):
        self.Du_dx = self.to_tensor(dudx).to(device)
        self.Du_dy = self.to_tensor(dudy).to(device)
        self.Dv_dx = self.to_tensor(dvdx).to(device)
        self.Dv_dy = self.to_tensor(dvdy).to(device)
        
        self.mask = roi
        # Get indices where the mask is greater than 0, returning (y, x) coordinates
        self.XY_roi = np.column_stack(np.where(self.mask > 0))
        self.XY_roi = torch.tensor(self.XY_roi)
        self.XY_roi = self.XY_roi.to(device)
    
    def __len__(self):
        N, _ = self.XY_roi.shape
        return N
    
    def __getitem__(self, idx):
        # Get the (y, x) coordinates corresponding to the index
        # y, x = self.XY_roi[idx, :]
        # Extract the u and v values based on the (y, x) coordinates
        # dudx = self.Du_dx[y, x]
        # dudy = self.Du_dy[y, x]
        # dvdx = self.Dv_dx[y, x]
        # dvdy = self.Dv_dy[y, x]
        # return dudx, dudy, dvdx, dvdy, (y, x)
        return idx

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.tensor(array, dtype=torch.float32)
        elif isinstance(array, (int, float)):
            return torch.tensor([array], dtype=torch.float32)
        else:
            raise TypeError("Unsupported type for to_tensor")
        
    def data_collect(self, device):
        return 0
        
def collate_fn_S(batch):
    # Separate the u, v samples and (y, x) coordinates from the batch and convert to numpy arrays
    # dudx = torch.tensor([item[0] for item in batch])
    # dudy = torch.tensor([item[1] for item in batch])
    # dvdx = torch.tensor([item[2] for item in batch])
    # dvdy = torch.tensor([item[3] for item in batch])
    # xy_batch = torch.tensor([item[4] for item in batch])
    # return dudx, dudy, dvdx, dvdy, xy_batch
    idx = torch.tensor([item for item in batch])
    return idx






# # Load data
# displacement = sio.loadmat('fpb_displacement.mat')
# u = displacement['u']
# v = displacement['v']
# roi = np.ones_like(u)

# # Create dataset and dataloader
# train_dataset = strainDataset(u, v, roi)
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=512, shuffle=True,
#     collate_fn=collate_fn_S
# )

# # Get a batch of data from the dataloader
# data_iter = iter(train_loader)
# u_sample, v_sample, xy_sample  = next(data_iter)

# print(u_sample.shape)  # Print the shape of u samples
# print(v_sample.shape)  # Print the shape of v samples
# print(xy_sample.shape) # Print the shape of selected ROI coordinates
# print(xy_sample)       # Print the actual selected ROI coordinates
