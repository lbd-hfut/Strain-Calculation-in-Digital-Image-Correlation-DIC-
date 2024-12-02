import torch
from PIL import Image
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from FCNN import MscaleDNN
from utils import sub_matrix

params = {
    "seed": 0,
    "dim": 2,
    "hidden_units": [50, 50, 50],
    "scales": [1, 2, 4, 8, 16, 32, 64],
    "activation": "phi",
    "roi_path": "./test_data/restructed_image/ROI.bmp",
    "displacement": './test_data/uvmat/star_displacement.mat',
    "checkpoint": [
        './checkpoint/checkpoint_adam_star_u.pth', 
        './checkpoint/checkpoint_adam_star_v.pth'
        ],
    
}

roi_path = params["roi_path"]
roi = Image.open(roi_path).convert('L')
roi = np.array(roi); roi = roi > 0
XY_roi = np.column_stack(np.where(roi > 0))

H,L = roi.shape
y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
IX, IY = np.meshgrid(x, y)

Ix = IX[XY_roi[:,0], XY_roi[:,1]]; Iy = IY[XY_roi[:,0], XY_roi[:,1]]
Ix = torch.tensor(Ix, requires_grad=True).float()
Iy = torch.tensor(Iy, requires_grad=True).float()

modelu = MscaleDNN(
    input_dim=params["dim"],
    hidden_units=params["hidden_units"],
    output_dim=1,
    scales=params["scales"],
    activation=params["activation"]
)
checkpoint_path = params["checkpoint"]
checkpoint = torch.load(checkpoint_path[0])
modelu.load_state_dict(checkpoint['model_state_dict'])

modelv = MscaleDNN(
    input_dim=params["dim"],
    hidden_units=params["hidden_units"],
    output_dim=1,
    scales=params["scales"],
    activation=params["activation"]
)
checkpoint_path = params["checkpoint"]
checkpoint = torch.load(checkpoint_path[1])
modelv.load_state_dict(checkpoint['model_state_dict'])

modelu.eval(); modelv.eval()
u = modelu(torch.stack((Ix, Iy), dim=1))
v = modelv(torch.stack((Ix, Iy), dim=1))
uD = u.detach().cpu().numpy(); vD = v.detach().cpu().numpy()

U = np.zeros((H, L)); V = np.zeros((H, L))
U[XY_roi[:,0], XY_roi[:,1]] = uD.squeeze()
V[XY_roi[:,0], XY_roi[:,1]] = vD.squeeze()

U = sub_matrix(U); V = sub_matrix(V)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

ax = axs[0]
im = ax.imshow(U, cmap='jet')
fig.colorbar(im, ax=ax)

ax = axs[1]
im = ax.imshow(V, cmap='jet')
fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('./result.png', bbox_inches='tight')
print(f"Figure saved to {'./result.png'}")