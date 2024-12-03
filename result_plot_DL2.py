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
    "scales": [1,2],
    "activation": "phi",
    "roi_path": "./test_data/restructed_image/ROI.bmp",
    "displacement": './test_data/uvmat/star_displacement.mat',
    "checkpoint": './checkpoint/checkpoint_adam_star.pth',
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

model = MscaleDNN(
    input_dim=params["dim"],
    hidden_units=params["hidden_units"],
    output_dim=2,
    scales=params["scales"],
    activation=params["activation"]
)
checkpoint_path = params["checkpoint"]
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
uv = model(torch.stack((Ix, Iy), dim=1))
u = uv[:, 0:1]
v = uv[:, 1:2]
# u_x = torch.autograd.grad(u, Ix, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
# u_y = torch.autograd.grad(u, Iy, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
# v_x = torch.autograd.grad(v, Ix, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
# v_y = torch.autograd.grad(v, Iy, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

# u_x = u_x*2/L; u_y = u_y*2/H; v_x = v_x*2/L; u_y = u_y*2/H; 

uD = u.detach().cpu().numpy(); vD = v.detach().cpu().numpy()
# ex = u_x.detach().cpu().numpy(); ey = v_y.detach().cpu().numpy()
# exy = (u_y + v_x)/2; exy = exy.detach().cpu().numpy()

U = np.zeros((H, L)); V = np.zeros((H, L))
Ex = np.zeros((H, L)); Ey = np.zeros((H, L)); Exy = np.zeros((H, L))

U[XY_roi[:,0], XY_roi[:,1]] = uD.squeeze()
V[XY_roi[:,0], XY_roi[:,1]] = vD.squeeze()
# Ex[XY_roi[:,0], XY_roi[:,1]] = ex
# Ey[XY_roi[:,0], XY_roi[:,1]] = ey
# Exy[XY_roi[:,0], XY_roi[:,1]] = exy

U = sub_matrix(U); V = sub_matrix(V)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

ax = axs[0]
im = ax.imshow(U, cmap='jet')
fig.colorbar(im, ax=ax)

ax = axs[1]
im = ax.imshow(V, cmap='jet')
fig.colorbar(im, ax=ax)

plt.tight_layout()
length = len(params["hidden_units"]); unit = params["hidden_units"][0]
plt.savefig(f'./result/result_{length}_{unit}.png', bbox_inches='tight')
print(f"Figure saved to {f'./result/result_{length}_{unit}.png'}")