import torch
from PIL import Image
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from FCNN import DNN
from utils import sub_matrix

roi_path = "./test_data/restructed_image/ROI.bmp"
roi = Image.open(roi_path).convert('L')
roi = np.array(roi); roi = roi > 0
XY_roi = np.column_stack(np.where(roi > 0))

H,L = roi.shape
y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
IX, IY = np.meshgrid(x, y)

Ix = IX[XY_roi[:,0], XY_roi[:,1]]; Iy = IY[XY_roi[:,0], XY_roi[:,1]]
Ix = torch.tensor(Ix, requires_grad=True).float()
Iy = torch.tensor(Iy, requires_grad=True).float()

layers = [2, 50, 50, 50, 2]
model = DNN(layers)
checkpoint_path = './checkpoint/checkpoint_adam.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
uv = model(torch.stack((Ix, Iy), dim=1))
u = uv[:, 0:1]
v = uv[:, 1:2]

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


