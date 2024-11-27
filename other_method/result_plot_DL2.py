import torch
from PIL import Image
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from FCNN import DNN

def compute_derivatives(Z, dx=1, dy=1):
    # Compute the partial derivative in the x direction (forward difference)
    dZ_dx = np.zeros_like(Z)
    dZ_dx[:-1, :] = (Z[1:, :] - Z[:-1, :]) / dx
    dZ_dx[-1, :] = (Z[-1, :] - Z[-2, :]) / dx
    # Compute the partial derivative in the y direction (forward difference)
    dZ_dy = np.zeros_like(Z)
    dZ_dy[:, :-1] = (Z[:, 1:] - Z[:, :-1]) / dy
    dZ_dy[:, -1] = (Z[:, -1] - Z[:, -2]) / dy
    return dZ_dx, dZ_dy

roi_path = "./restructed_image/ROI.bmp"
roi = Image.open(roi_path).convert('L')
roi = np.array(roi); roi = roi > 0;

displacement = sio.loadmat('star_displacement.mat')
u = displacement['u']
v = displacement['v']
du_dx, du_dy = compute_derivatives(u)
dv_dx, dv_dy = compute_derivatives(v)

H,L = roi.shape
y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
IX, IY = np.meshgrid(x, y)

IX = torch.tensor(IX, requires_grad=True).float()
IY = torch.tensor(IY, requires_grad=True).float()
Ix = IX.flatten()[:, None]
Iy = IX.flatten()[:, None]
xy = torch.cat((Ix, Iy), dim=1)

layers = [2, 50, 50, 50, 4]
model = DNN(layers)
checkpoint_path = './checkpoint/checkpoint_adam.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
derivatives = model(xy)
u_x,u_y,v_x,v_y = \
    derivatives[:, 0],derivatives[:, 1],derivatives[:, 2],derivatives[:, 3]

u_x,u_y,v_x,v_y = \
    u_x.detach().cpu().numpy(), u_y.detach().cpu().numpy(), \
    v_x.detach().cpu().numpy(), v_y.detach().cpu().numpy()
    
u_x = u_x.reshape(H,L); u_y = u_y.reshape(H,L)
v_x = u_x.reshape(H,L); v_y = u_x.reshape(H,L)

ex = u_x; ey = v_y; exy = (u_y + v_x)/2

fig, axs = plt.subplots(4, 2, figsize=(15, 10))

ax=axs[0, 0];im=ax.imshow(u_x, cmap='jet');fig.colorbar(im, ax=ax)
ax=axs[0, 1];im=ax.imshow(u_y, cmap='jet');fig.colorbar(im, ax=ax)
ax=axs[1, 0];im=ax.imshow(v_x, cmap='jet');fig.colorbar(im, ax=ax)
ax=axs[1, 1];im=ax.imshow(v_y, cmap='jet');fig.colorbar(im, ax=ax)

ax=axs[2, 0];im=ax.imshow(du_dx, cmap='jet');fig.colorbar(im, ax=ax)
ax=axs[2, 1];im=ax.imshow(du_dy, cmap='jet');fig.colorbar(im, ax=ax)
ax=axs[3, 0];im=ax.imshow(dv_dx, cmap='jet');fig.colorbar(im, ax=ax)
ax=axs[3, 1];im=ax.imshow(dv_dy, cmap='jet');fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('./result.png', bbox_inches='tight')
print(f"Figure saved to {'./result.png'}")


