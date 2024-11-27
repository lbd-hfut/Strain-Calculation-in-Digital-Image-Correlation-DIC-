import torch
from PIL import Image
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from FCNN import DNN, PositionalEncoding
from utils import sub_matrix

roi_path = "./restructed_image/ROI.bmp"
roi = Image.open(roi_path).convert('L')
roi = np.array(roi); roi = roi > 0;
XY_roi = np.column_stack(np.where(roi > 0))

H,L = roi.shape
y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
IX, IY = np.meshgrid(x, y)

Ix = IX[XY_roi[:,0], XY_roi[:,1]]; Iy = IY[XY_roi[:,0], XY_roi[:,1]]
Ix = torch.tensor(Ix, requires_grad=True).float()
Iy = torch.tensor(Iy, requires_grad=True).float()

num_freq = 10
encoder = PositionalEncoding(num_frequencies=num_freq, input_dims=2)
layers = [2+num_freq*2*2, 100, 100, 100, 2]
model = DNN(layers)
checkpoint_path = './checkpoint/checkpoint_adam_fpb_ed.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
xy = torch.stack((Ix, Iy), dim=1)
encoded_positions = encoder.encode(xy)
uv = model(encoded_positions)
u = uv[:, 0:1]
v = uv[:, 1:2]
u_x = torch.autograd.grad(u, Ix, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
u_y = torch.autograd.grad(u, Iy, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
v_x = torch.autograd.grad(v, Ix, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
v_y = torch.autograd.grad(v, Iy, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

uD = u.detach().cpu().numpy(); vD = v.detach().cpu().numpy()
ex = u_x.detach().cpu().numpy(); ey = v_y.detach().cpu().numpy()
exy = (u_y + v_x)/2; exy = exy.detach().cpu().numpy()

U = np.zeros((H, L)); V = np.zeros((H, L))
Ex = np.zeros((H, L)); Ey = np.zeros((H, L)); Exy = np.zeros((H, L))

U[XY_roi[:,0], XY_roi[:,1]] = uD.squeeze()
V[XY_roi[:,0], XY_roi[:,1]] = vD.squeeze()
Ex[XY_roi[:,0], XY_roi[:,1]] = ex
Ey[XY_roi[:,0], XY_roi[:,1]] = ey
Exy[XY_roi[:,0], XY_roi[:,1]] = exy

U = sub_matrix(U); V = sub_matrix(V)
Ex = sub_matrix(Ex); Ey = sub_matrix(Ey); Exy = sub_matrix(Exy)

# 设置图形和子图布局
fig, axs = plt.subplots(3, 2, figsize=(15, 10))

# 第一个子图
ax = axs[0, 0]
im = ax.imshow(U, cmap='jet')
fig.colorbar(im, ax=ax)

# 第二个子图
ax = axs[0, 1]
im = ax.imshow(V, cmap='jet')
fig.colorbar(im, ax=ax)

# 第三个子图
ax = axs[1, 0]
im = ax.imshow(Ex, cmap='jet')
fig.colorbar(im, ax=ax)

# 第四个子图
ax = axs[1, 1]
im = ax.imshow(Ey, cmap='jet')
fig.colorbar(im, ax=ax)

# 第五个子图
ax = axs[2, 0]
im = ax.imshow(Exy, cmap='jet')
fig.colorbar(im, ax=ax)

# 第六个子图（虽然只需要五个，但为了完整展示六个）
ax = axs[2, 1]
im = ax.imshow(np.zeros_like(Exy), cmap='jet')
fig.colorbar(im, ax=ax)

# 调整子图间距
plt.tight_layout()
plt.savefig('./result.png', bbox_inches='tight')
print(f"Figure saved to {'./result.png'}")


