import torch
from PIL import Image
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from FCNN import DNN, PositionalEncoding
from utils import sub_matrix, compute_derivatives

roi_path = "./restructed_image/ROI.bmp"
roi = Image.open(roi_path).convert('L')
roi = np.array(roi); roi = roi > 0;
XY_roi = np.column_stack(np.where(roi > 0))

displacement = sio.loadmat('star_displacement.mat')
u = displacement['u']
v = displacement['v']
du_dx, du_dy = compute_derivatives(u)
dv_dx, dv_dy = compute_derivatives(v)

H,L = roi.shape
y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
IX, IY = np.meshgrid(x, y)

Ix = IX[XY_roi[:,0], XY_roi[:,1]]; Iy = IY[XY_roi[:,0], XY_roi[:,1]]
Ix = torch.tensor(Ix, requires_grad=True).float()
Iy = torch.tensor(Iy, requires_grad=True).float()

num_freq = 10
encoder = PositionalEncoding(num_frequencies=num_freq, input_dims=2)
layers = [num_freq*2*2, 100, 100, 100, 4]
model = DNN(layers)
checkpoint_path = './checkpoint/checkpoint_adam_fpb_ed.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
xy = torch.stack((Ix, Iy), dim=1)
encoded_positions = encoder.encode(xy)
outputs = model(encoded_positions[:,2:])
u_x,u_y,v_x,v_y=outputs[:, 0],outputs[:, 1],outputs[:, 2],outputs[:, 3]

u_x = u_x.detach().cpu().numpy(); u_y = u_y.detach().cpu().numpy()
v_x = v_x.detach().cpu().numpy(); v_y = v_y.detach().cpu().numpy()

Du_x = np.zeros((H, L)); Du_y = np.zeros((H, L))
Dv_x = np.zeros((H, L)); Dv_y = np.zeros((H, L))

Du_x[XY_roi[:,0], XY_roi[:,1]] = u_x
Du_y[XY_roi[:,0], XY_roi[:,1]] = u_y
Dv_x[XY_roi[:,0], XY_roi[:,1]] = v_x
Dv_y[XY_roi[:,0], XY_roi[:,1]] = v_y


Du_x = sub_matrix(Du_x); Du_y = sub_matrix(Du_y)
Dv_x = sub_matrix(Dv_x); Dv_y = sub_matrix(Dv_y)

fig, axs = plt.subplots(4, 2, figsize=(15, 10))

ax=axs[0, 0];im=ax.imshow(Du_x, cmap='jet');fig.colorbar(im, ax=ax)
ax=axs[0, 1];im=ax.imshow(Du_y, cmap='jet');fig.colorbar(im, ax=ax)
ax=axs[1, 0];im=ax.imshow(Dv_x, cmap='jet');fig.colorbar(im, ax=ax)
ax=axs[1, 1];im=ax.imshow(Dv_y, cmap='jet');fig.colorbar(im, ax=ax)

ax=axs[2, 0];im=ax.imshow(du_dx, cmap='jet');fig.colorbar(im, ax=ax)
ax=axs[2, 1];im=ax.imshow(du_dy, cmap='jet');fig.colorbar(im, ax=ax)
ax=axs[3, 0];im=ax.imshow(dv_dx, cmap='jet');fig.colorbar(im, ax=ax)
ax=axs[3, 1];im=ax.imshow(dv_dy, cmap='jet');fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('./result.png', bbox_inches='tight')
print(f"Figure saved to {'./result.png'}")


