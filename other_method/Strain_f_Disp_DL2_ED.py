import torch
from PIL import Image
import numpy as np
import scipy.io as sio
from FCNN import DNN, PositionalEncoding
from EarlyStop import EarlyStopping
from torch.utils.data import DataLoader
from Strain_readData import StrainDataset, collate_fn_S
from utils import compute_derivatives

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''Data preparation'''
roi_path = "./restructed_image/ROI.bmp"
roi = Image.open(roi_path).convert('L')
roi = np.array(roi); roi = roi > 0;

H,L = roi.shape
y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L);
IX, IY = np.meshgrid(x, y)

displacement = sio.loadmat('star_displacement.mat')
u = displacement['u']
v = displacement['v']
du_dx, du_dy = compute_derivatives(u)
dv_dx, dv_dy = compute_derivatives(v)

early_stop_adam = EarlyStopping(
        patience=20, delta=0.001, 
        path='./checkpoint/checkpoint_adam.pth')

'''Create dataset and dataloader'''
train_dataset = StrainDataset(du_dx, du_dy, dv_dx, dv_dy, roi)
N = len(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn_S)

'''To tensor'''
IX = torch.tensor(IX, requires_grad=True).float().to(device); IX = IX.flatten()[:, None]
IY = torch.tensor(IY, requires_grad=True).float().to(device); IY = IX.flatten()[:, None]
du_dx = torch.tensor(du_dx).float().to(device); du_dx = du_dx.flatten()[:, None]
du_dy = torch.tensor(du_dy).float().to(device); du_dy = du_dy.flatten()[:, None]
dv_dx = torch.tensor(dv_dx).float().to(device); dv_dx = dv_dx.flatten()[:, None]
dv_dy = torch.tensor(dv_dy).float().to(device); dv_dy = dv_dy.flatten()[:, None]

'''Initialize model, loss function, and optimizer'''
num_freq = 10
encoder = PositionalEncoding(num_frequencies=num_freq, input_dims=2)
layers = [num_freq*2*2, 100, 100, 100, 4]
model = DNN(layers).to(device)
criterion = torch.nn.MSELoss()
mae = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training start!")
# Training loop
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    j_loader = 0.0
    for idx in train_loader:
        x, y, ux, uy, vx, vy = IX[idx,:], IY[idx,:], du_dx[idx,:], du_dy[idx,:], dv_dx[idx,:], dv_dy[idx,:]
        derivatives = torch.cat((ux, uy, vx, vy), dim=1)
        xy = torch.cat((x, y), dim=1)
        encoded_positions = encoder.encode(xy)
        outputs = model(encoded_positions[:,2:])
        u_x,u_y,v_x,v_y=outputs[:, 0],outputs[:, 1],outputs[:, 2],outputs[:, 3]
        u_xy = torch.autograd.grad(u_x, y, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yx = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        v_xy = torch.autograd.grad(v_x, y, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        v_yx = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
        loss1 = criterion(u_xy,u_yx) + criterion(v_xy,v_yx)
        loss2 = criterion(outputs, derivatives)
        loss = loss1 * 0.0 + loss2
        mae_loss = mae(outputs, derivatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += mae_loss.item()
        j_loader += 1
    epoch_loss = running_loss / j_loader
    early_stop_adam(epoch_loss, model, optimizer)
    if early_stop_adam.early_stop:
        print("Early stopping triggered")
        break
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Compatibility: {loss1.item():.4f}')
early_stop_adam.save_checkpoint(epoch_loss, model, optimizer)
print("Training complete!")

