import torch, os
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import scipy.io as sio
from FCNN import DNN, PositionalEncoding
from EarlyStop import EarlyStopping
from Strain_readData import DisplacementDataset, collate_fn_D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''Data preparation'''
roi_path = "./restructed_image/ROI.bmp"
roi = Image.open(roi_path).convert('L')
roi = np.array(roi); roi = roi > 0;

H,L = roi.shape
y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
IX, IY = np.meshgrid(x, y)

displacement = sio.loadmat('fpb_displacement.mat')
u = displacement['u']
v = displacement['v']

early_stop_adam = EarlyStopping(
        patience=10, delta=0.001,
        path='./checkpoint/checkpoint_adam.pth')

'''To tensor'''
IX = torch.tensor(IX).float().to(device)
IY = torch.tensor(IY).float().to(device)

'''Create dataset and dataloader'''
train_dataset = DisplacementDataset(u, v, roi)
N = len(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn_D)

'''Initialize model, loss function, and optimizer'''
num_freq = 10
encoder = PositionalEncoding(num_frequencies=num_freq, input_dims=2)
layers = [2+num_freq*2*2, 100, 100, 100, 2]
model = DNN(layers).to(device)
criterion = torch.nn.MSELoss()
mae = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print("Training start!")
# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    j_loader = 0.00001
    for u_batch, v_batch, xy_batch in train_loader:
        # Forward pass
        x = IX[xy_batch[:,0], xy_batch[:,1]]; y = IY[xy_batch[:,0], xy_batch[:,1]]
        xy = torch.stack((x, y), dim=1)
        encoded_positions = encoder.encode(xy)
        outputs = model(encoded_positions)
        loss = criterion(outputs, torch.stack((u_batch, v_batch), dim=1))
        mae_loss = mae(outputs, torch.stack((u_batch, v_batch), dim=1))
        # Backward pass and optimization
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
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
early_stop_adam.save_checkpoint(epoch_loss, model, optimizer)
print("Training complete!")
