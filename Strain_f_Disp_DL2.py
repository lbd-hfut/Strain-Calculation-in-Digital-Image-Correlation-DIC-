import torch, os
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import scipy.io as sio
from FCNN import MscaleDNN
from EarlyStop import EarlyStopping
from Strain_readData import DisplacementDataset, collate_fn_D

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''Training parameters'''
params = {
    "seed": 0,
    "dim": 2,
    "hidden_units": [50, 50, 50],
    "scales": [1,2,4,8],
    "activation": "sin",
    "learning_rate": 1e-3,
    "total_steps": 15,
    "batch_size": 256*4,
    "roi_path": "./test_data/restructed_image/ROI.bmp",
    "displacement": './test_data/uvmat/star_displacement.mat',
    "log_prefix": 'test',
    "ep_patience": 20,
    "ep_delta": 0.0001,
    "checkpoint": './checkpoint/checkpoint_adam_star.pth',
    "shuffle": True, 
}

set_seed(params["seed"])

'''Data preparation'''
roi_path = params["roi_path"]
roi = Image.open(roi_path).convert('L')
roi = np.array(roi); roi = roi > 0

H,L = roi.shape
y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
IX, IY = np.meshgrid(x, y)

displacement = sio.loadmat(params["displacement"])
u = displacement['u']
v = displacement['v']

early_stop_adam = EarlyStopping(
        patience=params["ep_patience"], 
        delta=params["ep_delta"], 
        path=params["checkpoint"])

'''To tensor'''
IX = torch.tensor(IX, requires_grad=False).float().to(device)
IY = torch.tensor(IY, requires_grad=False).float().to(device)

'''Create dataset and dataloader'''
train_dataset = DisplacementDataset(u, v, roi)
N = len(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=params["shuffle"], collate_fn=collate_fn_D)

'''Initialize model, loss function, and optimizer'''
model = MscaleDNN(
    input_dim=params["dim"],
    hidden_units=params["hidden_units"],
    output_dim=2,
    scales=params["scales"],
    activation=params["activation"]
).to(device)

criterion = torch.nn.MSELoss()
mae = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

print("Training start!")
# Training loop
num_epochs = params["total_steps"]
layer = len(params["hidden_units"])
neuron = params["hidden_units"][0]
acfun = params["activation"]
scale = len(params["scales"])
print(params["hidden_units"]); print(params["scales"])
file = open('./log/'+params["log_prefix"]+f'_{layer}_{neuron}_{acfun}_{scale}.txt', 'w')
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    j_loader = 0.00001
    for u_batch, v_batch, xy_batch in train_loader:
        # Forward pass
        x = IX[xy_batch[:,0], xy_batch[:,1]]; y = IY[xy_batch[:,0], xy_batch[:,1]]
        xy = torch.stack((x, y), dim=1)
        outputs = model(xy)
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
    file.write(f"\tEpoch:{epoch+1:02d}/{15}\tMAE\t{epoch_loss:.4f}\n")
early_stop_adam.save_checkpoint(epoch_loss, model, optimizer)
print("Training complete!")