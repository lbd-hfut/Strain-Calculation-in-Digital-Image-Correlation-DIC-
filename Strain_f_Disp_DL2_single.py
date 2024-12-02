import torch, os
import torch.optim as optim
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
    "scales": [1, 2, 4, 8, 16, 32, 64],
    "activation": "phi",
    "learning_rate": 1e-3,
    "eta_min": 1e-4,
    "total_steps": 15,
    "batch_size": 256,
    "scheduler_T": 15,
    "roi_path": "./test_data/restructed_image/ROI.bmp",
    "displacement": './test_data/uvmat/star_displacement.mat',
    "ep_patience": 5,
    "ep_delta": 0.0005,
    "checkpoint": [
        './checkpoint/checkpoint_adam_star_u.pth', 
        './checkpoint/checkpoint_adam_star_v.pth'
        ],
    "shuffle": True, 
}


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

'''To tensor'''
IX = torch.tensor(IX, requires_grad=False).float().to(device)
IY = torch.tensor(IY, requires_grad=False).float().to(device)

'''Create dataset and dataloader'''
train_dataset = DisplacementDataset(u, v, roi)
N = len(train_dataset)
train_loader = DataLoader(
    train_dataset, batch_size=params["batch_size"], 
    shuffle=params["shuffle"], collate_fn=collate_fn_D
    )

'''Initialize model, loss function, and optimizer'''
modelu = MscaleDNN(
    input_dim=params["dim"],
    hidden_units=params["hidden_units"],
    output_dim=1,
    scales=params["scales"],
    activation=params["activation"]
).to(device)

modelv = MscaleDNN(
    input_dim=params["dim"],
    hidden_units=params["hidden_units"],
    output_dim=1,
    scales=params["scales"],
    activation=params["activation"]
).to(device)

criterion = torch.nn.MSELoss()
mae = torch.nn.L1Loss()
optimizeru = torch.optim.Adam(modelu.parameters(), lr=params["learning_rate"])
optimizerv = torch.optim.Adam(modelv.parameters(), lr=params["learning_rate"])
scheduleru = optim.lr_scheduler.CosineAnnealingLR(
    optimizeru, T_max=params["scheduler_T"], eta_min=params["eta_min"])
schedulerv = optim.lr_scheduler.CosineAnnealingLR(
    optimizerv, T_max=params["scheduler_T"], eta_min=params["eta_min"])

modelu.Earlystop_set(patience=params["ep_patience"], 
        delta=params["ep_delta"], 
        path=params["checkpoint"][0])
modelv.Earlystop_set(patience=params["ep_patience"], 
        delta=params["ep_delta"], 
        path=params["checkpoint"][1])

print("Training start!")
# Training loop
num_epochs = params["total_steps"]
modelu.train(); modelv.train()
for epoch in range(num_epochs):
    running_loss_u = 0.0
    running_loss_v = 0.0
    j_loader = 0.00001
    for u_batch, v_batch, xy_batch in train_loader:
        # Forward pass
        x = IX[xy_batch[:,0], xy_batch[:,1]]; y = IY[xy_batch[:,0], xy_batch[:,1]]
        xy = torch.stack((x, y), dim=1)
        outputsu = modelu(xy); outputsv = modelv(xy)
        target = torch.stack((u_batch, v_batch), dim=1)
        outputs = torch.cat((outputsu, outputsv), dim=1)
        loss = criterion(outputs, target)
        mae_loss_u = mae(outputs[:,0], target[:,0])
        mae_loss_v = mae(outputs[:,1], target[:,1])
        # Backward pass and optimization
        optimizeru.zero_grad()
        optimizerv.zero_grad()
        loss.backward()
        optimizeru.step()
        optimizerv.step()
        scheduleru.step()
        schedulerv.step()
        
        running_loss_u += mae_loss_u.item()
        running_loss_v += mae_loss_v.item()
        j_loader += 1
    epoch_loss_u = running_loss_u / j_loader
    epoch_loss_v = running_loss_v / j_loader
    modelu.Earlystop(epoch_loss_u, modelu, epoch)
    modelv.Earlystop(epoch_loss_v, modelu, epoch)
    
    if modelu.early_stop or modelv.early_stop:
        if modelu.early_stop:
            modelu.freeze_all_parameters()
        if modelv.early_stop:
            modelv.freeze_all_parameters()
        if modelu.early_stop and modelv.early_stop:
            print("Early stopping triggered")
            break
    print(f'Epoch [{epoch+1}/{num_epochs}], Lossu: {epoch_loss_u:.4f}, Lossv: {epoch_loss_v:.4f}')
modelu.save_checkpoint(epoch_loss_u, modelu)
modelv.save_checkpoint(epoch_loss_v, modelv)
print("Training complete!")