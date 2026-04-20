import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


#Class definition for early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=25, min_delta=1e-5, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


#Class definition for the GRU network architecture
class RobotGRU(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=6):
        super(RobotGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    n = param.size(0) // 3
                    param.data.fill_(0)
                    with torch.no_grad():
                        bias_vals = torch.arange(1, n + 1) * 0.2
                        param.data[n:2*n].copy_(bias_vals) 

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc1(out[:, -1, :])
        out = self.tanh(out)
        return self.fc2(out)


#Data loading and preprocessing
def prepare_benchmark_data(mat_path, seq_len):
    data = scipy.io.loadmat(mat_path)
    u_all = data['u_train'].T
    y_all = data['y_train'].T
    
    scaler_u = MinMaxScaler(feature_range=(-1, 1)).fit(u_all)
    scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(y_all)
    
    u_s = scaler_u.transform(u_all)
    y_s = scaler_y.transform(y_all)

    def create_sequences(u, y, l):
        Xs, ys = [], []
        for i in range(len(u) - l):
            Xs.append(u[i:(i + l)])
            ys.append(y[i + l])
        return torch.tensor(np.array(Xs), dtype=torch.float32), \
               torch.tensor(np.array(ys), dtype=torch.float32)


    split_idx = int(len(u_s) * 0.8)
    tX, ty = create_sequences(u_s[:split_idx], y_s[:split_idx], seq_len)
    vX, vy = create_sequences(u_s[split_idx:], y_s[split_idx:], seq_len)
    
    return tX, ty, vX, vy, scaler_y

#Network and training configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 120
HIDDEN_DIM = 64
LR = 0.01
EPOCHS = 300
NUM_RESTARTS = 20
PATIENCE = 40
DATA_FILE = r'Robot_Identification_Benchmark_Without_Raw_Data\forward_identification_without_raw_data.mat'
LOG_FILE = "training_log.txt"

#Multi-start training loop
tX, ty, vX, vy, scaler_y = prepare_benchmark_data(DATA_FILE, SEQ_LEN)
train_loader = DataLoader(TensorDataset(tX, ty), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(vX, vy), batch_size=128, shuffle=False)

best_overall_val_loss = float('inf')
final_model_path = f"best_of_{NUM_RESTARTS}_runs.pth"

print(f"Starting Multi-start session ({NUM_RESTARTS} initializations) on {DEVICE}")

#Log initialization
with open(LOG_FILE, "w") as log_f:
    log_f.write(f"{datetime.now()} - Multi-start session ({NUM_RESTARTS} initializations) on {DEVICE}\n")
    log_f.write(f"Configuration: SEQ_LEN={SEQ_LEN}, HIDDEN_DIM={HIDDEN_DIM}, LR={LR}, EPOCHS={EPOCHS}\n\n")

for run in range(NUM_RESTARTS):
    run_start_time = time.time()
    model = RobotGRU(6, HIDDEN_DIM, 6).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    early_stopping = EarlyStopping(patience=PATIENCE, path=f"temp_run_{run}.pth")
    
    for epoch in range(EPOCHS):
        model.train()
        for b_X, b_y in train_loader:
            b_X, b_y = b_X.to(DEVICE), b_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(b_X), b_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bv_X, bv_y in val_loader:
                val_loss += criterion(model(bv_X.to(DEVICE)), bv_y.to(DEVICE)).item()
        
        avg_v = val_loss / len(val_loader)
        early_stopping(avg_v, model)
        if early_stopping.early_stop: break
 
    if early_stopping.best_loss < best_overall_val_loss:
        best_overall_val_loss = early_stopping.best_loss
        torch.save(model.state_dict(), final_model_path)
    
    print(f"Run {run+1}/{NUM_RESTARTS}. Best Val Loss: {early_stopping.best_loss:.6f}")
    with open(LOG_FILE, "a") as log_f:
        log_f.write(f"{datetime.now()} - Run {run+1}/{NUM_RESTARTS} - Best Val Loss: {early_stopping.best_loss:.6f} - Time: {time.time() - run_start_time:.2f}s\n")
        

#Metrics computation on the best model
model.load_state_dict(torch.load(final_model_path))
model.eval()
with torch.no_grad():
    y_pred_s = model(vX.to(DEVICE)).cpu().numpy()
    y_true_s = vy.numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_s)
    y_true = scaler_y.inverse_transform(y_true_s)
    
    r2_all = []
    print("\n--- Result R2 Joint ---")
    for n in range(y_true.shape[1]):
        res_ss = np.sum((y_true[:, n] - y_pred[:, n])**2)
        tot_ss = np.sum((y_true[:, n] - np.mean(y_true[:, n]))**2)
        
        r2 = 100 * (1 - (res_ss / tot_ss))
        r2_all.append(r2)
        print(f"Joint {n+1}: R2 = {r2:.2f}%")

# Calcolo della media totale
r2_mean = np.mean(r2_all)
print(f"\nTotal R2 mean: {r2_mean:.2f}%")