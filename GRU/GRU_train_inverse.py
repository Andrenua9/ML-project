import torch
import torch.nn as nn
import numpy as np
import scipy.io
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

#Best fit rate calculation function
def calculate_bfr(y_true, y_pred):
    numerator = np.linalg.norm(y_true - y_pred)
    denominator = np.linalg.norm(y_true - np.mean(y_true))
    return (1 - (numerator / denominator)) * 100

#Data preprocessing function
def prepare_inverse_data(mat_path, seq_len=50):
    data = scipy.io.loadmat(mat_path)
    X_train_raw, y_train_raw = data['u_train'].T, data['y_train'].T
    X_test_raw, y_test_raw = data['u_test'].T, data['y_test'].T

    scaler_x = StandardScaler().fit(X_train_raw)
    scaler_y = StandardScaler().fit(y_train_raw)

    X_train, y_train = scaler_x.transform(X_train_raw), scaler_y.transform(y_train_raw)
    X_test, y_test = scaler_x.transform(X_test_raw), scaler_y.transform(y_test_raw)

    def create_sequences(X, y, l):
        Xs, ys = [], []
        for i in range(len(X) - l):
            Xs.append(X[i:(i + l)])
            ys.append(y[i + l])
        return torch.tensor(np.array(Xs), dtype=torch.float32), \
               torch.tensor(np.array(ys), dtype=torch.float32)

    tX, ty = create_sequences(X_train, y_train, seq_len)
    vX, vy = create_sequences(X_test, y_test, seq_len)
    return tX, ty, vX, vy, scaler_y


#Class definition for the GRU network architecture
class RobotGRU(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, output_dim=6, n_layers=3):
        super(RobotGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


#Network and training configuration
FILE_PATH = r'Robot_Identification_Benchmark_Without_Raw_Data\inverse_identification_without_raw_data.mat'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN, BATCH_SIZE, EPOCHS = 120, 128, 200
HIDDEN_DIM, N_LAYERS = 64, 3
PATIENCE = 20
MODEL_NAME = f"robot_inverse_gru_h{HIDDEN_DIM}_l{N_LAYERS}.pth"
LOG_FILENAME = "training_log_inverse.txt"


tX, ty, vX, vy, scaler_y = prepare_inverse_data(FILE_PATH, SEQ_LEN)
train_loader = DataLoader(TensorDataset(tX, ty), batch_size=BATCH_SIZE, shuffle=True) 
val_loader = DataLoader(TensorDataset(vX, vy), batch_size=BATCH_SIZE, shuffle=False)

model = RobotGRU(input_dim=18, hidden_dim=HIDDEN_DIM, output_dim=6, n_layers=N_LAYERS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


best_val_loss = float('inf')
epochs_without_improvement = 0
start_time = time.time()

print(f"Starting training on {DEVICE}...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for b_vX, b_vy in val_loader:
            b_vX, b_vy = b_vX.to(DEVICE), b_vy.to(DEVICE)
            val_outputs = model(b_vX)
            v_loss = criterion(val_outputs, b_vy)
            val_loss += v_loss.item()
    
    avg_val_loss = val_loss / len(val_loader)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), MODEL_NAME)
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping attivato all'epoca {epoch+1}. Miglior Val Loss: {best_val_loss:.6f}")
        break

model.load_state_dict(torch.load(MODEL_NAME))
train_time_min = (time.time() - start_time) / 60

#Metrics computation on the best model
model.eval()
with torch.no_grad():
    raw_pred = model(vX.to(DEVICE)).cpu().numpy()
    y_pred = scaler_y.inverse_transform(raw_pred)
    y_true = scaler_y.inverse_transform(vy.numpy())

bfr_results = []
r2_results = []

#Loggin results on file and console
with open(LOG_FILENAME, "a") as f:
    header = (f"\n{'='*60}\n"
              f"SESSION: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
              f"Model: {MODEL_NAME} | Time: {train_time_min:.2f} min\n"
              f"Config: Hidden={HIDDEN_DIM}, Layers={N_LAYERS}, Seq={SEQ_LEN}, Best Val Loss={best_val_loss:.6f}\n"
              f"{'-'*60}\n")
    f.write(header)
    print(header)
    
    for i in range(6):
        bfr = calculate_bfr(y_true[:, i], y_pred[:, i])
        bfr_results.append(bfr)

        ss_res = np.sum((y_true[:, i] - y_pred[:, i])**2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i]))**2)
        r2 = 1 - (ss_res / ss_tot)
        r2_results.append(r2 * 100)

        sigma = np.std(y_true[:, i])
        nrmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i])**2)) / sigma
        res_line = f"Joint {i+1} | BFR: {bfr:6.2f}% | NRMSE: {nrmse:.4f} | R2: {r2_results[i]:.2f}%\n"
        f.write(res_line)
        print(res_line, end="")

    avg_bfr = np.mean(bfr_results)
    avg_r2 = np.mean(r2_results)

    footer = f"{'-'*60}\nTotal BFR mean: {avg_bfr:.2f}%\n{'='*60}\nTotal R2 mean: {avg_r2:.2f}%\n"
    f.write(footer)
    print(f"\nTotal Average BFR Inverse: {avg_bfr:.2f}%\nTotal Average R2 Inverse: {avg_r2:.2f}%\n")