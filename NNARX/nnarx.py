import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time
import shutil
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Architettura NNARX con Scheduled Sampling per diminuire l'accumulo di errore durante la fase di training
class NNARX(nn.Module):
    def __init__(self, input_dim_u, input_dim_y, hidden_dim, output_dim):
        super(NNARX, self).__init__()
        self.gru_cell = nn.GRUCell(input_dim_u + input_dim_y, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, u_seq, y_init, y_true_seq=None, teacher_forcing_ratio=0.0):
        batch_size, seq_len, _ = u_seq.size()
        h = torch.zeros(batch_size, self.gru_cell.hidden_size).to(u_seq.device)
        current_y = y_init
        predictions = []

        for t in range(seq_len):
            u_t = u_seq[:, t, :]
            # Concateniamo l'ingresso esogeno (motore U) con lo stato autoregressivo corrente (posizione Y) per il NNARX
            nnarx_input = torch.cat([u_t, current_y], dim=1)
            
            h = self.gru_cell(nnarx_input, h)
            hidden_features = self.relu(self.fc1(h))
            pred_y = self.fc2(hidden_features)
            predictions.append(pred_y.unsqueeze(1))
            # riceve un aiuto con la vera posizione
            if y_true_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current_y = y_true_seq[:, t, :] 
            else:
                #si adatta da sola se è maggiore della soglia
                current_y = pred_y               
                
        return torch.cat(predictions, dim=1)

class Dataset(Dataset):
    def __init__(self, X, Y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len - 1
    def __getitem__(self, idx):
        u_seq = self.X[idx : idx + self.seq_len]
        y_init = self.Y[idx]
        y_target_seq = self.Y[idx + 1 : idx + self.seq_len + 1]
        return u_seq, y_init, y_target_seq

class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-5, path='/kaggle/working/temp_model.pth'):
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
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def calculate_metrics(y_true, y_pred):
    sigma = np.std(y_true, axis=0)
    sigma[sigma == 0] = 1e-8
    mse = np.mean((y_true - y_pred)**2, axis=0)
    nrmse = np.sqrt(mse / (sigma**2))
    mean_true = np.mean(y_true, axis=0)
    ss_res = np.sum((y_true - y_pred)**2, axis=0)
    ss_tot = np.sum((y_true - mean_true)**2, axis=0)
    r2 = 100 * (1 - (ss_res / ss_tot))
    bfr = 100 * (1 - np.sqrt(ss_res) / np.sqrt(ss_tot))
    return nrmse, r2, bfr


if __name__ == "__main__":
    TASK = 'INVERSE' #FORWARD o INVERSE    
    SEQ_LEN = 120
    BATCH_SIZE = 128
    HIDDEN_DIM = 64      
    LR = 0.002
    EPOCHS = 200         
    PATIENCE = 40        
    NUM_RESTARTS = 5     
    dim_u = 6 if TASK == 'FORWARD' else 18

    PATH_FORWARD = '/kaggle/input/datasets/contemarco/forwardandinverse/forward_identification_without_raw_data.mat'
    PATH_INVERSE = '/kaggle/input/datasets/contemarco/forwardandinverse/inverse_identification_without_raw_data.mat'
    FINAL_MODEL_PATH = f"/kaggle/working/BEST_ROBUST_FREERUN_{TASK}.pth"

    if TASK == 'FORWARD': 
        mat_data = sio.loadmat(PATH_FORWARD)
    else:
        mat_data = sio.loadmat(PATH_INVERSE)
    
    X_train, Y_train = mat_data['u_train'].T, mat_data['y_train'].T
    X_test, Y_test  = mat_data['u_test'].T, mat_data['y_test'].T

    mean_X, std_X = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    std_X[std_X == 0] = 1e-8
    mean_Y, std_Y = np.mean(Y_train, axis=0), np.std(Y_train, axis=0)
    std_Y[std_Y == 0] = 1e-8
    
    #Normalizzazione
    X_train_norm, X_test_norm = (X_train - mean_X) / std_X, (X_test - mean_X) / std_X
    Y_train_norm, Y_test_norm = (Y_train - mean_Y) / std_Y, (Y_test - mean_Y) / std_Y

    train_loader = DataLoader(Dataset(X_train_norm, Y_train_norm, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Dataset(X_test_norm, Y_test_norm, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)

    best_overall_val_loss = float('inf')
    best_run_idx = -1
    best_train_history, best_val_history = [], []
    best_stopping_epoch = 0

    print(f"Inizio Addestramento ({NUM_RESTARTS} Run)")
    start_total_time = time.time()
    
    for run in range(NUM_RESTARTS):
        run_start = time.time()
        model = NNARX(input_dim_u=dim_u, input_dim_y=6, hidden_dim=HIDDEN_DIM, output_dim=6).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        
        temp_path = f"/kaggle/working/temp_run_{run}.pth"
        early_stopping = EarlyStopping(patience=PATIENCE, path=temp_path)
        
        current_train_history, current_val_history = [], []
        
        for epoch in range(EPOCHS):
            model.train()
            epoch_train_loss = 0.0
            
            # Calcolo decrescente del Teacher Forcing (diminuisce entro il 70% delle epoche totali)
            tf_ratio = max(0.0, 1.0 - (epoch / (EPOCHS * 0.3)))
            
            for u_seq, y_init, y_target_seq in train_loader:
                u_seq, y_init, y_target_seq = u_seq.to(DEVICE), y_init.to(DEVICE), y_target_seq.to(DEVICE)
                optimizer.zero_grad()
                
                # Passiamo il tf_ratio al modello
                preds_seq = model(u_seq, y_init, y_true_seq=y_target_seq, teacher_forcing_ratio=tf_ratio)
                
                loss = criterion(preds_seq, y_target_seq)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            current_train_history.append(avg_train_loss)
            
            #VALIDATION
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for u_seq_val, y_init_val, y_target_seq_val in val_loader:
                    u_seq_val, y_init_val, y_target_seq_val = u_seq_val.to(DEVICE), y_init_val.to(DEVICE), y_target_seq_val.to(DEVICE)
                    
                    preds_seq_val = model(u_seq_val, y_init_val, teacher_forcing_ratio=0.0)
                    val_loss += criterion(preds_seq_val, y_target_seq_val).item()
            
            avg_val_loss = val_loss / len(val_loader)
            current_val_history.append(avg_val_loss)
            
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop: break
        
        if early_stopping.best_loss < best_overall_val_loss:
            best_overall_val_loss = early_stopping.best_loss
            best_run_idx = run + 1
            shutil.copyfile(temp_path, FINAL_MODEL_PATH)
            best_train_history = current_train_history.copy()
            best_val_history = current_val_history.copy()
            best_stopping_epoch = len(current_val_history) - early_stopping.counter - 1
            
        if os.path.exists(temp_path): os.remove(temp_path)

    print(f"\nRun migliore: {best_run_idx} con Loss: {best_overall_val_loss:.6f}")
    
    #Calcolo metriche
    model.load_state_dict(torch.load(FINAL_MODEL_PATH))
    model.eval()

    all_preds_120, all_targets_120 = [], []

    with torch.no_grad():
        for u_seq_val, y_init_val, y_target_seq_val in val_loader:
            u_seq_val, y_init_val = u_seq_val.to(DEVICE), y_init_val.to(DEVICE)
            preds_seq_val = model(u_seq_val, y_init_val, teacher_forcing_ratio=0.0)
            
            all_preds_120.append(preds_seq_val[:, -1, :].cpu().numpy())
            all_targets_120.append(y_target_seq_val[:, -1, :].cpu().numpy())

    all_preds_120 = np.concatenate(all_preds_120, axis=0)
    all_targets_120 = np.concatenate(all_targets_120, axis=0)

    all_preds_denorm = all_preds_120 * std_Y + mean_Y
    all_targets_denorm = all_targets_120 * std_Y + mean_Y

    nrmse_vals, r2_vals, bfr_vals = calculate_metrics(all_targets_denorm, all_preds_denorm)

    print("\nRISULTATI")
    for i in range(6):
        print(f"Giunto {i+1}: NRMSE = {nrmse_vals[i]:.4f} | R2 = {r2_vals[i]:.2f}% | BFR = {bfr_vals[i]:.2f}%")
    print("-" * 70)
    print(f"MEDIA NRMSE    : {np.mean(nrmse_vals):.2f}%")
    print(f"MEDIA R2    : {np.mean(r2_vals):.2f}%")
    print(f"MEDIA BFR    : {np.mean(bfr_vals):.2f}%")
    print("="*70)

    #Grafici
    plt.figure(figsize=(10, 6))
    plt.plot(best_train_history, label='Training Loss (Mixed)', color='#1f77b4', linewidth=2)
    plt.plot(best_val_history, label='Validation Loss (Pure Free-Run)', color='#ff7f0e', linewidth=2)
    plt.axvline(x=best_stopping_epoch, color='green', linestyle='--', linewidth=2, label=f'Miglior Modello (Epoca {best_stopping_epoch+1})')
    plt.title(f'Curve di Apprendimento - Free-Run Robusto (Run #{best_run_idx})', fontsize=14, fontweight='bold')
    plt.xlabel('Epoche', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE Loss)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plot_path = '/kaggle/working/Robust_FreeRun_Loss_Curve.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
