import torch
import torch.nn as nn
from scipy.io import loadmat
import numpy as np
import time

#Hyper-parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 64
batch_size = 64
epochs = 150
learning_rate = 1e-3
seq_len = 120
input_dim = 6
output_dim = 6

#Modello
class RobotRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RobotRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        prediction = self.fc(out[:, -1, :])
        return prediction

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
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


#Caricamento e normalizzazione
def normalization_data(datasetPath, seq_len, stats=None):
    import os
    if not os.path.exists(datasetPath):
        raise FileNotFoundError(f"Il file {datasetPath} non esiste!")

    data = loadmat(datasetPath)
    raw_input = data.get('u_train', data.get('u_test', data.get('u')))
    raw_target = data.get('y_train', data.get('y_test', data.get('y')))

    if raw_input.shape[0] < raw_input.shape[1]:
        raw_input, raw_target = raw_input.T, raw_target.T

    if stats is None:
        mean_in, std_in = raw_input.mean(axis=0), raw_input.std(axis=0)
        mean_out, std_out = raw_target.mean(axis=0), raw_target.std(axis=0)
    else:
        mean_in, std_in, mean_out, std_out = stats

    input_norm = (raw_input - mean_in) / (std_in + 1e-7)
    target_norm = (raw_target - mean_out) / (std_out + 1e-7)

    X, Y = [], []
    for i in range(len(input_norm) - seq_len):
        X.append(input_norm[i : i + seq_len])
        Y.append(target_norm[i + seq_len])

    return (torch.tensor(np.array(X), dtype=torch.float32), 
            torch.tensor(np.array(Y), dtype=torch.float32), 
            (mean_in, std_in, mean_out, std_out))

def get_official_report(model, dataloader, stats_out):
    model.eval()
    all_preds, all_targets = [], []
    mean_out, std_out = stats_out

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            all_preds.append(pred.cpu().numpy() * std_out + mean_out)
            all_targets.append(y.cpu().numpy() * std_out + mean_out)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
  
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-7)) 

    #NRMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    nrmse = rmse / (np.std(y_true, axis=0) + 1e-7)
    
    #BFR
    numerator = np.linalg.norm(y_true - y_pred, axis=0)
    denominator = np.linalg.norm(y_true - np.mean(y_true, axis=0), axis=0)
    bfr = 100 * (1 - (numerator / (denominator + 1e-7)))

    print("\n==================================================")
    print(" RISULTATI TEST (Simulation Error One-Step)")
    print("==================================================")
    for i in range(6):
        print(f"Giunto {i+1}: NRMSE = {nrmse[i]:.4f} | R2 = {r2[i]*100:.2f}% | BFR = {bfr[i]:.2f}%")
    print("-" * 50)
    print(f"MEDIA NRMSE : {np.mean(nrmse):.4f}")
    print(f"MEDIA R2    : {np.mean(r2)*100:.2f}%")
    print(f"MEDIA BFR   : {np.mean(bfr):.2f}%")
    print("==================================================")


#Training
X_train, Y_train, full_stats = normalization_data('forward_identification_without_raw_data.mat', seq_len)
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#Test
X_test, Y_test, _ = normalization_data('forward_identification_without_raw_data.mat', seq_len, stats=full_stats)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

early_stopping = EarlyStopping(patience=25, min_delta=1e-5, path='best_model.pth')

model = RobotRNN(input_dim, hidden_size, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

print(f"Inizio training su {device}...")
start_time = time.time()

for epoch in range(epochs):
    
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss_val = loss_fn(pred, y)
            val_loss += loss_val.item()
            
    avg_val_loss = val_loss / len(test_loader)

    print(f"Epoch [{epoch+1:03d}/{epochs}] -> Train Loss (MSE): {avg_train_loss:.6f} | Validation Loss (MSE): {avg_val_loss:.6f}")

    early_stopping(avg_val_loss, model)
    
    if early_stopping.early_stop:
        print(f"\n-> Early stopping attivato all'epoca {epoch+1}. Il modello ha smesso di migliorare.")
        break

train_time = time.time() - start_time
print(f"\nTraining completato in: {train_time:.2f} secondi")

print("Caricamento del miglior modello salvato dall'Early Stopping...")
model.load_state_dict(torch.load('best_model.pth'))

get_official_report(model, test_loader, (full_stats[2], full_stats[3]))
