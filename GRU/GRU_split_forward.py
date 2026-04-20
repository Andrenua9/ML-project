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

  def __init__(self, patience=30, min_delta=1e-5, path='best_model.pth'):

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

  
#Class definition for the split network architecture
class RobotSplitGRU(nn.Module):

  def __init__(self, input_dim=6, hidden_low=64, hidden_high=128, output_dim=6):

    super(RobotSplitGRU, self).__init__()


    self.gru_123 = nn.GRU(input_dim, hidden_low, num_layers=1, batch_first=True)

    self.fc_123 = nn.Sequential(

      nn.Linear(hidden_low, hidden_low),

      nn.Tanh(),

      nn.Linear(hidden_low, 3)

    )

    self.gru_456 = nn.GRU(input_dim, hidden_high, num_layers=1, batch_first=True)

    self.fc_456 = nn.Sequential(

      nn.Linear(hidden_high, hidden_high),

      nn.Tanh(),

      nn.Linear(hidden_high, 3)

    )

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

    out123, _ = self.gru_123(x)

    out456, _ = self.gru_456(x)

    y123 = self.fc_123(out123[:, -1, :])

    y456 = self.fc_456(out456[:, -1, :])

    return torch.cat((y123, y456), dim=1)


#Data loading and preprocessing
def prepare_benchmark_data(mat_path, seq_len):

  data = scipy.io.loadmat(mat_path)

  u_train, y_train = data['u_train'].T, data['y_train'].T

  u_test, y_test = data['u_test'].T, data['y_test'].T

  scaler_u = MinMaxScaler(feature_range=(-1, 1)).fit(u_train)

  scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(y_train)

  def create_sequences(u, y, l, s_u, s_y):

    Xs, ys = [], []

    u_scaled = s_u.transform(u)

    y_scaled = s_y.transform(y)

    for i in range(len(u_scaled) - l):

      Xs.append(u_scaled[i:(i + l)])

      ys.append(y_scaled[i + l])

    return torch.tensor(np.array(Xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)

  tX, ty = create_sequences(u_train, y_train, seq_len, scaler_u, scaler_y)

  vX, vy = create_sequences(u_test, y_test, seq_len, scaler_u, scaler_y)

  return tX, ty, vX, vy, scaler_y

  

#Network and training configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LEN = 120

HIDDEN_LOW = 64 # Neurons for joints 1,2,3

HIDDEN_HIGH = 128 # Neurons for joints 4,5,6

LR = 0.01

EPOCHS = 300

NUM_RESTARTS = 10

LOG_FILE = "training_log.txt"

DATA_FILE = r'Robot_Identification_Benchmark_Without_Raw_Data\forward_identification_without_raw_data.mat'

PATIENCE = 40 

#Log initialization
with open(LOG_FILE, "a") as log_f:

  log_f.write(f"{datetime.now()} - Inizio sessione Multi-start ({NUM_RESTARTS} inizializzazioni) su {DEVICE}\n")

  log_f.write(f"Configurazione: SEQ_LEN={SEQ_LEN}, HIDDEN_LOW={HIDDEN_LOW}, HIDDEN_HIGH={HIDDEN_HIGH}, LR={LR}, EPOCHS={EPOCHS}\n\n")

  

#Sequential data preparation

tX, ty, vX, vy, scaler_y = prepare_benchmark_data(DATA_FILE, SEQ_LEN)

train_loader = DataLoader(TensorDataset(tX, ty), batch_size=128, shuffle=True)

val_loader = DataLoader(TensorDataset(vX, vy), batch_size=128, shuffle=False)

  

best_overall_val_loss = float('inf')

final_model_path = "best_robot_gru_model.pth"

  

for run in range(NUM_RESTARTS):

  run_start_time = time.time()

  model = RobotSplitGRU(6, HIDDEN_LOW, HIDDEN_HIGH, 6).to(DEVICE)

  optimizer = torch.optim.Adam(model.parameters(), lr=LR)

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

  criterion = nn.MSELoss()

  early_stopping = EarlyStopping(patience=PATIENCE, path=f"temp_run_{run}.pth")

  for epoch in range(1, EPOCHS + 1):

    model.train()

    for b_X, b_y in train_loader:

      b_X, b_y = b_X.to(DEVICE), b_y.to(DEVICE)

      optimizer.zero_grad()

      loss = criterion(model(b_X), b_y)

      loss.backward()

      optimizer.step()

    scheduler.step()

    model.eval()

    v_loss = 0

    with torch.no_grad():

      for bv_X, bv_y in val_loader:

        v_loss += criterion(model(bv_X.to(DEVICE)), bv_y.to(DEVICE)).item()

    avg_v = v_loss / len(val_loader)

    early_stopping(avg_v, model)

    if early_stopping.early_stop: break

  

#Saving the best model and logging results both on file and on console

  if early_stopping.best_loss < best_overall_val_loss:

    best_overall_val_loss = early_stopping.best_loss

    torch.save(model.state_dict(), final_model_path)

  duration = time.time() - run_start_time

  log_entry = f"Run {run+1}/{NUM_RESTARTS} - Best Val Loss: {early_stopping.best_loss:.6f} - Time: {duration:.2f}s\n"

  print(log_entry)

  with open(LOG_FILE, "a") as log_f:

    log_f.write(log_entry)

  

#Final evaluation and computing R2 scores

model.load_state_dict(torch.load(final_model_path))

model.eval()

with torch.no_grad():

  y_pred_s = model(vX.to(DEVICE)).cpu().numpy()

  y_true_s = vy.numpy()

  y_pred = scaler_y.inverse_transform(y_pred_s)

  y_true = scaler_y.inverse_transform(y_true_s)

  

  r2_list = []

  for n in range(6):

    res_ss = np.sum((y_true[:, n] - y_pred[:, n])**2)

    tot_ss = np.sum((y_true[:, n] - np.mean(y_true[:, n]))**2)

    r2_list.append(100 * (1 - (res_ss / tot_ss)))

  

print(f"\nR2 mean: {np.mean(r2_list):.2f}%")