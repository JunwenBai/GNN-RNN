import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN_RNN(nn.Module):

    def __init__(self, args):
        super(CNN_RNN, self).__init__()
        self.z_dim = args.z_dim
        self.n_w = 52*6
        self.n_s = 10*10
        self.n_m = 14
        self.n_extra = 5
        # weather
        self.w_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=9, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 12, 3, 1), 
            nn.ReLU(),
            nn.AvgPool1d(2, 2), 
            nn.Conv1d(12, 16, 3, 1),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(16, 20, 3, 1),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
        )
        self.w_fc = nn.Sequential(
            nn.Linear(6*20, 40), 
            nn.ReLU(),
        )
        
        self.s_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(4, 8, 3, 1),
            nn.ReLU(),
            nn.Conv1d(8, 12, 2, 1),
            nn.ReLU(),
        )
        self.s_fc = nn.Sequential(
            nn.Linear(10*12, 40),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=99, hidden_size=self.z_dim, num_layers=1, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(args.z_dim, args.z_dim//2),
            nn.ReLU(),
            nn.Linear(args.z_dim//2, 1),
        )

    def forward(self, X, Y):
        n_batch, n_years, n_feat = X.shape
        X = X.reshape(-1, n_feat)

        X_weather = X[:, :self.n_w].reshape(-1, 1, 52) # [64*5*6, 1, 52]
        X_w = self.w_conv(X_weather).squeeze(-1) # [64*5*6, 20]
        X_w = X_w.reshape(n_batch*n_years, -1) # [64*5, 120]
        X_w = self.w_fc(X_w) # [64*5, 40]

        X_soil = X[:, self.n_w:self.n_w+self.n_s].reshape(-1, 1, 10) # [64*5*10, 1, 10]
        X_s = self.s_conv(X_soil).squeeze(-1) # [64*5*10, 12]
        X_s = X_s.reshape(n_batch*n_years, -1) # [64*5, 120]
        X_s = self.s_fc(X_s) # [64*5, 40]

        X_m = X[:, self.n_w+self.n_s:self.n_w+self.n_s+self.n_m] # [64*5, 14]
        X_extra = X[:, self.n_w+self.n_s+self.n_m:] # [64*5, 5]

        X_all = torch.cat((X_w, X_s, X_m, X_extra), dim=1) # [64*5, 40+40+14+5]
        X_all = X_all.reshape(n_batch, n_years, -1) # [64, 5, 99]

        out, (last_h, last_c) = self.lstm(X_all)
        #print("out:", out.shape) # [64, 5, 64]
        #print("last_h:", last_h.shape) # [1, 64, 64]
        #print("last_c:", last_c.shape) # [1, 64, 64]
        pred = self.regressor(out).squeeze(-1) # [64, 5]
        
        return pred
        
class RNN(nn.Module):

    def __init__(self, args):
        super(RNN, self).__init__()
        self.z_dim = args.z_dim
        self.n_w = 52*6
        self.n_s = 10*10
        self.n_m = 14
        self.n_extra = 5
        self.n = self.n_w + self.n_s + self.n_m + self.n_extra
        # weather
        self.fc = nn.Sequential(
            nn.Linear(self.n, self.z_dim), 
            nn.ReLU(),
            nn.Linear(self.z_dim, 99), 
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=99, hidden_size=self.z_dim, num_layers=1, batch_first=True, dropout=1.-args.keep_prob)
        self.regressor = nn.Sequential(
            nn.Linear(args.z_dim, args.z_dim//2),
            nn.ReLU(),
            nn.Linear(args.z_dim//2, 1),
        )


    def forward(self, X):
        n_batch, n_years, n_feat = X.shape
        X = X.reshape(-1, n_feat)
        X = self.fc(X)
        X_all = X.reshape(n_batch, n_years, -1) # [64, 5, 99]

        out, (last_h, last_c) = self.lstm(X_all)
        #print("out:", out.shape) # [64, 5, 64]
        #print("last_h:", last_h.shape) # [1, 64, 64]
        #print("last_c:", last_c.shape) # [1, 64, 64]
        pred = self.regressor(out).squeeze(-1) # [64, 5]
        
        return pred
 
