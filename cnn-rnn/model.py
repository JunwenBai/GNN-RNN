import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN_RNN(nn.Module):

    def __init__(self, args):
        super(CNN_RNN, self).__init__()
        self.z_dim = args.z_dim

        # Store dataset dimensions
        self.time_intervals = args.time_intervals
        self.soil_depths = args.soil_depths

        self.n_w = args.time_intervals*args.num_weather_vars  # Original: 52*6, new: 52*23
        self.n_s = args.soil_depths*args.num_soil_vars  # Original: 10*10, new: 6*20
        self.n_m = args.time_intervals*args.num_management_vars # Original: 14, new: 52*96
        self.n_extra = args.num_extra_vars + len(args.output_names) # Original: 4+1, new: 6+6
        # weather
        if args.time_intervals == 52:
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
            self.m_conv = nn.Sequential(
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
        elif args.time_intervals == 365:
            self.w_conv = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=8, kernel_size=9, stride=2),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.Conv1d(8, 12, 3, 2), 
                nn.ReLU(),
                nn.AvgPool1d(2, 2), 
                nn.Conv1d(12, 16, 3, 2),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(16, 20, 3, 1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
            )
            self.m_conv = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=8, kernel_size=9, stride=2),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.Conv1d(8, 12, 3, 2), 
                nn.ReLU(),
                nn.AvgPool1d(2, 2), 
                nn.Conv1d(12, 16, 3, 2),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(16, 20, 3, 1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
            )
        else:
            raise ValueError("Invalid value for time_intervals (should be 52 or 365)")

        self.w_fc = nn.Sequential(
            nn.Linear(args.num_weather_vars*20, 40), 
            nn.ReLU(),
        )
        self.m_fc = nn.Sequential(
            nn.Linear(args.num_management_vars*20, 40), 
            nn.ReLU(),
        )

        if args.soil_depths == 10:
            self.s_conv = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(4, 8, 3, 1),
                nn.ReLU(),
                nn.Conv1d(8, 12, 2, 1),
                nn.ReLU(),
            )
        elif args.soil_depths == 6:
            self.s_conv = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv1d(4, 8, 3, 1),
                nn.ReLU(),
                nn.Conv1d(8, 12, 2, 1),
                nn.ReLU(),
            )
        else:
            raise ValueError("Don't know how to deal with a number of soil_depths that is not 6 or 10")

        self.s_fc = nn.Sequential(
            nn.Linear(args.num_soil_vars*12, 40),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=120+self.n_extra, hidden_size=self.z_dim, num_layers=1, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(args.z_dim, args.z_dim//2),
            nn.ReLU(),
            nn.Linear(args.z_dim//2, len(args.output_names)),
        )

    def forward(self, X, Y):
        n_batch, n_years, n_feat = X.shape
        X = X.reshape(-1, n_feat)

        # Note: 64 is batch size, 5 is number of years fed into LSTM
        X_weather = X[:, :self.n_w].reshape(-1, 1, self.time_intervals) # [64*5*num_weather_vars, 1, time_intervals]
        X_w = self.w_conv(X_weather)
        X_w = X_w.squeeze(-1) # [64*5*num_weather_vars, 20]
        X_w = X_w.reshape(n_batch*n_years, -1) # [64*5, num_weather_vars*20]
        X_w = self.w_fc(X_w) # [64*5, 40]

        X_m = X[:, self.n_w:self.n_w+self.n_m].reshape(-1, 1, self.time_intervals) # [64*5, n_m]
        X_m = self.m_conv(X_m)
        X_m = X_m.squeeze(-1) # [64*5*num_management_vars, 20]
        X_m = X_m.reshape(n_batch*n_years, -1) # [64*5, num_management_vars*20]
        X_m = self.m_fc(X_m) # [64*5, 40]

        X_soil = X[:, self.n_w+self.n_m:self.n_w+self.n_m+self.n_s].reshape(-1, 1, self.soil_depths) # [64*5*num_soil_vars, 1, soil_depths]
        X_s = self.s_conv(X_soil).squeeze(-1) # [64*5*num_soil_vars, 12]
        X_s = X_s.reshape(n_batch*n_years, -1) # [64*5, num_soil_vars*12]
        X_s = self.s_fc(X_s) # [64*5, 40]

        X_extra = X[:, self.n_w+self.n_m+self.n_s:] # [64*5, n_extra]

        X_all = torch.cat((X_w, X_m, X_s, X_extra), dim=1) # [64*5, 40+40+n_m+n_extra]  TODO put X_m (progress data) back in
        X_all = X_all.reshape(n_batch, n_years, -1) # [64, 5, 40+40+n_m+n_extra]

        out, (last_h, last_c) = self.lstm(X_all)
        #print("out:", out.shape) # [64, 5, 64]
        #print("last_h:", last_h.shape) # [1, 64, 64]
        #print("last_c:", last_c.shape) # [1, 64, 64]
        pred = self.regressor(out)  #.squeeze(-1) # [64, 5]
        
        return pred


# TODO - n_w, n_s, n_m need to be updated! 
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
 
