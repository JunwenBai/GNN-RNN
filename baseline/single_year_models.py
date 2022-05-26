import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SingleYearCNN(nn.Module):
    def __init__(self, args):
        super(SingleYearCNN, self).__init__()
        self.z_dim = args.z_dim
        self.output_dim = len(args.output_names)

        # Store dataset dimensions
        self.time_intervals = args.time_intervals
        self.soil_depths = args.soil_depths
        self.progress_indices = args.progress_indices
        self.num_weather_vars = args.num_weather_vars  # Number of variables in weather data (for each day)
        self.num_soil_vars = args.num_soil_vars
        self.no_management = args.no_management
        if self.no_management:
            self.num_management_vars_this_crop = 0
        else:
            self.num_management_vars_this_crop = int(len(args.progress_indices) / args.time_intervals)  # NOTE - only includes management vars for this specific crop
        print("num management vars being used", self.num_management_vars_this_crop)

        self.n_w = args.time_intervals*args.num_weather_vars  # Original: 52*6, new: 52*23
        self.n_s = args.soil_depths*args.num_soil_vars  # Original: 10*10, new: 6*20
        self.n_m = args.time_intervals*args.num_management_vars # Original: 14, new: 52*96, This includes management vars for ALL crops.
        self.n_extra = args.num_extra_vars + len(args.output_names) # Original: 4+1, new: 6+1

        print("Processing weather and management data in same CNN!")
        if args.time_intervals == 52:  # Weekly data
            self.wm_conv = nn.Sequential(
                nn.Conv1d(in_channels=self.num_weather_vars + self.num_management_vars_this_crop, out_channels=64, kernel_size=9, stride=1),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.Conv1d(64, 128, 3, 1), 
                nn.ReLU(),
                nn.AvgPool1d(2, 2), 
                nn.Conv1d(128, 256, 3, 1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(256, 512, 3, 1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
            )
        elif args.time_intervals == 365:   # Daily data
            self.wm_conv = nn.Sequential(
                nn.Conv1d(in_channels=self.num_weather_vars + self.num_management_vars_this_crop, out_channels=64, kernel_size=9, stride=2),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.Conv1d(64, 128, 3, 2), 
                nn.ReLU(),
                nn.AvgPool1d(2, 2), 
                nn.Conv1d(128, 256, 3, 2),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(256, 512, 3, 1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
            )
        else:
            raise ValueError("args.time_intervals should be 52 or 365")
        
        self.wm_fc = nn.Sequential(
            nn.Linear(512, 80), 
            nn.ReLU(),
        )

        # Soil CNN
        if args.soil_depths == 10:
            self.s_conv = nn.Sequential(
                nn.Conv1d(in_channels=self.num_soil_vars, out_channels=16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(16, 32, 3, 1),
                nn.ReLU(),
                nn.Conv1d(32, 64, 2, 1),
                nn.ReLU(),
            )
        elif args.soil_depths == 6:
            self.s_conv = nn.Sequential(
                nn.Conv1d(in_channels=self.num_soil_vars, out_channels=16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, 3, 1),
                nn.ReLU(),
                nn.Conv1d(32, 64, 2, 1),
                nn.ReLU(),
            )
        else:
            raise ValueError("Don't know how to deal with a number of soil_depths that is not 6 or 10")
        self.s_fc = nn.Sequential(
            nn.Linear(64, 40),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80+40+self.n_extra, args.z_dim),
            nn.ReLU(),
            nn.Linear(args.z_dim, self.output_dim)
        )


    def forward(self, X):
        n_batch, n_feat = X.shape # [675, 431]
        X_w = X[:, :self.n_w].reshape(-1, self.num_weather_vars, self.time_intervals) # [128, num_weather_vars, time_intervals]
        
        if self.no_management:
            X_wm = X_w
        else:
            X_m = X[:, self.progress_indices].reshape(-1, self.num_management_vars_this_crop, self.time_intervals) # [128, num_management_vars_this_crop, time_intervals]
            X_wm = torch.cat((X_w, X_m), dim=1) # [128, num_weather_vars+num_management_vars_this_crop, time_intervals]

        X_wm = self.wm_conv(X_wm).squeeze(-1) # [128, 512]
        X_wm = self.wm_fc(X_wm) # [128, 80]

        X_soil = X[:, self.n_w+self.n_m:self.n_w+self.n_m+self.n_s].reshape(-1, self.num_soil_vars, self.soil_depths)  # [128, num_soil_vars, soil_depths]
        X_s = self.s_conv(X_soil).squeeze(-1) # [128, 64]
        X_s = self.s_fc(X_s) # [128, 40]

        X_extra = X[:, self.n_w+self.n_m+self.n_s:] # [128, n_extra]

        X_all = torch.cat((X_wm, X_s, X_extra), dim=1) # [128, 80+40+n_extra]
        pred = self.regressor(X_all)  # [128, output_dim]
        return pred


class SingleYearRNN(nn.Module):

    def __init__(self, args):
        super(SingleYearRNN, self).__init__()
        # Store dataset dimensions
        self.no_management = args.no_management
        if args.no_management:
            self.num_management_vars_this_crop = 0
        else:
            self.num_management_vars_this_crop = int(len(args.progress_indices) / args.time_intervals)  # NOTE - only includes management vars for this specific crop
        print("num management vars being used", self.num_management_vars_this_crop)
    
        self.n_w = args.time_intervals*args.num_weather_vars  # Original: 52*6, new: 52*23
        self.n_s = args.soil_depths*args.num_soil_vars  # Original: 10*10, new: 6*20
        self.n_m = args.time_intervals*args.num_management_vars # Original: 14, new: 52*96, This includes management vars for ALL crops.
        self.n_extra = args.num_extra_vars + len(args.output_names) # Original: 4+1, new: 6+1
        self.z_dim = args.z_dim
        self.output_dim = len(args.output_names)
        self.progress_indices = args.progress_indices
        self.time_intervals = args.time_intervals
        self.num_weather_vars = args.num_weather_vars
        self.num_soil_vars = args.num_soil_vars
        self.soil_depths = args.soil_depths
        self.model = args.model

        if args.model == "lstm":
            self.within_year_rnn = nn.LSTM(input_size=args.num_weather_vars+self.num_management_vars_this_crop,
                                            hidden_size=self.z_dim,
                                            num_layers=1,
                                            batch_first=True,
                                            dropout=1.-args.keep_prob)
        elif args.model == "gru":
            self.within_year_rnn = nn.GRU(input_size=args.num_weather_vars+self.num_management_vars_this_crop,
                                            hidden_size=self.z_dim,
                                            num_layers=1,
                                            batch_first=True,
                                            dropout=1.-args.keep_prob)
        else:
            raise ValueError("If using SingleYearRNN, args.model must be `lstm` or `gru`.")

        self.wm_fc = nn.Sequential(
            nn.Linear(self.z_dim, 80),
            nn.ReLU(),
        )

        # Soil CNN
        if args.soil_depths == 10:
            self.s_conv = nn.Sequential(
                nn.Conv1d(in_channels=self.num_soil_vars, out_channels=16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(16, 32, 3, 1),
                nn.ReLU(),
                nn.Conv1d(32, 64, 2, 1),
                nn.ReLU(),
            )
        elif args.soil_depths == 6:
            self.s_conv = nn.Sequential(
                nn.Conv1d(in_channels=self.num_soil_vars, out_channels=16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, 3, 1),
                nn.ReLU(),
                nn.Conv1d(32, 64, 2, 1),
                nn.ReLU(),
            )
        else:
            raise ValueError("Don't know how to deal with a number of soil_depths that is not 6 or 10")
        self.s_fc = nn.Sequential(
            nn.Linear(64, 40),
            nn.ReLU(),
        )

        # Final regressor - combine weather/management, soil, and extra features
        self.regressor = nn.Sequential(
            nn.Linear(80+40+self.n_extra, args.z_dim),
            nn.ReLU(),
            nn.Linear(args.z_dim, self.output_dim)
        )


    def forward(self, X):
        n_batch, n_feat = X.shape

        # Extract the weather/management features we're using
        X_w = X[:, :self.n_w]
        if self.no_management:
            X_wm = X_w
        else:
            X_m = X[:, self.progress_indices]
            X_wm = torch.cat((X_w, X_m), dim=1)

        # Reshape weather/management data into weekly time series,
        # and pass through LSTM and fully-connected layers
        X_wm = X_wm.reshape(n_batch, self.num_weather_vars + self.num_management_vars_this_crop, self.time_intervals)
        X_wm = X_wm.permute((0, 2, 1))  # Permute dimensions to [batch_size, time_intervals, num_variables]
        if self.model == "lstm":
            X_wm, (last_h, last_c) = self.within_year_rnn(X_wm)  # [128, z_dim]
        elif self.model == "gru":
            X_wm, h_n = self.within_year_rnn(X_wm)
        X_wm = self.wm_fc(X_wm[:, -1, :])  # [128, 80]

        # Process soil data
        X_s = X[:, self.n_w+self.n_m:self.n_w+self.n_m+self.n_s].reshape(n_batch, self.num_soil_vars, self.soil_depths)
        X_s = self.s_conv(X_s).squeeze(-1) # [64*5, 64]
        X_s = self.s_fc(X_s) # [64*5, 40]

        # Combine weather/management and soil representations, and extra variables. Pass them all through final regressor
        X_extra = X[:, self.n_w+self.n_m+self.n_s:]
        X = torch.cat((X_wm, X_s, X_extra), dim=1)  # [128, 80+40+n_extra]
        pred = self.regressor(X)  # [128, output_dim]
        return pred