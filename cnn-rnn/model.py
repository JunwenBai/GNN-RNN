import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN_RNN(nn.Module):

    def __init__(self, args):
        super(CNN_RNN, self).__init__()
        self.z_dim = args.z_dim
        self.share_conv_parameters = args.share_conv_parameters
        self.combine_weather_and_management = args.combine_weather_and_management

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

        if args.share_conv_parameters:  # Each variable shares same CNN parameters
            if args.time_intervals == 52:  # weekly data
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
            elif args.time_intervals == 365:  # daily data
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
                nn.Linear(self.num_management_vars_this_crop * 20, 40), 
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

        else:  # Each variable can have its own CNN parameters
            print('Each variable has own conv params!')

            if args.combine_weather_and_management:  # Process weather and management data in same CNN
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
            

            else:  # Process weather and management data in distinct CNNs
                if args.time_intervals == 52:  # Weekly data
                    self.w_conv = nn.Sequential(
                        nn.Conv1d(in_channels=self.num_weather_vars, out_channels=32, kernel_size=9, stride=1),
                        nn.ReLU(),
                        nn.AvgPool1d(kernel_size=2, stride=2),
                        nn.Conv1d(32, 64, 3, 1), 
                        nn.ReLU(),
                        nn.AvgPool1d(2, 2), 
                        nn.Conv1d(64, 128, 3, 1),
                        nn.ReLU(),
                        nn.AvgPool1d(2, 2),
                        nn.Conv1d(128, 256, 3, 1),
                        nn.ReLU(),
                        nn.AvgPool1d(2, 2),
                    )
                    self.m_conv = nn.Sequential(
                        nn.Conv1d(in_channels=self.num_management_vars_this_crop, out_channels=32, kernel_size=9, stride=1),
                        nn.ReLU(),
                        nn.AvgPool1d(kernel_size=2, stride=2),
                        nn.Conv1d(32, 64, 3, 1), 
                        nn.ReLU(),
                        nn.AvgPool1d(2, 2), 
                        nn.Conv1d(64, 128, 3, 1),
                        nn.ReLU(),
                        nn.AvgPool1d(2, 2),
                        nn.Conv1d(128, 256, 3, 1),
                        nn.ReLU(),
                        nn.AvgPool1d(2, 2),
                    )

                elif args.time_intervals == 365:  # Daily data
                    self.w_conv = nn.Sequential(
                        nn.Conv1d(in_channels=self.num_weather_vars, out_channels=32, kernel_size=9, stride=2),
                        nn.ReLU(),
                        nn.AvgPool1d(kernel_size=2, stride=2),
                        nn.Conv1d(32, 64, 3, 2), 
                        nn.ReLU(),
                        nn.AvgPool1d(2, 2), 
                        nn.Conv1d(64, 128, 3, 2),
                        nn.ReLU(),
                        nn.AvgPool1d(2, 2),
                        nn.Conv1d(128, 256, 3, 1),
                        nn.ReLU(),
                        nn.AvgPool1d(2, 2),
                    )
                    self.m_conv = nn.Sequential(
                        nn.Conv1d(in_channels=self.num_management_vars_this_crop, out_channels=32, kernel_size=9, stride=2),
                        nn.ReLU(),
                        nn.AvgPool1d(kernel_size=2, stride=2),
                        nn.Conv1d(32, 64, 3, 2), 
                        nn.ReLU(),
                        nn.AvgPool1d(2, 2), 
                        nn.Conv1d(64, 128, 3, 2),
                        nn.ReLU(),
                        nn.AvgPool1d(2, 2),
                        nn.Conv1d(128, 256, 3, 1),
                        nn.ReLU(),
                        nn.AvgPool1d(2, 2),
                    )
                else:
                    raise ValueError("Invalid value for time_intervals (should be 52 or 365)")

                self.w_fc = nn.Sequential(
                    nn.Linear(256, 40), 
                    nn.ReLU(),
                )
                self.m_fc = nn.Sequential(
                    nn.Linear(256, 40), 
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
            self.lstm = nn.LSTM(input_size=120+self.n_extra, hidden_size=self.z_dim, num_layers=1, batch_first=True)


        self.regressor = nn.Sequential(
            nn.Linear(args.z_dim, args.z_dim//2),
            nn.ReLU(),
            nn.Linear(args.z_dim//2, len(args.output_names)),
        )


    def forward(self, X):
        n_batch, n_years, n_feat = X.shape
        X = X.reshape(-1, n_feat)

        # Note: 64 is batch size, 5 is number of years fed into LSTM
        if self.share_conv_parameters:
            X_weather = X[:, :self.n_w].reshape(-1, 1, self.time_intervals) # [64*5*num_weather_vars, 1, time_intervals]
            X_w = self.w_conv(X_weather).squeeze(-1) # [64*5*num_weather_vars, 20]
            X_w = X_w.reshape(n_batch*n_years, -1) # [64*5, num_weather_vars*20]
            X_w = self.w_fc(X_w) # [64*5, 40]

            X_m = X[:, self.progress_indices].reshape(-1, 1, self.time_intervals) # [64*5, n_m]
            X_m = self.m_conv(X_m).squeeze(-1) # [64*5*num_management_vars, 20]
            X_m = X_m.reshape(n_batch*n_years, -1) # [64*5, num_management_vars*20]
            X_m = self.m_fc(X_m) # [64*5, 40]
            X_wm = torch.cat((X_w, X_m), dim=1)  # [64*5, 80]

            X_soil = X[:, self.n_w+self.n_m:self.n_w+self.n_m+self.n_s].reshape(-1, 1, self.soil_depths) # [64*5*num_soil_vars, 1, soil_depths]
            X_s = self.s_conv(X_soil).squeeze(-1) # [64*5*num_soil_vars, 12]
            X_s = X_s.reshape(n_batch*n_years, -1) # [64*5, num_soil_vars*12]
            X_s = self.s_fc(X_s) # [64*5, 40]
        else:
            X_w = X[:, :self.n_w].reshape(-1, self.num_weather_vars, self.time_intervals) # [64*5, num_weather_vars, time_intervals]
            if not self.no_management:
                X_m = X[:, self.progress_indices].reshape(-1, self.num_management_vars_this_crop, self.time_intervals) # [64*5, num_management_vars_this_crop, time_intervals]
            # print("X_M shape", X_m.shape)
            if self.combine_weather_and_management:
                if self.no_management:
                    X_wm = X_w
                else:
                    X_wm = torch.cat((X_w, X_m), dim=1)
                X_wm = self.wm_conv(X_wm).squeeze(-1) # [64*5, 256]
                X_wm = self.wm_fc(X_wm) # [64*5, 80]
            else:
                X_w = self.w_conv(X_w).squeeze(-1)
                X_w = self.w_fc(X_w)
                X_m = self.m_conv(X_m).squeeze(-1) # [64*5*num_management_vars, 20]
                X_m = self.m_fc(X_m) # [64*5, 40]
                X_wm = torch.cat((X_w, X_m), dim=1)  # [64*5, 80]

            X_soil = X[:, self.n_w+self.n_m:self.n_w+self.n_m+self.n_s].reshape(-1, self.num_soil_vars, self.soil_depths) # [64*5, num_soil_vars, soil_depths]
            X_s = self.s_conv(X_soil).squeeze(-1) # [64*5, 64]
            X_s = self.s_fc(X_s) # [64*5, 40]

        X_extra = X[:, self.n_w+self.n_m+self.n_s:] # [64*5, n_extra]

        X_all = torch.cat((X_wm, X_s, X_extra), dim=1) # [64*5, 40+40+40+n_extra]  TODO put X_m (progress data) back in
        X_all = X_all.reshape(n_batch, n_years, -1) # [64, 5, 40+40+40+n_extra]

        out, (last_h, last_c) = self.lstm(X_all)
        #print("out:", out.shape) # [64, 5, 64]
        #print("last_h:", last_h.shape) # [1, 64, 64]
        #print("last_c:", last_c.shape) # [1, 64, 64]
        pred = self.regressor(out)  #.squeeze(-1) # [64, 5, num_outputs]
        
        return pred


class RNN(nn.Module):

    def __init__(self, args):
        super(RNN, self).__init__()
        print("The RNN is being used!")
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
        self.n = self.n_w + self.n_s + self.num_management_vars_this_crop*args.time_intervals + self.n_extra
        self.z_dim = args.z_dim
        self.progress_indices = args.progress_indices
        self.model = args.model

        # Pass flattened weather/soil/management data through MLP
        self.fc = nn.Sequential(
            nn.Linear(self.n, self.z_dim), 
            nn.ReLU(),
            nn.Linear(self.z_dim, 99), 
            nn.ReLU(),
        )

        if args.model == "lstm":
            self.lstm = nn.LSTM(input_size=99, hidden_size=self.z_dim, num_layers=1, batch_first=True, dropout=1.-args.keep_prob)
        elif args.model == "gru":
            self.lstm = nn.GRU(input_size=99, hidden_size=self.z_dim, num_layers=1, batch_first=True, dropout=1.-args.keep_prob)
        else:
            raise ValueError("args.model must be `lstm` or `gru`")
        self.regressor = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim//2),
            nn.ReLU(),
            nn.Linear(self.z_dim//2, 1),
        )


    def forward(self, X):
        # print("RNN forward is being called!")
        # print("X shape", X.shape)

        # Extract the features we're using
        X_w = X[:, :, :self.n_w]
        X_m = X[:, :, self.progress_indices]
        X_s = X[:, :, self.n_w+self.n_m:self.n_w+self.n_m+self.n_s]
        X_extra = X[:, :, self.n_w+self.n_m+self.n_s:]
        if self.no_management:
            X = torch.cat((X_w, X_s, X_extra), dim=2)
        else:
            X = torch.cat((X_w, X_m, X_s, X_extra), dim=2)

        # Pass each year's feature vector through MLP
        # print("New X shape", X.shape)
        n_batch, n_years, n_feat = X.shape
        X = X.reshape(-1, n_feat)
        # print("After initial reshape", X.shape)
        X = self.fc(X)
        X_all = X.reshape(n_batch, n_years, -1) # [64, 5, 99]
        # print("After FC and reshape", X_all.shape)

        # Now pass the sequence of year features through the LSTM
        if self.model == "lstm":
            out, (last_h, last_c) = self.lstm(X_all)  # [128, z_dim]
        elif self.model == "gru":
            out, h_n = self.lstm(X_all)

        # print("out:", out.shape) # [64, 5, 64]
        # print("last_h:", last_h.shape) # [1, 64, 64]
        # print("last_c:", last_c.shape) # [1, 64, 64]
        pred = self.regressor(out)  #.squeeze(-1) # [64, 5]
        
        return pred
 
