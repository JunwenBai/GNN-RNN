import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import dgl.nn.pytorch as dglnn

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.z_dim = args.z_dim

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

    def forward(self, X):
        n_batch, n_feat = X.shape # [675, 6315]
        X_w = X[:, :self.n_w].reshape(-1, self.num_weather_vars, self.time_intervals) # [675, num_weather_vars, time_intervals]
        if self.no_management:
            X_wm = X_w
        else:
            X_m = X[:, self.progress_indices].reshape(-1, self.num_management_vars_this_crop, self.time_intervals) # [675, num_management_vars_this_crop, time_intervals]
            X_wm = torch.cat((X_w, X_m), dim=1)

        X_wm = self.wm_conv(X_wm).squeeze(-1) # [675, 512]
        X_wm = self.wm_fc(X_wm) # [675, 80]

        X_soil = X[:, self.n_w:self.n_w+self.n_s].reshape(-1, self.num_soil_vars, self.soil_depths)  # [675*10, num_soil_vars, soil_depths]
        X_s = self.s_conv(X_soil).squeeze(-1) # [675, 64]
        X_s = self.s_fc(X_s) # [675, 40]

        X_extra = X[:, self.n_w+self.n_s+self.n_m:] # [675, n_extra]

        X_all = torch.cat((X_wm, X_s, X_extra), dim=1) # [675, 80+40+n_extra]
        
        return X_all


class SAGE_RNN(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super(SAGE_RNN, self).__init__()
        self.cnn = CNN(args)
        
        self.n_layers = args.n_layers
        self.n_hidden = args.z_dim
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(127, self.n_hidden, 'mean'))
        for i in range(1, self.n_layers - 1):
            self.layers.append(dglnn.SAGEConv(self.n_hidden, self.n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(self.n_hidden, self.n_hidden, 'mean'))
        self.dropout = nn.Dropout(args.dropout)

        self.lstm = nn.LSTM(input_size=self.n_hidden+out_dim, hidden_size=self.n_hidden, num_layers=1)
        self.regressor = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden//2),
            nn.ReLU(),
            nn.Linear(self.n_hidden//2, out_dim),
        )

    def forward(self, blocks, x, y):
        n_batch, n_seq, n_outputs = y.shape
        y_pad = torch.zeros(n_batch, n_seq+1, n_outputs).to(y.device)
        y_pad[:, 1:] = y
        # print("x:", x.shape) # [675, 5, 6315]
        # print("y_pad:", y_pad.shape) # [675, 5, 1]
        hs = []
        for i in range(n_seq+1):
            h = self.cnn(x[:, i, :]) # [675, 127]
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                # We need to first copy the representation of nodes on the RHS from the
                # appropriate nodes on the LHS.
                # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
                # would be (num_nodes_RHS, D)
                h_dst = h[:block.number_of_dst_nodes()]
                # Then we compute the updated representation on the RHS.
                # The shape of h now becomes (num_nodes_RHS, D)
                h = layer(block, (h, h_dst))
                #if l != len(self.layers) - 1:
                if l != len(self.layers):
                    h = F.relu(h)
                    h = self.dropout(h)
            hs.append(torch.cat((h, y_pad[:, i, :]), 1)) # [n_batch, n_hidden+out_dim]
            # hs.append(torch.cat((h, y_pad[:, i:i+1]), 1)) # [n_batch, n_hidden+out_dim]
        hs = torch.stack(hs, dim=0) # [5, n_batch, n_hidden+out_dim]
        if torch.isnan(hs).any():
            print("Some hs were nan")
            print("X")
            print(x)
            print("y")
            print(y)
            exit(1)

        out, (last_h, last_c) = self.lstm(hs)
        if torch.isnan(hs).any():
            print("Some out states were nan")
            print("X")
            print(x)
            print("y")
            print(y)
            exit(1)
        # print(out.shape) # [5, 64, 64]
        # print(last_h.shape) # [1, 64, 64]
        # print(last_c.shape) # [1, 64, 64]
        pred = self.regressor(out[-1]) # [64, 1]

        return pred

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        print('======================= Inference!!! =========================')
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes).to(device)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

            x = y
        return y

