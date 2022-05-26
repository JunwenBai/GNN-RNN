import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import dgl.nn.pytorch as dglnn
import tqdm

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
        n_batch, n_feat = X.shape # [675, 431]
        X_w = X[:, :self.n_w].reshape(-1, self.num_weather_vars, self.time_intervals) # [675, num_weather_vars, time_intervals]
        
        if self.no_management:
            X_wm = X_w
        else:
            X_m = X[:, self.progress_indices].reshape(-1, self.num_management_vars_this_crop, self.time_intervals) # [675, num_management_vars_this_crop, time_intervals]
            X_wm = torch.cat((X_w, X_m), dim=1)

        X_wm = self.wm_conv(X_wm).squeeze(-1) # [675, 256]
        X_wm = self.wm_fc(X_wm) # [675, 80]

        X_soil = X[:, self.n_w+self.n_m:self.n_w+self.n_m+self.n_s].reshape(-1, self.num_soil_vars, self.soil_depths)  # [675, soil_vars, soil_depths]
        X_s = self.s_conv(X_soil).squeeze(-1) # [675, 64]
        X_s = self.s_fc(X_s) # [675, 40]

        X_extra = X[:, self.n_w+self.n_m+self.n_s:] # [675, n_extra]

        X_all = torch.cat((X_wm, X_s, X_extra), dim=1) # [675, 80+40+n_extra]
        
        return X_all


class SmallerCNN(nn.Module):
    def __init__(self, args):
        super(SmallerCNN, self).__init__()
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
                nn.Conv1d(in_channels=self.num_weather_vars + self.num_management_vars_this_crop, out_channels=16, kernel_size=9, stride=1),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.Conv1d(16, 32, 3, 1), 
                nn.ReLU(),
                nn.AvgPool1d(2, 2), 
                nn.Conv1d(32, 64, 3, 1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(64, 128, 3, 1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
            )
        elif args.time_intervals == 365:   # Daily data
            self.wm_conv = nn.Sequential(
                nn.Conv1d(in_channels=self.num_weather_vars + self.num_management_vars_this_crop, out_channels=16, kernel_size=9, stride=2),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.Conv1d(16, 32, 3, 2), 
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(32, 64, 3, 2),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(64, 128, 3, 1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
            )
        else:
            raise ValueError("args.time_intervals should be 52 or 365")
        
        self.wm_fc = nn.Sequential(
            nn.Linear(128, 80),
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
        n_batch, n_feat = X.shape # [675, 431]
        X_w = X[:, :self.n_w].reshape(-1, self.num_weather_vars, self.time_intervals) # [675, num_weather_vars, time_intervals]
        
        if self.no_management:
            X_wm = X_w
        else:
            X_m = X[:, self.progress_indices].reshape(-1, self.num_management_vars_this_crop, self.time_intervals) # [675, num_management_vars_this_crop, time_intervals]
            X_wm = torch.cat((X_w, X_m), dim=1)

        X_wm = self.wm_conv(X_wm).squeeze(-1) # [675, 128]
        X_wm = self.wm_fc(X_wm) # [675, 80]

        X_soil = X[:, self.n_w+self.n_m:self.n_w+self.n_m+self.n_s].reshape(-1, self.num_soil_vars, self.soil_depths)  # [675, soil_vars, soil_depths]
        X_s = self.s_conv(X_soil).squeeze(-1) # [675, 64]
        X_s = self.s_fc(X_s) # [675, 40]

        X_extra = X[:, self.n_w+self.n_m+self.n_s:] # [675, n_extra]

        X_all = torch.cat((X_wm, X_s, X_extra), dim=1) # [675, 80+40+n_extra]
        
        return X_all


class SAGE(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super(SAGE, self).__init__()
        if args.encoder_type == "cnn":
            self.cnn = CNN(args)
        elif args.encoder_type == "smaller_cnn":
            self.cnn = SmallerCNN(args)
        else:
            raise ValueError("Invalid encoder type")
        self.n_layers = args.n_layers
        self.n_hidden = args.z_dim
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(127, self.n_hidden, args.aggregator_type))
        for i in range(1, self.n_layers - 1):
            self.layers.append(dglnn.SAGEConv(self.n_hidden, self.n_hidden, args.aggregator_type))
        self.layers.append(dglnn.SAGEConv(self.n_hidden, out_dim, args.aggregator_type))
        self.dropout = nn.Dropout(args.dropout)
        self.out_dim = out_dim

    def forward(self, blocks, x):
        #print("x:", x.shape) # [675, 431]
        h = self.cnn(x) # [675, 96]
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

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
        x = self.cnn(x) # [675, 96]
        nodes = torch.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.out_dim)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes).to(device)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

            x = y
        return y

