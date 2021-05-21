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

    def forward(self, X):
        n_batch, n_feat = X.shape # [675, 431]
        X_weather = X[:, :self.n_w].reshape(-1, 1, 52) # [675*6, 1, 52]
        X_w = self.w_conv(X_weather).squeeze(-1) # [675*6, 20]
        X_w = X_w.reshape(n_batch, -1) # [675, 120]
        X_w = self.w_fc(X_w) # [675, 40]

        X_soil = X[:, self.n_w:self.n_w+self.n_s].reshape(-1, 1, 10) # [675*10, 1, 10]
        X_s = self.s_conv(X_soil).squeeze(-1) # [675*10, 12]
        X_s = X_s.reshape(n_batch, -1) # [675, 120]
        X_s = self.s_fc(X_s) # [675, 40]

        X_m = X[:, self.n_w+self.n_s:self.n_w+self.n_s+self.n_m] # [675, 14]
        X_extra = X[:, self.n_w+self.n_s+self.n_m:] # [675, 5]

        X_all = torch.cat((X_w, X_s, X_m, X_extra), dim=1) # [675, 40+40+14+5]
        
        return X_all


class SAGE_RNN(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super(SAGE_RNN, self).__init__()
        self.cnn = CNN(args)
        
        self.n_layers = args.n_layers
        self.n_hidden = args.z_dim
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(99, self.n_hidden, 'mean'))
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
        n_batch, n_seq = y.shape
        y_pad = torch.zeros(n_batch, n_seq+1).to(y.device)
        y_pad[:, 1:] = y
        #print("x:", x.shape) # [711, 5, 431]
        #print("y_pad:", y_pad.shape) # [711, 5]
        hs = []
        for i in range(n_seq+1):
            h = self.cnn(x[:, i, :]) # [711, 431]
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
            hs.append(torch.cat((h, y_pad[:, i:i+1]), 1)) # [n_batch, n_hidden+out_dim]
        hs = torch.stack(hs, dim=0) # [5, n_batch, n_hidden+out_dim]

        out, (last_h, last_c) = self.lstm(hs)
        #print(out.shape) # [5, 64, 64]
        #print(last_h.shape) # [1, 64, 64]
        #print(last_c.shape) # [1, 64, 64]
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

