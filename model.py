import torch
import torch.nn as nn

class MultiScaleDecomposition(nn.Module):
    def __init__(self, kernel_sizes):
        super(MultiScaleDecomposition, self).__init__()
        self.moving_avg = [nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0) for kernel_size in kernel_sizes] # not trainable so nn.ModuleList is not needed
        self.mean = lambda x: sum(x) / len(x)
    
    def forward(self, x):
        avgs = []
        diffs = []
        
        for avg in self.moving_avg:
            # Padding to preserve the output shape
            front = x[:, 0:1, :].repeat(1, (avg.kernel_size[0] - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (avg.kernel_size[0] - 1) // 2, 1)
            x_padded = torch.cat([front, x, end], dim=1)
            
            avgs.append(avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1))
            diffs.append(x - avgs[-1])
        return self.mean(diffs).float().permute(0, 2, 1), self.mean(avgs).float().permute(0, 2, 1)

class MultiscaleIsometricConvolutionLayer(nn.Module):
    def __init__(self, d_model, isometric_kernel=[18, 6], decomp_kernel=[32], conv_kernel=[24], device=torch.device("cpu")):
        super(MultiscaleIsometricConvolutionLayer, self).__init__()
        self.conv_kernel = conv_kernel

        self.device = device

        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=i, padding=0, stride=1) for i in isometric_kernel])
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=i, padding=i//2, stride=i) for i in conv_kernel])
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=i, padding=0, stride=i) for i in conv_kernel])
        self.moving_avg = [nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0) for kernel_size in decomp_kernel] # not trainable so nn.ModuleList is not needed
        self.conv2d = torch.nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(len(conv_kernel), 1))

        self.linear1 = nn.Linear(d_model, d_model*4) # the paper assigns custom weights
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_model*4, d_model) # the paper assigns custom weights
        self.norm_linear = torch.nn.LayerNorm(d_model)
        
        self.norm = torch.nn.LayerNorm(d_model)

        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)
    
    def forward(self, x):
        different_scales = []  
        for i in range(len(self.conv_kernel)):
            # avg
            front = x[:, 0:1, :].repeat(1, (self.moving_avg[i].kernel_size[0] - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.moving_avg[i].kernel_size[0] - 1) // 2 + 1, 1)
            x_padded = torch.cat([front, x, end], dim=1)

            scale_i_src = x - self.moving_avg[i](x_padded.permute(0, 2, 1)).permute(0, 2, 1)

            # isometric conv
            skip = self.drop(self.act(self.conv[i](scale_i_src.permute(0, 2, 1))))
            scale_i = skip

            padding = skip.shape[2] - 2*scale_i.shape[2] + self.isometric_conv[i].kernel_size[0] - 1
            zeros = torch.zeros((scale_i.shape[0], scale_i.shape[1], scale_i.shape[2]+padding), device=self.device)
            scale_i = torch.cat((zeros, scale_i), dim=-1)
            scale_i = self.drop(self.act(self.isometric_conv[i](scale_i)))
            scale_i = self.norm((scale_i+skip).permute(0, 2, 1)).permute(0, 2, 1)

            scale_i = self.drop(self.act(self.conv_trans[i](scale_i)))
            scale_i = scale_i[:, :, :x.shape[1]]

            scale_i = self.norm(scale_i.permute(0, 2, 1) + scale_i_src)

            different_scales.append(scale_i)

        # merge all the captured scales
        merged = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            merged = torch.cat((merged, different_scales[i].unsqueeze(1)), dim=1)
        merged = self.conv2d(merged.permute(0,3,1,2)).squeeze(-2).permute(0,2,1)
        
        mg_projected = self.norm_linear(self.linear2(self.dropout(self.relu(self.linear1(merged)))))
        
        return self.norm(merged + mg_projected)

class SeasonalPredictionBlock(nn.Module):
    def __init__(self, d_model, num_layers, device=torch.device("cpu")):
        super(SeasonalPredictionBlock, self).__init__()
        self.layers = nn.ModuleList([MultiscaleIsometricConvolutionLayer(d_model, device=device) for i in range(num_layers)])
        self.linear = nn.Linear(d_model, 21)

    def forward(self, dec):
        for mic_layer in self.layers:
            dec = mic_layer(dec)
        return self.linear(dec)

def PositionalEmbeddings(d_model, len):
    positions = torch.arange(0, len).unsqueeze_(1)
    embeddings = torch.zeros(len, d_model)

    denominators = torch.pow(10000.0, 2*torch.arange(0, d_model//2)/d_model)
    embeddings[:, 0::2] = torch.sin(positions/denominators)
    embeddings[:, 1::2] = torch.cos(positions/denominators)

    return embeddings

class Embeddings(nn.Module):
    def __init__(self, d_model, tot_len, dropour_prob=0.1):
        super(Embeddings, self).__init__()
        self.TFE = nn.Linear(4, d_model)
        self.PE = PositionalEmbeddings(d_model, tot_len)
        self.VE = nn.Conv1d(in_channels=21, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular')
        self.dropout = nn.Dropout(dropour_prob)
    
    def forward(self, x, time_tokens):
        tfe = self.TFE(time_tokens)
        ve = self.VE(x).transpose(1,2)
        return self.dropout(tfe + self.PE + ve)

class MICN(nn.Module):
    def __init__(self, seq_len, pred_len, n_layers=1, d_model=256, device=torch.device("cpu")):
        super(MICN, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.device = device

        self.decomposition = MultiScaleDecomposition([33])
        self.regre = nn.Linear(seq_len, pred_len) # The paper also assign custom weights

        self.embeddings = Embeddings(self.d_model, seq_len + pred_len)
        self.embeddings.PE = self.embeddings.PE.to(device)

        self.conv_trans = SeasonalPredictionBlock(d_model, n_layers, device=device)

        self.to(device)

    
    def forward(self, x, x_time_tokens, y_time_tokens):
        Xs, Xt = self.decomposition(x)
        
        Yt = self.regre(Xt).permute(0, 2, 1)[:, -self.pred_len:, :]

        Xzero = torch.zeros((x.shape[0], x.shape[2], self.pred_len), device=self.device)
        Xs = torch.cat([Xs[:, :, -self.seq_len:], Xzero], dim=2)
        embedded = self.embeddings(Xs, torch.cat([x_time_tokens, y_time_tokens], dim=1))

        Ys = self.conv_trans(embedded)[:, -self.pred_len:, :]

        return Ys  + Yt