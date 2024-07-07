import torch
import torch.nn as nn

class MultiScaleDecomposition(nn.Module):
    def __init__(self, kernel_sizes):
        super(MultiScaleDecomposition, self).__init__()
        self.avg_pool = [nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0) for kernel_size in kernel_sizes] # not trainable so nn.ModuleList is not needed
        self.mean = lambda x: sum(x) / len(x)
    
    def forward(self, x):
        avgs = []
        diffs = []
        
        for avg in self.avg_pool:
            # Padding to preserve the output shape
            front = x[:, 0:1, :].repeat(1, (avg.kernel_size[0] - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (avg.kernel_size[0] - 1) // 2, 1)
            x_padded = torch.cat([front, x, end], dim=1)
            
            avgs.append(avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1))
            diffs.append(x - avgs[-1]) # Xs = X - Xt
        return self.mean(diffs).float().permute(0, 2, 1), self.mean(avgs).float().permute(0, 2, 1)

class MultiscaleIsometricConvolutionLayer(nn.Module):
    def __init__(self, d_model, isometric_kernel=[18, 6], decomp_kernel=[32], conv_kernel=[24], device=torch.device("cpu")):
        super(MultiscaleIsometricConvolutionLayer, self).__init__()
        self.conv_kernel = conv_kernel

        self.device = device

        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=i, padding=0, stride=1) for i in isometric_kernel])
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=i, padding=i//2, stride=i) for i in conv_kernel])
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=i, padding=0, stride=i) for i in conv_kernel])
        self.avg_pool = [nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0) for kernel_size in decomp_kernel] # not trainable so nn.ModuleList is not needed
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
            front = x[:, 0:1, :].repeat(1, (self.avg_pool[i].kernel_size[0] - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.avg_pool[i].kernel_size[0] - 1) // 2 + 1, 1)
            x_padded = torch.cat([front, x, end], dim=1)

            scale_i_src = x - self.avg_pool[i](x_padded.permute(0, 2, 1)).permute(0, 2, 1)

            # isometric conv
            skip = self.drop(self.act(self.conv[i](scale_i_src.permute(0, 2, 1))))
            scale_i = skip

            padding = skip.shape[2] - 2*scale_i.shape[2] + self.isometric_conv[i].kernel_size[0] - 1    # adaptive padding to match different input_size - pred_len combinations
            zeros = torch.zeros((scale_i.shape[0], scale_i.shape[1], scale_i.shape[2]+padding), device=self.device)
            scale_i = torch.cat((zeros, scale_i), dim=-1)
            scale_i = self.drop(self.act(self.isometric_conv[i](scale_i))) # isometric convolution
            scale_i = self.norm((scale_i+skip).permute(0, 2, 1)).permute(0, 2, 1)

            scale_i = self.drop(self.act(self.conv_trans[i](scale_i))) # conv trans to upscale back
            scale_i = scale_i[:, :, :x.shape[1]]

            scale_i = self.norm(scale_i.permute(0, 2, 1) + scale_i_src)

            different_scales.append(scale_i)

        # merge all the captured scales with a 2d convolution in order to weight differently each scale
        merged = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            merged = torch.cat((merged, different_scales[i].unsqueeze(1)), dim=1)
        merged = self.conv2d(merged.permute(0,3,1,2)).squeeze(-2).permute(0,2,1)
        
        mg_projected = self.norm_linear(self.linear2(self.dropout(self.relu(self.linear1(merged))))) # projection with a small ffw network of the merged tensor maintaining the d_model
        
        return self.norm(merged + mg_projected)

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
        self.TFE = nn.Linear(5, d_model) # time features embeddings: in this implementation is just a linear projection of ['month','day','weekday','hour','minute'] to the dimension of the model
        self.PE = PositionalEmbeddings(d_model, tot_len) # sine cosine positional embeddings
        self.VE = nn.Conv1d(in_channels=21, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular') # value embeddings for the input data
        self.dropout = nn.Dropout(dropour_prob)
    
    def forward(self, x, time_tokens):
        tfe = self.TFE(time_tokens)
        ve = self.VE(x).transpose(1,2)
        return self.dropout(tfe + self.PE + ve)

class MICN(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=256, device=torch.device("cpu")):
        super(MICN, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.device = device

        self.decomposition = MultiScaleDecomposition([33])
        self.regre = nn.Linear(seq_len, pred_len) # The paper also assign custom weights

        self.embeddings = Embeddings(self.d_model, seq_len + pred_len)
        self.embeddings.PE = self.embeddings.PE.to(device)

        self.seasonalPred =  nn.Sequential(MultiscaleIsometricConvolutionLayer(d_model, device=device), nn.Linear(d_model, 21))

        self.to(device)

    
    def forward(self, x, x_time_tokens, y_time_tokens):
        Xs, Xt = self.decomposition(x) # separate the trend cyclical and seasonal parts of the input x
        
        Yt = self.regre(Xt).permute(0, 2, 1)[:, -self.pred_len:, :] # trend cyclical prediction block: a simple linear regression is enough to make a prediction based on the trend-cyclical information

        Xzero = torch.zeros((x.shape[0], x.shape[2], self.pred_len), device=self.device)
        Xs = torch.cat([Xs[:, :, -self.seq_len:], Xzero], dim=2)
        embedded = self.embeddings(Xs, torch.cat([x_time_tokens, y_time_tokens], dim=1)) # embeds the seasonal x concatenated with a tensor of zero to have a total length of seq_len + pred_len
        
        Ys = self.seasonalPred(embedded)[:, -self.pred_len:, :]

        return Ys  + Yt