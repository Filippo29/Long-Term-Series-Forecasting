import torch
import torch.nn as nn

def mean(x): return sum(x) / len(x)

class MultiScaleDecomposition(nn.Module):
    def __init__(self, kernel_sizes):
        super(MultiScaleDecomposition, self).__init__()
        self.moving_avg = [nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0) for kernel_size in kernel_sizes]
    
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
        return mean(diffs).float().permute(0, 2, 1), mean(avgs).float().permute(0, 2, 1)

class MultiscaleIsometricConvolutionLayer(nn.Module):
    def __init__(self):
        super(MultiscaleIsometricConvolutionLayer, self).__init__()
    
    def forward(self, x):
        pass

class SeasonalPredictionBlock(nn.Module):
    def __init__(self, num_layers=1, conv_kernels=[18, 6]):
        super(SeasonalPredictionBlock, self).__init__()
    
    def forward(self, x):
        pass

def PositionalEmbeddings(d_model, max_len=1000, n=10000.0):
    if d_model % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(d_model))

    positions = torch.arange(0, max_len).unsqueeze_(1)
    embeddings = torch.zeros(max_len, d_model)

    denominators = torch.pow(n, 2*torch.arange(0, d_model//2)/d_model) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings

class TemporalEmbeddings(nn.Module):
    def __init__(self, seq_len, d_model):
        # minute_size = 4
        # hour_size = 24
        # weekday_size = 7
        # day_size = 32
        # month_size = 13
        emb_sizes = [4, 24, 7, 32, 13]
        embeddings = []

        for emb_size in emb_sizes:
            embeddings.append(nn.Embedding(emb_size, d_model))
            weight = PositionalEmbeddings(emb_size, d_model)[:, :embeddings[-1].weight.size(1)]
            embeddings[-1].weight = nn.Parameter(weight, requires_grad=False)
        
    def forward(self, x):
        pass

class ValueEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ValueEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular')
        nn.init.kaiming_normal_(self.tokenConv.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x).transpose(1,2)
        return x

class Embeddings(nn.Module):
    def __init__(self, seq_len, d_model, dropour_prob=0.1):
        super(Embeddings, self).__init__()
        self.TFE = nn.Linear(4, d_model)
        self.PE = PositionalEmbeddings(d_model)
        self.VE = ValueEmbedding(21, d_model)
        self.dropout = nn.Dropout(dropour_prob)
    
    def forward(self, x, time_tokens):
        tfe = self.TFE(time_tokens)
        pe = self.PE[:x.size(2), :]
        ve = self.VE(x)
        print(tfe.shape, pe.shape, ve.shape)
        return self.dropout(tfe + pe + ve)

class MICN(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=256):
        super(MICN, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.decomposition = MultiScaleDecomposition([33])
        self.regre = nn.Linear(seq_len, pred_len) # The paper also assign custom weights

        self.embeddings = Embeddings(seq_len, self.d_model)

        self.conv_trans = SeasonalPredictionBlock()

    
    def forward(self, x, x_time_tokens, y_time_tokens):
        Xs, Xt = self.decomposition(x)
        Yt = self.regre(Xt)

        Xzero = torch.zeros((x.shape[0], x.shape[2], self.pred_len))
        Xs = torch.cat([Xs[:, :, -self.seq_len:], Xzero], dim=2)
        embedded = self.embeddings(Xs, torch.cat([x_time_tokens, y_time_tokens], dim=1))
        print(embedded.shape)

        return dec_out