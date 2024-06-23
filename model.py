import torch
import torch.nn as nn

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
        return sum(diffs) / len(diffs), sum(avgs) / len(avgs)    