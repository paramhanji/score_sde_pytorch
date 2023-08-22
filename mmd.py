import numpy as np
from scipy.spatial.distance import cdist

def gaussian_rbf_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

def batched_mmd(x, y, batch_size=100, sigma=1):
    mmd_sum = 0.0
    num_samples = min(len(x), len(y))
    
    for i in range(0, num_samples, batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        
        kernel_sum = 0.0
        for xi in x_batch:
            for yi in y_batch:
                kernel_sum += gaussian_rbf_kernel(xi, yi, sigma)
        
        mmd_sum += kernel_sum / (len(x_batch) * len(y_batch))
    
    return mmd_sum / (num_samples / batch_size)


def rbf_kernel(X, Y, sigma):
    pairwise_dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-pairwise_dists / (2 * sigma**2))

def mmd(X, Y, sigma):
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)
    
    n = X.shape[0]
    m = Y.shape[0]
    
    mmd = np.sum(K_XX) / (n * (n - 1)) + np.sum(K_YY) / (m * (m - 1)) - 2 * np.sum(K_XY) / (n * m)
    
    return mmd

def main():
    # Example usage
    np.random.seed(42)
    X = np.random.randn(100, 1)
    Y = np.random.randn(100, 1)

    batch_size = 10  # Adjust the batch size as needed
    sigma = 1.0  # RBF kernel bandwidth

    mmd_value = mmd(X, Y, sigma)
    print("MMD value:", mmd_value)

    Y = np.random.rand(100, 1)
    mmd_value = mmd(X, Y, sigma)
    print("MMD value:", mmd_value)
    mmd_value = batched_mmd(X, Y, batch_size, sigma)
    print("Batched MMD:", mmd_value)


import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


if __name__ == '__main__':
    main()
