import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import MiniBatchDictionaryLearning

# Helper functions for abs weight pruning
def sorted_mat(matrix):
    temp = list(abs(matrix).flatten())
    temp.sort()
    return temp


def prune(matrix, mat_sort, to_prune):
    if to_prune != 0:
        alpha = mat_sort[int(to_prune * 0.1 * len(mat_sort))]
        matrix[abs(matrix) <= alpha] = 0
    return matrix


def rank(matrix):
    np_matrix = np.array(matrix)
    return np.linalg.matrix_rank(np_matrix)/min(list(np_matrix.shape))


# What percentage can be pruned by weight
def sparsity(matrix, alpha):
    abs_matrix = abs(matrix)
    filtered_matrix = abs_matrix[abs_matrix < alpha]
    return len(filtered_matrix)/matrix.size


def viz_rank_change(rank_list,name):
    fig = plt.figure()
    plt.plot(rank_list)
    plt.savefig(name)

def GaussLowpassFilter(shape, d0):
    H = np.zeros(shape, dtype=float)
    IH = shape[0]
    IW = shape[1]
    for i in range(IH):
        for j in range(IW):
            d = np.sqrt((i - IH / 2) ** 2 + (j - IW / 2) ** 2)
            H[i, j] = np.exp(-d * d / (2 * d0 * d0))
    return H

def GaussHighpassFilter(shape, d0):
    H = GaussLowpassFilter(shape, d0)
    return 1 - H

def ButterworthLowpassFilter(shape, d0, n=1):
    H = np.zeros(shape, dtype=float)
    IH = shape[0]
    IW = shape[1]
    for i in range(IH):
        for j in range(IW):
            d = np.sqrt((i - IH / 2) ** 2 + (j - IW / 2) ** 2)
            H[i, j] = 1 / (1 + (d / d0) ** (2 * n))
    return H

def ButterworthHighpassFilter(shape, d0, n=1):
    H = ButterworthLowpassFilter(shape, d0, n)
    return 1 - H
#########################

# Helper functions for rank reduction

def do_low_pass(weight, k, debug=False, niter=4):
    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k)

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")
        
    weight_approx = np.array(weight.clone().detach())
    HIm = ButterworthLowpassFilter(weight_approx.shape, 50)
    P2 = np.fft.fft2(weight_approx)
    P2 = np.fft.fftshift(P2)    
    P3 = P2 * HIm
    P3 = np.fft.ifftshift(P3)
    P3 = np.fft.ifft2(P3)
    weight_approx = torch.tensor(np.abs(P3))
    weight_approx = torch.tensor(weight_approx)
    
    print("Low Pass: This weight_approx.shape",weight_approx.shape)

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    
    weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx

def do_high_pass(weight, k, debug=False, niter=4):
    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k)

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")
        
    weight_approx2 = np.array(weight.clone().detach())
    HIm = ButterworthHighpassFilter(weight_approx2.shape, 40)
    P2 = np.fft.fft2(weight_approx2)
    P2 = np.fft.fftshift(P2)    
    P3 = P2 * HIm
    P3 = np.fft.ifftshift(P3)
    P3 = np.fft.ifft2(P3)
    weight_approx2 = torch.tensor(np.abs(P3))

    if debug:
        print(f"New matrix has shape {weight_approx2.shape}")

    print("High Pass: This weight_approx.shape",weight_approx2.shape)
    assert weight_approx2.shape[0] == weight.shape[0] and weight_approx2.shape[1] == weight.shape[1]
    
    weight_approx2 = torch.nn.Parameter(weight_approx2)

    return weight_approx2

from sklearn.cluster import KMeans

def do_quant(weight, k, debug=False, niter=4):
    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k)

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")
    
    print("weight.shape",weight.shape)
    
    K = int(weight.shape[1]*0.05)
    X = np.array(weight.clone().detach().permute(1,0))
    # X = np.array(weight.clone().detach())
    kmeans = KMeans(n_clusters=K, init='k-means++', random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_
    weight_approx = centroids[labels]
    # weight_approx = weight_approx.permute(1,0)
    weight_approx = torch.tensor(np.transpose(weight_approx,(1,0)))
    # weight_approx = torch.tensor(weight_approx)
    print("do_quant : This weight_approx.shape",weight_approx.shape)

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx



def do_low_rank(weight, k, debug=False, niter=4):
    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k)

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")

    results = torch.svd_lowrank(weight,
                                q=desired_rank,
                                niter=niter)
    weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx


def do_ksvd(weight, k, debug=False, niter=4):
    assert weight.ndim == 2

    K = int(weight.shape[1]*k)
    # K = int(weight.shape[0]*0.05)
    print("weight.shape",weight.shape,"comp",K)
    dico = MiniBatchDictionaryLearning(n_components=K, alpha=1, max_iter=6000,transform_algorithm="lasso_cd")
    
    X = np.array(weight.cpu().clone().detach().permute(1,0))
    # X = np.array(weight.cpu().clone().detach())
    V = dico.fit(X).components_

    code = dico.transform(X)
    weight_approx = np.dot((code), V)

    
    weight_approx = torch.tensor(np.transpose(weight_approx,(1,0)))
    # weight_approx = torch.tensor(weight_approx)
    print("do_ksvd : This weight_approx.shape",weight_approx.shape)

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx

import torch.fft as fft


def Fourier_filter_low(x, threshold, scale):
    dtype = x.dtype
    H, W = x.shape
    # Non-power of 2 images must be float32
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x)
    x_freq = fft.fftshift(x_freq)
    
    H, W = x_freq.shape
    mask = scale*torch.ones((H, W)).to(x.device) 
    # print(H,W)
    crow, ccol = H // 2, W //2
    mask[crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq)
    x_filtered = fft.ifftn(x_freq).real
    
    x_filtered = x_filtered.type(dtype)
    return x_filtered


def do_low_pass(weight, k,threshold1 , debug=False, niter=20):
    assert weight.ndim == 2
    
    # weight_approx = weight - Fourier_filter(weight, threshold=1, scale=k)
    weight_approx = Fourier_filter_low(weight, threshold= threshold1, scale=k)
    weight_approx = weight_approx - weight_approx.mean() + weight.mean()
    # weight_approx = (weight_approx)/(weight_approx.norm()) * weight.norm()
    # weight_approx = (weight_approx - weight_approx.mean())/(weight_approx.std()) * weight.std() + weight.mean()
    
    # weight_approx = high_fre
    # 
    print("Low Frequency:", weight_approx.max(), weight_approx.min(),weight_approx.mean(),weight_approx.std())
    print("Weight:", weight.max(), weight.min(),weight.mean(),weight.std())
    print("DIFF:",torch.sqrt(torch.sum((weight-weight_approx)*(weight-weight_approx))))
    
    weight_approx = torch.nn.Parameter(weight_approx)
    
    # print("weight.mean(),weight.std():",weight.mean(),weight.std())

    # print("weight_approx.mean(),weight_approx.std():",weight_approx.mean(),weight_approx.std())
    return weight_approx

def do_high_pass(weight, k,threshold1 , debug=False, niter=4):
    assert weight.ndim == 2
    
    temp = weight
    weight_approx = Fourier_filter_low(weight, threshold= threshold1, scale=k)
    weight_approx = weight_approx - weight_approx.mean() + weight.mean()
    print("Low Frequency:", weight_approx.max(), weight_approx.min(),weight_approx.mean(),weight_approx.std())
    print("Weight:", weight.max(), weight.min(),weight.mean(),weight.std())
    print("DIFF:",torch.sqrt(torch.sum((weight-weight_approx)*(weight-weight_approx))))
    weight_approx = temp - weight_approx

    # weight_approx = weight - Fourier_filter_low(weight, threshold= threshold1, scale=k)

    # # weight_approx = (weight_approx)/(weight_approx.norm()) * weight.norm()
    # # weight_approx = (weight_approx - weight_approx.mean())/(weight_approx.std()) * weight.std() + weight.mean()
    
    # # weight_approx = high_fre
    # # 
    
    
    weight_approx = torch.nn.Parameter(weight_approx)
    
    # print("weight.mean(),weight.std():",weight.mean(),weight.std())

    # print("weight_approx.mean(),weight_approx.std():",weight_approx.mean(),weight_approx.std())
    return weight_approx


# def do_low_rank(weight, k, debug=False, niter=4):
#     assert weight.ndim == 2

#     max_rank = min(weight.shape[0], weight.shape[1])
#     desired_rank = int(max_rank * k)

#     if debug:
#         print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")
        
#     # weight_approx = np.array(weight.clone().detach())
#     # HIm = ButterworthLowpassFilter(weight_approx.shape, 40)
#     # P2 = np.fft.fft2(weight_approx)
#     # P2 = np.fft.fftshift(P2)    
#     # P3 = P2 * HIm
#     # P3 = np.fft.ifftshift(P3)
#     # P3 = np.fft.ifft2(P3)
#     # weight_approx = torch.tensor(np.abs(P3))
#     ###############
    
#     print("weight.shape",weight.shape)
#     results = torch.svd_lowrank(weight,
#                                 q=desired_rank,
#                                 niter=niter)

#     weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T
    
#     print("This weight_approx.shape",weight_approx.shape)

#     if debug:
#         print(f"New matrix has shape {weight_approx.shape}")

#     assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
#     weight_approx = torch.nn.Parameter(weight_approx)

#     return weight_approx
