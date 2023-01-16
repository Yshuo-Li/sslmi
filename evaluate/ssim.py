import numpy as np
import math


# SSIM(structural similarity index)，结构相似性，是一种衡量两幅图像相似度的指标。
# 该指标首先由德州大学奥斯丁分校的图像和视频工程实验室(Laboratory for Image and Video Engineering)提出。
# SSIM使用的两张图像中，一张为未经压缩的无失真图像，另一张为失真后的图像

from PIL import Image
from scipy.signal import convolve2d

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def ssimOne(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))

def ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):
    np1 = np.array(im1)
    np2 = np.array(im2)
    if len(im1.shape)>2:
        ssim1 = ssimOne(np1[:,:,0], np2[:,:,0], k1=k1, k2=k2, win_size=win_size, L=L)
        ssim2 = ssimOne(np1[:,:,0], np2[:,:,0], k1=k1, k2=k2, win_size=win_size, L=L)
        ssim3 = ssimOne(np1[:,:,0], np2[:,:,0], k1=k1, k2=k2, win_size=win_size, L=L)
        return (ssim1+ssim2+ssim3)/3.0
    else:
        return ssimOne(np1, np2, k1=k1, k2=k2, win_size=win_size, L=L)
    #print(compute_ssim(np.array(im1), np.array(im2)))



'''
def ssim(target, ref):
    # assume RGB images
    target_data = np.array(target)
    #target_data = target_data[scale:-scale, scale:-scale]

    ref_data = np.array(ref)
    #ref_data = ref_data[scale:-scale, scale:-scale]

    k1 = 0.01
    k2 = 0.03
    L = 1
    c1 = (k1*L) ** 2.
    c2 = (k2*L) ** 2.
    c3 = c2/2.

    target = target.flatten('C')  # 折叠成一维的数组
    ref = ref.flatten('C')  # 折叠成一维的数组
    ux = np.mean(target)    #平均
    ox2 = np.var(target,ddof=1)    #方差
    ox = math.sqrt(ox2)    #标准差
    uy = np.mean(ref)    #平均
    oy2 = np.var(ref,ddof=1)    #方差
    oy = math.sqrt(oy2)    #标准差
    cxy = ox*oy

    ssimXY = (2.*ux*uy + c1) * (2.*cxy + c2) \
             / (ux*ux + uy*uy +c1) / (ox2 + oy2 + c2)

    return ssimXY
'''

if __name__ == '__main__':
    print(np.mean([1, 2, 3]))
    print(math.sqrt(np.var([0, 4, 1, 2, 3],ddof=1)))
    print(np.std([0, 4, 1, 2, 3],ddof=1))
    