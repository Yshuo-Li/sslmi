import numpy as np
import math


# Peak Signal to Noise Ratio
# peak中文意思是顶点。而ratio的意思是比率或比列的。整个意思就是到达噪音比率的顶点信号，psnr一般是用于最大值信号和背景噪音之间
# 的一个工程项目。通常在经过影像压缩之后，输出的影像都会在某种程度与原始影像不同。为了衡量经过处理后的影像品质，我们通常会参考
# PSNR值来衡量某个处理程序能否令人满意。它是原图像与被处理图像之间的均方误差相对于(2^n-1)^2的对数值(信号最大值的平方，n是每个
# 采样值的比特数)，它的单位是dB。 MATLAB用法的公式如下：
# PSNR=10*log10((2^n-1)^2/MSE)
# 其中，MSE是原图像与处理图像之间均方误差
# 此处n=1
def psnr(target, ref, bit=1):
    # assume RGB images
    target_data = np.array(target)
    #target_data = target_data[scale:-scale, scale:-scale]

    ref_data = np.array(ref)
    #ref_data = ref_data[scale:-scale, scale:-scale]

    diff = ref_data - target_data
    diff = diff.flatten('C')  # 折叠成一维的数组
    rmse = math.sqrt(np.mean(diff ** 2.))
    if rmse==0:
        return float('inf')
    return 20 * math.log10(1.0 / rmse)

if __name__ == '__main__':
    a=np.ones([100,100])*1.0
    b=np.array(a/2)
    b[1,1]=1.1
    print(a)
    print(b)
    print(psnr(a,b))
