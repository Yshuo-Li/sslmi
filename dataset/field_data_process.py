import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as scio


def jinchuan():
    data = scio.loadmat('data/field_data/field.mat')['magg']
    with open('data/field_data/field.txt', 'w', encoding='utf-8') as f:
        for d in data:
            f.write(f'{d[0]}\t{d[1]}\t{d[2]}\n')
    print(data.shape, data[0])
    n, m = 100, 62
    n_min, n_max = data[:, 0].min(), data[:, 0].max()
    print(n_max, n_min)
    m_min, m_max = data[:, 1].min(), data[:, 1].max()
    print(m_max, m_min)
    # print()
    # for j in range(1, 100):
    #     print(data[j][0]-data[j-1][0])
    # print()
    # for i in range(1, 62):
    #     print(data[i*100][1]-data[i*100-100][1])
    # matrix = np.zeros((m, n), dtype=np.float32)
    # for i, j, v in data:
    #     # print(i, j, n_min, n_max, n, (i-n_min), (n_max-n_min))
    #     i = round((i-n_min)/(n_max-n_min)*(n+1))
    #     j = round((j-m_min)/(m_max-m_min)*(m+1))
    #     print(i, j)
    #     if matrix[j, i] != 0:
    #         input()
    #     matrix[j, i] = v
    # k = 0
    # for i in range(n):
    #     for j in range(m):
    #         matrix[i, j] = data[k][2]
    #         k += 1
    matrix = data[:, 2]
    matrix = matrix.reshape(m, n).T
    plt.imsave('data/field_data/field.png', matrix, cmap='jet')
    scio.savemat('data/field_data/field_data.mat', dict(ma=matrix))


if __name__ == '__main__':
    jinchuan()
