import os

from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


def show(path):
    data = loadmat(path)
    mag = data['mag']
    # g = data['g']
    dir_path = path[:-4]
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if 'ma' in data.keys():
        ma = data['ma']
        plt.imsave(dir_path + '/ma.png', ma, cmap='jet')
    vmin, vmax = mag.min(), mag.max()
    for i in range(mag.shape[2]):
        plt.imsave(
            dir_path + f'/mag{i:03d}.png',
            mag[..., i],
            cmap='jet',
            vmin=vmin,
            vmax=vmax)
    savemat(f'{dir_path}/data.mat', data)


def show_zl():
    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    pred = loadmat('data/field_data/field_ref_v2.mat')
    ori = loadmat('data/field_data/field_data.mat')['ma']
    ma = pred['ma']
    mag = pred['mag']

    plt.imshow(ma, cmap='gray')
    plt.title('重建地表磁异常')
    plt.yticks((0, 14, 29, 43, 57, 71, 86, 99), (0, 1, 2, 3, 4, 5, 6, 7))
    plt.ylabel('Y / km')
    plt.xticks((0, 15, 31, 46, 62), (0, 1, 2, 3, 4))
    plt.xlabel('X / km')
    plt.savefig('data/field_data/field_ref_v2/ma_.png')

    plt.imshow(ori, cmap='gray')
    plt.title('原始地表磁异常')
    plt.yticks((0, 14, 29, 43, 57, 71, 86, 99), (0, 1, 2, 3, 4, 5, 6, 7))
    plt.ylabel('Y / km')
    plt.xticks((0, 15, 31, 46, 62), (0, 1, 2, 3, 4))
    plt.xlabel('X / km')
    plt.savefig('data/field_data/field_ref_v2/ma_ori.png')

    plt.close()
    print(mag.shape)
    mag_data = mag[65].T
    plt.imshow(mag_data, cmap='gray')
    plt.title('磁化率剖面')
    plt.xticks((0, 15, 31, 46, 61), (0, 1, 2, 3, 4))
    plt.xlabel('X / km')
    # plt.xticks((0, 14, 29, 43, 57, 71, 86, 99), (0, 1, 2, 3, 4, 5, 6, 7))
    # plt.xlabel('Y / km')
    plt.yticks((0, 9.5, 19), (0, 1, 2))
    plt.ylabel('Depth / km')
    plt.savefig('data/field_data/field_ref_v2/mag.png')


def draw_loss():
    file1 = 'work_dirs/ref_v1_self/weights/log.txt'
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = f.read().split('\n')
    file2 = 'work_dirs/ref_v2_self/weights/log.txt'
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = f.read().split('\n')
    print(len(data1))
    loss1 = []
    for data in data1:
        items = data.split('\t')
        if len(items) < 2:
            continue
        loss1.append(float(items[2].split(' ')[1]))
    loss2 = []
    for data in data2:
        items = data.split('\t')
        if len(items) < 2:
            continue
        loss2.append(float(items[2].split(' ')[1]))
    print(len(loss1), len(loss2))
    x = list(range(1, 3001))
    plt.plot(x, loss2, 'r')
    plt.plot(x, loss1, 'b')
    plt.axis([-10, 3001, -0.005, 5.8])
    plt.title('The influence of `tan activate` on training')
    plt.legend(['SSLMI', 'without `tan`'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('data/field_data/loss_en.png')


if __name__ == '__main__':
    # show('data/field_data/field_ref_v1_65.mat')
    # show('data/field_data/field_ref_v2_65.mat')
    # show_zl()
    draw_loss()
