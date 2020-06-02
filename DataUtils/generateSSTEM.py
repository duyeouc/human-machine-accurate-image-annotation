import scipy.io as scio
import glob
import numpy as np
import os
from PIL import Image



def generate():
    dir = 'C:/Users/DELL/Desktop/毕设2020/数据集/医学数据2groundtruth-drosophila-vnc-master/stack1/'
    save_dir = dir + 'new/'

    raw_img_dir = dir + 'raw/'
    class1_label_dir = dir + 'mitochondria/'
    class2_label_dir = dir + 'synapses/'

    # 分类1
    save_img_dir1 = save_dir + 'img1/'
    save_label_dir1 = save_dir + 'label1/'

    # 分类2
    save_img_dir2 = save_dir + 'img2/'
    save_label_dir2 = save_dir + 'label2/'

    if not os.path.exists(save_img_dir1):
        os.makedirs(save_img_dir1)
    if not os.path.exists(save_img_dir2):
        os.makedirs(save_img_dir2)
    if not os.path.exists(save_label_dir1):
        os.makedirs(save_label_dir1)
    if not os.path.exists(save_label_dir2):
        os.makedirs(save_label_dir2)

    files = glob.glob(class1_label_dir + '*')
    # 先提取分类1
    for f in files:
        # 读取
        img_lbl = Image.open(f)
        W = img_lbl.width
        H = img_lbl.height
        # img_lbl.show()
        x = np.array(img_lbl)  # (H, W)
        # print(x.shape)
        used = np.zeros((H, W))
        cur_img_name = f.split('\\')[-1][:-4]
        # print(cur_img_name)
        cnt = 0
        for i in range(H):
            for j in range(W):
                if x[i, j] == 255 and used[i, j] == 0:
                    # 寻找与(i,j)相邻的所有255，设置其used=1, 并添加到new_map
                    used[i, j] = 1
                    new_map = np.zeros((H, W))
                    new_map[i, j] = 255
                    queue = []
                    queue.append((i, j))
                    dirs = [[0, 1], [0, -1], [1, 0], [-1, 0], [-1, -1], [1, 1], [-1, 1], [1, -1]]
                    while queue:
                        x_now, y_now = queue.pop(0)
                        for dx, dy in dirs:
                            nx, ny = x_now + dx, y_now + dy
                            if 0 <= nx < H and 0 <= ny < W and x[nx, ny] == 255 and used[nx, ny] == 0:
                                queue.append((nx, ny))
                                used[nx, ny] = 1
                                new_map[nx, ny] = 255
                    # 结束, 保存一个new_map
                    new_img = Image.fromarray(new_map.astype('uint8'))
                    new_img.save(save_label_dir1 + cur_img_name + str(cnt) + '.png')
                    cnt += 1
    files = glob.glob(class2_label_dir + '*')
    # print(files)
    for f in files:
        # 读取
        img_lbl = Image.open(f)
        W = img_lbl.width
        H = img_lbl.height
        # img_lbl.show()
        x = np.array(img_lbl)  # (H, W, C)
        # print(x.shape)
        used = np.zeros((H, W))
        cur_img_name = f.split('\\')[-1][:-4]
        print(cur_img_name)
        cnt = 0
        for i in range(H):
            for j in range(W):
                if x[i, j] !=0 and used[i, j] == 0:
                    # 寻找与(i,j)相邻的所有255，设置其used=1, 并添加到new_map
                    used[i, j] = 1
                    new_map = np.zeros((H, W))
                    new_map[i, j] = 255
                    queue = []
                    queue.append((i, j))
                    dirs = [[0, 1], [0, -1], [1, 0], [-1, 0], [-1, -1], [1, 1], [-1, 1], [1, -1]]
                    while queue:
                        x_now, y_now = queue.pop(0)
                        for dx, dy in dirs:
                            nx, ny = x_now + dx, y_now + dy
                            if 0 <= nx < H and 0 <= ny < W and x[nx, ny] !=0 and used[nx, ny] == 0:
                                queue.append((nx, ny))
                                used[nx, ny] = 1
                                new_map[nx, ny] = 255
                    # 结束, 保存一个new_map
                    new_img = Image.fromarray(new_map.astype('uint8'))
                    save_file = save_label_dir2 + cur_img_name + str(cnt) + '.png'
                    print(save_file)
                    new_img.save(save_file)
                    cnt += 1















if __name__ == '__main__':
    generate()
    x = 'C:/Users/DELL/Desktop/毕设2020/数据集/医学数据2groundtruth-drosophila-vnc-master/stack1/new/label1/000.png'
    img = Image.open(x)
    z = np.array(img)
    print(z.shape)
    print(z[:, :, 1][z[:, :, 1] == 255])