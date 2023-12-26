import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio
# 文件夹路径
folder_path = r'/media/ders/Fjmnew/newdataPuse/test/labels/'
out_path = r'/media/ders/Fjmnew/newdataPuse/vistest/'
# 获取文件夹内所有文件名
file_names = os.listdir(folder_path)
label_colors = {
    0: (255, 255, 255),  # 背景
    1: (0, 0, 255),  # 类别1
    2: (0, 255, 255),  # 类别2
    3: (0, 255, 0),  # 类别3
    4: (255, 204, 0),  # 类别3
    5: (255, 0, 0)  # 类别3
}
# 遍历文件夹内所有文件
for file_name in file_names:
    # 判断文件是否为图片
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        # 读取图片
        img_path = os.path.join(folder_path, file_name)
        img = Image.open(img_path)

        # 将图像转换为numpy数组
        pred_mask = np.array(img)

        # 定义标签颜色映射

        # 创建一个彩色图像，与标签掩码图像大小相同
        colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)

        # 遍历标签掩码图像的每个像素，并根据标签颜色映射进行上色
        for label in label_colors:
            colored_mask[pred_mask == label] = label_colors[label]

        image_id = file_name[:-4]# 不带后缀的所有文件名
        imageio.imwrite(out_path + image_id + '.png', colored_mask.astype(np.uint8))
        # plt.imshow(img_array)
        # plt.show()

print('图片处理完成')

