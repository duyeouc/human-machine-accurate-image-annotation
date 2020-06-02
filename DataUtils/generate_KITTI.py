import os
import glob
from PIL import Image
data_path = 'D:/pycharm-project/FPN_Model/data/tracklet_roi'
image = glob.glob(os.path.join(data_path, '*.png'))
print(len(image))
cnt = 0
for im in image:
    im = Image.open(im)
    height = im.height
    width = im.width
    squrare = height * width
    if squrare >= 45*45:
        cnt += 1

print(45*45)
print(cnt)







