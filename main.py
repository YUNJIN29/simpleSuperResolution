import argparse
import os

import torch
from torchvision.utils import save_image
from PIL import Image

from utils.imageUtil import ImageSplitter
from Module import SRCNN


def getOpts():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", default='', help="输入图片路径")
    parser.add_argument('output', default='.', help="输出存储目录")
    parser.add_argument('-s', '--scale', default=2, type=int, help="放大倍率，默认2")
    parser.add_argument('-n', '--name', default='output.png', help="输出文件文件名，默认output.png")
    parser.add_argument('-c', '--checkpoint', required=True, help="使用的checkpoint")
    parser.add_argument('--border-size', default=6, type=int, help="图像切片边框宽度")
    return parser.parse_args()


opts = getOpts()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_filetype = ('jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp')


def loadModel(checkopint):
    model = SRCNN()
    model = model.to(device)
    model.load_state_dict(torch.load(checkopint).get('model'))
    return model


def calcImg(model, pic, border_pad_size=6):
    img_splitter = ImageSplitter(border_pad_size=border_pad_size)
    img_patchs = img_splitter.split_img_tensor(pic)
    with torch.no_grad():
        out = [model(i.to(device)) for i in img_patchs]
    return img_splitter.merge_img_tensor(out)


if __name__ == '__main__':
    model = loadModel(opts.checkpoint)
    if not os.path.isfile(opts.input):
        raise TypeError('输入必须为图片文件')
    if not opts.input.rsplit('.', maxsplit=1)[-1].lower() in _filetype:
        raise TypeError('不支持此文件格式')
    if not os.path.isdir(opts.output):
        raise ValueError('存储目录不存在')
    img = Image.open(opts.input)
    img = img.resize((img.size[0] * opts.scale, img.size[1] * opts.scale), Image.Resampling.BICUBIC)
    final = calcImg(model, img, opts.border_size)
    save_image(final, os.path.join(opts.output, opts.name))
