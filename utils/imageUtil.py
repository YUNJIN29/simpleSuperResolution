import copy

import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class ImageSplitter:
    def __init__(self, seg_size=48, scale_factor=2, border_pad_size=3):
        self.seg_size = seg_size
        self.scale_factor = scale_factor
        self.pad_size = border_pad_size
        self.height = 0
        self.width = 0
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.to_tensor = transforms.ToTensor()

    def split_img_tensor(self, pil_img, img_pad=0):
        img_tensor = pil_img
        if isinstance(pil_img, Image.Image):
            img_tensor = self.to_tensor(pil_img.convert('RGB')).unsqueeze(0)
        # else:
        #     img_tensor = pil_img.unsqueeze(0)
        img_tensor = nn.ReplicationPad2d(self.pad_size)(img_tensor)
        batch, channel, height, width = img_tensor.size()
        self.height = height
        self.width = width
        patchs = []
        if height % self.seg_size < self.pad_size or width % self.seg_size < self.pad_size:
            self.seg_size += self.scale_factor * self.pad_size

        for i in range(self.pad_size, height, self.seg_size):
            for j in range(self.pad_size, width, self.seg_size):
                part = img_tensor[:, :, (i - self.pad_size):min(i + self.pad_size + self.seg_size, height),
                                        (j - self.pad_size):min(j + self.pad_size + self.seg_size, width)]
                if img_pad > 0:
                    part = nn.ZeroPad2d(img_pad)(part)
                patchs.append(part)
        return patchs

    def merge_img_tensor(self, img_tensor_list):
        out = torch.zeros((1, 3, self.height, self.width))
        img_tensors = copy.copy(img_tensor_list)
        rem = self.pad_size

        pad_size = self.pad_size
        seg_size = self.seg_size
        height = self.height
        width = self.width
        for i in range(pad_size, height, seg_size):
            for j in range(pad_size, width, seg_size):
                part = img_tensors.pop(0)
                part = part[:, :, rem:-rem, rem:-rem]
                if len(part.size()) > 3:
                    _, _, p_h, p_w = part.size()
                    out[:, :, i:i + p_h, j:j + p_w] = part
        out = out[:, :, rem:-rem, rem:-rem]
        return out
