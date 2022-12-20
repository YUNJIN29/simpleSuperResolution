import argparse

import torch.cuda


# train parameters
class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._setParameters()

    def __call__(self):
        self.parser.parse_args()

    def _setParameters(self):
        self.parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                                 help='训练使用的设备，如:cuda, cpu, cuda:0；默认为cuda(若cuda不可用，则默认为cpu)')
        self.parser.add_argument('--train-dir', default='dataset/train', help="训练集目录，默认dataset/train")
        self.parser.add_argument('--test-dir', default='dataset/test', help="验证集目录，默认dataset/test")
        self.parser.add_argument('--input-folder', default='LR', help="数据集输入文件夹，默认LR")
        self.parser.add_argument('--target-folder', default='HR', help="数据集目标文件夹，默认HR")
        self.parser.add_argument('--checkpoint', default='', help="载入checkpoint")
        self.parser.add_argument('--log-dir', default='logs/train_log', help="训练日志存放位置，默认logs/train_log")
        self.parser.add_argument('--lr', default=1e-4, type=float, help='学习率，默认1e-4')
        self.parser.add_argument('--epoch', default=5, type=int, help="在数据集上的训练轮数，默认5")
        self.parser.add_argument('--seq-size', default=48, type=int, help="图像切片大小(不含边框)，默认48")
        self.parser.add_argument('--scale', default=2, type=int, help="图像放大倍率，默认2")
        self.parser.add_argument('--border', default=3, type=int, help="图像切片边框大小，默认3")
        self.parser.add_argument('--test-cycle', default=100, type=int, help="模型验证周期，默认100")
        self.parser.add_argument('--save-cycle', default=500, type=int, help="checkpoint保存周期，默认500")
        self.parser.add_argument('--save-dir', default='checkpoint', help="checkpoint保存目录，默认checkpoint")
        self.parser.add_argument('--data-prefix', default='', help="数据集中输入文件的文件名前缀")
        self.parser.add_argument('--data-subfix', default='', help="数据集中输入文件的文件名后缀")
        self.parser.add_argument('--train-prefix', default='', help="训练集中输入文件的文件名前缀")
        self.parser.add_argument('--train-subfix', default='', help="训练集中输入文件的文件名后缀")
        self.parser.add_argument('--test-prefix', default='', help="验证集中输入文件的文件名前缀")
        self.parser.add_argument('--test-subfix', default='', help="验证集中输入文件的文件名后缀")
        self.parser.add_argument('--no-patchs', default=0, action='count', help="不使用切片训练")
        self.parser.add_argument('--disable-patchs-eval', default=0, action='count', help="不使用切片验证，仅在启用切片训练时生效")
        self.parser.add_argument('--reset-counter', default=0, action='count', help="重置训练计数器")
        self.parser.add_argument('--disable-img-record', default=0, action='count', help="日志不记录验证机输出")

    def getOpts(self):
        parse = self.parser.parse_args()
        parse.train_prefix = parse.data_prefix if parse.train_prefix == '' else parse.train_prefix
        parse.train_subfix = parse.data_subfix if parse.train_subfix == '' else parse.train_subfix
        parse.test_prefix = parse.data_prefix if parse.test_prefix == '' else parse.test_prefix
        parse.test_subfix = parse.data_subfix if parse.test_subfix == '' else parse.test_subfix
        return parse
