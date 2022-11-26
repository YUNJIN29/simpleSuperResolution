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
                                 help='The device used for train. eg:cuda, cpu, cuda:0')
        self.parser.add_argument('--train-dir', default='dataset/train')
        self.parser.add_argument('--test-dir', default='dataset/test')
        self.parser.add_argument('--input-folder', default='LR')
        self.parser.add_argument('--target-folder', default='HR')
        self.parser.add_argument('--checkpoint', default='')
        self.parser.add_argument('--log-dir', default='logs/train_log')
        self.parser.add_argument('--lr', default=1e-4, type=float, help='The learning rate')
        self.parser.add_argument('--epoch', default=5, type=int)
        self.parser.add_argument('--seq-size', default=48, type=int)
        self.parser.add_argument('--scale', default=2, type=int)
        self.parser.add_argument('--border', default=3, type=int)
        self.parser.add_argument('--test-cycle', default=100, type=int)
        self.parser.add_argument('--save-cycle', default=500, type=int)
        self.parser.add_argument('--save-dir', default='checkpoint')
        self.parser.add_argument('--data-prefix', default='')
        self.parser.add_argument('--data-subfix', default='')
        self.parser.add_argument('--train-prefix', default='')
        self.parser.add_argument('--train-subfix', default='')
        self.parser.add_argument('--test-prefix', default='')
        self.parser.add_argument('--test-subfix', default='')
        self.parser.add_argument('--patchs', default=0, action='count')
        self.parser.add_argument('--disable-patchs-eval', default=0, action='count')
        self.parser.add_argument('--reset-counter', default=0, action='count')
        self.parser.add_argument('--disable-img-record', default=0, action='count')

    def getOpts(self):
        parse = self.parser.parse_args()
        parse.train_prefix = parse.data_prefix if parse.train_prefix == '' else parse.train_prefix
        parse.train_subfix = parse.data_subfix if parse.train_subfix == '' else parse.train_subfix
        parse.test_prefix = parse.data_prefix if parse.test_prefix == '' else parse.test_prefix
        parse.test_subfix = parse.data_subfix if parse.test_subfix == '' else parse.test_subfix
        return parse
