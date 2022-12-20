from PIL import Image
from torchvision.datasets.vision import *
from torchvision.transforms import ToTensor
import glob
import os


class ImgDataset(VisionDataset):
    _filetype = ('jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp')

    def __init__(self, root: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, scale=2, LR_dir: str = 'LR', HR_dir: str = 'HR',
                 prefix: str = '', subfix: str = '') -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.scale = scale
        self.root = root
        self.lr_dir = os.path.join(self.root, LR_dir)
        self.hr_dir = os.path.join(self.root, HR_dir)
        # LR_files = glob.glob(self.lr_dir + '/**', recursive=True)
        HR_files = glob.glob(self.hr_dir + '/**', recursive=True)
        self.imgs = []
        for file in HR_files:
            if os.path.isdir(file):
                continue
            filename = os.path.basename(file).rsplit('.', maxsplit=1)
            self.imgs.append((os.path.join(self.lr_dir, prefix + filename[0] + subfix + '.' + filename[-1]), file))
        self.length = len(self.imgs)
        self.toTensor = ToTensor()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        imgFile = self.imgs[index]
        img = Image.open(imgFile[0]).convert('RGB')
        target = Image.open(imgFile[1]).convert('RGB')
        img = img.resize((img.size[0] * self.scale, img.size[1] * self.scale), Image.Resampling.BICUBIC)
        if self.transform is not None:
            img, target = self.transforms(img, target)
        else:
            img = self.toTensor(img)
            target = self.toTensor(target)
        return img, target

    def __len__(self) -> int:
        return self.length
