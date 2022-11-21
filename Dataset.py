from PIL import Image
from torchvision.datasets.vision import *
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import ToTensor
import glob


class ImgDataset(VisionDataset):
    _filetype = ('jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp')

    def __init__(self, root: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, LR_dir: str = 'LR', HR_dir: str = 'HR') -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.lr_dir = LR_dir
        self.hr_dir = HR_dir
        LR_files = glob.glob(self.root + '/' + self.lr_dir + '/**', recursive=True)
        HR_files = glob.glob(self.root + '/' + self.hr_dir + '/**', recursive=True)
        self.LR_imgs = list(filter(lambda f: f.split('.')[-1] in self._filetype, LR_files))
        self.HR_imgs = list(filter(lambda f: f.split('.')[-1] in self._filetype, HR_files))
        self.length = len(self.HR_imgs)
        self.toTensor = ToTensor()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.LR_imgs[index]).convert('RGB')
        target = Image.open(self.HR_imgs[index]).convert('RGB')
        img = img.resize((img.size[0] * 2, img.size[1] * 2), Image.Resampling.BICUBIC)
        if self.transform is not None:
            img, target = self.transforms(img, target)
        else:
            img = self.toTensor(img)
            target = self.toTensor(target)
        return img, target

    def __len__(self) -> int:
        return self.length

# class ImgDataSet(VisionDataset):
#     def __init__(self, root: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None,
#                  target_transform: Optional[Callable] = None, lr_class='LR', hr_class='HR') -> None:
#         super().__init__(root, transforms, transform, target_transform)
#         self.imgs = ImageFolder(root)
#         self.lr = self.imgs.class_to_idx.get(lr_class)
#         self.hr = self.imgs.class_to_idx.get(hr_class)
#         if self.lr is None or self.hr is None:
#             raise RuntimeError('Dataset class not exists')
#
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         img = self.imgs[self.lr][index]
#         target = self.imgs[self.hr][index]
#         if self.transforms is not None:
#             img, target = self.transforms(img, target)
#         else:
#             img = to_tensor(img)
#             target = to_tensor(target)
#         return img, target
#
#     def __len__(self) -> int:
#         return len(self.imgs[self.lr])
