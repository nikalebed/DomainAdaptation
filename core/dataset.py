import os
import sys
import dlib
import PIL

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as transforms

from utils.common import align_face

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class ImagesDataset(Dataset):

    def __init__(self, opts, image_path=None, face_predictor=None, align_input=False):

        if type(image_path) == list:
            self.image_paths = image_path
        elif os.path.isdir(image_path):
            self.image_paths = sorted(make_dataset(image_path))
        elif os.path.isfile(image_path):
            self.image_paths = [image_path]
        else:
            sys.exit('Invalid Input')

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.opts = opts
        self.align_input = align_input
        self.predictor = face_predictor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        im_path = self.image_paths[index]

        if self.align_input and self.predictor is not None:
            im_H = align_face(im_path, self.predictor, output_size=self.opts.size)
        else:
            im_H = Image.open(im_path).convert('RGB')
        im_L = im_H.resize((256, 256), PIL.Image.LANCZOS)
        im_name = os.path.splitext(os.path.basename(im_path))[0]
        if self.image_transform:
            im_H = self.image_transform(im_H)
            im_L = self.image_transform(im_L)

        return im_H, im_L, im_name
