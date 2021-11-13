import torchvision.transforms as transforms

from os import sys, path
sys.path.append('../')
from datasets import CubDADataset

def testCubDA():
    da_root = '/data3/private/mxy/projects/mmda/data/v4-0.9-80k/sample_images'
    base_size = 64
    imsize = base_size * 4
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    ds = CubDADataset(da_root,
                      base_size=base_size,
                      transform=image_transform)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    testCubDA()