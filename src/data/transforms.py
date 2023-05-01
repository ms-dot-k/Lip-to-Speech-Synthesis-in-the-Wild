import random
import torchvision.transforms.functional as F

class StatefulRandomHorizontalFlip():
    def __init__(self, probability=0.5):
        self.probability = probability
        self.rand = random.random()

    def __call__(self, img):
        if self.rand < self.probability:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(probability={})'.format(self.probability)


class Crop(object):
    def __init__(self, crop):
        self.crop = crop

    def __call__(self, img):
        return img.crop(self.crop)
