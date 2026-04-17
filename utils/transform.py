import torch
from torchvision import transforms

def get_transform(rotation=0, noise=False):
    transform_list = []

    if rotation != 0:
        transform_list.append(transforms.RandomRotation((rotation, rotation)))

    transform_list.append(transforms.ToTensor())

    if noise:
        transform_list.append(AddNoise())

    return transforms.Compose(transform_list)


class AddNoise:
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * 0.1
        return tensor + noise