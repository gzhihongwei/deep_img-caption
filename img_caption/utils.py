"""Title: Image captioning
Author: Persson, Aladdin
Date: 2020
Availability: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning
"""

import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, dataset):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    model.eval()

    test_img1 = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(0)
    print("Example 1 caption: Dog on a beach by the ocean")
    print(f"Example 1 prediction: {' '.join(model.caption_image(test_img1.to(device), dataset.vocab))}")

    test_img2 = transform(Image.open("test_examples/child.jpg").convert("RGB")).unsqueeze(0)
    print("Example 2 caption: Child holding red frisbee outdoors")
    print(f"Example 2 prediction: {' '.join(model.caption_image(test_img2.to(device), dataset.vocab))}")

    test_img3 = transform(Image.open("test_examples/bus.png").convert("RGB")).unsqueeze(0)
    print("Example 3 caption: Bus driving by parked cars")
    print(f"Example 3 prediction: {' '.join(model.caption_image(test_img3.to(device), dataset.vocab))}")

    test_img4 = transform(Image.open("test_examples/boat.png").convert("RGB")).unsqueeze(0)
    print("Example 4 caption: A small boat in the ocean")
    print(f"Example 4 prediction: {' '.join(model.caption_image(test_img4.to(device), dataset.vocab))}")

    test_img5 = transform(Image.open("test_examples/horse.png").convert("RGB")).unsqueeze(0)
    print("Example 5 caption: A cowboy riding a horse in the desert")
    print(f"Example 5 prediction: {' '.join(model.caption_image(test_img5.to(device), dataset.vocab))}")

    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint["step"]
