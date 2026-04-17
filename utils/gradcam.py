import torch
import cv2
import numpy as np

def gradcam(model, image, target_class):
    model.eval()

    image = image.unsqueeze(0)

    output = model(image)
    loss = output[0, target_class]

    loss.backward()

    gradients = model.conv2.weight.grad
    activations = model.conv2(image)

    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * activations, dim=1)

    cam = cam.detach().numpy()[0]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cv2.resize(cam, (28, 28))