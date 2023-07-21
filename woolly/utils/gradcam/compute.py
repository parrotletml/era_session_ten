# Create dataset to load images

import torch
from torch.utils import data

from woolly.utils.gradcam.gradcam import GradCAM
from woolly.utils.gradcam.util import (
    get_prediction_for_image,
    generate_heat_map,
    apply_heatmap_to_image,
    plot_output,
)

import cv2 as cv


def compute_gradcam(
    model, class_map, image, impath, label, channel_size=128, device="cpu"
):
    # initialize the GradCAM model
    gradcam = GradCAM(model).to(device)

    # To device
    img = image.clone().to(device)
    # Read the original image
    origimg = cv.imread(impath)
    origimg = cv.cvtColor(origimg, cv.COLOR_BGR2RGB)
    # dim = (100, int(origimg.shape[0] * (100.0 / origimg.shape[1])))
    # dim = (int(origimg.shape[1] * (100.0 / origimg.shape[0])), 100)
    # origimg = cv.resize(origimg, dim)

    # Get Predictions
    prediction = get_prediction_for_image(
        gradcam, torch.reshape(img, (1, 3, 32, 32)), device
    )
    # Get Heatmap for this prediction
    heatmap = generate_heat_map(
        gradcam,
        prediction,
        torch.reshape(img, (1, 3, 32, 32)),
        channel_size=channel_size,
    )
    # Get Overlayed image
    superimposed = apply_heatmap_to_image(image.squeeze(), heatmap)

    # Plot Images
    plot_output(
        image.squeeze(),
        heatmap,
        superimposed,
        label,
        prediction.argmax(dim=1),
        class_map,
        origimg,
    )
