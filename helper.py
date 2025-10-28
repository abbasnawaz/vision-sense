import torch
import cv2 as cv
import os
import json


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json():
    with open("config.json") as f:
        config = json.load(f)

    model_name = config["model"]
    confidence_threshold = config["confidence_threshold"]
    stream = config["stream"]
    return model_name, confidence_threshold, stream



def load_json_variable(variable):
    with open("config.json") as f:
        config = json.load(f)

    return config[variable]

