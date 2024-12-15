from torchvision import transforms
import torch
import numpy as np


# Mean and standard deviation used for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def extract_frames(video_path):
    """ Extracts frames from video """
    frames = []
    video = av.open(video_path)
    for frame in video.decode(0):
        yield frame.to_image()


def gram_matrix(y):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram


def train_transform(image_size):
    """ Transforms for training images """
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform


def style_transform(image_size=None):
    """ Transforms for style image """
    resize = [transforms.Resize(image_size)] if image_size else []
    transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transform


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors


def deprocess(image_tensor):
    """ Denormalizes and rescales image tensor """
    image_tensor = denormalize(image_tensor)[0]
    image_tensor *= 255
    image_np = torch.clamp(image_tensor, 0, 255).cpu().numpy().astype(np.uint8)
    image_np = image_np.transpose(1, 2, 0)
    return image_np



import av
import numpy as np
from PIL import Image
import cv2

def save_video(frames, output_path, fps, size):
    """
    Saves the processed frames into an MP4 video file using OpenCV.

    :param frames: List of processed frames (numpy arrays or PIL images).
    :param output_path: The output path to save the MP4 video.
    :param fps: Frames per second for the output video.
    :param size: Tuple of (width, height) for the video resolution.
    """
    # Create a VideoWriter object to write frames into an MP4 file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    for frame in frames:
        # If the frame is a PIL image, convert it to numpy array
        if isinstance(frame, Image.Image):
            frame = np.array(frame)

        # Ensure the frame is in the correct format (BGR for OpenCV)
        if len(frame.shape) == 3 and frame.shape[2] == 3:  # Check if the frame is RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write the frame to the video file
        writer.write(frame)

    # Release the video writer
    writer.release()

    print(f"Video saved at {output_path}")
