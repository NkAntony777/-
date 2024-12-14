from models import TransformerNet
from utils import extract_frames, style_transform, deprocess, save_video
import torch
from torch.autograd import Variable
import argparse
import os
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to video")
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    parser.add_argument("--output_path", type=str, default="stylized_output.mp4", help="Path to save stylized video")
    args = parser.parse_args()

    os.makedirs("images/outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and prepare the model
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(args.checkpoint_model))
    transformer.eval()

    # Read video properties
    stylized_frames = []
    transform = style_transform()
    frame_height, frame_width = None, None
    fps = 24  # Default fps, replace if actual fps info is required

    for frame in tqdm.tqdm(extract_frames(args.video_path), desc="Processing frames"):
        if frame_height is None or frame_width is None:
            frame_height, frame_width = frame.size[1], frame.size[0]

        # Prepare input frame
        image_tensor = Variable(transform(frame)).to(device).unsqueeze(0)

        # Stylize image
        with torch.no_grad():
            stylized_image = transformer(image_tensor)

        # Collect stylized frame
        stylized_frames.append(deprocess(stylized_image))

    # Save the stylized video
    output_path = args.output_path
    save_video(stylized_frames, output_path, fps, (frame_width, frame_height))

    print(f"Stylized video saved at {output_path}")
