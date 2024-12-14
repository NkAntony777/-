import streamlit as st
from PIL import Image
import torch
from torchvision.utils import save_image
from models import TransformerNet
from utils import style_transform, denormalize, deprocess
import os
import skvideo.io
import tqdm
import av

# Load model and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = TransformerNet().to(device)

# Predefined models
STYLE_MODELS = {
    "Cuphead": "cuphead_10000.pth",
    "Starry Night": "starry_night_10000.pth",
    "Mosaic": "mosaic_10000.pth"
}

def load_model(model_path):
    transformer.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    transformer.eval()

def extract_frames(video_path):
    """ Extracts only video frames from the input video """
    video = av.open(video_path)
    video_stream = next(s for s in video.streams if s.type == 'video')  # Select video stream
    for frame in video.decode(video_stream):
        yield frame.to_image()

# Streamlit App Interface
st.title("快速风格迁移应用")

st.sidebar.header("上传与风格设置")

# Upload content image
content_image_file = st.sidebar.file_uploader("上传要风格迁移的图像", type=["jpg", "jpeg", "png"])
if content_image_file is not None:
    content_image = Image.open(content_image_file)
    st.image(content_image, caption="上传的内容图像", use_container_width=True)

# Upload video
video_file = st.sidebar.file_uploader("上传要风格迁移的视频", type=["mp4", "avi", "mov"])

# Select style model
selected_style = st.sidebar.selectbox("选择风格", options=list(STYLE_MODELS.keys()))

# Apply Style Transfer to Image
if st.sidebar.button("开始图像风格迁移"):
    if content_image_file is None:
        st.sidebar.error("请上传内容图片")
    else:
        with st.spinner("正在飞速加载..."):
            # Load the selected style model
            model_path = STYLE_MODELS[selected_style]
            load_model(model_path)

            # Transform content image
            transform = style_transform()
            content_tensor = transform(content_image).unsqueeze(0).to(device)

            with torch.no_grad():
                stylized_tensor = transformer(content_tensor)
            stylized_image = denormalize(stylized_tensor.cpu())[0]

            # Save and display result
            output_path = "stylized_output.jpg"
            save_image(stylized_image, output_path)
            result_image = Image.open(output_path)
            st.image(result_image, caption="风格化后的图像", use_container_width=True)
            
            # Add save instruction
            st.info("长按或右键点击图片进行保存。")

# Apply Style Transfer to Video
if st.sidebar.button("开始视频风格迁移"):
    if video_file is None:
        st.sidebar.error("请上传视频文件")
    else:
        with st.spinner("正在处理视频..."):
            # Load the selected style model
            model_path = STYLE_MODELS[selected_style]
            load_model(model_path)

            # Process video
            video_path = f"uploaded_{video_file.name}"
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            stylized_frames = []
            for frame in tqdm.tqdm(extract_frames(video_path), desc="Processing frames"):
                try:
                    image_tensor = style_transform()(frame).unsqueeze(0).to(device)
                    with torch.no_grad():
                        stylized_image = transformer(image_tensor)
                    stylized_frames.append(deprocess(stylized_image))
                except Exception as e:
                    st.warning(f"跳过无法处理的帧：{e}")

            # Save video
            output_video_path = f"stylized_{video_file.name}.gif"
            writer = skvideo.io.FFmpegWriter(output_video_path)
            for frame in tqdm.tqdm(stylized_frames, desc="Writing to video"):
                writer.writeFrame(frame)
            writer.close()

            st.video(output_video_path)
            st.success("视频风格迁移完成！")
