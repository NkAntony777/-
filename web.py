import streamlit as st
from PIL import Image
import torch
from torchvision.utils import save_image
from models import TransformerNet
from utils import style_transform, denormalize

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
    transformer.load_state_dict(torch.load(model_path, map_location=device))
    transformer.eval()

# Streamlit App Interface
st.title("快速风格迁移应用")

st.sidebar.header("上传与风格设置")

# Upload content image
content_image_file = st.sidebar.file_uploader("上传要风格迁移的图像", type=["jpg", "jpeg", "png"])
if content_image_file is not None:
    content_image = Image.open(content_image_file)
    st.image(content_image, caption="上传的内容图像", use_column_width=True)

# Select style model
selected_style = st.sidebar.selectbox("选择风格", options=list(STYLE_MODELS.keys()))

# Apply Style Transfer
if st.sidebar.button("开始风格迁移"):
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
            st.image(result_image, caption="风格化后的图像", use_column_width=True)
            
            # Add save button
            if st.button("保存风格化图像"):
                with open(output_path, "rb") as file:
                    btn = st.download_button(
                        label="下载风格化图像",
                        data=file,
                        file_name="stylized_image.jpg",
                        mime="image/jpeg"
                    )
