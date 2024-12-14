import streamlit as st
from PIL import Image
import torch
from torchvision.utils import save_image
from models import TransformerNet
from utils import style_transform, denormalize

# Load model and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = TransformerNet().to(device)

def load_model(model_path):
    transformer.load_state_dict(torch.load(model_path, map_location=device))
    transformer.eval()

# Streamlit App Interface
st.title("Style Transfer App")

st.sidebar.header("Upload and Settings")

# Upload content image
content_image_file = st.sidebar.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
if content_image_file is not None:
    content_image = Image.open(content_image_file)
    st.image(content_image, caption="Uploaded Content Image", use_column_width=True)

# Upload style model
style_model_file = st.sidebar.file_uploader("Upload Style Model (.pth)", type=["pth"])
if style_model_file is not None:
    load_model(style_model_file)
    st.sidebar.success("Style model loaded successfully")

# Apply Style Transfer
if st.sidebar.button("Apply Style"):
    if content_image_file is None:
        st.sidebar.error("Please upload a content image.")
    elif style_model_file is None:
        st.sidebar.error("Please upload a style model.")
    else:
        with st.spinner("Applying style..."):
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
            st.image(result_image, caption="Stylized Image", use_column_width=True)
            st.sidebar.success(f"Stylized image saved as {output_path}")
