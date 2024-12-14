import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision.utils import save_image
from models import TransformerNet
from utils import style_transform, denormalize

class StyleTransferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Style Transfer GUI")

        # Load model and set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = TransformerNet().to(self.device)

        # GUI elements
        self.content_image_label = tk.Label(root, text="No content image selected")
        self.content_image_label.pack()

        self.style_image_label = tk.Label(root, text="No style model loaded")
        self.style_image_label.pack()

        self.select_image_button = tk.Button(root, text="Select Content Image", command=self.load_content_image)
        self.select_image_button.pack()

        self.load_model_button = tk.Button(root, text="Load Style Model", command=self.load_style_model)
        self.load_model_button.pack()

        self.convert_button = tk.Button(root, text="Apply Style", command=self.apply_style, state=tk.DISABLED)
        self.convert_button.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

        self.result_canvas = tk.Label(root)
        self.result_canvas.pack()

        self.content_image_path = None
        self.output_image_path = "stylized_output.jpg"

    def load_content_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.content_image_path = file_path
            content_image = Image.open(self.content_image_path)
            content_image.thumbnail((300, 300))
            content_image = ImageTk.PhotoImage(content_image)
            self.content_image_label.config(image=content_image, text="")
            self.content_image_label.image = content_image
            self.check_ready_state()

    def load_style_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if file_path:
            self.transformer.load_state_dict(torch.load(file_path, map_location=self.device))
            self.transformer.eval()
            self.style_image_label.config(text="Style model loaded successfully")
            self.check_ready_state()

    def check_ready_state(self):
        if self.content_image_path and self.transformer:
            self.convert_button.config(state=tk.NORMAL)

    def apply_style(self):
        if not self.content_image_path:
            messagebox.showerror("Error", "No content image selected")
            return

        # Transform image and apply style
        transform = style_transform()
        content_image = Image.open(self.content_image_path)
        content_tensor = transform(content_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            stylized_tensor = self.transformer(content_tensor)
        stylized_image = denormalize(stylized_tensor.cpu())[0]

        save_image(stylized_image, self.output_image_path)

        # Display result
        stylized_image = Image.open(self.output_image_path)
        stylized_image.thumbnail((300, 300))
        stylized_image = ImageTk.PhotoImage(stylized_image)
        self.result_canvas.config(image=stylized_image, text="")
        self.result_canvas.image = stylized_image
        self.result_label.config(text=f"Stylized image saved at: {self.output_image_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StyleTransferApp(root)
    root.mainloop()
