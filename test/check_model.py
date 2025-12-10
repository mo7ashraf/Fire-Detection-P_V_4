from ultralytics import YOLO
model = YOLO("runs/detect/exp_y10n_t1000_stable7/weights/best.pt")
print(model.model)  # Print model info
#Pring image data
print('Image Data')
# Open and inspect the image
from PIL import Image
img = Image.open("test/fire.24.png")
img.show()
print(img.size)  # Check resolution
from PIL import Image
src = r"test\\fire.24.png"
dst = r"test\\fire_resized.png"
img = Image.open(src)
w = 512
h = int(w * img.height / img.width)
img = img.resize((w, h), Image.LANCZOS)
img.save(dst)
print("Saved", dst, "size=", img.size)