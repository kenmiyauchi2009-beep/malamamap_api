import time

import torch
import open_clip
import PIL.Image

name = "hf-hub:imageomics/bioclip-2.5-vith14"

t0 = time.time()
model, _, preprocess = open_clip.create_model_and_transforms(name)
model = model.to("cuda").eval()
print("load sec:", round(time.time() - t0, 1))

torch.cuda.reset_peak_memory_stats()
img = preprocess(PIL.Image.open("images.jpg").convert("RGB")).unsqueeze(0).to("cuda")
with torch.no_grad():
    f = model.encode_image(img)
print("image feat shape:", tuple(f.shape))
print("peak VRAM (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))
print("total VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
