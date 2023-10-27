from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
print("Model loading...")
model = Blip2ForConditionalGeneration.from_pretrained("blip/", torch_dtype=torch.float16, local_files_only=True)
model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)


# [ 4,  0,  2,  8, 23, 37, 35, 35, 28, 16, 20,  0, 33, 35, 37, 3

labels = ["bookshelf", "airplane", "bed"]
img_pc = torch.load('./raw_images.pt').cuda() # 160, 3, 224, 224

for k in range(3):
    for i in range(10):
        prompt = f"Question: Is this a {labels[k]}? Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
        inputs['pixel_values'] = img_pc[None, i, :]


        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f'Num {i}: {prompt} {generated_text}')

breakpoint()