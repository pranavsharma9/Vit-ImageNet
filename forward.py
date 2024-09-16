import numpy as np
from PIL import Image
import torch

k = 10

imagenet_labels = dict(enumerate(open("data/classes.txt")))

model = torch.load("model.pth")
model.eval()
# img=Image.open("data/goldebn_ret.jpg").resize((384, 384))
img=Image.open("data/cat.png")
print(img.size)

img = (np.array(img) / 128) - 1  
inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
logits = model(inp)
probs = torch.nn.functional.softmax(logits, dim=-1)

top_probs, top_ixs = probs[0].topk(k)

for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
    ix = ix_.item()
    prob = prob_.item()
    cls = imagenet_labels[ix].strip()
    print(f"{i}: {cls:<45} --- {prob:.4f}")