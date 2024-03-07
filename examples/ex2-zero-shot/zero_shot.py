import os
import clip
import torch
from torchvision.datasets import CIFAR100
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load('ViT-B/32', device)
model, preprocess = clip.load('ViT-L/14', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
# image = Image.open("../testing_images/closed_door_4.png")
# image.show()
# image_input = preprocess(image).unsqueeze(0).to(device)
image_input = preprocess(image).unsqueeze(0).to(device)
# text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
#   Normalize image and text features
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
# Calculate the similarity between image features and each text feature
#   matrix multiplication
#   -> creates a similarity matrix, where each element represents 
#       the similarity between an image feature and a specific text label.
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
print(type(values), " ", type(indices))
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")