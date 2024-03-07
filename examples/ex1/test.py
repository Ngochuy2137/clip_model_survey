import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Returns the model and the TorchVision transform needed by the model
# It will download the model as necessary
model, preprocess = clip.load("ViT-L/14",device=device,jit=False)
print("model type: ", type(model))
# Preprocess step:
#   Prepeocess image: resize, normalize, and sometimes crop images to fit the model's expected input
image_file = Image.open("../testing_images/closed_door_1.png")
image_file.show()
image = preprocess(image_file).unsqueeze(0).to(device)
#   Prepeocess text: Tokenization breaks text into smaller units (usually words or subwords) 
#   and then converts them into vocabulary IDs
token_list = ["a closed door", "an opened door", "a girl", "a house", "a car"]
text = clip.tokenize(token_list).to(device)

with torch.no_grad():
    # Find feature of image -> a feature vector
    image_features = model.encode_image(image)
    # Find feature of texts -> each text is convert to a feature vector
    text_features = model.encode_text(text)

    # returns two Tensors, containing the logit scores 
    # corresponding to each image and text input
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:")
prob_list = probs.tolist()[0]

print(type(prob_list))
for token, prob in zip(token_list, prob_list):
    print(token, " : ",  prob)