import argparse
import torch
from torchvision import models
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Function to load a checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if 'vgg16' in filepath:
        model = models.vgg16(pretrained=True)
        model.classifier = checkpoint['classifier']
    else:
        model = models.resnet18(pretrained=True)
        model.fc = checkpoint['classifier']
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Function to process an image
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image_path)
    
    # Resize
    width, height = im.size
    if width > height:
        im.thumbnail((256, 256 * width // height))
    else:
        im.thumbnail((256 * height // width, 256))
        
    # Crop
    left = (im.width - 224) / 2
    top = (im.height - 224) / 2
    right = left + 224
    bottom = top + 224
    im = im.crop((left, top, right, bottom))
    
    # Normalize
    np_image = np.array(im) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

# Function to display an image along with its top 5 predictions
def display_predictions(image_path, probs, class_names):
    ''' Display an image along with the top 5 predicted class names and probabilities '''
    
    # Plot the image
    im = Image.open(image_path)
    plt.figure(figsize=(6,10))
    ax = plt.subplot(2, 1, 1)
    
    plt.imshow(im)
    plt.axis('off')
    
    # Plot the top 5 classes with probabilities
    ax = plt.subplot(2, 1, 2)
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, probs)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()  # Invert y-axis to have the highest probability at the top
    
    plt.show()

# Command-line arguments
parser = argparse.ArgumentParser(description="Predict image class using a trained deep learning model")
parser.add_argument('image_path', type=str, help='Path to the image file')
parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available for prediction')

args = parser.parse_args()

# Load the checkpoint
model = load_checkpoint(args.checkpoint)

# Move model to device (either GPU or CPU)
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Load category names if provided
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    cat_to_name = None

# Define the prediction function
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    model.eval()
    image_tensor = torch.from_numpy(process_image(image_path)).unsqueeze_(0).float()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logps = model.forward(image_tensor)
    ps = torch.exp(logps)
    
    top_p, top_class = ps.topk(topk, dim=1)
    top_p = top_p.cpu().numpy().flatten()
    top_class = top_class.cpu().numpy().flatten()
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_class]
    
    return top_p, top_classes

# Run the prediction function
probs, classes = predict(args.image_path, model, topk=args.top_k)

# Convert class indices to names if category names are provided
if cat_to_name:
    class_names = [cat_to_name[str(cls)] for cls in classes]
else:
    class_names = classes

# Print the results
print(f"\nTop {args.top_k} Predicted Classes and Probabilities:\n")
for i in range(args.top_k):
    print(f"Class: {class_names[i]} | Probability: {probs[i]:.4f}")

# Display the image with predictions
display_predictions(args.image_path, probs, class_names)
