import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

# Function to load the data
def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets using ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    return dataloaders['train'], dataloaders['valid'], dataloaders['test'], image_datasets['train']

# Function to save the checkpoint
def save_checkpoint(model, train_data, optimizer, save_dir, epochs):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(checkpoint, save_dir)

# Function to initialize classifier dynamically
def initialize_classifier(model, hidden_units, output_features):
    if hasattr(model, 'classifier'):  # For architectures like VGG
        in_features = model.classifier[0].in_features
    else:  # For architectures like ResNet
        in_features = model.fc.in_features

    classifier = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, output_features),
        nn.LogSoftmax(dim=1)
    )
    return classifier

# Command-line arguments
parser = argparse.ArgumentParser(description="Train a neural network on a dataset")
parser.add_argument('data_dir', type=str, help='Directory of the dataset')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16 or resnet18)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in classifier')
parser.add_argument('--output_features', type=int, default=102, help='Number of output features (e.g., classes)')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')

args = parser.parse_args()

# Loading model based on architecture
if args.arch == 'vgg16':
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
elif args.arch == 'resnet18':
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
else:
    raise ValueError("Unsupported architecture. Please choose 'vgg16' or 'resnet18'.")

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Initialize and replace the classifier dynamically based on the architecture
if hasattr(model, 'classifier'):  # VGG-like models
    model.classifier = initialize_classifier(model, args.hidden_units, args.output_features)
else:  # ResNet-like models
    model.fc = initialize_classifier(model, args.hidden_units, args.output_features)

# Set the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters() if hasattr(model, 'classifier') else model.fc.parameters(), lr=args.learning_rate)

# Loading the data
trainloader, validloader, testloader, train_data = load_data(args.data_dir)

# Move model to device (either GPU or CPU)
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Model Training
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0  # Reset running loss at the start of each epoch
    
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        logps = model.forward(inputs)  # Forward pass
        loss = criterion(logps, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize

        running_loss += loss.item()

        # Validation phase (every `print_every` steps)
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()  # Set model to evaluation mode for validation
            
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            
            running_loss = 0
            model.train()  # Set model back to training mode after validation

# Save the checkpoint
save_checkpoint(model, train_data, optimizer, args.save_dir, epochs)
print("Model checkpoint saved.")
