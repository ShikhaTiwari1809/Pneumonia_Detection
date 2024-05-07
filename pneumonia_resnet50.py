'''
Shikha Tiwari
CS 5330 Spring 2024
pneumonia_resnet50.py - This code will import pretrained RESNET50 model and train it on pneumonia dataset.
'''

# import statements
import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torchsummary import summary
from torchvision.models import resnet50, ResNet50_Weights
import time
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Function to visulaize image
def imshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()

# Main function
def main(argv):

    # Setting random seed and disable CUDA
    random_seed = 20
    # Check if CUDA is available and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device) 
    # Find out if a GPU is available
    use_cuda = torch.cuda.is_available()

    # Load model from the saved location
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features 
    print('Number of features from pre-trained model', num_features)
    # freezes the parameters for the whole network
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)    
    
    # print the model
    print("**************** Structure and names of each layer **************** \n")
    #print(model)

     # DataLoader for the Train and test data set
    # Set up path for data after downloading
    train_dir = "/Users/shikhatiwari/Desktop/CS 5330 CVPR/Projects/FinalProject/chest_xray/train"
    test_dir = "/Users/shikhatiwari/Desktop/CS 5330 CVPR/Projects/FinalProject/chest_xray/test"
    # Create transform function
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),   
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), # data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),   
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(train_dir, transforms_train)
    test_dataset = datasets.ImageFolder(test_dir, transforms_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)

    print('Train dataset size:', len(train_dataset))
    print('Test dataset size:', len(test_dataset))
    class_names = train_dataset.classes
    print('Class names:', class_names)

    # Looking at first six example images
    examples = enumerate(train_dataloader)
    batch_idx, (example_data, example_targets) = next(examples)

    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams.update({'font.size': 20})

    # load a batch of train image
    iterator = iter(train_dataloader)
    # visualize a batch of train image
    inputs, classes = next(iterator)
    out = torchvision.utils.make_grid(inputs[:4])
    imshow(out, title=[class_names[x] for x in classes[:4]])

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # Set the random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    #### Train model
    train_loss=[]
    train_accuary=[]
    test_loss=[]
    test_accuary=[]

    num_epochs = 30   #(set no of epochs)
    start_time = time.time() #(for showing time)
    # Start loop
    for epoch in range(num_epochs): #(loop for every epoch)
        print("Epoch {} running".format(epoch)) #(printing message)
        """ Training Phase """
        model.train()    #(training model)
        running_loss = 0.   #(set loss 0)
        running_corrects = 0 
        # load a batch data of images
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device) 
            # forward inputs and get output
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # get loss value and update the network weights
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset) * 100.
        # Append result
        train_loss.append(epoch_loss)
        train_accuary.append(epoch_acc)
        # Print progress
        print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time() -start_time))
        """ Testing Phase """
        model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()
            epoch_loss = running_loss / len(test_dataset)
            epoch_acc = running_corrects / len(test_dataset) * 100.
            # Append result
            test_loss.append(epoch_loss)
            test_accuary.append(epoch_acc)
            # Print progress
            print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time()- start_time))
    
    save_path = 'custom-classifier_resnet_50.pth'
    torch.save(model.state_dict(), save_path)

    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(np.arange(1,num_epochs+1), train_accuary,'-o')
    plt.plot(np.arange(1,num_epochs+1), test_accuary,'-o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Test'])
    plt.title('Train vs Test Accuracy over time')
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv) 