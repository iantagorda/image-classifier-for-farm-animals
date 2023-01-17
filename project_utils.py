import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from PIL import Image
from IPython import display

from sklearn.metrics import f1_score, accuracy_score

# show_sample_images gets 6 random samples from the dataset
# and displays them in a grid format
def show_sample_images(dataset):
    class_dict = dataset.classes
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)
    batch = next(iter(dataloader))
    images, labels = batch
    
    grid = torchvision.utils.make_grid(images[:6], nrow = 3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    print("labels: ", [class_dict[l] for l in labels[:6]])

# predict_image uses the model to predict the label of the input image
def predict_image(model, image_path, test_transforms, class_dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = test_transforms(Image.open(image_path))
    image = image.view(1, 3, 96, 96)
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(image.to(device))
        image = image.cpu()[0]
        plt.figure(figsize=(5,5))
        plt.imshow(image.permute(1, 2, 0))
        predictions = predictions.argmax(dim=1)
        print("predicted: ", class_dict[predictions.item()])
    
    # send model back to cpu to free gpu space
    model.to("cpu")
    
    # Uncomment code below if you want to view image activation
    # This is applicable for some selected models only
    # show_activation(model, image)
        

# evaluate_models calculates the accuracy and f1 score of the passed models
# on the given dataset. It print the results and plots a graph of the accuracies for
# easy comparison
def evaluate_models(models, model_checkpoints, model_names, dataset):
    class_dict = dataset.classes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accuracy_list = []
    dataloader = DataLoader(dataset, batch_size=10)
    
    for i in range(len(models)):
        model = models[i]
        model.to(device)
        if model_checkpoints != None:
            model.load_state_dict(model_checkpoints[i])
        model.eval()
        
        predictions = []
        targets = []
        with torch.no_grad():   
            for images, labels in dataloader:
                images = images.to(device)
                predictions += model(images).cpu().argmax(dim=1)
                targets += labels

        f1 = f1_score(targets, predictions, average=None)
        f1 = [round(score, 3) for score in f1]
        f1_with_class = dict(zip([class_dict[ind] for ind in range(len(f1))], f1))
        accuracy = accuracy_score(targets, predictions)
        accuracy_list.append(accuracy)
        
        print(f"{model_names[i]} has f1 scores of {f1_with_class}")
        print(f"{model_names[i]} has an accuracy score of {round(accuracy*100, 2)}%")
        print()
        
        # send model back to cpu to free gpu space
        model.to("cpu")

    plt.figure(figsize=(10, 10))
    plt.bar(model_names, accuracy_list)
    plt.xlabel("Model")
    plt.xticks(rotation="vertical")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.show()

        
# plot_train_stats plots the accuracy and loss statics from the
# ModelTrainer.train_model output
def plot_train_stats(train_stats, fig_name=""):
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = train_stats
    
    fig, (loss_ax, acc_ax) = plt.subplots(1,2,figsize=(15,5))
    loss_ax.plot(train_loss_list, label="Train loss")
    loss_ax.plot(val_loss_list, label="Val loss")
    loss_ax.legend()
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")

    acc_ax.plot(train_acc_list, label="Train acc")
    acc_ax.plot(val_acc_list, label="Val acc")
    acc_ax.legend()
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    plt.show()
    
    if fig_name != "":
        fig.savefig(fig_name)
        
        
# load_plot displays a saved plot. It first converts the passed in
# hyperparameters and model name to the format the files were saved
def load_plot(name, lr, bs):
    filepath = f"saved_plots/{name}_lr{lr}_bs{bs}".replace(".", "-")
    display.display(display.Image(filename=f"{filepath}.png"))

# get_checkpoint loads a saved model state_dict. It first converts the passed in
# hyperparameters and model name to the format the files were saved
def get_checkpoint(name, lr, bs):
    filepath = f"saved_checkpoints/{name}_lr{lr}_bs{bs}".replace(".", "-")
    return torch.load(f"{filepath}.pt")

# show_activation plots the output of the first layer of a model
def show_activation(origModel, image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = list(list(origModel.children())[0].children())[0]
    model.eval()
    model.to(device)
    with torch.no_grad():
        image = image.to(device)
        predictions = model(image.view(1, 3, 96, 96))
        image = image.cpu()
        activation = predictions.squeeze().cpu().reshape(-1, 1, 96, 96)
        
        fig = plt.figure(figsize=(18, 7))
        fig.add_subplot(3, 5, 1)
        plt.imshow(np.transpose(image, (1,2,0)))
        plt.axis("off")
        plt.title("Normalized Image")
        
        for i in range(2, 16):      
            fig.add_subplot(3, 5, i)
            plt.imshow(np.transpose(activation[i], (1,2,0)))
            plt.axis("off")
            plt.title(f"Channel {i}")
    # send model back to cpu to free gpu space
    model.to("cpu")
    
    
# ModelTrainer abstracts away the training process
class ModelTrainer:
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train_model(self, model, lr=0.0003, ne=100, bs=32):
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_loader = DataLoader(self.train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=bs)
        
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        start_time = time.time()
        for epoch in range(ne):
            train_loss = 0
            val_loss = 0
            
            ####################### TRAINING #################################
            model.train()
            predicted_labels = []
            target_labels = []
            for images, labels in train_loader:
                target_labels += labels
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                predictions = model(images)
                loss = criterion(predictions, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                predicted_labels += predictions.cpu().argmax(dim=1)
            
            train_loss /= len(train_loader)
            train_acc = accuracy_score(target_labels, predicted_labels)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            #########################################################################
            
            
            #################### VALIDATION #########################################
            model.eval()
            predicted_labels = []
            target_labels = []
            with torch.no_grad():   
                for images, labels in val_loader:
                    target_labels += labels
                    
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    predictions = model(images)
                    loss = criterion(predictions, labels)
                    val_loss += loss.item()
                    
                    predicted_labels += predictions.cpu().argmax(dim=1)
                    
            val_loss /= len(val_loader)
            val_acc = accuracy_score(target_labels, predicted_labels)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            #########################################################################
            
            
            print(f"Epoch {epoch+1} \t Train Loss: {train_loss:.3f}\t Train Acc: {train_acc:.3f}\t " +
                  f"Val Loss: {val_loss:.3f}\t Val Acc: {val_acc:.3f}")
        
        print(f"Finished Training! Total time taken: {(time.time() - start_time):.2f}")
        # send model back to cpu to free gpu space
        model.to("cpu")

        return model, (train_loss_list, train_acc_list, val_loss_list, val_acc_list)
        

        



        
