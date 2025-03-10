import medmnist
from medmnist import INFO
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
import mlxtend
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import pandas as pd

# Import necessary metrics from sklearn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

data_flag = 'bloodmnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# Define data transformations
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load datasets
train_data = DataClass(split='train', transform=data_transform, download=True)
val_data = DataClass(split='val', transform=data_transform, download=True)
test_data = DataClass(split='test', transform=data_transform, download=True)

# Display an example image and label
img = train_data[0][0]
label = train_data[0][1]
print(f"Image:\n{img}")
print(f"Label:\n{label}")
print(f"Image shape: {img.shape}")
print(f"Label: {label}")

# Number of channels and classes
n_channels = info['n_channels']
print(f"Number of channels: {n_channels}")
n_classes = len(info['label'])
print(f"Number of classes: {n_classes}")
class_names = info['label']
print(f"Class names: {class_names}")

# Plot a few sample images from the training set
for i in range(5):
    img = train_data[i][0]
    label = train_data[i][1]
    plt.figure(figsize=(3, 3))
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"Label: {label}")
    plt.axis(False)
    plt.show()

BATCH_SIZE = 64
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# Define accuracy function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / len(y_pred)) * 100 
    return acc

# Training step function
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    
    for batch, (X, y) in enumerate(data_loader):
        y = y.squeeze().long()
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    
    return train_loss, train_acc

# Test (or validation) step function
def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for X, y in data_loader:
            y = y.squeeze().long()
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        return test_loss, test_acc

# Evaluation function to get predictions, targets, and probabilities
def eval_func(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    
    eval_loss, eval_acc = 0, 0
    model.to(device)
    model.eval()
    y_preds = []
    y_targets = []
    y_probs = []  # store probabilities for ROC curves
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(data_loader)):
            y = y.squeeze().long()
            X, y = X.to(device), y.to(device)
            eval_pred = model(X)
            eval_loss += loss_fn(eval_pred, y)
            eval_acc += accuracy_fn(y_true=y, y_pred=eval_pred.argmax(dim=1))
            eval_labels = torch.argmax(torch.softmax(eval_pred, dim=1), dim=1)
            y_preds.append(eval_labels)
            y_targets.append(y)
            y_probs.append(torch.softmax(eval_pred, dim=1))
            
        eval_loss /= len(data_loader)
        eval_acc /= len(data_loader)
        y_preds = torch.cat(y_preds).cpu()
        y_targets = torch.cat(y_targets).cpu()
        y_probs = torch.cat(y_probs).cpu()
        
        return {"model_name": model.__class__.__name__, 
                "loss": eval_loss.item(),
                "accuracy": eval_acc,
                "predictions": y_preds,
                "targets": y_targets,
                "probs": y_probs}

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# Define CNN model
class cnn(torch.nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units, 
                      kernel_size=3),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=3),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units*4, 
                      kernel_size=3),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*4, 
                      out_channels=hidden_units*4, 
                      kernel_size=3),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*4, 
                      out_channels=hidden_units*4, 
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(hidden_units*4 * 4 * 4, hidden_units*8),
            nn.ReLU(),
            nn.Linear(hidden_units*8, hidden_units*8),
            nn.ReLU(),
            nn.Linear(hidden_units*8, n_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create the model
model = cnn(input_shape=n_channels, 
            hidden_units=16,
            output_shape=n_classes).to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print(model)

torch.manual_seed(42)
train_time_start_model = timer()

iteration_loss_list = []
iteration_accuracy_list = []

# Lists to store epoch-based metrics
train_losses = []
train_accs = []
val_losses = []
val_accs = []

epochs = 10
best_loss = float("inf")

for epoch in tqdm(range(epochs)):
    # Train step
    train_loss, train_acc = train_step(model=model,
                                       data_loader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       accuracy_fn=accuracy_fn,
                                       device=device)
    
    # Validation step
    val_loss, val_acc = test_step(data_loader=val_dataloader,
                                  model=model,
                                  loss_fn=loss_fn,
                                  accuracy_fn=accuracy_fn,
                                  device=device)
    
    # Store epoch-based metrics
    train_losses.append(train_loss.item())
    train_accs.append(train_acc)
    val_losses.append(val_loss.item())
    val_accs.append(val_acc)
    
    # For iteration-based visualization (optional)
    for iteration, (x, y) in enumerate(train_dataloader):
        iteration_loss_list.append(train_loss.item())
        iteration_accuracy_list.append(train_acc)
    
    print(f"Epoch: {epoch+1} | "
          f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f} | "
          f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}")

    # Save the best model based on validation loss
    if val_loss < best_loss:
        best_loss = val_loss
        print(f"Saving best model at epoch: {epoch+1}")
        torch.save(model.state_dict(), "./model.pth")
        
train_time_end_model = timer()
total_train_time_model = print_train_time(start=train_time_start_model,
                                          end=train_time_end_model,
                                          device=device)

# Load the best model
loaded_model = cnn(input_shape=n_channels,
                   hidden_units=16,
                   output_shape=n_classes).to(device)
loaded_model.load_state_dict(torch.load("./model.pth"))

# Evaluate on the test set
model_results = eval_func(data_loader=test_dataloader,
                          model=loaded_model,
                          loss_fn=loss_fn,
                          accuracy_fn=accuracy_fn,
                          device=device)
print(model_results)

y_targets = model_results['targets']
y_preds = model_results['predictions']
y_probs = model_results['probs']

# Plot confusion matrix
confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
confmat_tensor = confmat(preds=y_preds, target=y_targets)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)
plt.show()

########################################################################
# 1) Plot Loss and Accuracy vs. Epochs (side-by-side)
########################################################################
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 5))

# Left plot: Train Loss vs. Val Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, 'b-o', label='Train Loss')
plt.plot(epochs_range, val_losses, 'r-o', label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Right plot: Train Accuracy vs. Val Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, 'b-o', label='Train Accuracy')
plt.plot(epochs_range, val_accs, 'r-o', label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

########################################################################
# 2) (Optional) Plot iteration vs. loss and accuracy (existing approach)
########################################################################
plt.figure(figsize=(10, 5)) 
plt.semilogy(iteration_loss_list, label='Training Loss (per iteration)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Iteration vs. Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.semilogx(iteration_accuracy_list, label='Training Accuracy (per iteration)')  
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Iteration vs. Accuracy')
plt.legend() 
plt.show()

########################################################################
# 3) Additional: Compute metrics on the test set and plot ROC curves
########################################################################
y_true = y_targets.numpy()
y_pred = y_preds.numpy()
y_prob = y_probs.numpy()

overall_accuracy = accuracy_score(y_true, y_pred)
precision_per_class = precision_score(y_true, y_pred, average=None, labels=np.arange(n_classes))
recall_per_class = recall_score(y_true, y_pred, average=None, labels=np.arange(n_classes))
f1_per_class = f1_score(y_true, y_pred, average=None, labels=np.arange(n_classes))

metrics_table = pd.DataFrame({
    "Accuracy": [overall_accuracy] * n_classes,
    "Precision": precision_per_class,
    "Recall": recall_per_class,
    "F1-Score": f1_per_class
}, index=[f"Class {i}" for i in range(n_classes)])

print("Metrics table on the test set:")
print(metrics_table)

# Binarize the true labels for ROC computation
y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

# Plot ROC curves for each class and micro-average ROC curve on the same plot
plt.figure(figsize=(8, 6))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple']

# Plot ROC curve for each class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
             label=f'Class {i} (AUC = {roc_auc:.2f})')

# Compute and plot the micro-average ROC curve
fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, color='navy', lw=2, linestyle='--',
         label=f'Micro-average (AUC = {roc_auc_micro:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class with Micro-average')
plt.legend(loc="lower right")
plt.show()

