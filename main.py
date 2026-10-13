import numpy as np
import pandas as pd
import glob
import torch
from torch import optim
from torch.nn.modules import dropout
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
#for fancy loading bar terminal graphics
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from early_stopper import EarlyStopper
from feature_transformer import FeatureTransformer

#allow PyTorch to be run on GPU if available for faster training
print("pytorch version: " + torch.__version__)
print("cuda version: " + str(torch.version.cuda))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#hyperparameters for training the model
learning_rate = 0.001
dropout = 0.1
batch_size = 64
max_epochs = 50
embed_dim = 32
head_count = 4
encode_layer_count = 2

#handle dataset
features = ["Protocol Type", "Rate", "ack_count", "syn_count", "rst_count",
    "Tot sum", "Min", "Max", "Std", "AVG", "IAT"]

#read in only the first csv file from the data et
#df = pd.read_csv("MERGED_CSV/Merged01.csv", usecols=features + ["Label"])
#read in many csv files from the dataset
csv_files = glob.glob('CIC_IOT_Dataset2023/Merged0*.csv')
file_chunks = []
for file in csv_files:
    with pd.read_csv(file, chunksize = 100000) as reader:
        for chunk in reader:
            chunk = chunk[features + ["Label"]]
            file_chunks.append(chunk)
df = pd.concat(file_chunks, ignore_index=True)

#use if all features of dataset will be included (except labels)
#df = pd.read_csv("MERGED_CSV/Merged01.csv")
#features = df.columns.tolist()
#features.remove("Label")

#clean data of infinite values
df = df.replace([np.inf, -np.inf, np.nan], 0)

#normalize features
feature_means = df[features].mean().values
feature_stds = df[features].mean().values
feature_stds = np.where(np.isinf(feature_stds) | (feature_stds == 0), 1., feature_stds)
feature_stds[features == 0] = 1.
df[features] = (df[features] - feature_means) / feature_stds

#label mappings
#df["Label"] = df["Label"].astype("category")
#labels = torch.tensor(df["Label"].cat.codes.values, dtype=torch.long)
label_map = {
 "DDOS-ACK_FRAGMENTATION": 0,
    "DDOS-HTTP_FLOOD": 0,
    "DDOS-ICMP_FLOOD": 0,
    "DDOS-ICMP_FRAGMENTATION": 0,
    "DDOS-PSHACK_FLOOD": 0,
    "DDOS-RSTFINFLOOD": 0,
    "DDOS-SLOWLORIS": 0,
    "DDOS-SYN_FLOOD": 0,
    "DDOS-SYNONYMOUSIP_FLOOD": 0,
    "DDOS-TCP_FLOOD": 0,
    "DDOS-UDP_FLOOD": 0,
    "DDOS-UDP_FRAGMENTATION": 0,
    "DOS-HTTP_FLOOD": 1,
    "DOS-SYN_FLOOD": 1,
    "DOS-TCP_FLOOD": 1,
    "DOS-UDP_FLOOD": 1,
    "MIRAI-GREETH_FLOOD": 2,
    "MIRAI-GREIP_FLOOD": 2,
    "MIRAI-UDPPLAIN": 2,
    "RECON-HOSTDISCOVERY": 3,
    "RECON-OSSCAN": 3,
    "RECON-PINGSWEEP": 3,
    "RECON-PORTSCAN": 3,
    "VULNERABILITYSCAN": 3,
    "DNS_SPOOFING": 4,
    "MITM-ARPSPOOFING": 4,
    "DICTIONARYBRUTEFORCE": 5,
    "BACKDOOR_MALWARE": 6,
    "BROWSERHIJACKING": 6,
    "COMMANDINJECTION": 6,
    "SQLINJECTION": 6,
    "UPLOADING_ATTACK": 6,
    "XSS": 6,
    "BENIGN": 7
}

labels = torch.tensor(df["Label"].map(label_map).values, dtype=torch.long)

dataset = TensorDataset(
    torch.tensor(df[features].values, dtype=torch.float32),
    labels
)

train_dataset, test_dataset = random_split(
    dataset,
    [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))]
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

#initialize model.
# 7 super classes + 1 benign class. Use nunique() on labels to get all 34 classes.
classes = 8 #df["Label"].nunique()
model = FeatureTransformer(embed_dim, head_count, encode_layer_count, len(features), classes, dropout)
model.to(device)

early_stopper = EarlyStopper(patience=3, min_delta=0.01)
#loss function for optimizing classification that heavily
#penalizes confident, incorrect predictions.
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#Adam optimizer adjusts learning rate per weight in node, allowing for dynamic
#larger learning steps in some areas, and smaller learning steps in others.
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#tracks metric (val_loss) and forces learning rate to decrease if metric
#does not improve over patience number of epochs
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

def train(model, train_loader, criterion, optimizer, schedular = None, device = "cpu"):
    running_loss = 0
    for batch_index, (data, targets) in enumerate(tqdm(train_loader, colour='green')):
        # Move data and targets to the device (GPU/CPU)
        data = data.to(device)
        # Forward pass: compute the model output
        scores = model(data)
        targets = targets.to(device)
        loss = criterion(scores, targets)
        running_loss += loss.item() * data.size(0)
        # Backward pass: compute the gradients
        optimizer.zero_grad()
        loss.backward()
        # Optimization step: update the model parameters
        optimizer.step()
    if schedular is not None:
        schedular.step(running_loss)

    return running_loss

def test(model, test_loader, device = "cpu"):
    num_correct = 0
    num_samples = 0
    # Set the model to evaluation mode
    model.eval()
    running_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples
            running_loss += criterion(scores, y).item() * x.size(0)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Calculate accuracy
    accuracy = float(num_correct) / float(num_samples) * 100
    print(f"\nGot {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")

    #set model back into training mode
    model.train()

    return (running_loss, all_preds, all_labels)

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_report(predictions, labels):
    # generate report
    print(classification_report(
        labels,
        predictions,
        digits=4
    ))
    # create and output confusion matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10)
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

train_losses = []
val_losses = []
for epoch in range(max_epochs):
    print(f"Epoch [{epoch + 1}/{max_epochs}]")

    #train
    loss = train(model, train_loader, criterion, optimizer, scheduler, device)
    train_losses.append(loss / len(train_loader.dataset))

    #test/validate
    (loss, preds, labels) = test(model, test_loader, device)
    val_losses.append(loss / len(test_loader.dataset))

    if(early_stopper.early_stop(loss)):
        print(f"Stopping early at epoch [{epoch + 1}/{max_epochs}]")
        generate_report(preds, labels)
        break
plot_loss(train_losses, val_losses)

model.to('cpu')
torch.save(model.state_dict(), "model_weights/transformer_model.pth")