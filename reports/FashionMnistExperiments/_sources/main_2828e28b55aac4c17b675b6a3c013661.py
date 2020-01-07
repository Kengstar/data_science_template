import os
import torch
from torch import optim
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.utils import save_image

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver, MongoObserver
from model import simple_cnn


##sacred stuff
ex = Experiment("Fashion Modell MTL")
ex.observers.append(FileStorageObserver.create('reports\\FashionMnistExperiments'))
ex.observers.append(MongoObserver())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.CrossEntropyLoss()
#ex.observers.append()

@ex.config
def config():
    batch_size = 128
    epochs = 15
    lr = 1e-3
    train_split_ratio = 0.8
    path = ""   ##output path for saving images for example
    notes = "Experiment description"
    cfg_dict = {} ## if there are to many hyper_params, collect here and dependy inject the dict





@ex.capture
def train(model, optimizer, epoch, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx == 5:
            break

    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
    ex.log_scalar('train_mean_loss', train_loss / len(train_loader.dataset))

@ex.capture
def test(model, data_loader, val=True):
    model.eval()
    test_loss = 0
    correct = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            preds = torch.argmax(output, dim=1)
            correct += torch.sum(preds == target).item()
            break
    test_loss /= len(data_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    correct /= len(data_loader.dataset)
    print('====> Test set acc: {:.4f}'.format(correct))
    if val:
        ex.log_scalar('val_mean_loss', test_loss)
        ex.log_scalar('val_acc', correct)
    else:
        ex.log_scalar('test_mean_loss', test_loss)
        ex.log_scalar('test_acc', correct)
    return test_loss




@ex.automain
def run_experiment(batch_size, path, epochs, train_split_ratio):
    print("Run Experiment")

    model = simple_cnn.small_cnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=ex.current_run.config['lr'])
    print(ex.current_run.config)

    ex.info["model"] = repr(model)
    print(ex.current_run.config)
    print(ex.current_run.config['lr'])


    ## prepare datasets and loaders
    full_train_set = FashionMNIST('../data', train=True, download=True,
                                  transform=transforms.ToTensor())
    train_size = int(train_split_ratio * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_set, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    # training, validation, testing
    print("test")
    best_val_loss = 1000000.0
    for epoch in range(epochs):
        train(model, optimizer, epoch, train_loader=train_loader)
        cur_val_loss = test(model, data_loader=val_loader)
        if cur_val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(ex.observers[0].dir, "weights.pt"))

    test(model, data_loader=test_loader, val=False)





