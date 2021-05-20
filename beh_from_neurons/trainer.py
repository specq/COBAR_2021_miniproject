import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time


def train(model, trainloader, valloader, epochs=25, lr=1e-3):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    lr_lambda = lambda e: 0.99**e
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    steps = 0
    running_loss = 0
    print_every = 20
    for e in range(epochs):
        start = time.time()
        for x, y in iter(trainloader):
            steps += 1
            output = model.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            model.zero_grad()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                stop = time.time()
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    acc = compute_accuracy(model, valloader)
                    print("Epoch: {}/{}..".format(e+1, epochs),
                          "lr: {:.4f}..".format(optimizer.state_dict()['param_groups'][0]['lr']),
                          "Loss train: {:.4f}..".format(running_loss/print_every),
                          "Accuracy val: {:.4f} %..".format(acc),
                          "{:.4f} s/batch".format((stop - start)/print_every)
                         )
                model.train()
                running_loss = 0
                start = time.time()
        scheduler.step()
        
def compute_accuracy(model, loader):
    acc = 0
    for i, (x,y) in enumerate(loader):
        output = model.forward(x)
        logits = F.softmax(output, dim=1)
        y_pred = torch.argmax(logits, dim=1)
        acc += (y_pred == y).float().mean()
    return acc/(i+1)*100