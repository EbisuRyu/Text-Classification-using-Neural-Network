from model import *
from dataset import *
from tqdm import tqdm
import torch.nn as nn
import os

class Saver():
    
    def __init__(self, model_path, save_epoch=2):
        self.model_path = model_path
        self.save_epoch = save_epoch
        self.bestModel = None
        self.bestLoss = 0
        
    def save(self, epoch, model, loss):
        if epoch % self.save_epoch == 0:
            self.model_filename = model.__class__.__name__ 
            save_path = os.path.join(self.model_path ,self.model_filename + f"_{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with loss: {loss}")
        if self.bestModel is None:
            self.bestModel = model.state_dict()
            self.bestLoss = loss
        elif loss < self.bestLoss:
            self.bestModel = model.state_dict()
            self.bestLoss = loss
            print(f"Best model updated with loss: {loss}")
    
    def save_best(self):
        save_path = os.path.join(self.model_path ,self.model_filename+ f"_best.pt")
        torch.save(self.bestModel, save_path)
        print(f"Best model saved with loss: {self.bestLoss}")


class Trainer():
    
    def __init__(self, save_path, save_epoch=2, log_interval=2, device='cpu'):
        self.device = device
        self.log_interval = log_interval
        self.saver = Saver(save_path, save_epoch)

    def compute_loss(self, output, label):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(output, label)
    
    def prepare_model(self, vocab_size, embed_dim, hidden_dim, num_class):
        model = TextClassificationModel(vocab_size, embed_dim, hidden_dim, num_class)
        model.weight_initilization()
        model.to(self.device)
        return model
    
    def configure_optimizers(self, model, lr=0.001):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def load_model(self, model, model_path):
        model.load_state_dict(torch.load(model_path))
        return model
    
    def fit_epoch(self, epoch, model, dataloader):
        model.train()
        total_acc, total_count = 0, 0
        total_loss = 0
        iteratorer = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}", leave=False)
        for (text, label, offsets) in iteratorer:
            self.optimizer.zero_grad()
            output = model(text, offsets)
            loss = self.compute_loss(output, label)
            iteratorer.set_postfix({'loss': f"{loss.item():6.3f}"})
            loss.backward()
            self.optimizer.step()
            total_acc += (output.argmax(1) == label).sum().item()
            total_count += label.size(0)
            total_loss += loss.item()
                
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = total_acc / total_count
        return epoch_loss, epoch_acc
    
    def train(self, epochs, model, train_dataloader):
        for epoch in range(epochs):
            train_loss, train_acc = self.fit_epoch(epoch, model, train_dataloader)
            self.saver.save(epoch, model, train_loss)
            if epoch % self.log_interval == 0:
                print("-" * 80)
                print("| End of epoch: {:3d} | Train Accuracy: {:8.3f} | Train Loss: {:8.3f}".format(epoch, train_acc, train_loss))
        self.saver.save_best()
        return model
    
    def evaluate(self, model, test_dataloader):
        with torch.no_grad():
            total_acc, total_count = 0, 0
            total_loss = 0
            iteratorer = tqdm(test_dataloader, total=len(test_dataloader), leave=False)
            for (text, label, offsets) in iteratorer:
                output = model(text, offsets)
                loss = self.compute_loss(output, label)
                iteratorer.set_postfix({'loss': f"{loss.item():6.3f}"})
                total_acc += (output.argmax(1) == label).sum().item()
                total_count += label.size(0)
                total_loss += loss.item()
            test_loss, test_acc = total_loss / len(test_dataloader), total_acc / total_count
            print("Test Accuracy: {:8.3f} | Test Loss: {:8.3f}".format(test_loss, test_acc))