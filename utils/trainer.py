import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from utils.dkd import DKD

from tqdm import tqdm


class Cifar10_ResNetTrainer:
    def __init__(self, model,
                optimizer, 
                criterion,
                scheduler,
                checkpoint_root_path = "./Checkpoints/",
                model_name = "ResNet8",
                num_epochs = 200,
                split_size = 0.9,
                root_dir = "./Data", 
                batch_size = 128, 
                device = "cpu"):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint = checkpoint_root_path
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.split_size = split_size
        self.epoch = num_epochs
        self.scheduler = scheduler
        self.model_name = model_name

        self.load_data()
        self.model.to(self.device)

    def load_data(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        full_train_dataset = torchvision.datasets.CIFAR10(root=self.root_dir, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=self.root_dir, train=False, download=True, transform=transform_test)

        train_size = int(self.split_size * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def train(self):
        train_losses = []
        val_losses = []
        train_accuracies = []  
        val_accuracies = []
        for epoch in range(self.epoch):
            train_loss, train_accuracy = self.train_step()
            val_loss, val_accuracy = self.eval_step(self.val_loader)
            
            if len(val_losses) > 0 and val_loss < min(val_losses):
                self.save(f"{self.checkpoint}{self.model_name}_{val_accuracy}.pth")

            val_losses.append(val_loss)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            self.scheduler.step(train_loss)

            print(f"Epoch: {epoch+1}/{self.epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return train_losses, val_losses, train_accuracies, val_accuracies


    def predict(self, path):
        self.load(path)
        test_loss, test_accuracy = self.eval_step(self.test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


    def train_step(self):
        self.model.train()  
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for images, labels in tqdm(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)       
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs.detach(), dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += images.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def eval_step(self, dataloader):
        self.model.eval()  
        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)  
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs.detach(), dim=1)
                total_correct += torch.sum(preds == labels).item()
                total_samples += images.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.model.eval()





class DKDTrainer:
    def __init__(self, 
                student, 
                teacher, 
                optimizer, 
                scheduler,
                root_dir = "./Data",
                split_size = 0.9,
                batch_size = 128,
                checkpoint_root_path = "./Checkpoints/",
                model_name = "ResNet8_DKD_ResNet18",
                num_epochs = 50,
                device = "cpu", 
                alpha=0.5, 
                beta=0.5, 
                temperature=1.0):
        
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.epoch = num_epochs
        self.checkpoint = checkpoint_root_path
        self.model_name = model_name
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.split_size = split_size
        
        self.dkd = DKD(self.student, self.teacher, self.alpha, self.beta, self.temperature)
        self.dkd.to(self.device)

        self.student.to(self.device)
        self.teacher.to(self.device)

        self.load_data()

    def load_data(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        full_train_dataset = torchvision.datasets.CIFAR10(root=self.root_dir, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=self.root_dir, train=False, download=True, transform=transform_test)

        train_size = int(self.split_size * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)


    def predict(self, path):
        self.load(path)
        test_loss, test_accuracy = self.eval_step(self.test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    def train(self):
        train_losses = []
        val_losses = []
        train_accuracies = []  
        val_accuracies = []
        for epoch in range(self.epoch):
            train_loss, train_accuracy = self.train_step()
            val_loss, val_accuracy = self.eval_step(self.val_loader)
            
            if len(val_losses) > 0 and val_loss < min(val_losses):
                self.save(f"{self.checkpoint}{self.model_name}_{val_accuracy}.pth")

            val_losses.append(val_loss)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            self.scheduler.step(train_loss)

            print(f"Epoch: {epoch+1}/{self.epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return train_losses, val_losses, train_accuracies, val_accuracies


    def train_step(self):
        self.student.train()
        self.teacher.eval()
        self.dkd.train()

        total_loss, total_correct, total_samples = 0.0, 0, 0

        for images, labels in tqdm(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            # outputs = self.model(images)       
            # loss = self.criterion(outputs, labels)
            loss, outputs = self.dkd(images, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += images.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def eval_step(self, dataloader):
        self.student.eval()
        self.teacher.eval()
        self.dkd.eval()

        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # outputs = self.student(images)  
                # loss = self.criterion(outputs, labels)
                loss, outputs = self.dkd(images, labels)
                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs.detach(), dim=1)
                total_correct += torch.sum(preds == labels).item()
                total_samples += images.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def save(self, path):
        torch.save(self.student.state_dict(), path)

    def load(self, path):
        self.student.load_state_dict(torch.load(path))
        self.student.to(self.device)
        self.student.eval()
    


    