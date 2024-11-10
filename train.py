import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleNet
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import multiprocessing
import argparse

class Trainer:
    def __init__(self, model, model_path='model.pth', early_stop_tolerance=5, batch_size=16, lr=0.001, max_epochs=50, visualize=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_path = model_path
        self.early_stop_tolerance = early_stop_tolerance
        self.batch_size = batch_size
        self.learning_rate = lr
        self.max_epochs = max_epochs
        self.visualize = visualize
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.train_loader = self._load_data()

    def _load_data(self):
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        return DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def _save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            print(f"Loaded model from '{self.model_path}'. Continuing training.")
        else:
            print(f"No model '{self.model_path}' found. Creating and training a new model.")

    def train(self):
        self.load_model()
        display_process = None

        if self.visualize:
            display_process = multiprocessing.Process(target=self.display_samples)
            display_process.start()

        for epoch in range(self.max_epochs):
            total_loss = 0
            self.model.train()

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {avg_loss:.4f}")

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.epochs_no_improve = 0
                self._save_model()
                print(f"New best model saved with loss {avg_loss:.4f}")
            else:
                self.epochs_no_improve += 1
                print(f"No improvement. Early stop: {self.epochs_no_improve}/{self.early_stop_tolerance}")

                if self.epochs_no_improve >= self.early_stop_tolerance:
                    print("Early stop triggered.")
                    break

        if display_process:
            display_process.terminate()

    def display_samples(self):
        last_modified_time = 0
        plt.ion()
        fig, axes = plt.subplots(2, 6, figsize=(12, 6))
        fig.canvas.manager.set_window_title("Live Sample Predictions")
        correct_patch = mpatches.Patch(color='green', label='Correct Prediction')
        incorrect_patch = mpatches.Patch(color='red', label='Incorrect Prediction')
        fig.legend(handles=[correct_patch, incorrect_patch], loc='upper right')

        while True:
            if os.path.exists(self.model_path):
                modified_time = os.path.getmtime(self.model_path)
                if modified_time != last_modified_time:
                    self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
                    self.model.eval()
                    last_modified_time = modified_time

            images, labels = next(iter(self.train_loader))
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                certainties, predictions = torch.max(probabilities, 1)

            images = images.cpu().numpy()
            labels = labels.cpu().numpy()
            predictions = predictions.cpu().numpy()
            certainties = certainties.cpu().numpy() * 100

            for i, ax in enumerate(axes.flatten()):
                if i < len(images):
                    ax.imshow(images[i].squeeze(), cmap='gray')
                    color = 'green' if predictions[i] == labels[i] else 'red'
                    ax.set_title(f'Pred: {predictions[i]} ({certainties[i]:.1f}%)\nActual: {labels[i]}', color=color)
                    ax.axis('off')
                else:
                    ax.axis('off')

            plt.draw()
            plt.pause(1)

            if not plt.fignum_exists(fig.number):
                break

        plt.ioff()
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a SimpleNet model on MNIST with optional live visualization.")
    parser.add_argument('--visualize', action='store_true', help="Enable live visualization of sample predictions.")
    args = parser.parse_args()

    model = SimpleNet()
    trainer = Trainer(model, visualize=args.visualize)
    trainer.train()