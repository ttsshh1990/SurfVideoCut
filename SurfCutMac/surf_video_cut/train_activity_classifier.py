#!/usr/bin/env python3
"""
Train an activity classifier to distinguish between active surfing vs sitting/waiting
on person+surfboard detection crops.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


class SurfingActivityDataset(Dataset):
    """Dataset for surfing activity classification"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            # Return a black image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_training_data(data_dir: Path) -> Tuple[List[str], List[int]]:
    """Load image paths and labels from training data directory"""
    
    positive_dir = data_dir / 'positive'
    negative_dir = data_dir / 'negative'
    
    image_paths = []
    labels = []
    
    # Load positive examples (active surfing = 1)
    if positive_dir.exists():
        for video_dir in positive_dir.iterdir():
            if video_dir.is_dir():
                for img_path in video_dir.glob('*.jpg'):
                    image_paths.append(str(img_path))
                    labels.append(1)
    
    # Load negative examples (sitting/waiting = 0)
    if negative_dir.exists():
        for video_dir in negative_dir.iterdir():
            if video_dir.is_dir():
                for img_path in video_dir.glob('*.jpg'):
                    image_paths.append(str(img_path))
                    labels.append(0)
    
    print(f"Loaded {len(image_paths)} images:")
    print(f"  Positive (active surfing): {sum(labels)}")
    print(f"  Negative (sitting/waiting): {len(labels) - sum(labels)}")
    
    return image_paths, labels


def create_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create activity classification model"""
    
    # Use EfficientNet-B0 as base model
    model = models.efficientnet_b0(pretrained=pretrained)
    
    # Replace classifier head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    
    return model


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                device: torch.device, num_epochs: int = 10, learning_rate: float = 1e-3) -> Dict:
    """Train the activity classification model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    model = model.to(device)
    
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss {loss.item():.4f}')
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Save training history
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f'New best validation accuracy: {val_acc:.2f}%')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return training_history


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict:
    """Evaluate model on test set"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = 100 * sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['Sitting/Waiting', 'Active Surfing']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels
    }


def main():
    parser = argparse.ArgumentParser(description='Train activity classifier for surfing detection')
    parser.add_argument('--data-dir', default='training_data',
                       help='Directory containing positive and negative training examples')
    parser.add_argument('--output-dir', default='models',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Fraction of data for testing')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Fraction of training data for validation')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device selection
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {data_dir}")
    
    image_paths, labels = load_training_data(data_dir)
    
    if len(image_paths) == 0:
        raise ValueError("No training data found!")
    
    # Check class balance
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    print(f"Class balance: {positive_count/(positive_count+negative_count)*100:.1f}% positive")
    
    if positive_count < 10 or negative_count < 10:
        print("Warning: Very few examples for one class. Consider collecting more data.")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=args.test_split, random_state=args.seed, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=args.val_split/(1-args.test_split), 
        random_state=args.seed, stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples") 
    print(f"  Test: {len(X_test)} samples")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SurfingActivityDataset(X_train, y_train, train_transform)
    val_dataset = SurfingActivityDataset(X_val, y_val, val_transform)
    test_dataset = SurfingActivityDataset(X_test, y_test, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create model
    model = create_model(num_classes=2, pretrained=True)
    print(f"Created EfficientNet-B0 based model")
    
    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    training_history = train_model(
        model, train_loader, val_loader, device, 
        num_epochs=args.epochs, learning_rate=args.learning_rate
    )
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_results = evaluate_model(model, test_loader, device)
    
    # Save model
    model_path = output_dir / 'activity_classifier.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'efficientnet_b0',
        'num_classes': 2,
        'training_history': training_history,
        'test_results': test_results,
        'class_names': ['sitting_waiting', 'active_surfing']
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Save training metadata
    metadata = {
        'training_args': vars(args),
        'dataset_info': {
            'total_samples': len(image_paths),
            'positive_samples': positive_count,
            'negative_samples': negative_count,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        },
        'training_history': training_history,
        'test_results': {
            'accuracy': test_results['accuracy']
        }
    }
    
    metadata_path = output_dir / 'training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training metadata saved to: {metadata_path}")
    
    print(f"\nTraining completed!")
    print(f"Final test accuracy: {test_results['accuracy']:.2f}%")


if __name__ == '__main__':
    main()