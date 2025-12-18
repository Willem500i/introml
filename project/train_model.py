"""
Train CNN and traditional ML models for vehicle-curb proximity classification.

Models:
1. CNN (transfer learning with ResNet18)
2. Logistic Regression on extracted features
3. SVM on extracted features

Evaluation:
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Optional: GradCAM visualization
"""

import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def load_data(data_dir: str, split: str):
    """Load preprocessed data for a split."""
    X = np.load(os.path.join(data_dir, f'{split}_images.npy'))
    y = np.load(os.path.join(data_dir, f'{split}_labels.npy'))
    features = np.load(os.path.join(data_dir, f'{split}_feature_vectors.npy'))
    return X, y, features


def create_cnn_model(num_classes: int = 2, pretrained: bool = True):
    """Create a CNN model using transfer learning."""
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    return model


def train_cnn(model, train_loader, val_loader, device, 
              epochs: int = 10, lr: float = 0.001):
    """Train the CNN model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.4f}, Val Loss={val_loss/len(val_loader):.4f}, "
              f"Val Acc={val_acc:.4f}")
        
        scheduler.step()
    
    return history


def evaluate_model(model, test_loader, device):
    """Evaluate CNN model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def train_traditional_models(X_train, y_train, X_val, y_val):
    """Train logistic regression and SVM models."""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models_results = {}
    
    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_val_scaled)
    models_results['Logistic Regression'] = {
        'model': lr_model,
        'scaler': scaler,
        'preds': lr_preds,
        'accuracy': accuracy_score(y_val, lr_preds)
    }
    print(f"Logistic Regression Accuracy: {models_results['Logistic Regression']['accuracy']:.4f}")
    
    # SVM
    print("\nTraining SVM...")
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_preds = svm_model.predict(X_val_scaled)
    models_results['SVM'] = {
        'model': svm_model,
        'scaler': scaler,
        'preds': svm_preds,
        'accuracy': accuracy_score(y_val, svm_preds)
    }
    print(f"SVM Accuracy: {models_results['SVM']['accuracy']:.4f}")
    
    return models_results


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['In Lane', 'Near Curb'],
                yticklabels=['In Lane', 'Near Curb'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_history(history: dict, save_path: str):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    """Main training pipeline."""
    data_dir = 'data/processed'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if data exists
    if not os.path.exists(os.path.join(data_dir, 'train_images.npy')):
        print("Preprocessed data not found!")
        print("Run: python extract_features.py")
        print("Then: python preprocess_data.py")
        return
    
    # Load data
    print("Loading data...")
    X_train, y_train, feat_train = load_data(data_dir, 'train')
    X_val, y_val, feat_val = load_data(data_dir, 'val')
    X_test, y_test, feat_test = load_data(data_dir, 'test')
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # ============================
    # Traditional ML Models
    # ============================
    print("\n" + "="*50)
    print("Training Traditional ML Models")
    print("="*50)
    
    trad_results = train_traditional_models(feat_train, y_train, feat_val, y_val)
    
    # Evaluate on test set
    for name, result in trad_results.items():
        X_test_scaled = result['scaler'].transform(feat_test)
        test_preds = result['model'].predict(X_test_scaled)
        
        print(f"\n{name} Test Results:")
        print(classification_report(y_test, test_preds, 
                                    target_names=['In Lane', 'Near Curb']))
        
        plot_confusion_matrix(
            y_test, test_preds, 
            f'{name} - Confusion Matrix',
            os.path.join(output_dir, f'{name.lower().replace(" ", "_")}_cm.png')
        )
    
    # ============================
    # CNN Model
    # ============================
    print("\n" + "="*50)
    print("Training CNN Model")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare PyTorch datasets
    # Convert from (N, H, W, C) to (N, C, H, W)
    X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_val_tensor = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create and train model
    model = create_cnn_model(num_classes=2)
    history = train_cnn(model, train_loader, val_loader, device, epochs=10)
    
    # Plot training history
    plot_training_history(history, os.path.join(output_dir, 'cnn_training_history.png'))
    
    # Evaluate on test set
    print("\nCNN Test Results:")
    cnn_preds, cnn_labels = evaluate_model(model, test_loader, device)
    print(classification_report(cnn_labels, cnn_preds, 
                                target_names=['In Lane', 'Near Curb']))
    
    plot_confusion_matrix(
        cnn_labels, cnn_preds,
        'CNN - Confusion Matrix',
        os.path.join(output_dir, 'cnn_cm.png')
    )
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'cnn_model.pth'))
    print(f"\nModel saved to {output_dir}/cnn_model.pth")
    
    # ============================
    # Summary
    # ============================
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    
    results_summary = {
        'Logistic Regression': accuracy_score(y_test, trad_results['Logistic Regression']['scaler'].transform(feat_test).dot(trad_results['Logistic Regression']['model'].coef_.T).flatten() > 0),
        'SVM': accuracy_score(y_test, trad_results['SVM']['model'].predict(trad_results['SVM']['scaler'].transform(feat_test))),
        'CNN': accuracy_score(cnn_labels, cnn_preds)
    }
    
    for name, acc in results_summary.items():
        print(f"{name}: {acc:.4f}")
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
