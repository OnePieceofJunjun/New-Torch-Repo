# Task VIII - X & Bonus 1
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_batch_predictions(model, batch, device, title):
    # Task VIII: Plot input images and network predictions
    model.eval()
    with torch.no_grad():
        # Move batch to device and get predictions
        batch = {k: v.to(device) for k, v in batch.items()}
        output_dict = model(batch)
        predictions = (output_dict['output'] > 0.5).float()
        
        # Set up the plot
        batch_size = len(batch['img1'])
        n_rows = int(np.sqrt(batch_size))
        n_cols = int(np.ceil(batch_size / n_rows))
        
        plt.figure(figsize=(15, 10))
        plt.suptitle(title)
        
        for i in range(batch_size):
            plt.subplot(n_rows, n_cols, i + 1)
            # Show both images side by side
            combined_img = torch.cat([batch['img1'][i][0], batch['img2'][i][0]], dim=1)
            plt.imshow(combined_img.cpu(), cmap='gray')
            pred = predictions[i].item()
            true = batch['same_class'][i].item()
            plt.title(f'Pred: {pred:.2f}\nTrue: {true}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'images/{title.lower().replace(" ", "_")}.png', dpi=200)
        plt.close()

def plot_training_progress(train_losses, val_accuracies, test_accuracy):
    plt.figure(figsize=(12, 5))
    
    # Task IX: Training loss plot with EMA smoothing
    plt.subplot(1, 2, 1)
    # Original loss curve
    plt.plot(train_losses, alpha=0.3, color='blue', label='Raw Loss')
    
    # Bonus 1
    # Calculate EMA
    beta = 0.98  # Smoothing factor
    ema = []
    ema_val = 0
    for i, loss in enumerate(train_losses):
        ema_val = beta * ema_val + (1 - beta) * loss
        # Bias correction
        ema_val_corrected = ema_val / (1 - beta ** (i + 1))
        ema.append(ema_val_corrected)
    
    # Plot smoothed curve
    plt.plot(ema, color='red', label='EMA Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    
    # Task X: Validation accuracy plot (unchanged)
    plt.subplot(1, 2, 2)
    epochs = range(1, len(val_accuracies) + 1)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.axhline(y=test_accuracy, color='r', linestyle='--', 
                label=f'Test Accuracy: {test_accuracy:.2f}%')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation and Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('images/training_progress.png')
    plt.close()
