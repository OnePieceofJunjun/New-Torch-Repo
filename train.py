# Task III & VII & Bonus 3
import torch
from tabulate import tabulate

def evaluate(model, data_loader, device):
    # Task VII & Bonus 3
    model.eval()
    tp = fp = tn = fn = 0
    
    with torch.no_grad():
        for input_dict in data_loader:
            input_dict = {k: v.to(device) for k, v in input_dict.items()}
            output_dict = model(input_dict)
            predictions = (output_dict['output'] > 0.5).float()
            true_labels = input_dict['same_class']
            
            tp += ((predictions == 1) & (true_labels == 1)).sum().item()
            tn += ((predictions == 0) & (true_labels == 0)).sum().item()
            fp += ((predictions == 1) & (true_labels == 0)).sum().item()
            fn += ((predictions == 0) & (true_labels == 1)).sum().item()
    
    precision = tp / (tp + fp) * 100
    recall = tp / (tp + fn) * 100
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    
    model.train()
    return { 'precision': precision, 'recall': recall, 'accuracy': accuracy }

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    train_losses = []
    val_accuracies = []
    # Training loop
    for epoch in range(num_epochs):
        for i, input_dict in enumerate(train_loader):
            input_dict = {k: v.to(device) for k, v in input_dict.items()}
            output_dict = model(input_dict)
            loss = criterion(output_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if (i+1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Step [{i+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, "
                )
        # Validation loop
        metric = evaluate(model, val_loader, device)
        val_accuracies.append(metric['accuracy'])
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(tabulate([metric], headers='keys'))
    return train_losses, val_accuracies

def main():
    # my functions
    from dataset import get_loaders
    from model import MyNeuralNet
    from criterion import MyCriterion
    from utils import model_summary, init_weights
    from visualize import plot_batch_predictions, plot_training_progress
    import config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyNeuralNet(config.hidden_size).to(device)
    model.apply(init_weights)
    model_summary(model)
    train_loader, val_loader, test_loader = get_loaders(config.batch_size)
    criterion = MyCriterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Save initial batch visualizations
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    plot_batch_predictions(model, train_batch, device, "Initial Training Batch")
    plot_batch_predictions(model, val_batch, device, "Initial Validation Batch")

    train_losses, val_accuracies = train(
        model, train_loader, val_loader, 
        criterion, optimizer, device, config.num_epochs
    )

    # Final batch visualizations
    plot_batch_predictions(model, train_batch, device, "Final Training Batch")
    plot_batch_predictions(model, val_batch, device, "Final Validation Batch")
    test_metric = evaluate(model, test_loader, device)
    print("Test Set Metrics")
    print(tabulate([test_metric], headers='keys'))
    plot_training_progress(train_losses, val_accuracies, test_metric['accuracy'])

if __name__ == '__main__':
    main()
