# Task II & VI
import torch
import torch.nn as nn

class MyNeuralNet(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # Siamese network architecture
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*7*7, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, input_dict: dict):
        # Task VI
        # Process both images through the same network
        img1 = input_dict['img1']
        img2 = input_dict['img2']
        
        # Extract features from both images
        feat1 = self.feature_extractor(img1)
        feat2 = self.feature_extractor(img2)
        
        # Compute the absolute difference between the two features
        diff = torch.abs(feat1 - feat2)
        
        # Final classification
        output = self.classifier(diff)
        
        return {
            'output': output.squeeze(),
            'same_class': input_dict['same_class']
        }

if __name__ == '__main__':
    net = MyNeuralNet(500)
    from utils import model_summary, init_weights
    model_summary(net)
    net.apply(init_weights)
    
    # Test forward pass
    x1 = torch.randn(16, 1, 28, 28)
    x2 = torch.randn(16, 1, 28, 28)
    same = torch.randint(0, 2, (16,)).float()
    out = net({'img1': x1, 'img2': x2, 'same_class': same})
    print(out['output'].shape)
    print(out['same_class'].shape)
