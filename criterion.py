# Task III & V
import torch
import torch.nn as nn

class MyCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, input_dict):
        # input_dict['output'] should be a single value between 0 and 1
        # input_dict['same_class'] is the ground truth (0 or 1)
        return self.loss(input_dict['output'], input_dict['same_class'])
    

if __name__ == '__main__':
    criterion = MyCriterion()
    print(criterion)
    
    # Test forward pass
    out = torch.randn(16, 10, requires_grad=True)
    target = torch.randint(0, 10, (16,))
    loss = criterion({'output': out, 'target': target})
    print(loss)
