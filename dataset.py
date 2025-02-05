# Task I & III & IV & VII & Bonus 2
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import random

class MyMNIST(Dataset):
    def __init__(self, train=True, use_dict=True):
        super().__init__()
        self.sequential_pairs = False  # Add this flag
        mnist = MNIST(root='./data', train=train, transform=transforms.ToTensor(), download=True)
        class_indices = {}
        for _class in mnist.classes:
            class_idx = mnist.class_to_idx[_class]
            class_indices[class_idx] = [
                i for i in range(len(mnist)) if mnist.targets[i] == class_idx
            ]

        self.mnist = mnist
        self.class_indices = class_indices
        self.use_dict = use_dict        

    def __len__(self):
        return len(self.mnist)
    
    def get_second_img_idx(self, class1: int, same_class: bool):
        if same_class:
            idx2 = random.choice(self.class_indices[class1])
        else:
            diff_classes = [c for c in self.class_indices.keys() if c != class1]
            diff_class = random.choice(diff_classes)
            idx2 = random.choice(self.class_indices[diff_class])
        return idx2
    
    def __getitem__(self, idx):
        # Task IV
        if self.sequential_pairs:
            # For test set: pair sequential images
            pair_idx = idx // 2 * 2  # Round down to even number
            img1, class1 = self.mnist[pair_idx]
            img2, class2 = self.mnist[pair_idx + 1]
            same_class = (class1 == class2)
        else:
            # Original random pairing logic
            img1, class1 = self.mnist[idx]
            same_class = random.random() > 0.5
            idx2 = self.get_second_img_idx(class1, same_class)
            img2, class2 = self.mnist[idx2]
            
        if self.use_dict:
            return {
                'img1': img1, 
                'img2': img2,
                'class1': class1,
                'class2': class2,
                'same_class': torch.tensor(float(same_class))
            }
        else:
            return img1, img2, class1, class2, torch.tensor(float(same_class))
    
def get_loaders(batch_size):
    # Task VII
    train_set = MyMNIST(train=True)
    test_set = MyMNIST(train=False)

    # Split the training set into training and validation
    train_size = int(0.8 * len(train_set))
    assert train_size == 48000, "Train size should be 48000"
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])
    test_set.sequential_pairs = True  # Set this flag for test set

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )

def visualize_batch(batch):
    import matplotlib.pyplot as plt
    import numpy as np
    
    img1_data = batch['img1']
    img2_data = batch['img2']
    class1_labels = batch['class1']
    class2_labels = batch['class2']
    same_class = batch['same_class']
    batch_size = len(img1_data)

    n_rows = int(np.sqrt(batch_size))
    n_cols = int(np.ceil(batch_size / n_rows))

    plt.figure(figsize=(12, 8))
    plt.suptitle('MNIST Batch')
    for i in range(batch_size):
        plt.subplot(n_rows, n_cols, i + 1)
        combined_img = torch.cat([img1_data[i][0], img2_data[i][0]], dim=1)
        plt.imshow(combined_img.cpu(), cmap='gray')
        is_same = same_class[i].item()
        title = f'Class {class1_labels[i]} vs. Class {class2_labels[i]}\n{"Same" if is_same else "Different"}'
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def time_dataloader(dataset, num_iterations=1000, batch_size=32):
    import time
    from tqdm import tqdm
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    start_time = time.time()
    iterator = iter(dataloader)
    for _ in tqdm(range(num_iterations), desc='Timing...', unit='iter'):
        try:
            _ = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            _ = next(iterator)
    end_time = time.time()
    
    return end_time - start_time

def _test_visualize():
    batch_size = 16  # You can change this to 32 if needed
    train_loader, val_loader, test_loader = get_loaders(batch_size)
    
    visualize_batch(next(iter(train_loader)))
    visualize_batch(next(iter(val_loader)))
    visualize_batch(next(iter(test_loader)))

def _test_timing():
    # Bonus 2
    num_iterations = 1000
    print(f"Running timing comparison for {num_iterations} iterations...")

    tuple_time = time_dataloader(MyMNIST(train=True, use_dict=False))
    dict_time = time_dataloader(MyMNIST(train=True, use_dict=True))

    print("\nResults:")
    print(f"Tuple-based implementation: {tuple_time:.2f} seconds")
    print(f"Dictionary-based implementation: {dict_time:.2f} seconds")
    print(f"Difference: {(dict_time - tuple_time):.2f} seconds")
    print(f"Dictionary overhead: {((dict_time - tuple_time) / tuple_time * 100):.1f}%")

    
if __name__ == '__main__':
    # _test_visualize()
    _test_timing()
