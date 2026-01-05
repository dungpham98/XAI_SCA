import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    def __init__(self, input_length=700, num_classes=256):
        super(BaseCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(1, 64, kernel_size=11, stride=2, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        
        # Block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=11, stride=2, padding=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        
        # Block 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=11, stride=2, padding=5)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AvgPool1d(kernel_size=2)
        
        # Block 4
        self.conv4 = nn.Conv1d(256, 512, kernel_size=11, stride=2, padding=5)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.AvgPool1d(kernel_size=2)
        
        self.dropout = nn.Dropout(0.5)
        
        # --- Dynamic Linear Layer Size Calculation ---
        # We pass a dummy input through the conv layers to see what the output size is.
        # This avoids manual math errors (like the 1536 vs 10752 mismatch).
        self._to_linear = None
        self.convs_only(torch.randn(1, 1, input_length))
        
        # Define Fully Connected Layers
        self.fc1 = nn.Linear(self._to_linear, 4096) 
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def convs_only(self, x):
        """Helper to run just the conv parts and calculate size."""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).shape[1]
        
        return x

    def forward(self, x):
        # x shape: [Batch, 1, Length]
        x = self.convs_only(x)
        
        x = x.view(x.size(0), -1) # Flatten
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x