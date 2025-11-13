import torch
from torch import nn, optim


class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 5)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(5, 3)
        self.relu1 = nn.Sigmoid()
        self.linear2= nn.Linear(3, 2)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x

model = MyMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(3, 4)
print("Input shape: ", x.shape)

for epoch in range(10):
    target = torch.tensor([1, 0, 1])
    output = model(x)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    print()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Output shape: ", output.shape)
print("Output:\n", output)