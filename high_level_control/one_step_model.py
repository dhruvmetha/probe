import torch

# this class takes in an observation and an action and predicts the next observation
class OneStepModel(torch.nn.Module):
    def __init__(self, latent_dim, action_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(latent_dim + action_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, latent_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x