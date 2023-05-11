import torch

# this class takes in an observation and an action and predicts the next observation
class OneStepModel(torch.nn.Module):
    def __init__(self, latent_dim, action_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(latent_dim + action_dim, 64)
        # self.action_fc1 = torch.nn.Linear(action_dim, 32)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 8)
        self.fc4 = torch.nn.Linear(8, 32)
        self.fc5 = torch.nn.Linear(32, 64)
        self.fc6 = torch.nn.Linear(64, latent_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, observation):
        x = self.relu(self.fc1(observation))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class OneStepTransition(torch.nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(action_dim, 8)
        self.fc2 = torch.nn.Linear(latent_dim+8, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, latent_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, observation, action):
        x = self.relu(self.fc1(action))
        x = torch.cat([observation, x], dim=1)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(latent_dim, 32)
        self.fc2 = torch.nn.Linear(32, 64)
        self.fc3 = torch.nn.Linear(64, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, latent):
        x = self.relu(self.fc1(latent))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class OneStep(torch.nn.Module):
    def __init__(self, input_dim, action_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.transition_model = OneStepTransition(latent_dim, action_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, obs, action, next_obs):
        latent_obs = self.encoder(obs)
        latent_next_obs = self.encoder(next_obs)
        pred_latent_next_obs = self.transition_model(latent_obs, action)

        decoded_obs = self.decoder(latent_obs)
        decoded_next_obs = self.decoder(latent_next_obs)
        decoded_pred_next_obs = self.decoder(pred_latent_next_obs)
        
        return decoded_obs, decoded_next_obs, decoded_pred_next_obs, pred_latent_next_obs

    def transition(self, obs, action):
        latent_obs = self.encoder(obs)
        pred_latent_next_obs = self.transition_model(latent_obs, action)
        return pred_latent_next_obs
       
        

        