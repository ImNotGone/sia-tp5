import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.optim import Adam
from torchvision.utils import save_image, make_grid
from dataset_loaders import load_font_data

class FontMatrixDataset(Dataset):
    def __init__(self, font_matrix):
        self.data = torch.FloatTensor(font_matrix).reshape(-1, 35)  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class VAE(nn.Module):
    def __init__(self, input_dim=35, hidden_dim=25, latent_dim=12):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim *2),  
        )

        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = x[:, :12], x[:, 12:]  
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var
    
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def train(model, optimizer, dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        if(epoch%1000==0):
            print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (batch_idx + 1))


def generate_digit(mean, var):
    latent_dim = 12  # Cambiar segun latent 
    z_sample = torch.tensor([[mean, var] * (latent_dim // 2)], dtype=torch.float).to(device)
    x_decoded = model.decode(z_sample)
    digit = x_decoded[0].detach().cpu().reshape(7, 5)
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()
    


def plot_latent_space_characters(model, scale=1.0, n=15, figsize=15):
    padding = 1
    digit_size_x = 5  
    digit_size_y = 7  
    padded_digit_size_x = digit_size_x + padding
    padded_digit_size_y = digit_size_y + padding

    figure = np.zeros((padded_digit_size_y * n, padded_digit_size_x * n))

    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi] * (12 // 2)], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size_y, digit_size_x)
            y_start = i * padded_digit_size_y
            y_end = (i + 1) * padded_digit_size_y
            x_start = j * padded_digit_size_x
            x_end = (j + 1) * padded_digit_size_x
            figure[y_start:y_end - padding, x_start:x_end - padding] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = padding // 2
    end_range = n * padded_digit_size_y + start_range
    pixel_range = np.arange(start_range, end_range, padded_digit_size_y)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()



    

transform = transforms.Compose([transforms.ToTensor()])

font_matrix = load_font_data()


font_dataset = FontMatrixDataset(font_matrix)
font_loader = DataLoader(dataset=font_dataset, batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)


train(model, optimizer, font_loader, epochs=20000)

#generate_digit(0.0, 1.0), generate_digit(1.0, 0.0)
plot_latent_space_characters(model)
