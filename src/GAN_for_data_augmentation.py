import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_size = 64
hidden_size = 256
image_size = 28 * 28
batch_size = 100
lr = 0.0002
epochs = 10

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])


# Generator network
class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img


# Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        return out


# Initialize networks
generator = Generator(latent_size).to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()


# Helper function to generate images
def generate_images(generator, num_images):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_size, device=device)
        fake_images = generator(z).cpu()
    generator.train()
    return fake_images.view(num_images, 28, 28)


# Training loop
def train_gan(generator, discriminator, train_loader, epochs):
    d_losses, g_losses = [], []
    for epoch in range(epochs):
        for i, (images, _) in enumerate(train_loader):
            current_batch_size = images.size(0)
            real_images = images.view(current_batch_size, -1).to(device)
            real_labels = torch.ones(current_batch_size, 1, device=device)
            fake_labels = torch.zeros(current_batch_size, 1, device=device)

            z = torch.randn(current_batch_size, latent_size, device=device)
            fake_images = generator(z)

            optimizer_d.zero_grad()
            real_outputs = discriminator(real_images)
            fake_outputs = discriminator(fake_images.detach())
            d_loss = criterion(real_outputs, real_labels) + criterion(fake_outputs, fake_labels)
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            if (i + 1) % 200 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], "
                      f"d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, "
                      f"D(x): {real_outputs.mean().item():.2f}, D(G(z)): {fake_outputs.mean().item():.2f}")

    return d_losses, g_losses


if __name__ == "__main__":
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    d_losses, g_losses = train_gan(generator, discriminator, train_loader, epochs)

    os.makedirs('docs', exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator loss', alpha=0.7)
    plt.plot(g_losses, label='Generator loss', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('GAN training losses')
    plt.legend()
    plt.tight_layout()
    plt.savefig('docs/losses.png', dpi=120)
    plt.show()

    images = generate_images(generator, 64)
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow((img + 1) / 2, cmap='gray')
        ax.axis('off')
    plt.suptitle(f'Generated digits after {epochs} epochs')
    plt.tight_layout()
    plt.savefig('docs/generated_digits.png', dpi=120)
    plt.show()
