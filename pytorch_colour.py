import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Загрузка CIFAR-10
train_set = torchvision.datasets.CIFAR10(
    root='./CIFAR', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(
    root='./CIFAR', train=False, download=True, transform=transform)

dog_class = 5 # Класс собак в CIFAR-10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False)

# вход - изображение (3, 32, 32)
# выход - классификация (число от 0 до 1)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # при свертке размер сохраняется
            # размерность на входе - (batch_size, 3, 32, 32)
            # размерность после сверток - (batch_size, 64, 32, 32)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5,
                      stride=2, padding=2),  # padding="SAME"
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),

            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),

            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


discriminator = Discriminator().to(device=device)
summary(discriminator, (3, 32,32))
# вход - шум
# выход - тензор (3, 32, 32)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.BatchNorm1d(128 * 4 * 4),  # нормализация данных
            nn.Unflatten(1, (128, 4, 4)),  # одномерный вектор в многомерный

            # увеличение размера с 4x4 до 8x8
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 128, kernel_size=4,
                               stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 32x32, каналы - 128 -> 3
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # x - вектор шума
        return self.model(x)


generator = Generator().to(device=device)
summary(generator, (100,))

# Сглаживание меток

def smooth_positive(y):
    # true labels will are marked with values between 0.8 and 1
    return y - 0.2 + (torch.rand(y.shape) * 0.2)


def smooth_negative(y):
    # Fake labels are marked with values between 0 and 0.3
    return y + torch.rand(y.shape) * 0.3

# Обучение
lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)


for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    for n, (real_samples, _) in enumerate(train_loader):
        batch_size = real_samples.size(0)

        # данные для дискриминатора
        random_samples = torch.rand((batch_size, 100)).to(device=device)

        real_samples_labels = smooth_positive(torch.ones((batch_size, 1))).to(device=device)
        generated_samples_labels = smooth_negative(torch.zeros((batch_size, 1))).to(device=device)

        generated_samples = generator(random_samples)
        all_samples = torch.cat((real_samples, generated_samples))
        all_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # обучение дискриминатора
        discriminator.zero_grad()
        discriminator_output = discriminator(all_samples)
        discriminator_loss = loss_function(discriminator_output, all_labels)

        discriminator_loss.backward()
        optimizer_discriminator.step()

        # данные для генератора
        random_samples = torch.rand((batch_size, 100)).to(device=device)

        # обучение генератора
        generator.zero_grad()
        generated_samples = generator(random_samples)


        discriminator_output_generated = discriminator(generated_samples)
        generator_loss = loss_function(discriminator_output_generated, real_samples_labels)

        generator_loss.backward()
        optimizer_generator.step()

        if (n == batch_size - 1):
            print(f"Epoch: {epoch + 1} Loss D.: {discriminator_loss}")
            print(f"Epoch: {epoch + 1} Loss G.: {generator_loss}")

            plt.clf()
            plt.title(f"After {epoch + 1} epoch(s)")
            generated_samples = generated_samples.cpu().detach()
            img = (generated_samples + 1)/2
            for i in range(32):
                ax = plt.subplot(8, 4, i + 1)
                plt.imshow(img[i].permute(1, 2, 0).numpy())
                plt.xticks([])
                plt.yticks([])
                plt.pause(0.001)

torch.save(generator, 'generator_colour.pth')
torch.save(discriminator, 'discriminator_colour.pth')