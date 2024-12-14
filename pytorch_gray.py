import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import torch

def smooth_positive(y):
    return y - 0.2 + (torch.rand_like(y) * 0.2)

def smooth_negative(y):
    return y + (torch.rand_like(y) * 0.3)

torch.manual_seed(42)
plt.figure(figsize=(6, 4))
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = torchvision.datasets.MNIST(root='.',train=True, download=True,transform=transform)
test_set = torchvision.datasets.MNIST(root='.',train=False, download=True,transform=transform)
batch_size=32

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, x): 
        x = x.view(x.size(0), 784)  # Преобразуем входной тензор (32, 1, 28, 28) в тензор размерности (32, 784)
        output = self.model(x)
        return output

discriminator = Discriminator().to(device=device)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(1024, 784),
            nn.LeakyReLU(0.2, inplace=True), #выходные коэффициенты от -1 до 1
        )
    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28) # Преобразуем выход в изображение размером 28x28
        return output
    
generator = Generator().to(device=device)

lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    for n, (real_samples, mnist_labels) in enumerate(train_loader):

        #Данные для дискриминатора
        real_samples = real_samples.to(device=device)
        random_samples = torch.rand((batch_size, 100)).to(device=device)

        real_samples_labels = smooth_positive(torch.ones((batch_size, 1))).to(device=device)
        generated_samples_labels = smooth_negative(torch.zeros((batch_size, 1))).to(device=device)
        
        generated_samples = generator(random_samples)

        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        #обучение дискриминатора
        discriminator.zero_grad()
        discriminator_output = discriminator(all_samples)
        discriminator_loss = loss_function(discriminator_output, all_samples_labels)

        discriminator_loss.backward()
        optimizer_discriminator.step()

        #данные для генератора
        random_samples = torch.rand((batch_size, 100)).to(device=device)

        #обучение генератора
        generator.zero_grad()
        generated_samples = generator(random_samples)

        discriminator_output_generated = discriminator(generated_samples)
        generator_loss = loss_function(discriminator_output_generated, real_samples_labels)

        generator_loss.backward()
        optimizer_generator.step()


        if n == batch_size - 1:
            print(f"Epoch: {epoch + 1} Loss D.: {discriminator_loss}")
            print(f"Epoch: {epoch + 1} Loss G.: {generator_loss}")

            plt.clf()
            plt.title(f"After {epoch + 1} epoch(s)")
            generated_samples = generated_samples.cpu().detach()
            for i in range(32):
                ax = plt.subplot(8, 4, i + 1)
                plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
                
                plt.xticks([])
                plt.yticks([])
            plt.pause(0.001)


torch.save(generator, 'generator_gray.pth')
torch.save(discriminator, 'discriminator_gray.pth')