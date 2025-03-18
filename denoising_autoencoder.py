from multiprocessing.spawn import freeze_support

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim

# this brings in the ChestMNIST dataset, so we can access training, validation and test images in pytorch dataloaders
from medmnist.dataset import ChestMNIST


# define the NN architecture
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # TODO
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),   
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=1),                      
            nn.Sigmoid()                             
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DenoisingAutoencoder().to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # lower the number of epochs during development
    NUM_EPOCHS = 30
    # batch size = number of simultaneously used images during mini-batch-based stochastic gradient descent
    BATCH_SIZE = 128
    # noise level as a variance, given that our input images are normalized to [0,1]
    # try halving for IMG_SIZE = 28, to reduce the noise effect on the smaller images!
    NOISE_LEVEL = 0.001
    # IMG_SIZE = 64 gives access to higher resolution images. for development
    # ATTENTION: stay with IMG_SIZE = 28, especially if you don't have a GPU!
    IMG_SIZE = 28
    VISUALIZE = True

    # makes sure images are converted to pytorch tensors
    data_transform = transforms.Compose([transforms.ToTensor()])

    training_dataset = ChestMNIST(split="train", download=True, root='./data', size=IMG_SIZE, transform=data_transform)
    training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # validation and testing datasets are analyzed in batches of 1 image (with these datasets, the model is applied, not trained!)
    validation_dataset = ChestMNIST(split="val", download=True, root='./data', size=IMG_SIZE, transform=data_transform)
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False, num_workers=0)
    testing_dataset = ChestMNIST(split="test", download=True, root='./data', size=IMG_SIZE, transform=data_transform)
    testing_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False, num_workers=0)

    #print(training_dataset)

    total_train_time = 0 
    results_train = {"epoch": [], "total time": [], "epoch loss": []}
    results_val = {"epoch": [], "val loss": []}
    batch_losses_train = []
    batch_losses_val = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss_train = 0
        for batch_idx, (batch_of_images, _) in enumerate(training_dataloader):
        #for batch_of_images, _ in training_dataloader:

            start = time.time()
            batch_of_images = batch_of_images.to(device)

            # Add noise
            noisy_images = batch_of_images + (NOISE_LEVEL ** 0.5) * torch.randn_like(batch_of_images).to(device)
            noisy_images = torch.clamp(noisy_images, 0., 1.)

            # if batch_idx % visualize_interval == 0:  # Visualize every 'visualize_interval' batches
            #     noisy_image_to_visualize = noisy_images[0].cpu().detach().numpy()  # Take the first image
            #     plt.imshow(noisy_image_to_visualize.transpose((1, 2, 0)))  # Handle potential channel order
            #     plt.title(f"Noisy Image - Epoch: {epoch}, Batch: {batch_idx}")
            #     plt.show()

            outputs = model(noisy_images)
            loss = criterion(outputs, batch_of_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_train += loss.item()
            batch_losses_train.append(loss.item())

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss_train/len(training_dataloader)}")
        end = time.time()
        total_train_time += (end-start)

        results_train["epoch"].append(epoch)
        results_train["total time"].append(total_train_time)
        results_train["epoch loss"].append(epoch_loss_train/len(training_dataloader))    


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (batch_of_images, _) in enumerate(validation_dataloader):
            
                batch_of_images = batch_of_images.to(device)
                noisy_images = batch_of_images + (NOISE_LEVEL ** 0.5) * torch.randn_like(batch_of_images).to(device)
                noisy_images = torch.clamp(noisy_images, 0., 1.)

                outputs = model(noisy_images)

                loss = criterion(outputs, batch_of_images)
                val_loss += loss.item() 
                batch_losses_val.append(loss.item())
            

                #if i == 5:  # Visualize 5 examples
                    # fig, axarr = plt.subplots(1, 3, figsize=(12, 4))
                    # axarr[0].imshow(batch_of_images[0].cpu().squeeze(0), cmap='gray')
                    # axarr[0].set_title('Original Image')
                    # axarr[1].imshow(noisy_images[0].cpu().squeeze(0), cmap='gray')
                    # axarr[1].set_title('Noisy Image')
                    # axarr[2].imshow(outputs[0].cpu().squeeze(0), cmap='gray')
                    # axarr[2].set_title('Reconstructed Image')
                    # plt.show()
        
            results_val["epoch"].append(epoch)
            results_val["val loss"].append(val_loss/len(validation_dataloader))
            print(f"Validation Loss: {val_loss/len(validation_dataloader)}")

    torch.save(model.state_dict(), "model_GG_5.pth")

    epochs = results_train["epoch"] 
    train_losses = results_train["epoch loss"]
    val_losses = results_val["val loss"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'g-x', label='Validation Loss')

    plt.annotate(f"{train_losses[0]:.6f}", 
                 (epochs[0], train_losses[0]),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center') 

    plt.annotate(f"{train_losses[-1]:.6f}",
                 (epochs[-1], train_losses[-1]),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center')

    plt.annotate(f"{val_losses[0]:.6f}",
                 (epochs[0], val_losses[0]),
                 textcoords="offset points",
                 xytext=(0,-20),
                 ha='center')

    plt.annotate(f"{val_losses[-1]:.6f}",
                 (epochs[-1], val_losses[-1]),
                 textcoords="offset points",
                 xytext=(0,-20),
                 ha='center')


    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig('train_val_loss_per_epoch.png')
    plt.show()

    
    model.eval()
    test_loss = 0
    ssim_scores = []
    with torch.no_grad():
        for batch_of_images, _ in testing_dataloader:
            batch_of_images = batch_of_images.to(device)
            noisy_images = batch_of_images + (NOISE_LEVEL ** 0.5) * torch.randn_like(batch_of_images).to(device)
            noisy_images = torch.clamp(noisy_images, 0., 1.)

            outputs = model(noisy_images)
            loss = criterion(outputs, batch_of_images)
            test_loss += loss.item() * batch_of_images.size(0)

            outputs_np = outputs.squeeze().cpu().numpy()
            batch_of_images_np = batch_of_images.squeeze().cpu().numpy()

            for i in range(outputs_np.shape[0]): 
                ssim_value = ssim(outputs_np[i], batch_of_images_np[i], data_range=1.0)
                ssim_scores.append(ssim_value)

    test_loss /= len(testing_dataloader.dataset)
    print(f"Test Loss: {test_loss}")
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    rmse = np.sqrt(test_loss)

    print(f"Test Loss (MSE): {test_loss}")
    print(f"Test RMSE: {rmse}")
    print(f"Average SSIM: {avg_ssim}")


    if VISUALIZE == True:
        # obtain one batch of training images
        dataiter = iter(training_dataloader)
        for i in range(0,3):
            batch_of_images, _ = next(dataiter)

            # apply noising to the whole batch of images
            noisy_images = batch_of_images + (NOISE_LEVEL ** 0.5) * torch.randn_like(batch_of_images)

            # get one orig image from the batch
            img_orig = np.squeeze(batch_of_images.numpy()[0])
            print(i, "img orig: min=", np.min(img_orig), "max=", np.max(img_orig), "mean=", np.mean(img_orig), "std=", np.std(img_orig))

            # get one noisy image from the batch
            img_noised = np.squeeze(noisy_images.numpy()[0])
            print(i, "img noised: min=", np.min(img_noised), "max=", np.max(img_noised), "mean=", np.mean(img_noised), "std=", np.std(img_noised))

            fig, axarr = plt.subplots(1, 2)

            axarr[0].imshow(np.squeeze(img_orig), cmap='gray')
            axarr[0].set_title('Original Image')
            axarr[1].imshow(np.squeeze(img_noised), cmap='gray')
            axarr[1].set_title('Noisy Image')
            
            plt.show()

