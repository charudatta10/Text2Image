from __future__ import print_function

import torch
import logging
import argparse
import torch.utils.data
import matplotlib.pyplot as plt

from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display

################################################################
# PARAMETERS: Define if visdom visualizations should be enabled.
################################################################
VISUALIZE, CURRENT_EPOCH = False, 0
if VISUALIZE: 
    from code.E2.visualizer import Visualizer
    # from visualizer import Visualizer
    visualizer = Visualizer("GAN MNIST Training")

# Define the batch and latent space size.
batch_size = 100
latent_size = 20

# Define the data, hidden and discriminator output layer sizes.
data_dim = 784
hidden_dim = 400
pred_dim = 1

# Determine if a GPU is available and set the device according.
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Define the number of workers and memory configurations.
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# Define the MNIST dataset loaders (train and test splits)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

class Generator(nn.Module):
    """
    The generator takes an input of size latent_size, and will produce 
    an output of size 784. It should have a single hidden linear layer 
    with 400 nodes using ReLU activations, and use Sigmoid activation 
    for its outputs.
    """

    def __init__(self):
        """
        """
        # Initialize the base torch model.
        super(Generator, self).__init__()
        
        # Define the generator logger.
        self._logger = logging.getLogger("GAN Generator Logger")

        # Construct the generator layers.
        self.__construct_generator(latent_size, hidden_dim, data_dim)

    def __construct_generator(self, input_dim : int, hidden_dim : int, output_dim : int) -> None:
        """
        """
        try:
            # Construt generator hidden layer (input dim to hidden dim).
            self.generator_hidden_layer = nn.Linear(input_dim, hidden_dim)
            # Define the generator hidden layer activations (ReLU).
            self.generator_hidden_activation = nn.ReLU()

            # Construct generator output layer (hidden dim to output_dim)
            self.generator_output_layer = nn.Linear(hidden_dim, output_dim)
            # Define the generator ouput layer activations (Sigmoid).
            self.generator_output_activation = nn.Sigmoid()

        except Exception as construct_generator_exception:
            self._logger.error(
                "ERROR: Issue occured while attempting to construct the generator layers.")
            raise construct_generator_exception

    def forward(self, z : torch.Tensor):
        """
        """
        try:
            # Pass the input through the hidden layer.
            z = self.generator_hidden_layer(z)
            z = self.generator_hidden_activation(z)

            # Pass the hidden layer output throught the output layer.
            x_hat = self.generator_output_layer(z)
            x_hat = self.generator_output_activation(x_hat)

            # Return the generator output tensor, corresponding to z.
            return x_hat

        except Exception as forward_exception:
            self._logger.error(
                "ERROR: Issue occured while attempting to conduct a forward pass on the generator.")
            raise forward_exception

class Discriminator(nn.Module):
    """
    The discriminator takes an input of size 784, and will 
    produce an output of size 1. It should have a single 
    hidden linear layer with 400 nodes using ReLU activations, 
    and use Sigmoid activation for its output.
    """

    def __init__(self):
        """
        """
        # Initialize the base torch model.
        super(Discriminator, self).__init__()
        
        # Define the generator logger.
        self._logger = logging.getLogger("GAN Discriminator Logger")

        # Construct the discriminator layers.
        self.__construct_discriminator(data_dim, hidden_dim, pred_dim)

    def __construct_discriminator(self, input_dim : int, hidden_dim : int, output_dim : int):
        try:
            # Construt discriminator hidden layer (input dim to hidden dim).
            self.discriminator_hidden_layer = nn.Linear(input_dim, hidden_dim)
            # Define the discriminator hidden layer activations (ReLU).
            self.discriminator_hidden_activation = nn.ReLU()

            # Construct discriminator output layer (hidden dim to output_dim)
            self.discriminator_output_layer = nn.Linear(hidden_dim, output_dim)
            # Define the discriminator ouput layer activations (Sigmoid).
            self.discriminator_output_activation = nn.Sigmoid()

        except Exception as construct_discrimnator_exception:
            self._logger.error(
                "ERROR: Issue occured while attempting to construct the discriminator layers.")
            raise construct_discrimnator_exception

    def forward(self, x : torch.Tensor):
        """
        """
        try:
            # Pass the input through the hidden layer.
            x = self.discriminator_hidden_layer(x)
            x = self.discriminator_hidden_activation(x)

            # Pass the hidden layer output throught the output layer.
            y_hat = self.discriminator_output_layer(x)
            y_hat = self.discriminator_output_activation(y_hat)
 
            # Return the discriminator output tensor, corresponding to x.
            return y_hat

        except Exception as forward_pass_exception:
            self._logger.error(
                "ERROR: Issue occurred while attempting to conduct a forward pass on the discriminator.")
            raise forward_pass_exception

def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    """
    Trains both the generator and discriminator for one 
    epoch on the training dataset. Returns the average 
    generator and discriminator loss (scalar values, 
    use the binary cross-entropy appropriately)
    """
    try:
        # Define the initial generator and discriminator losses.
        avg_generator_loss = 0
        avg_discriminator_loss = 0

        # Ensure that both models are in training mode.
        generator.train()
        discriminator.train()

        # Initialize the binary cross-entropy loss.
        binary_cross_entropy = nn.BCELoss()

        # Iterate through each batch in the training dataset.
        for batch, (features, _) in enumerate(train_loader):
            #################################################################################
            # Discriminator Training: max (1/n)*Sum(log(y_real)) + (1/m)*Sum(log(1 - y_fake))
            #################################################################################
            # Clear gradients for the discriminator optimizers. 
            discriminator_optimizer.zero_grad()

            # Generate a collection of fake MNIST images.
            x_hat = generator(torch.normal(0.0, 1.0, (batch_size, latent_size,)).to(device))

            # Pass the generated images through the discriminator.
            y_fake = discriminator(x_hat)

            # Pass the real images through the discriminator.
            y_real = discriminator(torch.flatten(features, start_dim = 1).to(device))

            # Determine the loss associated with the real predictions and compute the assoociated gradients.
            discriminator_loss_real = binary_cross_entropy(y_real, torch.ones((batch_size, pred_dim)).to(device))
            discriminator_loss_real.backward()
            
            # Determine the loss associated with fake predictions and compute the associated gradients.
            discriminator_loss_fake = binary_cross_entropy(y_fake, torch.zeros((batch_size, pred_dim)).to(device))
            discriminator_loss_fake.backward()

            # Update discriminator parameters using the gradients for the real and fake predictions.
            discriminator_optimizer.step()

            ####################################################
            # Generator Training: min (1/m)*Sum(log(1 - y_fake))
            ####################################################
            # Clear the gradients for the generator optimizer.
            generator_optimizer.zero_grad()

            # Generate a collection of fake MNIST images.
            x_hat = generator(torch.normal(0.0, 1.0, (batch_size, latent_size,)).to(device))

            # Pass the generated images through the discriminator.
            y_fake = discriminator(x_hat)

            # Determine the generator loss, which is attempting to essentially 
            # trying maximize fooling the discriminator.
            generator_loss =  binary_cross_entropy(y_fake, torch.ones((batch_size, pred_dim)).to(device))
            
            # Compute the generator gradients.
            generator_loss.backward()

            # Update the generator parameters using the gradients for the current batch.
            generator_optimizer.step()

            # Update the average losses.
            avg_generator_loss += generator_loss.item()
            avg_discriminator_loss += discriminator_loss_real.item() + discriminator_loss_fake.item()
            
            # If visualizations are enabled, visualize the loss.
            if VISUALIZE:
                visualizer.plot(
                    plot_name = "loss_over_epochs",
                    component_name = "train_generator_loss",
                    title = "Loss over Batches", 
                    xlabel = "Batches",
                    ylabel = "Loss",
                    x = [CURRENT_EPOCH + ((batch + 1) / len(train_loader))],
                    y = [generator_loss.item()/batch_size]
                )
                
                visualizer.plot(
                    plot_name = "loss_over_epochs",
                    component_name = "train_discriminator_loss",
                    title = "Loss over Batches", 
                    xlabel = "Batches",
                    ylabel = "Loss",
                    x = [CURRENT_EPOCH + ((batch + 1) / len(train_loader))],
                    y = [(discriminator_loss_real.item() + \
                        discriminator_loss_fake.item())/batch_size]
                )

        return avg_generator_loss / len(train_loader), avg_discriminator_loss / len(train_loader)

    except Exception as training_exception:
        logging.error(
            "ERROR: Issue occured while attempting to train the GAN on the loaded dataset.")
        raise training_exception

def test(generator, discriminator):
    """
    Runs both the generator and discriminator over the test dataset.
    Returns the average generator and discriminator loss (scalar 
    values, use the binary cross-entropy appropriately).
    """
    try:
        # Define the initial testing losses.
        avg_generator_loss = 0
        avg_discriminator_loss = 0

        # Initialize the binary cross-entropy loss.
        binary_cross_entropy = nn.BCELoss()

        # Ensure that both models are in evaluation mode.
        generator.eval()
        discriminator.eval()

        # Iterate through each batch in the testing dataset.
        for batch, (features, _) in enumerate(test_loader):
            # Generate a collection of fake MNIST images.
            x_hat = generator(torch.normal(0.0, 1.0, (batch_size, latent_size,)).to(device))

            # Pass the generated images through the discriminator.
            y_fake = discriminator(x_hat)

            # Pass the real images through the discriminator.
            y_real = discriminator(torch.flatten(features, start_dim = 1).to(device))

            # Determine the loss associated with the real predictions.
            discriminator_loss_real = binary_cross_entropy(y_real, torch.ones((batch_size, pred_dim)).to(device))
            
            # Determine the loss associated with fake predictions.
            discriminator_loss_fake = binary_cross_entropy(y_fake, torch.zeros((batch_size, pred_dim)).to(device))

            # Determine the generator loss.
            generator_loss = binary_cross_entropy(y_fake, torch.ones((batch_size, pred_dim)).to(device))

            # Update the average losses.
            avg_generator_loss += generator_loss.item()
            avg_discriminator_loss += discriminator_loss_real.item() + discriminator_loss_fake.item()
            
            # If visualizations are enabled, visualize the loss.
            if VISUALIZE:
                visualizer.plot(
                    plot_name = "loss_over_epochs",
                    component_name = "test_generator_loss",
                    title = "Loss over Batches", 
                    xlabel = "Batches",
                    ylabel = "Loss",
                    x = [CURRENT_EPOCH + ((batch + 1) / len(train_loader))],
                    y = [generator_loss.item()/batch_size]
                )
                
                visualizer.plot(
                    plot_name = "loss_over_epochs",
                    component_name = "test_discriminator_loss",
                    title = "Loss over Batches", 
                    xlabel = "Batches",
                    ylabel = "Loss",
                    x = [CURRENT_EPOCH + ((batch + 1) / len(train_loader))],
                    y = [(discriminator_loss_real.item() + \
                        discriminator_loss_fake.item())/batch_size]
                )

        return avg_generator_loss/ len(test_loader), avg_discriminator_loss / len(test_loader)

    except Exception as testing_exception:
        logging.error(
            "ERROR: Issue occured while attempting to test the generator and discriminator.")
        raise testing_exception

if __name__ == "__main__":
    # Define the number of epochs and losses over epochs.
    epochs = 50
    discriminator_avg_train_losses = []
    discriminator_avg_test_losses = []
    generator_avg_train_losses = []
    generator_avg_test_losses = []

    # Construct the generator and discriminator models.
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Construct the generator and discriminator optimizers.
    generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
    
    # During each epoch, train on the training dataset and evaluate on the 
    # testing dataset, additionally construct a collection of sample 
    # generated datapoints to observe the model's generative abilities.
    for epoch in range(1, epochs + 1):
        # Train and evaluate the gan model on the training and test datasets respectively.
        generator_avg_train_loss, discriminator_avg_train_loss = \
            train(generator, generator_optimizer, discriminator, discriminator_optimizer)
        generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)

        # Keep track of the average generator and discriminator losses.
        discriminator_avg_train_losses.append(discriminator_avg_train_loss)
        generator_avg_train_losses.append(generator_avg_train_loss)
        discriminator_avg_test_losses.append(discriminator_avg_test_loss)
        generator_avg_test_losses.append(generator_avg_test_loss)

        # Output the test losses at every epoch.
        print(F"Generator Loss: {generator_avg_test_loss}")
        print(F"Discriminator Loss: {discriminator_avg_test_loss}")

        # Construct a sample of generated images to observe the model's generative abilities.
        with torch.no_grad():
            sample = torch.randn(64, latent_size).to(device)
            sample = generator(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                    'results/gan_sample_' + str(epoch) + '.png')
            print('Epoch #' + str(epoch))
            display(Image('results/gan_sample_' + str(epoch) + '.png'))
            print('\n')

        # Update the global epoch tracker.
        CURRENT_EPOCH += 1

    # Plot the average generator and discriminator training loss over epochs.
    plt.plot(discriminator_avg_train_losses, label = "Discriminator")
    plt.plot(generator_avg_train_losses, label = "Generator")
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("results/gan_train_loss.png")
    plt.close()

    # Plot the average generator and discriminator testing loss over epochs.
    plt.plot(discriminator_avg_test_losses, label = "Discriminator")
    plt.plot(generator_avg_test_losses, label = "Generator")
    plt.title('Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("results/gan_test_loss.png")
    plt.close()
