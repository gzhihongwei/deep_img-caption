import math
import torch
import os
import glob
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN


# Empty the CUDA cache
torch.cuda.empty_cache()

# Hyperparameters
batch_size = 128
vocab_threshold = 5
vocab_from_file = True
load_checkpoint = True
embed_size = 400
hidden_size = 384
num_epochs = 3
save_every = 1
lr = 0.003
logging_file = "training.log"
encoder_dir = "inception_encoder"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform_train = transforms.Compose([
    transforms.Resize(328),
    transforms.RandomCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# Build data loader.
data_loader = get_loader(transform=transform_train,
                         mode="train",
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

# The size of the vocabulary
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder.
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

# Define the loss function.
criterion = nn.CrossEntropyLoss().to(device)

# The learnable parameters of the model
params = list(encoder.inception.fc.parameters()) + list(decoder.parameters())

# Define the optimizer
optimizer = torch.optim.Adam(params=params, lr=lr)

# Set the total number of training steps per epoch.
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

# Create new directory for encoder
if not os.path.exists(encoder_dir):
    os.mkdir(encoder_dir)

# Model Checkpoints directory
model_dir = os.path.join(encoder_dir, "models")

# Make models directory
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Open logging file
f = open(os.path.join(encoder_dir, logging_file), "w")

# Train from latest checkpoint
if load_checkpoint:
    # Get the latest encoder checkpoint
    encoder_files = glob.glob(os.path.join(model_dir, "encoder-?.pkl"))
    encoder_file = max(encoder_files, key=os.path.getctime)

    # Get the latest decoder checkpoint
    decoder_files = glob.glob(os.path.join(model_dir, "decoder-?.pkl"))
    decoder_file = max(decoder_files, key=os.path.getctime)

    # Load state dicts from checkpoints
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))

for epoch in range(num_epochs):
    for i_step in range(total_step):
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler

        # Obtain the batch.
        images, captions = next(iter(data_loader))

        # Move batch of images and captions to GPU if CUDA is available
        images = images.to(device)
        captions = captions.to(device)

        # Zero the gradients
        decoder.zero_grad()
        encoder.zero_grad()

        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)

        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()

        # Get training statistics.
        stats = f"Epoch[{epoch + 1}/{num_epochs}], Step [{i_step + 1}/{total_step}], Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):.4f}"

        # Print stats
        print(stats)

        # Print training statistics to file.
        f.write(stats + '\n')
        f.flush()

    # Save the weights
    if epoch % save_every == 0:
        torch.save(decoder.state_dict(), os.path.join(model_dir, f"decoder-{epoch}.pkl"))
        torch.save(encoder.state_dict(), os.path.join(model_dir, f"encoder-{epoch}.pkl"))

# Close log
f.close()
