import os
import json
import torch
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from torchvision import transforms
from tqdm import tqdm


def clean_sentence(output, data_loader):

    # Stores words corresponding to indices
    words_sequence = []

    # Iterates over each word in the output
    for i in output:

        # Skip <EOS> tokens
        if i == 1:
            continue

        # Otherwise add word to sentence
        words_sequence.append(data_loader.dataset.vocab.itos[i])

    # Skip end tokens
    words_sequence = words_sequence[1: -1]
    # Join sentence
    sentence = " ".join(words_sequence)
    # Capitalize sentence
    sentence = sentence.capitalize()

    return sentence


def build_json(encoder, decoder, data_loader, device, filename):

    # Set both encoder and decoder to evaluation mode
    encoder.eval()
    decoder.eval()

    # Stores all of the small dictionaries
    result = list()

    # No need for gradient
    with torch.no_grad():

        # Iterates over each batch
        for batch_idx, (img_id, image) in tqdm(enumerate(data_loader)):

            # Change to CUDA if possible
            image = image.to(device)
            # Add a dimension for timestep
            features = encoder(image).unsqueeze(0).unsqueeze(1)
            # Sample the sentence from decoder
            output = decoder.sample(features)
            # Clean up sentence so that it is human readable
            sentence = clean_sentence(output, data_loader)
            # Add resulting caption with its image id
            result.append({"image_id": img_id.item(), "caption": sentence})

    # Dump into specified file
    with open(filename + ".json", "w") as f:
        json.dump(result, f)


# Hyperparameters
batch_size = 1
vocab_threshold = 5
vocab_from_file = True
embed_size = 300
hidden_size = 512
encoder_dir = "inception_encoder"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Navigate to encoder_dir
os.chdir(encoder_dir)

# Transformations
transform_test = transforms.Compose([
    transforms.Resize(328),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# Validation loader
val_loader = get_loader(transform=transform_test,
                        mode="valid",
                        batch_size=batch_size,
                        vocab_threshold=vocab_threshold,
                        vocab_from_file=vocab_from_file)

# Test loader
test_loader = get_loader(transform=transform_test,
                         mode="test")

# The size of the vocabulary
vocab_size = len(test_loader.dataset.vocab)

# Initialize the encoder and decoder.
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to GPU if CUDA is available
encoder.to(device)
decoder.to(device)

# Saved checkpoints
encoder_file = "encoder-0.pkl"
decoder_file = "decoder-0.pkl"

# Load checkpoints
encoder.load_state_dict(torch.load(os.path.join("models", encoder_file)))
decoder.load_state_dict(torch.load(os.path.join("models", decoder_file)))

# Make predictions subdirectory
if not os.path.exists("predictions"):
    os.mkdir("predictions")

# Navigate to it
os.chdir("predictions")

# Building the output jsons for server evaluation
build_json(encoder, decoder, val_loader, device, "captions_val2014_inception-bidirectional_results")
build_json(encoder, decoder, test_loader, device, "captions_test2014_inception-bidirectional_results")
