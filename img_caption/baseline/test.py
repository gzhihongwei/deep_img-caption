"""Title: Automatic-Image-Captioning
Author: Kshirsagar, Krunal
Date: 2020
Availability: https://github.com/Noob-can-Compile/Automatic-Image-Captioning/
Creates testing captions and validation captions for COCO evaluation server
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from torchvision import transforms
from tqdm import tqdm


def clean_sentence(output, data_loader):
    words_sequence = []

    for i in output:
        if i == 1:
            continue
        words_sequence.append(data_loader.dataset.vocab.itos[i])

    words_sequence = words_sequence[1: -1]
    sentence = " ".join(words_sequence)
    sentence = sentence.capitalize()

    return sentence


def build_json(encoder, decoder, data_loader, device, filename):
    encoder.eval()
    decoder.eval()

    result = list()

    with torch.no_grad():
        for batch_idx, (img_id, image) in tqdm(enumerate(data_loader)):
            image = image.to(device)

            features = encoder(image).unsqueeze(1)
            output = decoder.sample(features)
            sentence = clean_sentence(output, data_loader)

            result.append({"image_id": img_id.item(), "captions": sentence})

    with open(filename + ".json", "w") as f:
        json.dump(result, f)


# def get_prediction(encoder, decoder, data_loader):
#     orig_image, image = next(iter(data_loader))
#     plt.imshow(np.squeeze(orig_image))
#     plt.title("Sample Image")
#     plt.show()
#     image = image.to(device)
#     features = encoder(image).unsqueeze(1)
#     output = decoder.sample(features)
#     sentence = clean_sentence(output)
#     print(sentence)


batch_size = 1
vocab_threshold = 5
vocab_from_file = True
embed_size = 300
hidden_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# val_loader = get_loader(transform=transform_test,
#                         mode="valid",
#                         batch_size=batch_size,
#                         vocab_threshold=vocab_threshold,
#                         vocab_from_file=vocab_from_file)

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

encoder_file = "encoder-2.pkl"
decoder_file = "decoder-2.pkl"

encoder.load_state_dict(torch.load(os.path.join("./models", encoder_file)))
decoder.load_state_dict(torch.load(os.path.join("./models", decoder_file)))

encoder.to(device)
decoder.to(device)

# build_json(encoder, decoder, val_loader, device, "validation_preds")
build_json(encoder, decoder, test_loader, device, "test_preds")
