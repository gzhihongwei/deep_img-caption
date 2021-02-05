import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    Encodes images into a feature vector using ResNet50
    """
    def __init__(self, embed_size=1024):
        super(EncoderCNN, self).__init__()

        # Using Inception V3 to encode images
        inception = models.inception_v3(pretrained=True)

        # Freezing model
        for param in inception.parameters():
            param.requires_grad_(False)

        # Change last fully connected layer to encode to embed_size
        inception.fc = nn.Linear(inception.fc.in_features, embed_size)

        # Setting the encoding model
        self.inception = inception

    def forward(self, images):
        # Encode the images
        features = self.inception(images)[0]

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # For embedding the words
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        # Main decoder
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        # For identifying which word in vocab to pick
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Exclude last word
        captions = captions[:, :-1]
        # Embed all of the captions
        embed = self.embedding_layer(captions)
        # Add a timestep entry in feature tensor and concatenate with embedding
        embed = torch.cat((features.unsqueeze(1), embed), dim=1)
        # Only outputs from LSTM are used for decoding
        lstm_outputs, _ = self.lstm(embed)
        # Feed through linear layer to get words
        out = self.linear(lstm_outputs)

        return out

    def sample(self, inputs, states=None, max_len=20):
        """Accepts pre-processed image tensor (inputs and returns predicted sentence (list of tensor ids of length max_len)"""

        # Stores output
        output_sentence = []

        # Loops over length longest allowed sentence
        for _ in range(max_len):
            # LSTM outputs and hidden states
            lstm_outputs, states = self.lstm(inputs, states)
            # Get rid of timestep
            lstm_outputs = lstm_outputs.squeeze(1)
            # Feed through linear layer
            out = self.linear(lstm_outputs)
            # Get index of most probable word
            last_pick = out.max(1)[1]
            # Add index to output sentence
            output_sentence.append(last_pick.item())
            # Repeat
            inputs = self.embedding_layer(last_pick).unsqueeze(1)

        return output_sentence
