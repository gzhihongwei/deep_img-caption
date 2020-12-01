"""Title: Automatic-Image-Captioning
Author: Kshirsagar, Krunal
Date: 2020
Availability: https://github.com/Noob-can-Compile/Automatic-Image-Captioning/
Refined some statements and handled data loading for test and validation sets
"""

import nltk
import numpy as np
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO


def get_loader(transform,
               mode="train",
               batch_size=1,
               vocab_threshold=None,
               vocab_file="./vocab.pkl",
               start_word="<SOS>",
               end_word="<EOS>",
               unk_word="<UNK>",
               vocab_from_file=True,
               num_workers=0,
               coco_loc="../../datasets/coco"):
    """
    Returns the data loader.
    :param transform: Image transform.
    :param mode: Either 'train' or 'test'.
    :param batch_size: Batch size (if testing, batch_size must be 1).
    :param vocab_threshold: Minimum word count threshold.
    :param vocab_file: File containing the vocabulary.
    :param start_word: Special word denoting sentence start.
    :param end_word: Special word denoting sentence end.
    :param unk_word: Special word denoting unknown words.
    :param vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                            If True, load vocab from existing vocab_file if it exists.
    :param num_workers: Number of subprocesses to use for data loading.
    :param coco_loc: Location of coco directory.
    :return: torch.utils.data.DataLoader for the COCO dataset
    """

    assert mode in {"train", "valid", "test"}, "Mode must be either 'train', 'valid', or 'test'."

    # Check that training mode is active since there is no vocab from file
    if not vocab_from_file:
        assert mode == "train", "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations file.
    if mode == "train":
        if vocab_from_file:
            assert os.path.exists(vocab_file), \
                "vocab_file does not exist. Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(coco_loc, "images/train2014/")
        annotations_file = os.path.join(coco_loc, "annotations/captions_train2014.json")
    else:
        # Sample each image one at a time
        assert batch_size == 1, "Please change batch_size to 1 if validating or testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file, "Change vocab_from_file to True."

        img_folder = os.path.join(coco_loc, "images/val2014/") \
            if mode == "valid" else os.path.join(coco_loc, "images/test2014/")
        annotations_file = os.path.join(coco_loc, "annotations/captions_val2014.json") \
            if mode == "valid" else os.path.join(coco_loc, "annotations/image_info_test2014.json")


    # COCO captions dataset.
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    if mode == "train":
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # Data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset,
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))

    else:
        # Sample an image at a time
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader


class CoCoDataset(data.Dataset):
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word,
                 end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        # Preprocessing transform
        self.transform = transform
        # Train, valid, or test
        self.mode = mode
        self.batch_size = batch_size
        # Dictionaries of stoi and itos for words
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
                                end_word, unk_word, annotations_file, vocab_from_file)
        # Where the images are
        self.img_folder = img_folder

        # Train on all captions
        if self.mode == "train":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())

            print("Obtaining caption lengths...")

            all_tokens = [
                nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower())
                for index in np.arange(len(self.ids))
            ]

            self.caption_lengths = [len(token) for token in all_tokens]

        # Caption all images
        else:
            self.coco = COCO(annotations_file)
            self.img_ids = list(self.coco.imgToAnns.keys()) if self.mode == "valid" else self.coco.getImgIds()

    def __getitem__(self, index):
        # Obtain image and caption if in training mode
        if self.mode == "train":
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = [self.vocab(self.vocab.start_word)]
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # return pre-processed image and caption tensors
            return image, caption

        # Obtain image id and the image itself if testing or validating
        else:
            img_id = self.img_ids[index]
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)

            return img_id, image

    def get_train_indices(self):
        # Gets indices of annotations where captions are same length
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))

        return indices

    def __len__(self):
        return len(self.ids) if self.mode == "train" else len(self.img_ids)
