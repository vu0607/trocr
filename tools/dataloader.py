import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset


def create_dataframe(path: str):
    df = pd.read_csv(path, sep='\t', header=None, names=['file_name', 'text'])
    return df


class ImageCustomDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # Prepare image (resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB").resize((384, 384))

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # Add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        labels = labels[:self.max_target_length]
        # Make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
