import os
import sys
import torch
import yaml
import argparse
import wandb

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from transformers import AdamW
from tqdm import tqdm
from datasets import load_metric

from dataloader import ImageCustomDataset, create_dataframe


def parser_args():
    """
    Initiating argument parser
    :return: args
    """
    parser = argparse.ArgumentParser(description='Training TrOCR Model')
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default='/config/config_train.yaml',
        help='path to file config')

    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, config: dict = None):

        # Declare config
        self.config = config

        # Init model
        self.model = VisionEncoderDecoderModel.from_pretrained(self.config["model"]["model_path"])
        self.processor = TrOCRProcessor.from_pretrained(self.config["model"]["processor_path"])

        # Move model to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Declare hypeparameters
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id

        # Make sure vocab size is set correctly
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # Set beam search parameters
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_new_tokens = self.config["model"]["max_length"]
        self.model.config.early_stopping = self.config["model"]["early_stopping"]
        self.model.config.no_repeat_ngram_size = self.config["model"]["no_repeat"]
        self.model.config.length_penalty = self.config["model"]["length_penalty"]
        self.model.config.num_beams = self.config["model"]["num_beams"]

        # Initiate loss and optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.config["model"]["learning_rate"])

        # Initiate dataloader

        # Create train and valid dataframe
        self.train_df = create_dataframe(self.config["train"]["train_label"])
        self.valid_df = create_dataframe(self.config["eval"]["valid_label"])

        # Create train and valid dataset
        self.train_dataset = ImageCustomDataset(
            root_dir=self.config["train"]["train_dir"],
            df=self.train_df,
            processor=self.processor,
        )
        self.valid_dataset = ImageCustomDataset(
            root_dir=self.config["eval"]["valid_dir"],
            df=self.valid_df,
            processor=self.processor,
        )
        # Create train and valid dataloader
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.config["train"]["batch_size"],
                                           shuffle=True
                                           )
        self.valid_dataloader = DataLoader(self.valid_dataset,
                                           batch_size=self.config["eval"]["batch_size"],
                                           )

        self.num_epochs = self.config["model"]["num_epochs"]
        self.best_valid_accuracy = 0.0

    def train(self):
        self.model.train()
        train_loss = 0.0
        for epoch in range(1, self.num_epochs + 1):
            for idx, batch in enumerate(tqdm(self.train_dataloader)):
                # get the inputs
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                # forward + backward + optimize
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss += loss.item()

                # print training information
                if (idx + 1) % self.config["model"]["print_batch_step"] == 0:
                    print(
                        f"Epoch [{epoch}/{self.num_epochs}] - Step [{idx + 1}/{len(self.train_dataloader)}] - Loss: {loss.item()}")

                if (idx + 1) % self.config["model"]["eval_batch_step"] == 0:
                    self.model.eval()
                    valid_cer = 0.0
                    correct = 0
                    with torch.no_grad():
                        for batch in tqdm(self.valid_dataloader):
                            # run batch generation
                            outputs = self.model.generate(batch["pixel_values"].to(self.device))
                            pred, label = outputs, batch["labels"]
                            pred, label = self.output_decoder(pred, label)

                            # After decode
                            pred, label = self.normalize(pred, label)

                            # compute metrics
                            cer = self.compute_cer(pred, label)
                            valid_cer += cer
                            correct += self.get_correct(pred, label)

                    iter_cer = valid_cer / len(self.valid_dataloader)
                    accuracy = correct / len(self.valid_dataset)
                    print(f"Accuracy = {accuracy:.4f}, CER = {iter_cer:.4f}")

                    # Logging metrics
                    wandb.log({"valid acc": accuracy, "cer": iter_cer})

                    # Choose best checkpoint
                    if accuracy > self.best_valid_accuracy:
                        self.best_valid_accuracy = accuracy
                        self.model.save_pretrained(self.config["model"]["save_path"])
                        print(f"save best metric at epoch {epoch}")
                    else:
                        print(f"Not pass the best metric, best metric acc = {self.best_valid_accuracy:.2f}")
                    print(f"Cur accuracy {self.best_valid_accuracy:.4f}")

    def compute_cer(self, pred_list, label_list):
        cer_metric = load_metric("cer")
        cer = cer_metric.compute(predictions=pred_list, references=label_list)
        return cer

    def output_decoder(self, pred_ids, label_ids):
        pred_list = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_list = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        return pred_list, label_list

    def normalize(self, pred_list: list, label_list: list):
        pred_list = [pred.replace(" ", "") for pred in pred_list]
        label_list = [label.replace(" ", "") for label in label_list]
        return pred_list, label_list

    def get_correct(self, pred_list: list, label_list: list) -> int:
        correct = 0
        if len(pred_list) == len(label_list):
            for i in range(len(label_list)):
                # print(f"Predict : {pred_list[i]}, label : {label_list[i]}")
                if pred_list[i] == label_list[i]:
                    correct += 1
                    # print(f"CORRECT PREDICT!!! ")
        return correct


def run(args):
    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Init wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="handwritten-ocr",
        name="v1.0.0",
        # track hyperparameters and run metadata
        config={
            "learning_rate": config["model"]["learning_rate"],
            "architecture": "VisionTransformers",
            "dataset": "handwritten",
            "epochs": config["model"]["num_epochs"],
        }
    )

    trainer = Trainer(config=config)
    trainer.train()
    wandb.finish()


def main():
    args = parser_args()
    run(args)


if __name__ == '__main__':
    main()
