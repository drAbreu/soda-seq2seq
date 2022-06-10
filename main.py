# -*- coding: utf-8 -*-
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from datasets import load_dataset
from data_utils import load_dataset_into_hf
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq,
                          BartForConditionalGeneration, T5ForConditionalGeneration,
                          TrainingArguments, Seq2SeqTrainer)
from typing import List
from setup_logger import logger
import logging

logger = logging.getLogger('seq2seq.main')


class Seq2Seq:
    """
    Class intended to easily train Seq2Seq models in
    different tasks using the HuggingFace 🤗 ecosystem.
    """

    def __init__(self,
                 # DATA AND MODELS
                 datapath: str,
                 delimiter: str,
                 from_pretrained: str,
                 task: str,
                 # DATA GENERATION
                 split: List[float] = [0.8, 0.1, 0.1],
                 skip_lines: int = 0,
                 # TOKENIZER PARAMETERS
                 max_input_length: int = 512,
                 max_target_length: int = 512,
                 # MODEL PARAMETERS
                 max_output_length: int = 128,
                 min_output_length: int = 32,
                 num_beams: int = 2,
                 # TRAINING PARAMETERS
                 output_training_folder: str = "training_log"
                 ):

        self.datapath = datapath
        self.delimiter = delimiter
        self.from_pretrained = from_pretrained
        self.task = task
        self.split = split
        self.skip_lines = skip_lines
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.skip_lines = skip_lines
        self.max_output_length = max_output_length
        self.min_output_length = min_output_length
        self.num_beams = num_beams
        self.output_training_folder = output_training_folder
        self.tokenizer = AutoTokenizer.from_pretrained(self.from_pretrained)

        try:
            logger.info(f"Obtaining data from the HuggingFace 🤗 Hub: {self.datapath}")
            self.dataset = load_dataset(self.datapath)
        except FileNotFoundError:
            assert self.datapath.split('.')[-1] in ['csv', 'txt', 'tsv'], \
                f"""The data format is not supported. Please upload a file with format {'csv', 'txt', 'tsv'}
                        or write a valid path to a dataset in HuggingFace 🤗 Hub."""
            logger.info(f"Obtaining data from the local file: {self.datapath}")
            self.dataset = load_dataset_into_hf(self.datapath, self.delimiter, self.task)

        logger.info(f"""Tokenizing the data using the tokenizer of model {self.from_pretrained}\n
                        {self.tokenizer}""")
        self.tokenized_dataset = self._tokenize_data()

        logger.info(f"Downloading the model based on: {self.from_pretrained}")
        #!TODO: Go into https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/configuration#transformers.PretrainedConfig and check how to generate a better dataclass for the sequence generation parameters that can go here
        if 'bart' in self.from_pretrained:
            self.model = BartForConditionalGeneration.from_pretrained(self.from_pretrained,
                                                                      max_length=self.max_output_length,
                                                                      min_length=self.min_output_length,
                                                                      num_beams=self.num_beams,)
        elif 't5' in self.from_pretrained:
            self.model = T5ForConditionalGeneration.from_pretrained(self.from_pretrained,
                                                                    max_length=self.max_output_length,
                                                                    min_length=self.min_output_length,
                                                                    num_beams=self.num_beams,)
        else:
            raise ValueError(f"""Please select a model that is compatible wit the 
                                conditional generation task: {['bart', 't5']}.""")

    def __str__(self):
        print(self.tokenizer)
        print(self.tokenized_dataset['train'][0])
        return "Everything is running on a good way"

    def __call__(self):

        logger.info(f"""Preparing the data collator""")
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                               model=self.model,
                                               padding=True,
                                               return_tensors='pt')

        training_args = TrainingArguments(self.output_training_folder)
        trainer = Seq2SeqTrainer(**training_args)
        trainer.train()

    def _preprocess_data(self, examples):
        """
        Method that will be used to tokenize the input and target data on a way
        that can be used by the Seq2Seq model for training and inference.
        Method to be used with `Dataset.map` or `DatasetDict.map`.
        :param examples: iterable elements of the dataset
        :return: tokenized examples for a `Dataset.map` or `DatasetDict.map`.
        """
        input_ = list(map(lambda orig_string: self.task + orig_string, examples['input']))
        model_inputs = self.tokenizer(
            input_, max_length=self.max_input_length, truncation=True
        )
        # Set up the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["target"], max_length=self.max_target_length, truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _tokenize_data(self):
        return self.dataset.map(self._preprocess_data,
                                batched=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DATA_PATH = "./data/sd-seq2seq-clean.csv"
    DELIMITER = '###tt9HHSlkWoUM###'
    FROM_PRETRAINED = 'facebook/bart-base'
    TASK = "Causal hypothesis: "
    seq2seq = Seq2Seq(DATA_PATH, DELIMITER, FROM_PRETRAINED, TASK)
    seq2seq()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
