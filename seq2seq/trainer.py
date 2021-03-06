from datasets import load_dataset
from data_utils import load_dataset_into_hf
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq,
                          BartForConditionalGeneration, T5ForConditionalGeneration,
                          Seq2SeqTrainer)
from typing import List
from setup_logger import logger
from data_classes import TrainingArgumentsSeq2Seq, ModelConfigSeq2Seq
from transformers.integrations import TensorBoardCallback
from callbacks import ShowExample, MyTensorBoardCallback
import logging
from torch.utils.data import DataLoader
import torch
from metrics import ClassificationSeq2Seq
from data_collator import MyDataCollatorForSeq2Seq
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

logger = logging.getLogger('seq2seq.trainer')


class SodaSeq2SeqTrainer:
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
                 from_local_checkpoint: str = None,
                 base_model: str = None,
                 # DATA GENERATION
                 split: List[float] = [0.8, 0.1, 0.1],
                 skip_lines: int = 0,
                 # TOKENIZER PARAMETERS
                 max_input_length: int = 512,
                 max_target_length: int = 512,
                 # MODEL PARAMETERS
                 model_param: ModelConfigSeq2Seq = ModelConfigSeq2Seq(),
                 # TRAINING PARAMETERS
                 training_args: TrainingArgumentsSeq2Seq = TrainingArgumentsSeq2Seq()
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
        self.model_param = model_param
        self.training_args = training_args
        self.from_local_checkpoint = from_local_checkpoint
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer, self.model = self._get_model_and_tokenizer()

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

    def __call__(self):

        logger.info(f"""Preparing the data collator""")
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                               model=self.model,
                                               padding=True,
                                               return_tensors='pt')
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=data_collator,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['eval'],
            tokenizer=self.tokenizer,
            callbacks=[ShowExample(self.tokenizer)]
        )

        if self.training_args.do_train:
            trainer.remove_callback(TensorBoardCallback)  # remove default Tensorboard callback
            trainer.add_callback(MyTensorBoardCallback)  # replace with customized callback
            trainer.train()

        if self.training_args.do_predict:
            output_predictions, output_labels = [], []
            test_dataloader = trainer.get_test_dataloader(self.tokenized_dataset['test'])
            logger.info("Getting the data predictions")
            logger.info(f"Data columns: {self.tokenized_dataset['test'].column_names}")
            logger.info(f"The device used at this point is {self.device}")
            with torch.no_grad():
                for batch in tqdm(test_dataloader):
                    outputs = self.model.generate(batch['input_ids'].to(self.device))
                    preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    output_predictions.append(preds)
            # Call to my metrics calculator
            metrics_role = ClassificationSeq2Seq(task="roles")
            metrics_ner = ClassificationSeq2Seq(task="ner")
            metrics_exp = ClassificationSeq2Seq(task="experiment")
            flat_predictions = list(np.concatenate(output_predictions).flat)
            logger.info("Metric evaluation for roles")
            role_labels, role_predictions = metrics_role(flat_predictions, self.tokenized_dataset['test']['target'])
            logger.info("Metric evaluation for NER")
            ner_labels, ner_predictions = metrics_ner(flat_predictions, self.tokenized_dataset['test']['target'])
            logger.info("Metric evaluation for experiments")
            jaccard_distance = metrics_exp(flat_predictions, self.tokenized_dataset['test']['target'])

            print(classification_report(np.array(role_labels),
                                        np.array(role_predictions),
                                        labels=["MEASURED_VAR", "CONTROLLED_VAR"]
                                        )
                  )
            print(classification_report(np.array(ner_labels),
                                        np.array(ner_predictions),
                                        labels=["gene", "protein", "molecule","cell", "organism", "tissue", "subcellular"]
                                        )
                  )

            print(f"""The Average Jaccard Distance experiment strings: {jaccard_distance}""")

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

    def _get_model_and_tokenizer(self):
        if self.from_local_checkpoint:
            logger.info(f"Downloading the model based on: {self.base_model} and checkpoint{self.from_local_checkpoint}")
            if 'bart' in self.base_model:
                model = BartForConditionalGeneration.from_pretrained(self.from_local_checkpoint,
                                                                          **self.model_param.__dict__)
            elif 't5' in self.base_model:
                model = T5ForConditionalGeneration.from_pretrained(self.from_local_checkpoint,
                                                                        **self.model_param.__dict__)
            else:
                raise ValueError(f"""Please select a model that is compatible with the 
                                    conditional generation task: {['bart', 't5']}.""")
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        else:
            logger.info(f"Downloading the model based on: {self.from_pretrained}")
            if 'bart' in self.from_pretrained:
                model = BartForConditionalGeneration.from_pretrained(self.from_pretrained,
                                                                          **self.model_param.__dict__)
            elif 't5' in self.from_pretrained:
                model = T5ForConditionalGeneration.from_pretrained(self.from_pretrained,
                                                                        **self.model_param.__dict__)
            else:
                raise ValueError(f"""Please select a model that is compatible wit the 
                                    conditional generation task: {['bart', 't5']}.""")
            tokenizer = AutoTokenizer.from_pretrained(self.from_pretrained)
        return tokenizer, model
