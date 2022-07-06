from transformers import BartForConditionalGeneration
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
# from metrics import ClassificationSeq2Seq
# from data_collator import MyDataCollatorForSeq2Seq
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

from blurb_benchmark import BlurbIOBToSeq2Seq


class BlurbTrainer(BlurbIOBToSeq2Seq):
    def __init__(self,
                 from_pretrained: str,
                 from_local_checkpoint: str = None,
                 base_model: str = None,
                 # TOKENIZER PARAMETERS
                 max_input_length: int = 512,
                 max_target_length: int = 512,
                 # MODEL PARAMETERS
                 model_param: ModelConfigSeq2Seq = ModelConfigSeq2Seq(),
                 # TRAINING PARAMETERS
                 training_args: TrainingArgumentsSeq2Seq = TrainingArgumentsSeq2Seq(),
                 **kw):
        super(BlurbTrainer, self).__init__(**kw)
        # Here generate the data using the BlurbBenchmark class
        self.from_pretrained = from_pretrained
        self.from_local_checkpoint = from_local_checkpoint
        self.base_model = base_model
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.model_param = model_param
        self.training_args = training_args
        self.blurb_data = BlurbIOBToSeq2Seq(dataset_name=self.dataset_name,
                                         task_name=self.task_name,
                                         model=self.model,
                                         separator=self.separator,
                                         label_mode=self.label_mode
                                        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer, self.model = self._get_model_and_tokenizer()
        self.tokenized_dataset = self._tokenize_data()
        print(self.tokenized_dataset['train'][0])

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
            eval_dataset=self.tokenized_dataset['validation'],
            tokenizer=self.tokenizer,
            callbacks=[ShowExample(self.tokenizer)]
        )

        if self.training_args.do_train:
            trainer.remove_callback(TensorBoardCallback)  # remove default Tensorboard callback
            trainer.add_callback(MyTensorBoardCallback)  # replace with customized callback
            trainer.train()

        if self.training_args.do_predict:
            output_predictions = []
            test_dataloader = trainer.get_test_dataloader(self.tokenized_dataset['test'])
            logger.info("Getting the data predictions")
            logger.info(f"Data columns: {self.tokenized_dataset['test'].column_names}")
            logger.info(f"The device used at this point is {self.device}")
            with torch.no_grad():
                for batch in tqdm(test_dataloader):
                    outputs = self.model.generate(batch['input_ids'].to(self.device))
                    preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    output_predictions.extend(preds)
            
            with open("demofile.csv", "w") as file_:
                for pred, label in zip(output_predictions, self.tokenized_dataset['test']):
                    file_.write(f"""{pred} ###separator### {label["targets"]}\n""")

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
            elif ('t5' in self.from_pretrained) or ('SciFive' in self.from_pretrained):
                model = T5ForConditionalGeneration.from_pretrained(self.from_pretrained,
                                                                        **self.model_param.__dict__)
            else:
                raise ValueError(f"""Please select a model that is compatible wit the 
                                    conditional generation task: {['bart', 't5', 'SciFive']}.""")
            tokenizer = AutoTokenizer.from_pretrained(self.from_pretrained)
        return tokenizer, model

    def _tokenize_data(self):
        return self.blurb_data.dataset_seq2seq.map(self._preprocess_data,
                                batched=True)

    def _preprocess_data(self, examples):
        """
        Method that will be used to tokenize the input and target data on a way
        that can be used by the Seq2Seq model for training and inference.
        Method to be used with `Dataset.map` or `DatasetDict.map`.
        :param examples: iterable elements of the dataset
        :return: tokenized examples for a `Dataset.map` or `DatasetDict.map`.
        """
        # input_ = list(map(lambda orig_string: self.task_name + orig_string, examples['inputs']))
        model_inputs = self.tokenizer(
            examples['inputs'], max_length=self.max_input_length, truncation=True
        )
        # Set up the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["targets"], max_length=self.max_target_length, truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
