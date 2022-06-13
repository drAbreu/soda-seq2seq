from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from random import randrange, sample
from typing import Dict

import torch
from transformers import RobertaTokenizerFast, TrainerCallback


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ShowExample(TrainerCallback):
    """Visualizes on the console the result of a prediction with the current state of the model.
    It uses a randomly picked input example and decodes input and output with the provided tokenizer.
    The predicted words are colored depending on whether the prediction is correct or not.
    If the prediction is incorrect, the expected word is displayed in square brackets.

    Args:

        tokenizer: the tokenizer used to generate the dataset.

    Class Attributes:

        COLOR_CHAR (Dict): terminal colors used to produced colored string
    """

    COLOR_CHAR = {}

    def __init__(self, tokenizer, *args, **kwargs):
        self.tokenizer = tokenizer

    def on_evaluate(
        self,
        *args,
        model=None,
        eval_dataloader: torch.utils.data.DataLoader = None,
        **kwargs
    ):
        """Method called when evaluating the model. Only the needed kwargs are unpacked.

        Args:
            model: the current model being trained.
            eval_dataloader (torch.utils.data.DataLoader): the DataLoader used to produce the evaluation examples
        """
        with torch.no_grad():
            examples = self.pick_random_example(eval_dataloader)
            labels = self.tokenizer.decode(examples['labels'][0], skip_special_tokens=True)
            outputs = model.generate(examples['input_ids'])
            pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            inputs = self.tokenizer.decode(examples['input_ids'][0], skip_special_tokens=True)

        self.to_console(inputs, pred, labels)

    @staticmethod
    def pick_random_example(dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        length = len(dataloader.dataset)
        dataset = dataloader.dataset
        rand_example_idx = randrange(length)
        batch = dataloader.collate_fn([dataset[rand_example_idx]])  # batch with a single random example
        inputs = {}
        for k, v in batch.items():
            inputs[k] = v.cuda() if torch.cuda.is_available() else v
        return inputs

    def to_console(self, inputs, pred, labels):
        print(50*"*")
        print(f"{BColors.OKGREEN} Example of prediction in an evaluation sentence{BColors.ENDC}")
        print(f"{BColors.OKGREEN} Input: {BColors.ENDC}")
        print(inputs)
        print(f"{BColors.OKGREEN} Output: {BColors.ENDC}")
        print(pred)
        print(f"{BColors.OKGREEN} Expected output: {BColors.ENDC}")
        print(labels)
        print(50*"*")
