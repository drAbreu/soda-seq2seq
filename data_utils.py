from datasets import Dataset, DatasetDict
from typing import List
import numpy as np
from setup_logger import logger
import logging

logger = logging.getLogger('seq2seq.data_utils')

def load_dataset_into_hf(datapath: str, delimiter:str, task: str, split: List[float] = [0.8, 0.1, 0.1],
                         skip_lines: int = 0) -> DatasetDict[Dataset]:
    """
    Loads a file encoded as a `csv` into a format that can be used
    by the `Seq2Seq` class.
    The `Seq2Seq` class accepts HuggingFace 🤗 datasets with features
    called `input` and `target`. Both features encoded as `str`.

    :param datapath: `str` path to the `csv` file
    :param delimiter: `str` delimiter between the two data fields
    :param task: `str` prefix for the inputs. To be used for training.
    :param split: `list` of `float` percentage of data splits for training, validation, and test
    :param skip_lines: `int` number of lines to skip at the beginning of the file

    :return: datasets.DatasetDict with the data split into `train`, `test`, and `eval` sets
    """

    assert sum(split) == 1, f"""The numbers of the split argument must sum up to one. They are 
                                {split}, summing up to {sum(split)}"""
    assert len(split) == 3, f"""The length of the split argument must be 3. You have 
                                {len(split)} elements in the list."""

    data_output = {'train': {'input': [], 'target': []},
                   'test': {'input': [], 'target': []},
                   'eval': {'input': [], 'target': []}}
    logger.info(f"Reading lines from file {datapath}")
    with open(datapath, 'r') as file_:
        for line in file_.readlines()[skip_lines:]:
            data_pair = line.split(delimiter)
            choice = np.random.choice(["train", "eval", "test"], p=split)
            try:
                input_, target = data_pair
                data_output[choice]['input'].append(input_)
                data_output[choice]['target'].append(target)
            except ValueError:
                logger.warning(f"""Wrong number of task--input in line. Possible lack of 
                                delimiter. Skipping line to next one. {line}""")
                continue

    return DatasetDict(
                        {
                            'train': Dataset.from_dict(data_output['train']),
                            'eval': Dataset.from_dict(data_output['eval']),
                            'test': Dataset.from_dict(data_output['test'])
                        }
                       )
