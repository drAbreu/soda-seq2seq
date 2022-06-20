# -*- coding: utf-8 -*-
from datasets import Dataset, DatasetDict
from typing import List, Tuple, Dict
import numpy as np
from setup_logger import logger
import logging
import re

logger = logging.getLogger('seq2seq.data_utils')


def load_dataset_into_hf(datapath: str, delimiter:str, task: str, split: List[float] = [0.8, 0.1, 0.1],
                         skip_lines: int = 0) -> DatasetDict:
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


def get_label_entity_pairs(text: str) -> List[Tuple[str, str]]:
    """
    This is a function intended mainly for supporting metrics evaluation. It will return a list with
    labels in the text. We define the labels as a pair `label` -> `entity` encoded as `label:entity`
    in the text.
    :param text: `str` text where the labels are to be found.
    :return: `list` of `tuple` `label` -> `entity`
    """
    label_entity_pair_reg = r"([a-z]+)\:(.[a-zA-Z0-9].+?(?=and|or|on|by|was|$))"
    return re.findall(label_entity_pair_reg, text)


def get_control_measure_exp_mentions(
        text: str,
        separators: List[str] = ["was tested for its influence", "by"]
        ) -> Dict[str, str]:
    """
    This is a function intended mainly for supporting metrics evaluation. It will return a list with
    labels in the text. We define the labels as a pair `label` -> `entity` encoded as `label:entity`
    in the text.
    :param text: `str` text where the labels are to be found.
    :param separators: `list` of `str` wirth the standarized text in the inputs.
                        Will be used to separate into controled, measured and experiment.
    :return: `list` of `tuple` `label` -> `entity`
    """
    if text.count("was tested") > 1:
        text = text.split('.')[0]
        if text.count("was tested") > 1:
            text = text.split('"')[0]

    regex_str = f"({separators[0]})|({separators[1]})"
    split_text = re.split(regex_str, text)
    separators += [None]

    output = {}
    output_list = [text for text in split_text if text not in separators]
    if len(output_list) == 3:
        output['control'] = output_list[0]
        output['measured'] = output_list[1]
        output['experiment'] = output_list[2]
    elif len(output_list) == 2:
        output['control'] = output_list[0]
        output['measured'] = output_list[1]
        output['experiment'] = ""
        logger.warning(f"The example: {text} has only two text outputs. It might have an infinite"
                       f"loop on the model prediction.")
    else:
        logger.warning(f"The example: {text} Might have an infinite loop of control entities")
        if text.count("was tested") < 1:
            output['control'] = output_list[0]
            output['measured'] = ""
            output['experiment'] = ""
        raise NotImplementedError
    return output


def get_labelled_data(
        text: str,
        separators: List[str] = ["was tested for its influence", "by"]
        ) -> Tuple[Tuple[str, str], Tuple[str, str], str]:
    """
    This is a function intended mainly for supporting metrics evaluation. It will return a list with
    labels in the text. We define the labels as a pair `label` -> `entity` encoded as `label:entity`
    in the text.
    :param text: `str` text where the labels are to be found.
    :param separators: `list` of `str` wirth the standarized text in the inputs.
                        Will be used to separate into controled, measured and experiment.
    :return: `list` of `tuple`, `tuple`, `str`
    """
    groups = get_control_measure_exp_mentions(text, separators=separators)

    label_entity_control = get_label_entity_pairs(groups['control'])
    label_entity_measured = get_label_entity_pairs(groups['measured'])

    return label_entity_control, label_entity_measured, groups['experiment']
