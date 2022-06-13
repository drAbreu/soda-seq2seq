# -*- coding: utf-8 -*-
import logging
from seq2seq.trainer import SodaSeq2SeqTrainer
from transformers import HfArgumentParser
from data_classes import TrainingArgumentsSeq2Seq, ModelConfigSeq2Seq
logger = logging.getLogger('seq2seq.main')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = HfArgumentParser((TrainingArgumentsSeq2Seq,
                               ModelConfigSeq2Seq), description="Training arguments for a Seq2Seq task.")
    parser.add_argument("data_loader", help="Path to the csv file or a 🤗 dataset repository.")
    parser.add_argument("--from_pretrained",
                        default="facebook/bart-base",
                        help="""🤗 model repository to be used as base model for training.
                        It can also be the path to a checkpoint that would be further trained.""")
    parser.add_argument("--from_local_checkpoint",
                        default="",
                        help="""Begin the training from a previous checkpoint stored locally. 
                        If the model is in the 🤗, then using from_pretrained would be enough.
                        Must be set with base_model.""")
    parser.add_argument("--base_model",
                        default="facebook/bart-base",
                        help="""Base-model of the checkpoint to be further trained. This will be used to instantiate
                        the model and the tokenizer.""")
    parser.add_argument("--task",
                        default="Causal hypothesis: ",
                        choices=["Causal hypothesis: ", "multitask"],
                        help="""Task to fine-tune the model. If set to multitask, the tasks must be 
                        defined at the beginning of the inputs in the dataset.
                        This would be the prefered way to do it if several tasks are to be included in the model.""")
    parser.add_argument("--delimiter",
                        default="###tt9HHSlkWoUM###",
                        help="Delimiter between inputs and labels in the dataset.")
    parser.add_argument("--split",
                        default=[0.8, 0.1, 0.1],
                        help="""3-element list with the relative size of the train, test, and eval datasets.
                        Only relevant if the dataset is enclosed in a single file. If the data set is already
                        split into three parts in a 🤗 repository, the argument will be ignored.""")
    parser.add_argument("--skip_lines",
                        default=0,
                        help="""Lines to be skipped when parsing the data. 
                        If the data set is already in a 🤗 repository, the argument will be ignored.""")
    parser.add_argument("--max_input_length",
                        default=512,
                        help="Maximum length (in tokens) of the sentence accepted by the tokenizer.")
    parser.add_argument("--max_target_length",
                        default=512,
                        help="Maximum length (in tokens) of the sentence output by the model.")

    training_args, model_args, args = parser.parse_args_into_dataclasses()
    data_loader = args.data_loader
    from_pretrained = args.from_pretrained
    task = args.task
    delimiter = args.delimiter
    split = args.split
    skip_lines = int(args.skip_lines)
    max_input_length = int(args.max_input_length)
    max_target_length = int(args.max_target_length)
    from_local_checkpoint = args.from_local_checkpoint
    base_model = args.base_model

    seq2seq = SodaSeq2SeqTrainer(data_loader,
                                 delimiter,
                                 from_pretrained,
                                 task,
                                 from_checkpoint=from_local_checkpoint,
                                 base_model=base_model,
                                 split=split,
                                 skip_lines=skip_lines,
                                 max_input_length=max_input_length,
                                 max_target_length=max_target_length,
                                 training_args=training_args,
                                 model_param=model_args)
    seq2seq()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
