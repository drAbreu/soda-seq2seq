# -*- coding: utf-8 -*-
import logging
from seq2seq.trainer import SodaSeq2SeqTrainer
from data_utils import get_label_entity_pairs, get_control_measure_exp_mentions, get_labelled_data
from metrics import ClassificationSeq2Seq
logger = logging.getLogger('seq2seq.main')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # LABEL = ['''gene:cyclin f or gene:e2f1 or gene:ubiquitin or molecule:ly2603618 or protein:chk1 was tested for its influence on protein:e2f1 or protein:ub by immunoblotted, immunoprecipitated "''',
    #          '''gene:igfbp1a was tested for its influence on cell:-cell or organism:zebrafish by fluorescence microscopy''']
    # OUTPUT = ['''gene:cyclin f or gene:ubiquitin or molecule:chk1i was tested for its influence on protein:e2f1 or protein:ubiquitin by immunoblotted, immunoprecipitated "''',
    #           '''gene:igfbp1a was tested for its influence on cell:-cell by regeneration assays.''']
    # SEPARATORS = ['was tested for its influence', 'by']
    #
    #
    # metrics = ClassificationSeq2Seq(task="experiment")
    # metrics(LABEL, OUTPUT)
    data_loader = "./data/sd-seq2seq-clean.csv"
    delimiter = "###tt9HHSlkWoUM###"
    from_pretrained = "###tt9HHSlkWoUM###"
    task = "facebook/bart-base"
    from_local_checkpoint = "./seq2seq_models/checkpoint-10000"
    base_model = "t5-base"

    seq2seq = SodaSeq2SeqTrainer(data_loader,
                                 delimiter,
                                 from_pretrained,
                                 task,
                                 from_local_checkpoint=from_local_checkpoint,
                                 base_model=base_model,
                                 split=split,
                                 skip_lines=skip_lines,
                                 max_input_length=max_input_length,
                                 max_target_length=max_target_length,
                                 training_args=training_args,
                                 model_param=model_args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
