# -*- coding: utf-8 -*-
import logging
from seq2seq.trainer import SodaSeq2SeqTrainer
logger = logging.getLogger('seq2seq.main')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DATA_PATH = "./data/sd-seq2seq-clean.csv"
    DELIMITER = '###tt9HHSlkWoUM###'
    FROM_PRETRAINED = 'facebook/bart-base'
    TASK = "Causal hypothesis: "
    seq2seq = SodaSeq2SeqTrainer(DATA_PATH, DELIMITER, FROM_PRETRAINED, TASK)
    seq2seq()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
