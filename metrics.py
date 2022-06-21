from typing import List
from sklearn.metrics import classification_report
from data_utils import get_labelled_data
import numpy as np
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import textdistance
from tqdm import tqdm

class ClassificationSeq2Seq:
    def __init__(self,
                task: str = 'roles',
                ner_labels: List[str] = ["gene", "protein", "molecule",
                                         "cell", "organism", "tissue", "subcellular"],
                roles_labels: List[str] = ["MEASURED_VAR", "CONTROLLED_VAR"]
                ):
        assert task in ['roles', 'ner', 'experiment'], f"""Please, choose one of {['roles', 'ner', 'experiment']}"""
        self.task = task
        self.ner_labels = ner_labels
        self.roles_labels = roles_labels

    def __call__(self,
                 predictions: List[str],
                 labels: List[str],
                 separators: List[str] = ["was tested for its influence", "by"]
                 ):
        assert len(predictions) == len(labels), """predictions and labels must have the same size"""

        if self.task == 'roles':
            cleaned_outputs = {'predictions': [], 'labels': []}
            for prediction, label in tqdm(zip(predictions, labels)):
                pred_controls, pred_measured, _ = get_labelled_data(prediction, separators=separators)
                label_controls, label_measured, _ = get_labelled_data(label, separators=separators)
                print(pred_controls, pred_measured)
                print(label_controls, label_measured)
                for label_pair in label_controls:
                    if label_pair in pred_controls:
                        cleaned_outputs['predictions'].append("CONTROLLED_VAR")
                        cleaned_outputs['labels'].append("CONTROLLED_VAR")
                        pred_controls.remove(label_pair)
                    if label_pair not in pred_controls:
                        cleaned_outputs['predictions'].append("O")
                        cleaned_outputs['labels'].append("CONTROLLED_VAR")
                for label_pair in pred_controls:
                    cleaned_outputs['predictions'].append("CONTROLLED_VAR")
                    cleaned_outputs['labels'].append("O")
                for label_pair in label_measured:
                    if label_pair in pred_measured:
                        cleaned_outputs['predictions'].append("MEASURED_VAR")
                        cleaned_outputs['labels'].append("MEASURED_VAR")
                        pred_measured.remove(label_pair)
                    if label_pair not in pred_measured:
                        cleaned_outputs['predictions'].append("O")
                        cleaned_outputs['labels'].append("MEASURED_VAR")
                for label_pair in pred_measured:
                    cleaned_outputs['predictions'].append("MEASURED_VAR")
                    cleaned_outputs['labels'].append("O")
            print(classification_report(np.array(cleaned_outputs['labels']),
                                        np.array(cleaned_outputs['predictions']),
                                        labels=self.roles_labels
                                        )
                  )
            return np.array(cleaned_outputs['labels']),np.array(cleaned_outputs['predictions'])

        elif self.task == 'ner':
            cleaned_outputs = {'predictions': [], 'labels': []}
            for prediction, label in tqdm(zip(predictions, labels)):
                pred_controls, pred_measured, _ = get_labelled_data(prediction, separators=separators)
                label_controls, label_measured, _ = get_labelled_data(label, separators=separators)
                pred_controls = list(set(pred_controls))
                pred_measured = list(set(pred_measured))
                all_preds = list(itertools.chain(pred_controls, pred_measured))
                all_labels = list(itertools.chain(label_controls, label_measured))
                for label_pair in all_labels:
                    cleaned_outputs['labels'].append(label_pair[0])
                    if label_pair in all_preds:
                        cleaned_outputs['predictions'].append(label_pair[0])
                        all_preds.remove(label_pair)
                    else:
                        cleaned_outputs['predictions'].append("O")
                for label_pair in all_preds:
                    cleaned_outputs['predictions'].append(label_pair[0])
                    cleaned_outputs['labels'].append("O")
            print(classification_report(np.array(cleaned_outputs['labels']),
                                        np.array(cleaned_outputs['predictions']),
                                        labels=self.ner_labels
                                        )
                  )
            return np.array(cleaned_outputs['labels']), np.array(cleaned_outputs['predictions'])

        elif self.task == 'experiment':
            jaccard_distance = []
            for prediction, label in tqdm(zip(predictions, labels)):
                _, _, pred_experiment = get_labelled_data(prediction, separators=separators)
                _, _, label_experiment = get_labelled_data(label, separators=separators)
                if type(pred_experiment) == list:
                    pred_experiment = " ".join(pred_experiment)
                if type(label_experiment) == list:
                    label_experiment = " ".join(label_experiment)
                jaccard_distance.append(textdistance.jaccard(
                    label_experiment.split(),
                    pred_experiment.split())
                )
            print(f"""The Average Jaccard Distance experiment strings: {np.array(jaccard_distance).mean()}""")
            return np.array(jaccard_distance).mean()
        else:
            pass
