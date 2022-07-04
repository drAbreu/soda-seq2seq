import datasets


class BlurbBenchmark:
    def __init__(self,
                 dataset_name: str = "BC5CDR-chem-IOB",
                 task_name: str = "Tag entities",
                 model: str = "facebook/bart-base"):
        self.base_data = "EMBO/BLURB"
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.model = model
        self.dataset = datasets.load_dataset(self.base_data, self.dataset_name)
        self.id2label, self.label2id = self._get_data_labels()

        self.dataset_seq2seq = self.dataset.map(
                                        self._convert_into_seq2seq,
                                        batched=True,
                                        remove_columns=["tokens", "ner_tags", "id"]
                                        )

    def _convert_into_seq2seq(self, examples):
        if self.task_name == "Tag entities":
            modified_data = self._ner_into_seq2seq(examples)
        else:
            raise NotImplemented
        return modified_data

    def _ner_into_seq2seq(self, examples):
        modified_data = {'inputs': [], 'targets': []}

        for i, t in zip(examples['tokens'], examples['ner_tags']):
            modified_data['inputs'].append(f"""{self.task_name}: {" ".join(i)}""")
            modified_data['targets'].append(self._get_text_labels(i, t))

        return modified_data

    def _get_data_labels(self):
        num_labels = self.dataset['train'].info.features['ner_tags'].feature.num_classes
        label_list = self.dataset['train'].info.features['ner_tags'].feature.names
        id2label, label2id = {}, {}
        for class_, label in zip(range(num_labels), label_list):
            id2label[class_] = label
            label2id[label] = class_
        print(f"\nTraining on {num_labels} features:")
        print(", ".join(label_list))
        return id2label, label2id

    def _get_text_labels(self, inputs, targets):
        output_string = ""
        for i, t in zip(inputs, targets):
            if t != 0:
                output_string += f"{self.id2label[t]}:{i} |separator| "
        if output_string == "":
            output_string = "No entities found"
        return output_string