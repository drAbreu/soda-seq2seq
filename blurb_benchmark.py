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
            modified_data['targets'].append(str(t))

        return modified_data

