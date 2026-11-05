
from transformers import AutoModelForSequenceClassification


def loadModel(modelName, labels):

    label2id = {
        label: index
        for index, label in enumerate(labels)
    }

    id2label = {
        index: label
        for index, label in enumerate(labels)
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        modelName,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        label2id=label2id,
        id2label=id2label
    )

    return model
