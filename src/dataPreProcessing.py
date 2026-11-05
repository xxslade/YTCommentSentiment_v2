
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class CommentDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(
            labels.values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encodings.items()
            if key in ["input_ids", "attention_mask"]
        }

        item["labels"] = self.labels[index]
        return item


def prepareData(xTrain, yTrain, xVal, yVal, xTest, yTest,
                modelName="distilbert-base-uncased",
                maxLength=128, batchSize=8):

    tokenizer = AutoTokenizer.from_pretrained(modelName)

    trainEncodings = tokenizer(
        xTrain.tolist(), truncation=True,
        padding="max_length", max_length=maxLength,
        return_tensors="pt"
    )

    valEncodings = tokenizer(
        xVal.tolist(), truncation=True,
        padding="max_length", max_length=maxLength,
        return_tensors="pt"
    )

    testEncodings = tokenizer(
        xTest.tolist(), truncation=True,
        padding="max_length", max_length=maxLength,
        return_tensors="pt"
    )

    trainDataset = CommentDataset(trainEncodings, yTrain)
    valDataset = CommentDataset(valEncodings, yVal)
    testDataset = CommentDataset(testEncodings, yTest)

    trainLoader = DataLoader(
        trainDataset, batch_size=batchSize, shuffle=True
    )
    valLoader = DataLoader(
        valDataset, batch_size=batchSize, shuffle=False
    )
    testLoader = DataLoader(
        testDataset, batch_size=batchSize, shuffle=False
    )

    return tokenizer, trainLoader, valLoader, testLoader
