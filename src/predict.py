import torch
import json
import os
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from model import loadModel

class CommentPredictor:
    def __init__(self, modelDir="outputs/finalModel"):
        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(modelDir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            modelDir
        )

        self.model.to(self.device)
        self.model.eval()

        with open(os.path.join(modelDir, "thresholds.json"), "r") as file:
            self.thresholds = json.load(file)

        self.labels = list(self.thresholds.keys())
    def predict(self, comment):
        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        inputIds = encoding["input_ids"].to(self.device)
        attentionMask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputIds,
                attention_mask=attentionMask
            )

        probabilities = torch.sigmoid(outputs.logits)[0].numpy()

        predictions = {}

        for i, label in enumerate(self.labels):
            predictions[label] = {
                "probability": float(probabilities[i]),
                "predicted": bool(probabilities[i] >= self.thresholds[label])
            }

        return predictions

if __name__ == "__main__":
    predictor = CommentPredictor()

    comment = "This is an amazing video! Keep up the great work."

    results = predictor.predict(comment)

    print("\nComment:", comment)
    print("\nPredictions:")

    for label, result in results.items():
        print(
            f"{label}: Probability={result['probability']:.4f}, "
            f"Predicted={result['predicted']}"
        )