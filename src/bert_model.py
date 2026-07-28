from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "bert-base-uncased"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )