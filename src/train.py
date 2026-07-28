from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


from transformers import Trainer


def get_training_args():
    return TrainingArguments(
        output_dir="../models/bert",

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=2e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        num_train_epochs=3,

        weight_decay=0.01,

        logging_steps=100,

        load_best_model_at_end=True,

        report_to="none",
    )


def create_trainer(
    model,
    training_args,
    train_dataset,
    val_dataset,
):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )


