"""Train both NLP and Vision models."""
from nlp_pipeline import fine_tune

# This will train the T5 model and save it to artifacts/nlp/t5_small_finetuned
fine_tune(
    num_train_samples=200,  # Number of training samples
    num_eval_samples=50,  # Number of validation samples
    num_train_epochs=3,  # Number of epochs (increase for better results)
    per_device_train_batch_size=4,
    learning_rate=5e-5
)
