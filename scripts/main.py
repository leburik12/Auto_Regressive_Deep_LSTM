import torch
import torch.nn as nn
import random
from src.data_utils import DataManager
from src.model import DeepRNN
from src.trainer import DeepRNNTrainer


def main():

    CONFIG = {
        "url": "https://www.gutenberg.org/files/100/100-0.txt",
        "max_chars": 50000,
        "seq_length": 100,
        "batch_size": 32,
        "hidden_size": 128,
        "num_layers": 2,
        "learning_rate": 0.002,
        "dropout": 0.2,
        "epochs": 20,
        "temperature": 0.7,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    print(f"🚀 Initializing experiment on device: {CONFIG['device']}")

    dm = DataManager(CONFIG["url"], CONFIG["max_chars"])
    dataloader, dataset = dm.get_dataloader(CONFIG["seq_length"], CONFIG["batch_size"])
    
    vocab_size = len(dataset.chars)
    print(f"📊 Dataset loaded. Vocabulary size: {vocab_size}")

    model = DeepRNN(
        input_size=vocab_size,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=vocab_size,
        dropout=CONFIG["dropout"]
    )
    
    model.fetcher = dataset 
    model.seq_length = CONFIG["seq_length"]

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    trainer = DeepRNNTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=CONFIG["device"]
    )

    print(f"🏋️  Training Deep LSTM for {CONFIG['epochs']} epochs...")
    for epoch in range(CONFIG["epochs"]):
        avg_loss = trainer.train_epoch(dataloader)

        if (epoch + 1) % 5 == 0:
            print(f"✨ Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f}")

            start_prompts = [
                "The quantum field theory",
                "As a scientist at MIT,",
                "The fundamental law",
                "Mathematically, we can"
            ]
            prompt = random.choice(start_prompts)
            
            generated = trainer.generate_text(
                start_text=prompt,
                length=200,
                temperature=CONFIG["temperature"],
                vocab=dataset
            )

            print(f"\n🧪 Inference Sample (T={CONFIG['temperature']}):")
            print(f"{'-'*50}\n{generated}\n{'-'*50}\n")

if __name__ == "__main__":
    main()