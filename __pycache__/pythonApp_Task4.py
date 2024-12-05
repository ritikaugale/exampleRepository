import pickle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import numpy as np

# Function to preprocess and load data
def load_data(file_path):
    try:
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        if np.isnan(data).any():
            raise ValueError("Dataset contains NaN values. Please preprocess the data.")
        inputs = data[:, :-1]
        targets = data[:, -1:]
        return inputs, targets
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

# Function to create training and testing datasets
def prepare_datasets(inputs, targets, split_ratio=0.8):
    dataset_size = len(inputs)
    split_index = int(dataset_size * split_ratio)
    train_inputs, test_inputs = inputs[:split_index], inputs[split_index:]
    train_targets, test_targets = targets[:split_index], targets[split_index:]

    train_ds = SupervisedDataSet(inputs.shape[1], 1)
    test_ds = SupervisedDataSet(inputs.shape[1], 1)

    for i in range(len(train_inputs)):
        train_ds.addSample(train_inputs[i], train_targets[i])
    for i in range(len(test_inputs)):
        test_ds.addSample(test_inputs[i], test_targets[i])

    return train_ds, test_ds

# Function to create a feedforward network
def create_network(input_size, hidden_size, output_size):
    return buildNetwork(input_size, hidden_size, output_size, bias=True)

# Function to train the model
def train_model(network, train_data, learningrate=0.01, max_epochs=10):
    trainer = BackpropTrainer(network, train_data, learningrate=learningrate)
    for epoch in range(max_epochs):
        error = trainer.train()
        print(f"Epoch {epoch + 1}/{max_epochs}, Training Error: {error}")

# Function to save the model
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}.")

# Function to load the model
def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}.")
    return model

# Main function
if __name__ == "__main__":
    # Load data
    file_path = "dataset03.csv"  # Ensure this file exists
    try:
        inputs, targets = load_data(file_path)
        print(f"Dataset loaded successfully with shape: Inputs {inputs.shape}, Targets {targets.shape}")
    except Exception as e:
        print("Failed to load dataset.")
        exit()

    # Prepare datasets
    train_data, test_data = prepare_datasets(inputs, targets)

    # Create and train the first model
    input_size = train_data.indim
    hidden_size = 5
    output_size = train_data.outdim
    ann1 = create_network(input_size, hidden_size, output_size)

    print("Training the first ANN model...")
    train_model(ann1, train_data, learningrate=0.005, max_epochs=100)

    # Save the first model
    model_file = "ann_model.pkl"
    save_model(ann1, model_file)

    # Load the saved model into a second instance
    ann2 = load_model(model_file)

    # Test both models on the same data entries
    sample_data = inputs[:2]  # Use the first two inputs for demonstration
    print("\nTesting model activations with sample data:")

    for idx, sample in enumerate(sample_data):
        result1 = ann1.activate(sample)
        result2 = ann2.activate(sample)
        print(f"Data Entry {idx + 1}:")
        print(f"  Model 1 Activation: {result1}")
        print(f"  Model 2 Activation: {result2}")