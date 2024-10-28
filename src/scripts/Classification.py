import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import os
import random
import wandb

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# Load your data
def load_data(input_path):
    # For each in the path, read the data and concatenate it
    data_frames = []
    for file in os.listdir(input_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_path, file), delimiter=';', header = 0)
            data_frames.append(df)
    data = pd.concat(data_frames, ignore_index=True)
    return data

def calculate_averages_and_dispersion(data, data_percentage=5):
    df = data
    results = []
    for (sample, freq), group in df.groupby(['Sample', 'Frequency (GHz)']):
        window_size = max(1, int(len(group) * data_percentage / 100))
        print(f"Processing sample: {sample}, frequency: {freq} with window size: {window_size}")
        for start in range(0, len(group), window_size):
            window_data = group.iloc[start:start + window_size]
            mean_values = window_data[['LG (mV)', 'HG (mV)']].mean()
            std_deviation_values = window_data[['LG (mV)', 'HG (mV)']].std()
            variance_values = window_data[['LG (mV)', 'HG (mV)']].var()
            results.append({
                'Frequency (GHz)': freq,
                'LG (mV) mean': mean_values['LG (mV)'],
                'HG (mV) mean': mean_values['HG (mV)'],
                'LG (mV) std deviation': std_deviation_values['LG (mV)'],
                'HG (mV) std deviation': std_deviation_values['HG (mV)'],
                'LG (mV) variance': variance_values['LG (mV)'],
                'HG (mV) variance': variance_values['HG (mV)'],
                'Thickness (mm)': window_data['Thickness (mm)'].iloc[0],
                'Sample': sample,
            })
    results_df = pd.DataFrame(results)
    # results_df.to_csv(output_file, sep=';', index=False)
    # print(f"Processed {input_file} and saved to {output_file}")
    return results_df


# Preprocess your data
def preprocess_data(data):
    # Windowing the data
    data = calculate_averages_and_dispersion(data)

    # Assuming the last column is the target
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Encode the target variable if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    return X, y

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim  # Define hidden_dim as an attribute
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')

def run_inference(model, X, device):
    '''Run inference on a single sample'''
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
    return predicted

def main(input_path, seed=42):

    # Initialize W&B run
    run = wandb.init(project='classification', entity='wandb', reinit=True)

    # Set seed for reproducibility
    set_seed(seed)

    # Load and preprocess data
    data = load_data(input_path)
    X, y = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    # Reshape data to fit LSTM input requirements [samples, time steps, features]
    # Assuming each sample is a sequence of length 1 (time steps = 1)
    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss function, and optimizer
    input_dim = X_train.shape[2]
    hidden_dim = 50
    output_dim = len(torch.unique(y))
    model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

# Simplify the config data using a dictionary
    config = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001    
    }

    wandb.config.update(config)

    run.watch(model)

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx % args.log_interval == 0:
        # 4. Log metrics to visualize performance
            run.log({"loss": loss})

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device)

    # Evaluate the model
    evaluate_model(model, test_loader, device)

    # Finish the W&B run
    wandb.finish()

    # Save the model as onnx
    torch.onnx.export(model, X_train, 'lstm_model.onnx')
    # Save the model state dict
    torch.save(model.state_dict(), 'lstm_model.pth')

if __name__ == "__main__":

    input_path = '../../data/experiment_1_plastics/processed_27s/training_file/'  
    main(input_path)