import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Determine available compute device
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {compute_device}")

# Helper function to parse space-delimited string columns
def parse_space_delimited_values(value):
    """Convert space-separated strings to numpy arrays"""
    if isinstance(value, str) and ' ' in value:
        return np.array([float(x) for x in value.split()])
    return value

# Load dataset from CSV file
racing_data = pd.read_csv('New DataSet copy.csv')

# Display dataset metadata
print("\nDataset Overview:")
print(f"Total samples: {len(racing_data)}")
print(f"Total features: {len(racing_data.columns)}")
print("\nFeature columns:", racing_data.columns.tolist())

# Data preparation section
print("\nPreparing data for model training...")

# Define model input and output features
model_inputs = ['Angle', 'CurLapTime', 'Damage', 'DistFromStart', 'DistRaced', 'WheelSpinVel', 'Z',
               'Fuel', 'LastLapTime', 'RPM', 'Speed X', 'Speed Y', 'Speed Z', 'TrackPos', 'Focus2', 'Gear2', 'RacePos']

# Categorize output features
binary_targets = ['Acceleration', 'Brake']  # Binary classification targets
continuous_targets = ['Meta', 'Focus', 'Gear', 'Steer', 'Clutch']  # Regression targets
model_outputs = continuous_targets + binary_targets

# Validate feature availability
missing_input_cols = [col for col in model_inputs if col not in racing_data.columns]
missing_output_cols = [col for col in model_outputs if col not in racing_data.columns]

if missing_input_cols or missing_output_cols:
    print("Warning: Dataset is missing some expected columns:")
    if missing_input_cols:
        print(f"Missing input features: {missing_input_cols}")
    if missing_output_cols:
        print(f"Missing output features: {missing_output_cols}")

    # Filter to only available columns
    model_inputs = [col for col in model_inputs if col in racing_data.columns]
    model_outputs = [col for col in model_outputs if col in racing_data.columns]
    binary_targets = [col for col in binary_targets if col in racing_data.columns]
    continuous_targets = [col for col in continuous_targets if col in racing_data.columns]

    print("\nUsing available features:")
    print(f"Input features: {model_inputs}")
    print(f"Output features: {model_outputs}")

# Function to expand array-type columns into multiple columns
def expand_array_columns(dataframe, column_name):
    """Convert array-type columns into multiple single-value columns"""
    if dataframe[column_name].dtype == 'object':
        sample_value = dataframe[column_name].dropna().iloc[0]
        if isinstance(sample_value, str) and ' ' in sample_value:
            array_size = len(sample_value.split())
            for i in range(array_size):
                new_col = f"{column_name}_{i}"
                dataframe[new_col] = dataframe[column_name].apply(
                    lambda x: float(x.split()[i]) if isinstance(x, str) else np.nan
                )
            if column_name in model_inputs:
                model_inputs.remove(column_name)
                model_inputs.extend([f"{column_name}_{i}" for i in range(array_size)])
            if column_name in model_outputs:
                model_outputs.remove(column_name)
                model_outputs.extend([f"{column_name}_{i}" for i in range(array_size)])
    return dataframe

# Process potential array columns
for feature in list(model_inputs) + list(model_outputs):
    if feature in racing_data.columns:
        racing_data = expand_array_columns(racing_data, feature)

print("\nAfter feature expansion:")
print(f"Input features: {model_inputs}")
print(f"Continuous targets: {continuous_targets}")
print(f"Binary targets: {binary_targets}")

# Handle missing data
missing_values = racing_data[model_inputs + model_outputs].isnull().sum()
print("\nMissing value counts:")
print(missing_values[missing_values > 0])

# Impute missing values
if missing_values.sum() > 0:
    print("Imputing missing values with median...")
    for col in model_inputs + model_outputs:
        if racing_data[col].isnull().sum() > 0:
            racing_data[col] = racing_data[col].fillna(racing_data[col].median())

# Process binary targets to ensure proper 0/1 encoding
for col in binary_targets:
    unique_vals = sorted(racing_data[col].unique())
    print(f"Unique values in {col}: {unique_vals}")

    if len(unique_vals) > 2:
        print(f"Converting {col} to binary with threshold detection")
        threshold = 0.5
        for i in range(len(unique_vals) - 1):
            if unique_vals[i+1] - unique_vals[i] > threshold:
                threshold = (unique_vals[i+1] + unique_vals[i]) / 2
                break

        print(f"Using threshold {threshold} for {col}")
        racing_data[col] = (racing_data[col] >= threshold).astype(int)
    elif len(unique_vals) == 2:
        racing_data[col] = (racing_data[col] == max(unique_vals)).astype(int)

# Prepare feature and target matrices
X = racing_data[model_inputs].values
y_cont = racing_data[continuous_targets].values if continuous_targets else np.zeros((len(racing_data), 0))
y_bin = racing_data[binary_targets].values

# Split data into training and test sets
X_train, X_test, y_cont_train, y_cont_test, y_bin_train, y_bin_test = train_test_split(
    X, y_cont, y_bin, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Feature scaling
print("\nApplying feature scaling...")
feature_scaler = MinMaxScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Scale continuous targets if present
if y_cont.shape[1] > 0:
    target_scaler = MinMaxScaler()
    y_cont_train_scaled = target_scaler.fit_transform(y_cont_train)
    y_cont_test_scaled = target_scaler.transform(y_cont_test)
else:
    y_cont_train_scaled = y_cont_train
    y_cont_test_scaled = y_cont_test

# Visualization of feature distributions
num_features = len(model_inputs)
cols_per_row = 4
rows_needed = (num_features + cols_per_row - 1) // cols_per_row

plt.figure(figsize=(20, 5*rows_needed))
for i, feature in enumerate(model_inputs):
    plt.subplot(rows_needed, cols_per_row, i+1)
    sns.histplot(racing_data[feature], kde=True)
    plt.title(feature)
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()

# Visualization of binary target distributions
plt.figure(figsize=(15, 5))
for i, feature in enumerate(binary_targets):
    plt.subplot(1, len(binary_targets), i+1)
    sns.countplot(x=racing_data[feature])
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig('binary_target_distributions.png')
plt.show()

# Feature correlation visualization
plt.figure(figsize=(14, 10))
correlation_matrix = racing_data[model_inputs].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig('feature_correlations.png')
plt.show()

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)

y_cont_train_tensor = torch.FloatTensor(y_cont_train_scaled) if y_cont_train_scaled.shape[1] > 0 else None
y_cont_test_tensor = torch.FloatTensor(y_cont_test_scaled) if y_cont_test_scaled.shape[1] > 0 else None

y_bin_train_tensor = torch.FloatTensor(y_bin_train)
y_bin_test_tensor = torch.FloatTensor(y_bin_test)

# Dataset and DataLoader configuration
batch_size = 128

class RacingDataset(torch.utils.data.Dataset):
    """Custom dataset for racing control prediction"""
    def __init__(self, features, continuous_targets=None, binary_targets=None):
        self.features = features
        self.continuous_targets = continuous_targets
        self.binary_targets = binary_targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature_vector = self.features[idx]
        target_list = []
        
        if self.continuous_targets is not None:
            target_list.append(self.continuous_targets[idx])
            
        if self.binary_targets is not None:
            for i in range(self.binary_targets.shape[1]):
                target_list.append(self.binary_targets[idx, i:i+1])
        
        return feature_vector, target_list

# Create datasets and data loaders
train_data = RacingDataset(X_train_tensor, y_cont_train_tensor, y_bin_train_tensor)
test_data = RacingDataset(X_test_tensor, y_cont_test_tensor, y_bin_test_tensor)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Neural network architecture
class RacingControlModel(nn.Module):
    """Multi-output neural network for racing control prediction"""
    def __init__(self, input_size, continuous_output_size, binary_output_count):
        super(RacingControlModel, self).__init__()
        
        # Shared feature extraction layers
        self.shared_network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # Continuous output branch
        self.continuous_output_size = continuous_output_size
        if continuous_output_size > 0:
            self.continuous_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.1),
                nn.Linear(64, continuous_output_size)
            )
        
        # Binary output branches
        self.binary_output_count = binary_output_count
        self.binary_heads = nn.ModuleList()
        
        for _ in range(binary_output_count):
            head = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            self.binary_heads.append(head)
    
    def forward(self, x):
        shared_features = self.shared_network(x)
        
        outputs = []
        
        if self.continuous_output_size > 0:
            continuous_output = self.continuous_head(shared_features)
            outputs.append(continuous_output)
        
        for head in self.binary_heads:
            binary_output = head(shared_features)
            outputs.append(binary_output)
            
        return outputs

# Initialize model
racing_model = RacingControlModel(X_train_scaled.shape[1], y_cont_train_scaled.shape[1], len(binary_targets))
racing_model = racing_model.to(compute_device)

print("\nModel Architecture:")
print(racing_model)

# Define loss functions
regression_loss = nn.MSELoss()
classification_loss = nn.BCELoss()

# Configure optimizer
model_optimizer = optim.Adam(racing_model.parameters(), lr=0.001)

# Learning rate scheduler
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    model_optimizer, mode='min', factor=0.5, patience=8, min_lr=0.00001
)

# Early stopping configuration
early_stop_patience = 15
best_validation_loss = float('inf')
early_stop_counter = 0
best_model_weights = None

# Training history tracking
training_history = {
    'training_loss': [],
    'validation_loss': [],
    'binary_metrics': {feature: {'train': [], 'val': []} for feature in binary_targets}
}

def run_training_epoch(model, data_loader, optimizer, device):
    """Execute one training epoch"""
    model.train()
    total_loss = 0.0
    correct_classifications = [0] * len(binary_targets)
    total_classifications = [0] * len(binary_targets)
    
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        device_targets = [t.to(device) for t in targets]
        
        # Forward pass
        predictions = model(inputs)
        
        # Calculate composite loss
        loss = 0
        
        # Regression loss
        if y_cont_train_scaled.shape[1] > 0:
            reg_loss = regression_loss(predictions[0], device_targets[0]) * 1.0
            loss += reg_loss
        
        # Classification losses
        for i in range(len(binary_targets)):
            pred_idx = i + (1 if y_cont_train_scaled.shape[1] > 0 else 0)
            class_loss = classification_loss(predictions[pred_idx], device_targets[pred_idx]) * 2.0
            loss += class_loss
            
            # Track classification accuracy
            class_predictions = (predictions[pred_idx] > 0.5).float()
            correct_classifications[i] += (class_predictions == device_targets[pred_idx]).sum().item()
            total_classifications[i] += device_targets[pred_idx].size(0)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
    
    epoch_loss = total_loss / len(data_loader.dataset)
    classification_accuracies = [correct / total for correct, total in zip(correct_classifications, total_classifications)]
    
    return epoch_loss, classification_accuracies

def evaluate_model(model, data_loader, device):
    """Evaluate model performance on validation/test data"""
    model.eval()
    total_loss = 0.0
    correct_classifications = [0] * len(binary_targets)
    total_classifications = [0] * len(binary_targets)
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            device_targets = [t.to(device) for t in targets]
            
            # Forward pass
            predictions = model(inputs)
            
            # Calculate composite loss
            loss = 0
            
            # Regression loss
            if y_cont_train_scaled.shape[1] > 0:
                reg_loss = regression_loss(predictions[0], device_targets[0]) * 1.0
                loss += reg_loss
            
            # Classification losses
            for i in range(len(binary_targets)):
                pred_idx = i + (1 if y_cont_train_scaled.shape[1] > 0 else 0)
                class_loss = classification_loss(predictions[pred_idx], device_targets[pred_idx]) * 2.0
                loss += class_loss
                
                # Track classification accuracy
                class_predictions = (predictions[pred_idx] > 0.5).float()
                correct_classifications[i] += (class_predictions == device_targets[pred_idx]).sum().item()
                total_classifications[i] += device_targets[pred_idx].size(0)
            
            total_loss += loss.item() * inputs.size(0)
    
    eval_loss = total_loss / len(data_loader.dataset)
    classification_accuracies = [correct / total for correct, total in zip(correct_classifications, total_classifications)]
    
    return eval_loss, classification_accuracies

# Create validation split from training data
train_subset_size = int(0.8 * len(train_data))
val_subset_size = len(train_data) - train_subset_size
train_subset, val_subset = torch.utils.data.random_split(train_data, [train_subset_size, val_subset_size])

train_subset_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_subset_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Model training loop
print("\nStarting model training...")
training_start = time.time()

for epoch in range(100):  # Maximum 100 epochs with early stopping
    train_loss, train_accuracies = run_training_epoch(racing_model, train_subset_loader, model_optimizer, compute_device)
    val_loss, val_accuracies = evaluate_model(racing_model, val_subset_loader, compute_device)
    
    # Adjust learning rate
    lr_scheduler.step(val_loss)
    
    # Record training history
    training_history['training_loss'].append(train_loss)
    training_history['validation_loss'].append(val_loss)
    
    for i, feature in enumerate(binary_targets):
        training_history['binary_metrics'][feature]['train'].append(train_accuracies[i])
        training_history['binary_metrics'][feature]['val'].append(val_accuracies[i])
    
    # Print epoch summary
    print(f'Epoch {epoch+1}/100 | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}')
    
    for i, feature in enumerate(binary_targets):
        print(f'{feature} Accuracy - Train: {train_accuracies[i]:.4f}, Val: {val_accuracies[i]:.4f}')
    
    # Early stopping logic
    if val_loss < best_validation_loss:
        best_validation_loss = val_loss
        early_stop_counter = 0
        best_model_weights = racing_model.state_dict().copy()
        torch.save(racing_model.state_dict(), 'racing_model_weights.pth')
        print("Saved improved model weights")
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")
        if early_stop_counter >= early_stop_patience:
            print("Triggering early stopping")
            break

training_duration = time.time() - training_start
print(f"\nTraining completed in {training_duration:.2f} seconds")

# Load best model weights
racing_model.load_state_dict(torch.load('racing_model_weights.pth'))

# Plot training history
plt.figure(figsize=(15, 10))

# Loss plot
plt.subplot(2, 2, 1)
plt.plot(training_history['training_loss'], label='Training Loss')
plt.plot(training_history['validation_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Binary accuracy plots
for i, feature in enumerate(binary_targets):
    plt.subplot(2, 2, i+2)
    plt.plot(training_history['binary_metrics'][feature]['train'], label='Training Accuracy')
    plt.plot(training_history['binary_metrics'][feature]['val'], label='Validation Accuracy')
    plt.title(f'{feature} Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.savefig('model_training_history.png')
plt.show()

# Final model evaluation
racing_model.eval()
test_loss, test_accuracies = evaluate_model(racing_model, test_loader, compute_device)
print(f"\nTest Set Loss: {test_loss:.4f}")

for i, feature in enumerate(binary_targets):
    print(f"{feature} Test Accuracy: {test_accuracies[i]:.4f}")

# Generate predictions for detailed evaluation
continuous_predictions = []
binary_predictions = []

racing_model.eval()
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(compute_device)
        outputs = racing_model(inputs)
        
        # Store predictions
        if y_cont_test_scaled.shape[1] > 0:
            if len(continuous_predictions) == 0:
                continuous_predictions = outputs[0].cpu().numpy()
            else:
                continuous_predictions = np.vstack((continuous_predictions, outputs[0].cpu().numpy()))
        
        # Store binary predictions
        start_index = 1 if y_cont_test_scaled.shape[1] > 0 else 0
        for i in range(len(binary_targets)):
            if len(binary_predictions) <= i:
                binary_predictions.append(outputs[start_index + i].cpu().numpy())
            else:
                binary_predictions[i] = np.vstack((binary_predictions[i], outputs[start_index + i].cpu().numpy()))

# Evaluate continuous predictions
if y_cont_test_scaled.shape[1] > 0:
    predicted_continuous = target_scaler.inverse_transform(continuous_predictions)
    actual_continuous = target_scaler.inverse_transform(y_cont_test_scaled)

    print("\nRegression Metrics:")
    for i, feature in enumerate(continuous_targets):
        mse = mean_squared_error(actual_continuous[:, i], predicted_continuous[:, i])
        mae = mean_absolute_error(actual_continuous[:, i], predicted_continuous[:, i])
        r2 = r2_score(actual_continuous[:, i], predicted_continuous[:, i])

        print(f"{feature}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")

# Evaluate binary predictions
print("\nClassification Metrics:")
for i, feature in enumerate(binary_targets):
    # Get predictions for current binary target
    pred_probs = binary_predictions[i]
    pred_classes = (pred_probs > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_bin_test[:, i], pred_classes)
    try:
        precision = precision_score(y_bin_test[:, i], pred_classes)
        recall = recall_score(y_bin_test[:, i], pred_classes)
        f1 = f1_score(y_bin_test[:, i], pred_classes)
    except:
        precision = recall = f1 = float('nan')

    # Additional regression metrics for comparison
    mse = mean_squared_error(y_bin_test[:, i], pred_probs)
    mae = mean_absolute_error(y_bin_test[:, i], pred_probs)
    r2 = r2_score(y_bin_test[:, i], pred_probs)

    print(f"{feature}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  MSE: {mse:.4f} (probability)")
    print(f"  MAE: {mae:.4f} (probability)")
    print(f"  R²: {r2:.4f} (probability)")

    # Classification report
    print(f"\nClassification Report for {feature}:")
    print(classification_report(y_bin_test[:, i], pred_classes))

    # Visualize prediction distributions
    plt.figure(figsize=(15, 5))
    
    # Plot probability distributions by actual class
    plt.subplot(1, 2, 2)
    class_0_probs = pred_probs[y_bin_test[:, i] == 0]
    class_1_probs = pred_probs[y_bin_test[:, i] == 1]

    sns.histplot(class_0_probs, color='blue', alpha=0.5, bins=50, label='Actual=0')
    sns.histplot(class_1_probs, color='red', alpha=0.5, bins=50, label='Actual=1')
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.title(f'{feature}: Predicted Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{feature}_probability_distribution.png')
    plt.show()

# Prediction function for new data
def predict_racing_controls(input_values, model, device):
    """
    Generate control predictions for new racing scenarios
    
    Args:
        input_values (dict): Dictionary of input feature values
        model: Trained PyTorch model
        device: Computation device
    
    Returns:
        dict: Predicted control parameters
    """
    # Prepare input array
    input_array = np.zeros((1, len(model_inputs)))
    
    for i, feature in enumerate(model_inputs):
        if feature in input_values:
            input_array[0, i] = input_values[feature]
    
    # Scale input
    scaled_input = feature_scaler.transform(input_array)
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(scaled_input).to(device)
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Format results
    results = {}
    
    # Process continuous outputs
    if len(continuous_targets) > 0:
        scaled_continuous = predictions[0].cpu().numpy()
        continuous_values = target_scaler.inverse_transform(scaled_continuous)
        
        for i, feature in enumerate(continuous_targets):
            results[feature] = continuous_values[0, i]
    
    # Process binary outputs
    output_start = 1 if len(continuous_targets) > 0 else 0
    for i, feature in enumerate(binary_targets):
        binary_prob = predictions[output_start + i].cpu().numpy()
        results[feature] = int(binary_prob[0, 0] > 0.5)
    
    return results

# Example prediction
print("\nSample Prediction:")
sample_input = {feature: racing_data[feature].iloc[0] for feature in model_inputs}
control_prediction = predict_racing_controls(sample_input, racing_model, compute_device)
print("Input Features:", {k: round(float(v), 4) if isinstance(v, (int, float, np.number)) else v
                for k, v in sample_input.items()})
print("Predicted Controls:", {k: round(float(v), 4) if isinstance(v, float) else int(v)
                     for k, v in control_prediction.items()})

# Save complete model and configuration
torch.save({
    'model_state': racing_model.state_dict(),
    'input_features': model_inputs,
    'continuous_targets': continuous_targets,
    'binary_targets': binary_targets,
}, 'model.pth')

# Save scalers
import pickle
with open('scalers.pkl', 'wb') as f:
    pickle.dump({
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler if y_cont.shape[1] > 0 else None
    }, f)

print("\nTraining and evaluation complete!")
print(f"Model saved as 'model.pth'")
print(f"Scalers saved as 'scalers.pkl'")