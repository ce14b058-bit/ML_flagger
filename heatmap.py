import os
import pickle
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
from Model_small_morenodes_xavierinit import Flagger  # make sure this is in the same folder or in PYTHONPATH

# --- 1. Define relative paths ---
base_dir = os.path.dirname(os.path.abspath(__file__))  # folder where this script is located

processed_file_path = os.path.join(base_dir, 'Processed_data', 'data_10')
scaler_path = os.path.join(base_dir, 'Scalers_10')
model_path = os.path.join(base_dir, 'Models_10')

model_type = '_varied'  # same as used during training

# --- 2. Load processed data ---
with open(os.path.join(processed_file_path,'data_vectors.pk1'), 'rb') as f:
    data_vectors = pickle.load(f)

with open(os.path.join(processed_file_path,'ID_onehot.pk1'), 'rb') as f:
    id_onehot = pickle.load(f)

# --- 3. Load scaler and model ---
with open(os.path.join(scaler_path,'scaler_10k'+model_type+'.pk1'), 'rb') as f:
    scaler = pickle.load(f)

model = Flagger()
model.load_state_dict(torch.load(os.path.join(model_path,'My_model'+model_type+'.pth')))
model.eval()  # set to evaluation mode

# --- 4. Standardize features ---
X_scaled = scaler.transform(data_vectors)
y = id_onehot

# --- 5. Sample 2000 examples ---
sample_size = 10000
total_samples = X_scaled.shape[0]
sample_size = min(sample_size, total_samples)

sample_indices = random.sample(range(total_samples), sample_size)
X_sample = X_scaled[sample_indices]
y_sample = y[sample_indices]

X_sample_tensor = torch.tensor(X_sample, dtype=torch.float32)

# --- 6. Make predictions ---
with torch.no_grad():
    outputs = model(X_sample_tensor)
    preds = torch.argmax(outputs, axis=1).numpy()

# Convert one-hot labels to class indices
y_true = np.argmax(y_sample, axis=1)

# --- 7. Confusion matrix ---
cm = confusion_matrix(y_true, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,2,3])
disp.plot(cmap='Blues', xticks_rotation='horizontal')
plt.title('Confusion Matrix (10k Sampled Data)')
plt.show()
