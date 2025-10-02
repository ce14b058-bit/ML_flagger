
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




from Model_small_morenodes_xavierinit import Flagger
from Training import train_loop,test_loop,train_accuracy
from matplotlib import pyplot as plt
import pickle



if torch.cuda.is_available():
    print("GPU is available")
else:
    print("Running on CPU")

class Load_dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#ReadingData
# Base folder = current script directory
base_path = os.path.dirname(__file__)

# Data folders
file_path = os.path.join(base_path, "GeneratedData")
sub_folder1 = "AllCurveData"
sub_folder2 = "AllCurveData_noCons"

# Scaler and Model paths
scaler_path = os.path.join(base_path, "Scalers_10")
model_path  = os.path.join(base_path, "Models_10")

# Model type
model_type = "_varied"


anet_10k = pd.read_excel(file_path+ sub_folder1 + 'Anet_noOutlier_10_10000.xlsx')
ci_10k = pd.read_excel(file_path + sub_folder1 + 'Ci_noOutlier_10_10000.xlsx')
id_10k = pd.read_excel(file_path + sub_folder1 + 'ID_noOutlier_10_10000.xlsx')

anet_5k_ID2 = pd.read_excel(file_path+ sub_folder1 + 'Anet_noOutlier_10_5000_ID2.xlsx')
ci_5k_ID2 = pd.read_excel(file_path + sub_folder1 + 'Ci_noOutlier_10_5000_ID2.xlsx')
id_5k_ID2 = pd.read_excel(file_path + sub_folder1 + 'ID_noOutlier_10_5000_ID2.xlsx')

anet_5k_ID3 = pd.read_excel(file_path+ sub_folder1 + 'Anet_noOutlier_10_5000_ID3.xlsx')
ci_5k_ID3 = pd.read_excel(file_path + sub_folder1 + 'Ci_noOutlier_10_5000_ID3.xlsx')
id_5k_ID3 = pd.read_excel(file_path + sub_folder1 + 'ID_noOutlier_10_5000_ID3.xlsx')

anet_nocons = pd.read_excel(file_path+ sub_folder2 + 'Anet_noOutlier_10_10000_noCons.xlsx')
ci_nocons = pd.read_excel(file_path + sub_folder2 + 'Ci_noOutlier_10_10000_noCons.xlsx')
id_nocons = pd.read_excel(file_path + sub_folder2 + 'ID_noOutlier_10_10000_noCons.xlsx')


anet = pd.concat([anet_10k,anet_5k_ID2,anet_5k_ID3,anet_nocons],axis = 1)
ci = pd.concat([ci_10k,ci_5k_ID2,ci_5k_ID3,ci_nocons],axis = 1)
id = pd.concat([id_10k,id_5k_ID2,id_5k_ID3,id_nocons],axis = 1)


n = len(ci.columns)
ci.columns = np.linspace(0,n-1,n)
id.columns = ci.columns
anet.columns = ci.columns
ci_anet = pd.concat([ci,anet],axis = 0,ignore_index=True)





#Flattening
data_vectors = np.array([[ci.iloc[i,j]] + [anet.iloc[i,j]] + list(ci_anet.iloc[:,j]) for j in range(len(ci_anet.columns)) for i in range(len(id))])


#Vectorisation
id_values = np.array([id.iloc[i,j] for j in range(len(id.columns)) for i in range(len(id))])
id_onehot = np.array([np.eye(3)[id_values[i]-1] for i in range(len(id_values))])

#Saving Processed data
processed_file_path = 'C:/Users/vsand/OneDrive/Desktop/Project/C3_Parameter_Estimation_Paper_ML/ML_flagger/Processed_data/data_10/'
os.makedirs(processed_file_path, exist_ok=True)
with open(processed_file_path+'data_vectors' + '.pk1','wb') as f:
    pickle.dump(data_vectors,f)

with open(processed_file_path+'ID_onehot' + '.pk1','wb') as f:
    pickle.dump(id_onehot,f)


#Test - Train Split
X_train_raw,X_test_raw,y_train,y_test = train_test_split(data_vectors,id_onehot,shuffle=True)



# Standardizing data
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

with open(scaler_path+'scaler_10k' + model_type+'.pk1','wb') as f:
    pickle.dump(scaler,f)



# Convert to 2D PyTorch tensors


training_set = Load_dataset(X_train,y_train)
testing_set = Load_dataset(X_test,y_test)

#Model
model = Flagger()

#Training
n_epochs = 100
learning_rate = 0.001
batchsize = 300
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

#Batching
train_loader = DataLoader(training_set,batch_size=batchsize,shuffle=True)
test_loader = DataLoader(testing_set)

epochs = []
history_loss = []
history_accuracy = []
for epoch in range(n_epochs):
    print(f"epoch:{epoch+1}\n")
    train_loop(train_loader,model,loss_fn,optimizer)
    print(f'Training Accuracy:{train_accuracy(train_loader,model)}%')
    a,b = test_loop(test_loader,model,loss_fn)
    history_loss.append(a*100)
    history_accuracy.append(b)
    epochs.append(epoch+1)


model_state_dict = model.state_dict()

torch.save(model_state_dict,model_path + 'My_model'+ model_type+'.pth')

plt.plot(epochs,history_accuracy)
plt.plot(epochs,history_loss)
plt.show()