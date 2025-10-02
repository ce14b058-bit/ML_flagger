import os
import torch
from Model_small_morenodes_xavierinit import Flagger
import numpy as np
import pickle
from matplotlib import pyplot as plt
from C3 import Photosynthesis_C3_ACi
from Optimize import photosynthesis_optimize
from pandas import DataFrame as df



# ---------------- Paths ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of this script

scaler_file = os.path.join(BASE_DIR, "scaler_10k_anetnoise.pk1")
model_path = os.path.join(BASE_DIR, "Models")
model_file = os.path.join(model_path, "My_model_anetnoise.pth")

# ---------------- Load Scaler ---------------- #
with open(scaler_file, "rb") as f:
    scaler = pickle.load(f)
   
def flag(data1, data2):
    data1 = np.array(data1)
    data2 = np.array(data2)
    data = np.array([[data1[i]] + [data2[i]] + list(data1) + list(data2) for i in range(len(data1))])
    scaled_data = scaler.transform(data)
    scaled_data_tensors = torch.tensor(scaled_data, dtype=torch.float)

    # Import model
    model = Flagger()
    load_state = torch.load(model_file, map_location="cpu")
    model.load_state_dict(load_state)
    model.eval()

    Out = model(scaled_data_tensors)
    flags = np.array([x.argmax(0).item() + 1 for x in Out])
    return Out.detach().numpy(), flags

ObsCi = [20.7,77.5,135,197,266,344,428,517,606,698,791]
ObsAnet = [-3.27,3.49,10.8,17.8,23.5,27.6,30.5,31.7,33,33.3,33.3]
Obsflags = [1,1,1,1,2,2,2,2,2,3,3]

weather = {
    'temperature' : [25],
    'pressure' : [101.325],
    'O2' : [210]
}

weather = df(weather)

skip = 1

Ci = ObsCi[:skip]+ObsCi[skip+1:11]
Anet = ObsAnet[:skip]+ObsAnet[skip+1:11]
flags = Obsflags[:skip]+Obsflags[skip+1:11]
observed = df(data=[Ci,Anet,flags],index=['Ci','Anet','flag']).T

photo_vec,n1,n2,n3 = photosynthesis_optimize(weather,observed)

prob,model_flags = flag(Ci,Anet)

predicted = df(data=[Ci,Anet,model_flags],index=['Ci','Anet','flag']).T

photo_vec_p,n1p,n2p,n3p = photosynthesis_optimize(weather,predicted)

Photosynthesis = {}
Photosynthesis['vcmax25'] = photo_vec[0] 
Photosynthesis['jmax25']= photo_vec[1] 
Photosynthesis['tpu25'] = photo_vec[2]
Photosynthesis['rd25']= photo_vec[3] 
Photosynthesis['gm25']= photo_vec[4]

Photosynthesis_pred = {}
Photosynthesis_pred['vcmax25'] = photo_vec_p[0] 
Photosynthesis_pred['jmax25']= photo_vec_p[1] 
Photosynthesis_pred['tpu25'] = photo_vec_p[2]
Photosynthesis_pred['rd25']= photo_vec_p[3] 
Photosynthesis_pred['gm25']= photo_vec_p[4]


weather_input = df()
weather_input.loc[:,'Ci'] = np.linspace(1,1000,50)
weather_input.loc[:,'pressure'] = weather.loc[0,'pressure']
weather_input.loc[:,'temperature'] = weather.loc[0,'temperature']
weather_input.loc[:,'O2'] = weather.loc[0,'O2']

photo_model = Photosynthesis_C3_ACi(weather_input,Photosynthesis)
photo_model_pred = Photosynthesis_C3_ACi(weather_input,Photosynthesis_pred)

LeafMassFlux,LeafState = photo_model.solve()
LeafMassflux_pred,LeafState_pred = photo_model_pred.solve()

# ---------------- Plotting ---------------- #
flag1_prob = [p[0]*100 for p in prob] 
flag2_prob = [p[1]*100 for p in prob] 
flag3_prob = [p[2]*100 for p in prob] 

fig,axes = plt.subplots(3,2)
# Probabilities
axes[0,0].plot(Ci,flag1_prob,label = 'flag1')
axes[0,0].plot(Ci,flag2_prob,label = 'flag2')
axes[0,0].plot(Ci,flag3_prob,label = 'flag3')
axes[0,0].scatter(Ci,flag1_prob)
axes[0,0].scatter(Ci,flag2_prob)
axes[0,0].scatter(Ci,flag3_prob)
axes[0,0].legend()
axes[0,0].set_xlabel('Ci')
axes[0,0].set_ylabel('Probability (predicted)')

axes[0,1].scatter(Ci,flags,color = 'red',label = 'observed')
axes[0,1].scatter(Ci,model_flags,marker = 'x',label = 'predicted')
axes[0,1].set_xlabel('Ci')
axes[0,1].set_ylabel('flag')
axes[0,1].legend()


temperature_kelvin = 25 + 273
Constants = {'R':8.314/1000}
gm = lambda p : p*np.exp(20.01 - (49600 / (Constants['R'] * 1000.0 * temperature_kelvin)))/(1+np.exp((1400 * temperature_kelvin - 437400) / (Constants['R'] * 1000.0 * temperature_kelvin))) #p4
gm1 = gm(Photosynthesis['gm25'])
gm1_pred = gm(Photosynthesis_pred['gm25'])

x = LeafState.loc[:,'cm']
ObsCc = np.array(Ci) - np.array(Anet)/gm1
axes[1,0].plot(x,(LeafMassFlux.loc[:,'ac']-LeafMassFlux.loc[:,'rd']), c = 'r',linewidth = 2,label = 'Rubisco')
axes[1,0].plot(x,(LeafMassFlux.loc[:,'aj']-LeafMassFlux.loc[:,'rd']),c = 'b',linewidth = 2,label = 'RuBP_regen')
axes[1,0].plot(x,(LeafMassFlux.loc[:,'ap']-LeafMassFlux.loc[:,'rd']),c= 'y',linewidth = 2,label = 'TPU')
axes[1,0].scatter(ObsCc,Anet,s=20,marker ='o',label = 'Aobs')
axes[1,0].set_xlabel('Cc ,Pa')
axes[1,0].set_ylabel('A')
axes[1,0].set_title('A/Cc Curve')
axes[1,0].legend()

x = LeafState_pred.loc[:,'cm']
ObsCc_pred = np.array(Ci) - np.array(Anet)/gm1_pred
axes[1,1].plot(x,(LeafMassflux_pred.loc[:,'ac']-LeafMassflux_pred.loc[:,'rd']), c = 'r',linewidth = 2,label = 'Rubisco')
axes[1,1].plot(x,(LeafMassflux_pred.loc[:,'aj']-LeafMassflux_pred.loc[:,'rd']),c = 'b',linewidth = 2,label = 'RuBP_regen')
axes[1,1].plot(x,(LeafMassflux_pred.loc[:,'ap']-LeafMassflux_pred.loc[:,'rd']),c= 'y',linewidth = 2,label = 'TPU')
axes[1,1].scatter(ObsCc_pred,Anet,s=20,marker ='o',label = 'Aobs')
axes[1,1].set_xlabel('Cc ,Pa')
axes[1,1].set_ylabel('A')
axes[1,1].set_title('A/Cc Curve predicted')
axes[1,1].legend()

x = LeafState.loc[:,'ci']
axes[2,0].plot(x,(LeafMassFlux.loc[:,'ac']-LeafMassFlux.loc[:,'rd']), c = 'r',linewidth = 2,label = 'Rubisco')
axes[2,0].plot(x,(LeafMassFlux.loc[:,'aj']-LeafMassFlux.loc[:,'rd']),c = 'b',linewidth = 2,label = 'RuBP_regen')
axes[2,0].plot(x,(LeafMassFlux.loc[:,'ap']-LeafMassFlux.loc[:,'rd']),c= 'y',linewidth = 2,label = 'TPU')
axes[2,0].scatter(Ci,Anet,s=20,marker ='o',label = 'Aobs')
axes[2,0].set_xlabel('Ci')
axes[2,0].set_ylabel('A')
axes[2,0].set_title('A/Ci Curve')
axes[2,0].legend()

x = LeafState_pred.loc[:,'ci']
axes[2,1].plot(x,(LeafMassflux_pred.loc[:,'ac']-LeafMassflux_pred.loc[:,'rd']), c = 'r',linewidth = 2,label = 'Rubisco')
axes[2,1].plot(x,(LeafMassflux_pred.loc[:,'aj']-LeafMassflux_pred.loc[:,'rd']),c = 'b',linewidth = 2,label = 'RuBP_regen')
axes[2,1].plot(x,(LeafMassflux_pred.loc[:,'ap']-LeafMassflux_pred.loc[:,'rd']),c= 'y',linewidth = 2,label = 'TPU')
axes[2,1].scatter(Ci,Anet,s=20,marker ='o',label = 'Aobs')
axes[2,1].set_xlabel('Ci')
axes[2,1].set_ylabel('A')
axes[2,1].set_title('A/Ci Curve predicted')
axes[2,1].legend()
plt.show()


for i in range(len(Obsflags)):   
    ci = np.delete(np.array(ObsCi),i)
    Anet = np.delete(np.array(ObsAnet),i)
    flags = np.delete(np.array(Obsflags),i)
    output,predicted_flags = flag(ci,Anet)
    print(f'Data:{flags}')
    print(f'Predicted{predicted_flags}')
    print(f'diff:{flags-predicted_flags}')
  
