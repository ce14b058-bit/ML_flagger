import numpy as np
from matplotlib import pyplot as plt
from C3 import Photosynthesis_C3_ACi
from Optimize import photosynthesis_optimize
from pandas import DataFrame as df
from matplotlib.gridspec import GridSpec


Ci = [20.7,77.5,135,197,266,344,428,517,606,698,791]
Anet = np.array([-3.27,3.49,10.8,17.8,23.5,27.6,30.5,31.7,33,33.3,33.3])
flags = [1,1,1,1,1,2,2,2,2,3,3]

def sse_anet(LeafMassFlux):
    sse = np.sum(np.square(LeafMassFlux.loc[:,'aNet'] - Anet))/len(Anet)
    return sse

weather = {
    'temperature' : [25],
    'pressure' : [101.325],
    'O2' : [210]
}

weather = df(weather)

observed = df(data=[Ci,Anet,flags],index=['Ci','Anet','flag']).T

photo_vec,n1,n2,n3 = photosynthesis_optimize(weather,observed)

weather_input = df()
weather_input.loc[:,'Ci'] = Ci
weather_input.loc[:,'pressure'] = weather.loc[0,'pressure']
weather_input.loc[:,'temperature'] = weather.loc[0,'temperature']
weather_input.loc[:,'O2'] = weather.loc[0,'O2']

vc_var = [(-0.5+0.1*i +1)*photo_vec[0] for i in range(11)]
j_var = [(-0.5+0.1*i +1)*photo_vec[1] for i in range(11)]
tpu_var = [(-0.5+0.1*i +1)*photo_vec[2] for i in range(11)]
rd_var = [(-0.5+0.1*i +1)*photo_vec[3] for i in range(11)]
gm_var = [(-0.5+0.1*i +1)*photo_vec[4] for i in range(11)]

Photosynthesis = {}
Photosynthesis['vcmax25'] = photo_vec[0] 
Photosynthesis['jmax25']= photo_vec[1] 
Photosynthesis['tpu25'] = photo_vec[2]
Photosynthesis['rd25']= photo_vec[3] 
Photosynthesis['gm25']= photo_vec[4]

photosynthesis_vc_var,photosynthesis_j_var,photosynthesis_tpu_var,photosynthesis_rd_var,photosynthesis_gm_var =[0] * len(vc_var),[0] * len(vc_var),[0] * len(vc_var),[0] * len(vc_var),[0] * len(vc_var)

for i in range(len(vc_var)):
    photosynthesis_vc_var[i] = Photosynthesis.copy()
    photosynthesis_j_var[i] = Photosynthesis.copy()
    photosynthesis_tpu_var[i] = Photosynthesis.copy()
    photosynthesis_rd_var[i] = Photosynthesis.copy()
    photosynthesis_gm_var[i] = Photosynthesis.copy()

    photosynthesis_vc_var[i]['vcmax25'] = vc_var[i]
    photosynthesis_j_var[i]['jmax25'] = j_var[i]
    photosynthesis_tpu_var[i]['tpu25'] = tpu_var[i]
    photosynthesis_rd_var[i]['rd25'] = rd_var[i]
    photosynthesis_gm_var[i]['gm25'] = gm_var[i]


photomodels_vc_var = [Photosynthesis_C3_ACi(weather_input,photosynthesis) for photosynthesis in photosynthesis_vc_var]
photomodels_j_var = [Photosynthesis_C3_ACi(weather_input,photosynthesis) for photosynthesis in photosynthesis_j_var]
photomodels_tpu_var = [Photosynthesis_C3_ACi(weather_input,photosynthesis) for photosynthesis in photosynthesis_tpu_var]
photomodels_rd_var = [Photosynthesis_C3_ACi(weather_input,photosynthesis) for photosynthesis in photosynthesis_rd_var]
photomodels_gm_var = [Photosynthesis_C3_ACi(weather_input,photosynthesis) for photosynthesis in photosynthesis_gm_var]

LeafMassFlux_vc_var = [model.solve()[0] for model in photomodels_vc_var]
LeafMassFlux_j_var = [model.solve()[0] for model in photomodels_j_var]
LeafMassFlux_tpu_var = [model.solve()[0] for model in photomodels_tpu_var]
LeafMassFlux_rd_var = [model.solve()[0] for model in photomodels_rd_var]
LeafMassFlux_gm_var = [model.solve()[0] for model in photomodels_gm_var]

sse_vc_var = [sse_anet(lm) for lm in LeafMassFlux_vc_var]
sse_j_var = [sse_anet(lm) for lm in LeafMassFlux_j_var]
sse_tpu_var = [sse_anet(lm) for lm in LeafMassFlux_tpu_var]
sse_rd_var = [sse_anet(lm) for lm in LeafMassFlux_rd_var]
sse_gm_var = [sse_anet(lm) for lm in LeafMassFlux_gm_var]

fig,axes = plt.subplots(3,2)
axes[0,0].plot(vc_var,sse_vc_var)
axes[0,0].set_xlabel('Vc')
axes[0,0].set_ylabel('sse')

axes[0,1].plot(j_var,sse_j_var)
axes[0,1].set_xlabel('j')
axes[0,1].set_ylabel('sse')

axes[1,0].plot(tpu_var,sse_tpu_var)
axes[1,0].set_xlabel('tpu')
axes[1,0].set_ylabel('sse')

axes[1,1].plot(rd_var,sse_rd_var)
axes[1,1].set_xlabel('rd')
axes[1,1].set_ylabel('sse')

axes[2,0].plot(gm_var,sse_gm_var)
axes[2,0].set_xlabel('gm')
axes[2,0].set_ylabel('sse')

plt.show()