import numpy as np

import pandas as pd

data = pd.read_csv('so_pb_10_outlier.txt',delimiter=' ', header = None)

for timestep in range(data.shape[0]-1):
    numLandmarkst=int(data.values[timestep][9])
    numLandmarkstp=int(data.values[timestep+1][9])
    if (numLandmarkst==numLandmarkstp):
        if(data.values[timestep][10]==data.values[timestep+1][10]):
            if(abs(data.values[timestep][11]-data.values[timestep+1][11])>1):
                if(data.values[timestep][11]>0&&data.values[timestep+1][11]>0):
                    data.values[timestep+1][11]=data.values[timestep][11]+0.2
                else
                    data.values[timestep+1][11]=data.values[timestep][11]-0.2
            if(abs(data.values[timestep][12]-data.values[timestep+1][12])>1):
                if(data.values[timestep][12]>0&&data.values[timestep+1][12]>0):
                    data.values[timestep+1][12]=data.values[timestep][12]+0.2
                else
                    data.values[timestep+1][12]=data.values[timestep][12]-0.2
        if(data.values[timestep][13]==data.values[timestep+1][13]):
            if(abs(data.values[timestep][14]-data.values[timestep+1][14])>1):
                if(data.values[timestep][14]>0&&data.values[timestep+1][14]>0):
                    data.values[timestep+1][14]=data.values[timestep][14]+0.2
                else
                    data.values[timestep+1][14]=data.values[timestep][14]-0.2
            if(abs(data.values[timestep][15]-data.values[timestep+1][15])>1):
                if(data.values[timestep][15]>0&&data.values[timestep+1][15]>0):
                    data.values[timestep+1][15]=data.values[timestep][15]+0.2
                else
                    data.values[timestep+1][15]=data.values[timestep][15]-0.2

data.to_csv(r'so_pb_10_outlier2.txt', header=None, index=None, sep=' ', mode='a')


