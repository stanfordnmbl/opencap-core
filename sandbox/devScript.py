# load a pickle file

import pickle

with open('C:/Users/hpl/Downloads/OpenCapData_039b9d88-b3a5-4199-94b2-c687d2f180ac/OpenCapData_039b9d88-b3a5-4199-94b2-c687d2f180ac/Videos/Cam2/cameraIntrinsicsExtrinsics.pickle', 'rb') as handle:

    data = pickle.load(handle)

print(data)
