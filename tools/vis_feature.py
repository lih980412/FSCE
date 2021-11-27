import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np

def vis_features(objectness, layer, batch_id):
    aa = objectness[layer].data.cpu().numpy()[batch_id].transpose(1, 2, 0)
    ff = objectness[layer].data.cpu().numpy()[batch_id].clip(0).transpose(1, 2, 0)
    aa0 = aa[:, :, 0]
    aa[:, :, 0] = (aa0 - np.min(aa0)) / (np.max(aa0) - np.min(aa0) + 0.000001)
    aa1 = aa[:, :, 1]
    aa[:, :, 1] = (aa1 - np.min(aa1)) / (np.max(aa1) - np.min(aa1) + 0.000001)
    aa2 = aa[:, :, 2]
    aa[:, :, 2] = (aa2 - np.min(aa2)) / (np.max(aa2) - np.min(aa2) + 0.000001)

    ff0 = ff[:, :, 0]
    ff[:, :, 0] = (ff0 - np.min(ff0)) / (np.max(ff0) - np.min(ff0) + 0.000001)
    ff1 = ff[:, :, 1]
    ff[:, :, 1] = (ff1 - np.min(ff1)) / (np.max(ff1) - np.min(ff1) + 0.000001)
    ff2 = ff[:, :, 2]
    ff[:, :, 2] = (ff2 - np.min(ff2)) / (np.max(ff2) - np.min(ff2) + 0.000001)



    aa = aa.astype(np.float64)
    ff = ff.astype(np.float64)
    plt.imshow(aa)
    plt.imshow(ff)

    plt.show()