
import torch
import torch.nn as nn
import numpy as np
from metric.inception import InceptionV3
from scipy import linalg

## Step 3 : Your Implementation Here ##

'''
def frechet_distance(u1, sig1, u2, sig2):

    diff = u1 - u2
    covmean, _
'''


def get_FIDScore(images):

    model = InceptionV3()
    model.cuda()
    model.eval()
    
    num_image = len(images)
    fake = np.empty((num_image, 2048))
    real = np.empty((num_image, 2048))

    for im in range(num_image):

        fake[im] = model(images[im]['fake'])[0].reshape(1, -1).cpu().data.numpy()
        real[im] = model(images[im]['real'])[0].reshape(1, -1).cpu().data.numpy()
    #fake = model(image['fake'])
    #real = model(image['real'])
    #fake = fake[0].reshape(fake[0].shape[0], -1).cpu().data.numpy()
    #real = real[0].reshape(real[0].shape[0], -1).cpu().data.numpy()
    

    
    #u_fake = fake.mean()
    #u_real = real.mean()
    u_fake = np.mean(fake, axis=0)
    u_real = np.mean(real, axis=0)

    #np_fake = fake.cpu().data.numpy()
    #np_real = real.cpu().data.numpy()
    
    sig1 = np.cov(fake, rowvar=False)
    #sig1 = np.cov(np_fake)
    sig2 = np.cov(real, rowvar=False)
    #sig2 = np.cov(np_real)
    diff = u_fake - u_real
    
    # use sqrtm to do square root for matrix
    covmean, _ = linalg.sqrtm(sig1.dot(sig2), disp=False)
    #covmean = np.sqrt(sig1 * sig2)
    
    
    # if covmean has complex value, change to real number
    if np.iscomplexobj(covmean):
        
        covmean = covmean.real
    
    tr_cov = np.trace(covmean)

    #covmean = np.trace(covmean)
    #covmean = torch.from_numpy(covmean).type(diff.type).to(diff.device)
    #return diff.dot(diff) + np.trace(sig1) + np.trace(sig2) - 2 * covmean
    return diff.dot(diff) + np.trace(sig1) + np.trace(sig2) - 2 * tr_cov



## Implement functions for fid score measurement using InceptionV3 network ##
