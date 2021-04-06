import numpy as np 
import matplotlib.pyplot as plt 
from skimage import io 
def plt_rgb(img:np.ndarray):
    img_np = img.copy().astype('uint8')
    plt.imshow(img_np)

def plt_rgb_histgram(img:np.ndarray): 
    img_np = img.copy().astype('uint8')
    #img = io.imread(img_np)
    