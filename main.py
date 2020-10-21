import os, sys, random
from glob import glob
import matplotlib.pyplot as plt
# %matplotlib inline
os.system('pip install -qr requirements.txt')  # install dependencies

## Add the path where you have stored the neccessary supporting files to run detect.py ##
## Replace this with your path.##
sys.path.insert(0, '') 
print(sys.path)
cwd = os.getcwd()
print(cwd)

## Single Image prediction
## Beware the contents in the output folder will be deleted for every prediction
output = os.system('''python detect.py 
          --source /INPUTS/10rupee.jpg 
          --weights best.pt 
           --output /OUTPUTS/ --device cpu''')
print(output)
img = plt.imread('./OUTPUTS/image.jpg')
plt.imshow(img)

## Folder Prediction
# output = !python 'detect.py' 
#           --source '/content/inputs/' 
#           --weights '/content/drive/My Drive/Machine Learning Projects/YOLO/SOURCE/best_BCCM.pt' 
#           --output '/content/OUTPUTS/' --device 'cpu'
          
print(output)