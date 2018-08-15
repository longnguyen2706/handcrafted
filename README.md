**Installation**

Anaconda 3.6

Opencv for anaconda 

Pyvlfeat 
https://pypi.org/project/pyvlfeat/

Boost Python for anaconda 
conda install -c anaconda boost 


Need to prepare the dataset carefully:
* First, the color dataset need to be converted to jpeg. Use utils/convert_images.py
* Second, need to check the name of dir and file between the color image dataset (used for CNN feature extraction) vs grayscale image dataset (used for handcrafted feature extraction)
    * Hep needs to manually change the name of file (using utils/rename.sh)
    
Install python matlab