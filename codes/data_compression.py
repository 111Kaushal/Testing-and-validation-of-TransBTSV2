import os,pandas as pd,glob

#Directory where you would like to dump the files after conversion from pickle to bz2
export="D:\\MICCAI_BraTS2020_ValidationData\\"


 #Enter the directory of the training dataset below which has all the folders and this code will read the pickle files,convert, and dump in the eport path with the same sub directories of the folders.
for path, subdirs, files in os.walk(r'C:\\Users\\skyvi\\Desktop\\class work\\ANC\\project\\Datasets\\MICCAI_BraTS2020_ValidationData\\'):
        for subdir in subdirs:
            for file in glob.glob(path+subdir+"\\*.pkl"):
                data = pd.read_pickle(file)
                path2= export+subdir
                if not os.path.exists(path2):
                    os.makedirs(path2) 
                pd.to_pickle(data,path2+"\\"+os.path.splitext(os.path.basename(file))[0]+".bz2",compression='bz2')
            print("Conversion completed")