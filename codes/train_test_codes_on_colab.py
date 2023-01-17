#Train and test on Colab

#Put the code 
######################
!pip uninstall torch
!pip install torch==1.6.0 torchvision
!pip install setproctitle
!pip install tensorboardx
!pip install pickle5
#go to /usr/local/lib/python3.7/dist-packages/pandas/io/pickle.py and change the first line as "import pickle5 as pickle" instead of just "import pickle"
######################

# Put the training code below in another cell.
!python -m torch.distributed.launch --nproc_per_node=1 /content/drive/MyDrive/TransBTS-main/train.py --resume_dir '/content/drive/MyDrive/TransBTS-main/checkpoint/TransBTS2022-03-29/model_epoch_935.pth' --load=True --save_freq=1 --num_workers=2 --batch_size=4 --train_dir '/content/drive/MyDrive/MICCAI_BraTS2020_TrainingData' --valid_dir '/content/drive/MyDrive/MICCAI_BraTS2020_ValidationData' --train_file '/content/drive/MyDrive/TransBTS-main/train.txt' --valid_file '/content/drive/MyDrive/TransBTS-main/valid.txt'
'''
Arguments explanation:
1. nproc_per_node: numer of gpus.
2. resume_dir: if need to continue training then link this argument to the directory of the previously trained model.
3. load: Flag for indicating if there will be continued training or not. If the model is not found in the resume_dir or if this flag is set to False then the trainig will start form epoch 0.
4. save_freq: number of epochs after which the model should be saved.
5. num_workers: Number of processes to launch for training. It is recommended to keep this value to 2. Higher the value longer the training time takes.
6. batch_size: batch size for the training data 
7. train_dir: Directory to training data folder
8. valid_dir: Directory to validation data folder
9. train.txt: Directory to the train.txt file located in the main folder. This text file contains the list of all the folders (sub directories) inside the main directory of training data folder where the train_dir directory is pointing to. If this file does not exist then you must create one.
10 valid.txt: Directory to the valid.txt file located in the main folder. This text file contains the list of all the folders (sub directories) inside the main directory of validation data folder where the valid_dir directory is pointing to. If this file does not exist then you must create one.
'''


# Put the testing code below in another cell after training is completed.
!python  /content/drive/MyDrive/TransBTS-main/test.py --snapshot=False --post_process=False --load_file '/content/drive/MyDrive/TransBTS-main/checkpoint/TransBTS2022-03-29/model_epoch_935.pth' --num_workers=2 --output_dir='/content/drive/MyDrive/' --valid_dir '/content/drive/MyDrive/MICCAI_BraTS2020_ValidationData' --valid_file '/content/drive/MyDrive/TransBTS-main/valid.txt'
'''
Arguments explanation:
1. snapshot: If set to True then the output of the model which is a series of time varying segmented images showing the tumour will be stored in a folder named visualization
2. post_process: Default False, when use postprocess, the score of dice_ET would be changed.
3. load_file: Directory to the trained model.
4. output_dir: Directory of the output where you like.
5. num_workers: Number of processes to launch for training. It is recommended to keep this value to 2. Higher the value longer the training time takes.
6. valid_dir: Directory to validation data folder
7. valid.txt: Directory to the valid.txt file located in the main folder. This text file contains the list of all the folders (sub directories) inside the main directory of validation data folder where the valid_dir directory is pointing to. If this file does not exist then you must create one.
'''