# RBEI

A Hackathon Project for Robert Bosch Engineering and Business Solutions

## Instructors to setup the lab environment

### Data Preparation

- Clone this folder into your workstation (GPU enabled), lets call this folder `workspace`. 
- Download the raw data zip from [here]() to the workspace folder. The raw data contains images, csv (training.csv) file containing a record per bounding box on the image. 
- Upload the zip file to your workspace and unzip. Ensure the variable `raw_data` points to the location of the raw data.

### Pre-processing

Open the Jupyter notebook, and run the below steps

- Declare the below varibles and ensure your command prompt is set to the workspace folder. 

```
training_data = '/train/'
validation_data = '/val/'
current = '.'
%cd [workspace_folder]
```
- Train Test split as per the required ratio
  
```
from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df, test_size = 0.1, shuffle=True, random_state=42)
```
- Clone the darknet directory 
- Split the data into train and validation folders, and creating corresponding train.txt, test.txt files for feeding to darknet.

### Training

- Compile darknet
- Training with Tiny Config
  - Copy the yolov4-tiny-custom.cfg to darknet/cfg
  - Download pre-training model 
    ```
    !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
    ```
  - Run the training
    ```
    !darknet/darknet detector train obj.data darknet/cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -map -dont_show```
- Trianing with full config
  - Copy the yolov4-custom.cfg to darknet/cfg
  - Download pre-training model 
    ```
    !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
    ```
  - Run the training
    ```
    !darknet/darknet detector train obj.data darknet/cfg/yolov4-custom.cfg yolov4.conv.137 -map -dont_show```

### Inference

- Run log with Tiny Configuration

![Results](images/yolov4-tiny-custom.png)

- Run log with Full Configuration

![Results](images/yolov4-custom.png)

### Miscelleneous

Code to remove augmentation files if -show_imgs is enabled while training. 

```
# # Code for cleaning up augmented files during training
# import os
# import tqdm
# for name in tqdm.tqdm(os.listdir('/content/drive/MyDrive/datasets/rbei/train_data_subsets/')):
#   if ('aug_' in name):
#     os.remove(os.path.join('/content/drive/MyDrive/datasets/rbei/train_data_subsets/', name))

```

