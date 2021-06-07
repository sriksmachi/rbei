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

- Clone and Compile darknet
- Training with Tiny Config
  - Copy the yolov4-tiny-custom.cfg to darknet/cfg
  - Download pre-training model 
    ```
    !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
    ```
  - Run the training
    ```
    !darknet/darknet detector train obj.data darknet/cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -map -dont_show```
- Training with full config
  - Copy the yolov4-custom.cfg to darknet/cfg
  - Download pre-training model 
    ```
    !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
    ```
  - Run the training
    ```
    !darknet/darknet detector train obj.data darknet/cfg/yolov4-custom.cfg yolov4.conv.137 -map -dont_show```

### Training Results

The below images show how the training has progressed for both Tiny and full configuration. 

- Run log with Tiny Configuration

![Results](images/yolov4-tiny-custom.png)

- Run log with Full Configuration

![Results](images/yolov4-custom.png)


### Inference

This section focuses on how do you run and test an image for identifying objects belonging to classes - furniture, small garments, wire and doors. 

1. Clone and Make the darknet code (refer to the notebook on steps to do this)
2. Download the obj.data and configuration from this repository (cfg/ contains the configuration)
3. Download the best model - [tiny model](https://drive.google.com/file/d/100I1cdX6SQfucPZIHE9MdS1S9lxUrywU/view?usp=sharing), [full model](https://drive.google.com/file/d/19AdjTg3l4Ihwy3BaxtX5rcXAsZRKDg-B/view?usp=sharing)
4. Run the below command

```
!./darknet detector test obj.data [path to .cfg file] [path to best model] [path to test image] -dont_show
```
Here is an example

```
# Testing with Tiny configuration and Best model using Tiny Configuration

!./darknet detector test obj.data cfg/yolov4-tiny-custom-test.cfg results/yolov4-tiny-custom_best.weights test/test-image-1.jpeg -dont_show

# Testing with Full configuration and Best moduel using Full configuration.

!./darknet detector test obj.data cfg/yolov4-custom-test.cfg results/yolov4-custom_best.weights test/test-image-3.jpeg.jpg -dont_show

```

5. The results of the test are stored in predictions.jpg

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

