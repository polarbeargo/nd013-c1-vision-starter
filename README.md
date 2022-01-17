# Object Detection in an Urban Environment

[image1]: ./images/1.png
[image2]: ./images/2.png
[image3]: ./images/3.png
[image4]: ./images/4.png
[image5]: ./images/5.png
[image6]: ./images/6.png
[image7]: ./images/7.png
[image8]: ./images/8.png
[image9]: ./images/barplot.png
[image10]: ./images/cyclist.png
[image11]: ./images/pedestrian.png
[image12]: ./images/vehicle.png
[image13]: ./images/step2500.gif
[image14]: ./images/loss.png
[image15]: ./images/trainingloss.png
[image16]: ./images/detection_box_precision.png
[image17]: ./images/detectionbox_recall.png
[image18]: ./images/learningRate.png
[image19]: ./images/eval.png
[image20]: ./images/aug1.png
[image21]: ./images/aug2.png
[image22]: ./images/aug3.png
[image23]: ./images/aug4.png
[image24]: ./images/aug5.png
[image25]: ./images/aug6.png
[image26]: ./images/aug7.png
[image27]: ./images/aug8.png
[image28]: ./images/aug9.png
[image29]: ./images/aug10.png
[image30]: ./images/aug11.png
[image31]: ./images/aug12.png
[image32]: ./images/aug31.png
[image33]: ./images/aug32.png
[image34]: ./images/aug33.png
[image35]: ./images/aug34.png
  
  
  
## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```
In the Udacity workspace:
```
python create_splits.py --source /home/workspace/data/waymo/training_and_validation/ --destination /home/workspace/data/waymo/
```
### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission Template

### Project overview
This section should contain a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self driving car systems?

### Set up
This section should contain a brief description of the steps to follow to run the code for this repository.  
#### In the Udacity workspace:  
- Run notebook  
```
cd /home/workspace
sudo apt-get update
sudo apt-get install firefox (Prevent firefox crash)
./launch_jupyter.sh
```
- Run create_splits.py  
```
python create_splits.py --source /home/workspace/data/waymo/training_and_validation/ --destination /home/workspace/data/waymo/
```  
- Run edit_config.py  
```
python edit_config.py --train_dir /home/workspace/data/waymo/train/ --eval_dir /home/workspace/data/waymo/val/ --batch_size 4 --checkpoint ./training/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map label_map.pbtxt
```
- Training
```
python experiments/model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config 
```
- Evaluation
```
python experiments/model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config --checkpoint_dir=training/reference/
```

- Watching Tensorboard
```
pip show tensorflow
```
```
WARNING: pip is being invoked by an old script wrapper. This will fail in a future version of pip.
Please see https://github.com/pypa/pip/issues/5599 for advice on fixing the underlying issue.
To avoid this problem you can invoke Python with '-m pip' instead of running pip directly.
Name: tensorflow
Version: 2.4.1
Summary: TensorFlow is an open source machine learning framework for everyone.
Home-page: https://www.tensorflow.org/
Author: Google Inc.
Author-email: packages@tensorflow.org
License: Apache 2.0Location: /data/virtual_envs/sdc-c1-gpu-augment/lib/python3.7/site-packages
Requires: tensorboard, keras-preprocessing, astunparse, protobuf, six, gast, wrapt, termcolor, typing-extensions, grpcio, absl-py, numpy, tensorflow-estimator, wheel, h5py, flatbuffers, google-pasta, opt-einsum
Required-by: tf-models-official
```
```
cd /data/virtual_envs/sdc-c1-gpu-augment/lib/python3.7/site-packages/tensorboard
python main.py --logdir=/home/workspace/training/reference
```
- Export the trained model
```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path training/reference/pipeline_new.config --trained_checkpoint_dir training/reference --output_directory training/experiment0/exported_model/
```
- Create a video of model's inferences for tf record file
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path training/experiment0/exported_model/saved_model --tf_record_path /home/workspace/data/waymo/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path training/reference/pipeline_new.config --output_path animation.gif
```  
### Dataset
#### Dataset analysis
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.   

| Display 5 images         |  Display 5 images        |
:-------------------------:|:-------------------------:
![][image1]   |  ![][image2] 
![][image3]   |  ![][image4] 
![][image5]   |  ![][image6]  
![][image7]   |  ![][image8]   

![][image9]    
- vehicle 
![][image12] 
- pedestrian
![][image11]  
- cyclist
![][image10]  

#### Cross validation
By using 100 tfrecord, we use [sklearn.model_selection.train_test_split¶](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) shuffle data by default=True to generalize data and create three splits from the processed records into train, val and test to reduce the imbalance class in each sample. Since we only use 100 tfrecords, we split the train data to 60 tfrecords and 20 tfrecords for test and validate dataset to ensure we have enough data for training and enough data for test and validate.


### Training
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances. 
- The batch size is 4 in the SSD Resnet 50 640x640 baseline model, training step up to 3200 in the Udacity workspace.
- I'm not able to run train and eval commands at the same time due to Udacity workspace will throw out of memory (OOM) this cause eval only have 1 blue dot(run eval command after training finished). 
- I also tried setting a smaller batch size from 4 to 2 still has the same problem.  
#### Loss
![][image14] 
![][image15]  
#### Precision
![][image16] 
#### Recall
![][image17]
#### Learning Rate
![][image18] 
#### Evaluation  
![][image19] 
  
- The default results were poor though the training loss approximate to 1 but the mean average precision mAP was 0.033340 which indicate need some modifications to improve the default baseline model.  
#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.
We use `Explore augmentations.ipynb` tried different data augmentation combinations to optimal for our dataset.  

In this section We applied the following Augmentations from [preprocessor.proto](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto)
such as add a random RGB to gray transform with a probability of 0.2:
```
data_augmentation_options {
    random_rgb_to_gray {
    probability: 0.2
    }
  }
```
- Add random_pixel_value_scale and random_image_scale to default value.
```
  data_augmentation_options {
    random_pixel_value_scale {
    }
  }
  data_augmentation_options {
    random_image_scale {
    }
  }
```
- Brightness adjusted to 0.3  
```
  data_augmentation_options {
    random_adjust_brightness {
    max_delta: 0.3
    }
  }
```
- Add random_adjust_hue, random_adjust_contrast and random_adjust_saturation to default value. 
```
 data_augmentation_options {
   random_adjust_hue {
   }
 }
 data_augmentation_options {
   random_adjust_contrast {
   }
 }
 data_augmentation_options {
   random_adjust_saturation {
   }
 }
```
-  Add a random contrast values between 0.6 ~ 1.0   
```
  data_augmentation_options {
    random_adjust_contrast {
    min_delta: 0.6
    max_delta: 1.0
    }
  }
```

- Add random_jpeg_quality with min_jpeg_quality to 80
```
data_augmentation_options {
    random_jpeg_quality {
      min_jpeg_quality: 80
    }
  }
```
![][image20]
![][image21]
![][image22]
![][image23]
![][image24]
![][image25]
![][image26]
![][image27]
![][image28]
![][image29]
![][image30]
![][image31]

- Only adjust random_rgb_to_gray, random_adjust_brightness and contrast  
```
  data_augmentation_options {
    random_rgb_to_gray {
    probability: 0.2
    }
  }
  data_augmentation_options {
    random_adjust_brightness {
    max_delta: 0.3
    }
  }
  data_augmentation_options {
    random_adjust_contrast {
    min_delta: 0.6
    max_delta: 1.0
    }
  }
```  
![][image32]
![][image33]
![][image34]
![][image35]  
