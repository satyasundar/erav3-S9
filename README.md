# Train Imagenet1k with Resnet50 on a mosaic-ml composer docker

### [Huggingface Spaces](https://huggingface.co/spaces/satyanayak/imagenet1k-resnet50-demo)

https://huggingface.co/spaces/satyanayak/imagenet1k-resnet50-demo

The objective of this code is to train the Imagenet1k dataset with Resnet50 model architecture

### Contents

1. [Aws Setup](#aws-setup)
2. [Data Preparation](#data-preparation)
3. [EBS Volume](#ebs-volume)
4. [Source Code Setup](#source-code-setup)
5. [Docker Preparation](#docker-preparation)
6. [Model Running](#model-running)
7. [Model Accuracy](#model-accuracy)
8. [Model Run Logs](#model-run-logs)

- Here is the approach:

  - #### AWS Setup

    - Used t3.medium ec2 insrance for data preparation and EBS volumen setup
    - Used g4dn.2xlarge ec2 insrance for initial test run.
    - Used g4dn.2xlarge for 1st model run. It Ran for 3 days almost.
    - Used g6.12xlarge for 2nd Model run. It ran for 18 hours.
    - As per [**benchmarking**](https://www.databricks.com/blog/mosaic-resnet), it can run `17 min - 80 min` in a P4 instance.

  - #### Data Preparation

    - I am using AWS. Connect to a EC2 instance, preferable t3.medium to download the dataset from kaggle
    - I have used kaggle cli to fetch the dataset.
    - I kept a copy of this dataset in s3 storage.
    - Now I make an EBS volume with the data which can be plugged into any of the EC2 machine while model running.

      ```
      $ kaggle competitions download -c imagenet-object-localization-challenge
      $ aws s3 cp  imagenet-object-localization-challenge.zip s3://my-bucket/
      ```

  - #### EBS volume
    - Create a new volume and attach to one of the ec2 instance.
    - My volume size was 250GB.
    - The val dataset is not proper in the original kaggle dataset. We need to download the val dataset with proper folder structure.
    - Extract both dataset in this volume and make the folder structure proper.
      ```
      Imagenet
      Imagenet/Train/
      Imagenet/Val/
      ...
      ```
    - Once volume is ready, you can detach it and re-attach when the gpu based ec2 instance is spinned up.
    ```
    $ sudo mkdir /mnt/volume
    $ sudo mount /dev/nvme2n1 /mnt/vol
    ```
  - #### Source Code setup

    - Clone the repo
      ```
       $ git clone https://github.com/mosaicml/examples.git
      ```
    - Also copy this code to EBS volume, so that it can be readily available.

  - #### Docker preparation

    - Mosaic has release one docker for this benchmark code to run. I will use this.
    - Pull the docker from this tag on [mosain ml pytorch images]('https://hub.docker.com/r/mosaicml/pytorch')
    - Tag will be "mosaicml/pytorch:1.12.1_cu116-python3.9-ubuntu20.04"

    ```
    $ docker pull mosaicml/pytorch:1.12.1_cu116-python3.9-ubuntu20.04
    ```

    - Once docker image is avalable, run a contaner as below.

    ```
    $ docker run -it -v /mnt/vol/Imagenet:/datasets/Imagenet \
            -v /mnt/vol/examples/examples/benchmarks/resnet_imagenet:/app \
            --gpus all --shm-size 5g <INMAGE_ID>
    ```

    - Install the packages from requirement.txt

    ```
    cd examples/examples/benchmarks/resnet_imagenet/
    $ pip install -r requirements.txt
    ```

  - #### Model Running
    - Change the resnet50.yaml as per required.
    - Change the recipe from "mild", "medium" or "hot".
    - Configure dataset path, batch size, number of epoch if you want to change, save folder path, load path etc.
    ```
    $ composer main.py yamls/resnet50.yaml
    ```
  - #### Model Accuracy

    - By default it was implemented `torchmetrics.classification MulticlassAccuracy`
    - Modified this code to implement Top-1 Accuracy using `torchmetrics.classification Accuracy`

      ```
          train_metrics = Accuracy(
          task='multiclass',
          num_classes=num_classes,
          top_k=1,
          compute_on_step=True,
          dist_sync_on_step=True
          )
      ```

    - In the first run it achived `76.50%` Top-1 accuracy in 36 epochs
    - In the second run it achived similar accuracy in less number of epochs.
    - As per [**benchmarking**](https://www.databricks.com/blog/mosaic-resnet) released by Mosaic ML, it can achive `75.2% - 78.1%` in 36 epochs.

  - #### Docker Commands

    ```
    $ docker images # show images
    $ docker ps     # show containers
    $ docker commit <CONTAINER_ID> <IMAGE_ID>
    $ docker save -o /mnt/vol/composer-img-docker.tar composer-imagenet
    $ docker load -i /path/to/volume/modified_image.tar
    $ docker stop $(docker ps -aq) # stop all the containers
    $ docker rm $(docker ps -aq) # remove all the containers
    $ docker exec -it my_container /bin/bash # a new shell in a running container

    ```

## Model Run Logs

I have ran the model 2 times. In the 1st attempt logs were not capltured clearly as I was using EC2 spot instances and it is getting interrputed. Second time modle Run log was fine. Both logs attached here.

1. [**Model Run Log - 1**](#model-run-log---1)
2. [**Model Run Log - 2**](#model-run-log---2)

### [Model Run Log - 1](#model-run-log---1)

```
/usr/local/lib/python3.9/dist-packages/composer/core/data_spec.py:39: UserWarning: Cannot split tensor of length 848 into batches of size 1024. As it is smaller, no splitting will be done. This may happen on the last batch of a dataset if it is a smaller size than the microbatch size.
  warnings.warn(f'Cannot split tensor of length {len(t)} into batches of size {microbatch_size}. '

eval           Epoch  29:  100%|█████████████████████████| 49/49 [01:43<00:00,  1.82s/ba, metrics/eval/Accuracy=0.7344]                 Train!
                                                                                                                                        /usr/local/lib/python3.9/dist-packages/composer/trainer/trainer.py:240: RuntimeWarning: CUDA out of memory detected. Train microbatch size will be decreased from 1024 -> 512.
  warnings.warn(

train          Epoch  30:  100%|█████████████████████████| 1251/1251 [1:16:18<00:00,  3.66s/ba, loss/train/total=2.9341]
eval           Epoch  30:  100%|█████████████████████████| 49/49 [01:43<00:00,  1.95s/ba, metrics/eval/Accuracy=0.7393]
train          Epoch  31:  100%|█████████████████████████| 1251/1251 [1:16:25<00:00,  3.68s/ba, loss/train/total=2.9261]
eval           Epoch  31:  100%|█████████████████████████| 49/49 [01:47<00:00,  1.92s/ba, metrics/eval/Accuracy=0.7456]
train          Epoch  32:  100%|█████████████████████████| 1251/1251 [1:16:23<00:00,  3.64s/ba, loss/train/total=2.8784]
eval           Epoch  32:  100%|█████████████████████████| 49/49 [01:45<00:00,  1.90s/ba, metrics/eval/Accuracy=0.7512]

  model.loss_name: binary_cross_entropy
  train_dataset.crop_size: 176
  eval_dataset.resize_size: 232
  max_duration: 270ep
save_folder: ./{run_name}/ckpt
save_interval: 1ep
save_num_checkpoints_to_keep: 2
load_path: ./asgn9-run1/ckpt/latest-rank0.pt

Run evaluation
******************************
Config:
blurpool/num_blurconv_layers: 6
blurpool/num_blurpool_layers: 1
node_name: unknown because NODENAME environment variable not set
num_gpus_per_node: 1
num_nodes: 1
rank_zero_seed: 17

******************************

eval           Epoch  29:   71%|█████████████████▊       | 35/49 [01:17<00:26,  1.90s/ba]                                               /usr/local/lib/python3.9/dist-packages/composer/core/data_spec.py:39: UserWarning: Cannot split tensor of length 848 into batches of size 1024. As it is smaller, no splitting will be done. This may happen on the last batch of a dataset if it is a smaller size than the microbatch size.
  warnings.warn(f'Cannot split tensor of length {len(t)} into batches of size {microbatch_size}. '

eval           Epoch  29:  100%|█████████████████████████| 49/49 [01:43<00:00,  1.82s/ba, metrics/eval/Accuracy=0.7344]                 Train!
                                                                                                                                        /usr/local/lib/python3.9/dist-packages/composer/trainer/trainer.py:240: RuntimeWarning: CUDA out of memory detected. Train microbatch size will be decreased from 1024 -> 512.
  warnings.warn(

train          Epoch  30:  100%|█████████████████████████| 1251/1251 [1:16:18<00:00,  3.66s/ba, loss/train/total=2.9341]
eval           Epoch  30:  100%|█████████████████████████| 49/49 [01:43<00:00,  1.95s/ba, metrics/eval/Accuracy=0.7393]
train          Epoch  31:  100%|█████████████████████████| 1251/1251 [1:16:25<00:00,  3.68s/ba, loss/train/total=2.9261]
eval           Epoch  31:  100%|█████████████████████████| 49/49 [01:47<00:00,  1.92s/ba, metrics/eval/Accuracy=0.7456]
train          Epoch  32:  100%|█████████████████████████| 1251/1251 [1:16:23<00:00,  3.64s/ba, loss/train/total=2.8784]
eval           Epoch  32:  100%|█████████████████████████| 49/49 [01:45<00:00,  1.90s/ba, metrics/eval/Accuracy=0.7512]                 ^[[A^[[A^[[A^[[A^[[B^[[B^[[B^[[B^[[B^[[B^[[B^[[B^[[B^[[B^[[B^[[B^[[B^[[B^[[B^[[B:05,  3.54s/ba, loss/train/total=2.8268]                drain          Epoch  33:   92%|███████████████████████  | 1152/1251 [1:10:25<06:06,  3.70s/ba, loss/train/total=2.7965]
train          Epoch  33:  100%|█████████████████████████| 1251/1251 [1:16:27<00:00,  3.57s/ba, loss/train/total=2.6606]
eval           Epoch  33:  100%|█████████████████████████| 49/49 [01:46<00:00,  1.94s/ba, metrics/eval/Accuracy=0.7545]
train          Epoch  34:  100%|█████████████████████████| 1251/1251 [1:16:25<00:00,  3.68s/ba, loss/train/total=2.6889]
eval           Epoch  34:  100%|█████████████████████████| 49/49 [01:45<00:00,  1.93s/ba, metrics/eval/Accuracy=0.7618]
train          Epoch  35:  100%|█████████████████████████| 1251/1251 [1:16:17<00:00,  3.68s/ba, loss/train/total=2.5828]
eval           Epoch  35:  100%|█████████████████████████| 49/49 [01:46<00:00,  1.94s/ba, metrics/eval/Accuracy=0.7650]

```

---

### [Model Run Log - 2](#model-run-log---2)

```
train          Epoch   0:  100%|█████████████████████████| 625/625 [10:11<00:00,  1.02ba/s, loss/train/total=6.2503]

eval           Epoch   0:  100%|█████████████████████████| 25/25 [00:26<00:00,  1.06s/ba, metrics/eval/Accuracy=0.0550]
                                                                                                                        `
train          Epoch   1:  100%|█████████████████████████| 625/625 [07:30<00:00,  1.39ba/s, loss/train/total=5.4919]

eval           Epoch   1:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.10s/ba, metrics/eval/Accuracy=0.2022]

train          Epoch   2:  100%|█████████████████████████| 625/625 [06:21<00:00,  1.64ba/s, loss/train/total=4.9385]

eval           Epoch   2:  100%|█████████████████████████| 25/25 [00:26<00:00,  1.07s/ba, metrics/eval/Accuracy=0.3079]

train          Epoch   3:  100%|█████████████████████████| 625/625 [05:40<00:00,  1.83ba/s, loss/train/total=4.6694]

eval           Epoch   3:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.11s/ba, metrics/eval/Accuracy=0.3908]

train          Epoch   4:  100%|█████████████████████████| 625/625 [06:18<00:00,  1.65ba/s, loss/train/total=4.4587]

eval           Epoch   4:  100%|█████████████████████████| 25/25 [00:25<00:00,  1.04s/ba, metrics/eval/Accuracy=0.4332]

train          Epoch   5:  100%|█████████████████████████| 625/625 [05:31<00:00,  1.88ba/s, loss/train/total=4.2514]

eval           Epoch   5:  100%|█████████████████████████| 25/25 [00:26<00:00,  1.07s/ba, metrics/eval/Accuracy=0.4771]

train          Epoch   6:  100%|█████████████████████████| 625/625 [05:34<00:00,  1.87ba/s, loss/train/total=4.2317]

eval           Epoch   6:  100%|█████████████████████████| 25/25 [00:28<00:00,  1.13s/ba, metrics/eval/Accuracy=0.4986]

train          Epoch   7:  100%|█████████████████████████| 625/625 [06:54<00:00,  1.51ba/s, loss/train/total=4.1712]

eval           Epoch   7:  100%|█████████████████████████| 25/25 [00:26<00:00,  1.05s/ba, metrics/eval/Accuracy=0.5172]

train          Epoch   8:  100%|█████████████████████████| 625/625 [05:42<00:00,  1.82ba/s, loss/train/total=4.1265]

eval           Epoch   8:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.08s/ba, metrics/eval/Accuracy=0.5321]

train          Epoch   9:  100%|█████████████████████████| 625/625 [05:30<00:00,  1.89ba/s, loss/train/total=4.0594]

eval           Epoch   9:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.10s/ba, metrics/eval/Accuracy=0.5454]

train          Epoch  10:  100%|█████████████████████████| 625/625 [07:10<00:00,  1.45ba/s, loss/train/total=3.8802]

eval           Epoch  10:  100%|█████████████████████████| 25/25 [00:26<00:00,  1.08s/ba, metrics/eval/Accuracy=0.5614]

train          Epoch  11:  100%|█████████████████████████| 625/625 [05:41<00:00,  1.83ba/s, loss/train/total=3.8310]

eval           Epoch  11:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.12s/ba, metrics/eval/Accuracy=0.5658]

train          Epoch  12:  100%|█████████████████████████| 625/625 [05:56<00:00,  1.75ba/s, loss/train/total=3.8082]

eval           Epoch  12:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.12s/ba, metrics/eval/Accuracy=0.5882]

train          Epoch  13:  100%|█████████████████████████| 625/625 [08:11<00:00,  1.27ba/s, loss/train/total=3.6285]

eval           Epoch  13:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.10s/ba, metrics/eval/Accuracy=0.6379]

train          Epoch  14:  100%|█████████████████████████| 625/625 [09:07<00:00,  1.14ba/s, loss/train/total=3.6143]

eval           Epoch  14:  100%|█████████████████████████| 25/25 [00:26<00:00,  1.06s/ba, metrics/eval/Accuracy=0.6570]

train          Epoch  15:  100%|█████████████████████████| 625/625 [09:22<00:00,  1.11ba/s, loss/train/total=3.3888]

eval           Epoch  15:  100%|█████████████████████████| 25/25 [00:26<00:00,  1.06s/ba, metrics/eval/Accuracy=0.6752]

train          Epoch  16:  100%|█████████████████████████| 625/625 [11:52<00:00,  1.14s/ba, loss/train/total=3.3867]

eval           Epoch  16:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.12s/ba, metrics/eval/Accuracy=0.6952]

train          Epoch  17:  100%|█████████████████████████| 625/625 [12:50<00:00,  1.23s/ba, loss/train/total=3.1986]

eval           Epoch  17:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.09s/ba, metrics/eval/Accuracy=0.7094]

train          Epoch  18:  100%|█████████████████████████| 625/625 [13:32<00:00,  1.30s/ba, loss/train/total=3.0901]

eval           Epoch  18:  100%|█████████████████████████| 25/25 [00:28<00:00,  1.12s/ba, metrics/eval/Accuracy=0.7171]

train          Epoch  19:  100%|█████████████████████████| 625/625 [14:42<00:00,  1.41s/ba, loss/train/total=3.0133]

eval           Epoch  19:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.11s/ba, metrics/eval/Accuracy=0.7256]

train          Epoch  20:  100%|█████████████████████████| 625/625 [17:54<00:00,  1.72s/ba, loss/train/total=2.8814]

eval           Epoch  20:  100%|█████████████████████████| 25/25 [00:28<00:00,  1.13s/ba, metrics/eval/Accuracy=0.7373]

train          Epoch  21:  100%|█████████████████████████| 625/625 [19:33<00:00,  1.88s/ba, loss/train/total=2.7297]

eval           Epoch  21:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.09s/ba, metrics/eval/Accuracy=0.7452]

train          Epoch  22:  100%|█████████████████████████| 625/625 [19:39<00:00,  1.89s/ba, loss/train/total=2.7039]

eval           Epoch  22:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.09s/ba, metrics/eval/Accuracy=0.7516]

train          Epoch  23:  100%|█████████████████████████| 625/625 [19:43<00:00,  1.89s/ba, loss/train/total=2.5767]

eval           Epoch  23:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.10s/ba, metrics/eval/Accuracy=0.7569]

train          Epoch  24:  100%|█████████████████████████| 625/625 [19:43<00:00,  1.89s/ba, loss/train/total=2.5412]

eval           Epoch  24:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.12s/ba, metrics/eval/Accuracy=0.7599]

train          Epoch  25:  100%|█████████████████████████| 625/625 [19:40<00:00,  1.89s/ba, loss/train/total=2.5106]

eval           Epoch  25:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.10s/ba, metrics/eval/Accuracy=0.7602]

train          Epoch  26:  100%|█████████████████████████| 625/625 [19:47<00:00,  1.90s/ba, loss/train/total=2.4409]

eval           Epoch  26:  100%|█████████████████████████| 25/25 [00:27<00:00,  1.10s/ba, metrics/eval/Accuracy=0.7610]
Saving model for HuggingFace...

```
