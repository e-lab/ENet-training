# Dataset loader description

Cityscapes dataset can be downloaded from their [website](https://www.cityscapes-dataset.com/).
In case of [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/), we use different structure which can be downloaded using this [link](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid) and modified it.
To make it compatible with our loader, just replace `train.txt` and `test.txt` with the one provided [CamVid](CamVid/) folder.
How to download [SUN RGBD](http://rgbd.cs.princeton.edu/) dataset and prepare it for our data loader has been explained in details [here](getTensorchunksSUN/).

## Folder/file structure for each dataset:

1. CamVid:

    ```
    CamVid/
    ├── test
    ├── train
    ├── testannot
    ├── trainannot
    ├── train.txt
    └── test.txt
    ```

2. Cityscapes:

    ```
    Cityscapes/
    └── leftImg8bit
        ├── train
        └── val
    ```

3. SUN RGBD

    ```
    SUN/
    ├── Images
    │   └── tensorImgsX.t7
    └── Labels
        └── tensorLabelsX.t7
    ```
