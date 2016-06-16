# Dataset loader description

Cityscapes dataset can be downloaded from [here](https://www.cityscapes-dataset.com/) and SUN RGBD can be found [here](http://rgbd.cs.princeton.edu/). In case of [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/), we use different structure which we downloaded from [here](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid) and modified it. CamVid dataset compatible with our loader can be found [here](CamVid).

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
    ├──
    ├──
    ├──
    └──
    ```
