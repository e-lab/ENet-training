# Training ENet

This work has been published in arXiv: [`ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation`](https://arxiv.org/abs/1606.02147).

Currently the network can be trained on three datasets:

| Datasets | Input Resolution | Output Resolution^ | # of classes |
|:--------:|:----------------:|:------------------:|:------------:|
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) | 480x360 | 60x45 | 11 |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 1024x512 | 128x64 | 19 |
| [SUN RGBD](http://rgbd.cs.princeton.edu/) | 256x200 | 32x25 | 37 |

^ is the encoder output resolution; decoder output resolution is the same as that of the input image. Folder arrangement of the datasets compatible with our data-loader has been explained in detail [here](data/README.md).

## Files/folders and their usage:

* [run.lua](run.lua)    : main file
* [opts.lua](opts.lua)  : contains all the input options used by the tranining script
* [data](data)          : data loaders for loading datasets
* models                : all the model architectures are defined here
* [train.lua](train.lua) : loading of models and error calculation
* [test.lua](test.lua)  : calculate testing error and save confusion matrices

## Example command for training encoder:

```
th run.lua --dataset cs --datapath /Cityscapes/dataset/path/ --model models/encoder.lua --save /save/trained/model/ --imHeight 256 --imWidth 512 --labelHeight 32 --labelWidth 64
```

## Example command for training decoder:

```
th run.lua --dataset cs --datapath /Cityscape/dataset/path/ --model models/decoder.lua --imHeight 256 --imWidth 512 --labelHeight 256 --labelWidth 512
```

Use `cachepath` option to save your loaded dataset in `.t7` format so that you won't have to load it again from scratch.
