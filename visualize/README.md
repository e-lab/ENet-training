# Visualize network output on images/videos

## Requirements

+ `image`
+ `imgraph`
+ [`video-decoder`](https://github.com/e-lab/torch-toolbox/tree/master/Video-decoder)

[frame](frame) is a subset of elab's [torch-toolbox/demo-core](https://github.com/e-lab/torch-toolbox/tree/master/demo-core).

## Examples

1. Images:

```
qlua run.lua -i image/folder/path -m network-model
```

2. Video:

```
qlua run.lua -i path/video.mp4 -m network-model
```

3. Camera:

```
qlua run.lua -i cam0 -m network-model
```

Here, 0 in cam'0' indicates camera id. Make sure that the number of classes in your categories.txt file generated during training and the number of classes in [`colorMap`](colorMap.lua) are the same.
