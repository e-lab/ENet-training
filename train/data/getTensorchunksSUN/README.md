To get data

```bash
sh getSUN.sh
```

Preprocess data to create 8 chunks of tesnsor for training

```bash
th preprocess.lua
```

This will create imgs and labels folder and save data

1~4 is training set and 5~8 is for validate or test

1-5, 2-6, 3-7, 4-8 are pairs
