--------------------------------------------------------------------------------
-- Contains options required by run.lua
--
-- Written by: Abhishek Chaurasia
-- Dated:      6th June, 2016
--------------------------------------------------------------------------------

local opts = {}

lapp = require 'pl.lapp'
function opts.parse(arg)
   local opt = lapp [[
   Command line options:
   Training Related:
   -r,--learningRate       (default 1e-3)        learning rate
   -d,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples)
   -w,--weightDecay        (default 5e-4)        L2 penalty on the weights
   -m,--momentum           (default 0.9)         momentum
   -b,--batchSize          (default 8)           batch size
   --maxepoch              (default 300)         maximum number of training epochs
   --plot                                        plot training/testing error
   --lrDecayEvery          (default 100)          Decay learning rate every X epoch by 1e-1

   Device Related:
   -t,--threads            (default 8)           number of threads
   -i,--devid              (default 1)           device ID (if using CUDA)
   --nGPU                  (default 2)           number of GPUs you want to train on
   --save                  (default /media/)     save trained model here

   Dataset Related:
   --channels              (default 3)
   --datapath              (default /media/Dataset/)
                           dataset location
   --dataset               (default cv)          dataset type: cv(CamVid)/cs(cityscapes)/su(SUN)
   --cachepath             (default /media/)     cache directory to save the loaded dataset
   --imHeight              (default 360)         image height  (360 cv/256 cs/256 su)
   --imWidth               (default 480)         image width   (480 cv/512 cs/328 su)
   --labelHeight           (default 45)          label height  (45  cv/32 cs/32 su)
   --labelWidth            (default 60)          label width   (60  cv/64 cs/41 su)
   --smallNet                                    reduce number of classes

   Model Related:
   --model                 (default models/encoder.lua)
                           Path of a model
   --CNNEncoder            (default /media/Models/CamVid/enc/model-100.net)
                           pretrained encoder for which you want to train your decoder

   Saving/Displaying Information:
   --noConfusion           (default skip)
                           skip: skip confusion, all: test+train, tes : test only
   --startfrom             (default .)           type dir/model-x.net
   --printNorm                                   For visualize norm factor while training
 ]]

   return opt
end

return opts
