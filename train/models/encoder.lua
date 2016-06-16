----------------------------------------------------------------------
-- Create model and calculate loss to optimize for encoder,
--
-- Adam Paszke,
-- May 2016.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
print '==> define parameters'

local histClasses = opt.datahistClasses
local classes = opt.dataclasses
local conClasses = opt.dataconClasses

----------------------------------------------------------------------
print '==> construct model'

local model = nn.Sequential()

function bottleneck(input, output, downsample)
   local internal = output / 4
   local input_stride = downsample and 2 or 1

   local sum = nn.ConcatTable()

   local main = nn.Sequential()
   local other = nn.Sequential()
   sum:add(main):add(other)

   main:add(cudnn.SpatialConvolution(input, internal, 1, 1, input_stride, input_stride, 0, 0):noBias())
   main:add(nn.SpatialBatchNormalization(internal, 1e-3))
   main:add(cudnn.ReLU(true))
   main:add(cudnn.SpatialConvolution(internal, internal, 3, 3, 1, 1, 1, 1))
   main:add(nn.SpatialBatchNormalization(internal, 1e-3))
   main:add(cudnn.ReLU(true))
   main:add(cudnn.SpatialConvolution(internal, output, 1, 1, 1, 1, 0, 0):noBias())
   main:add(nn.SpatialBatchNormalization(output, 1e-3))

   other:add(nn.Identity())
   if downsample then
      other:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   end
   if input ~= output then
      other:add(nn.Padding(1, output-input, 3))
   end

   return nn.Sequential():add(sum):add(nn.CAddTable()):add(cudnn.ReLU(true))
end

local initial_block = nn.ConcatTable(2)
initial_block:add(cudnn.SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1))
initial_block:add(nn.SpatialMaxPooling(2, 2, 2, 2))

model:add(initial_block)                                         -- 128x256
model:add(nn.JoinTable(2)) -- can't use Concat, because SpatialConvolution needs contiguous gradOutput
model:add(nn.SpatialBatchNormalization(16, 1e-3))
model:add(cudnn.ReLU(true))
model:add(bottleneck(16, 64, true))                              -- 64x128
model:add(bottleneck(64, 64))
model:add(bottleneck(64, 128, true))                             -- 32x64
for i = 1,6 do
   model:add(bottleneck(128, 128))
end
model:add(bottleneck(128, 128))
model:add(cudnn.SpatialConvolution(128, #classes, 1, 1))

local gpu_list = {}
for i = 1,cutorch.getDeviceCount() do gpu_list[i] = i end
model = nn.DataParallelTable(1):add(model:cuda(), gpu_list)
print(opt.nGPU .. " GPUs being used")

-- Loss: NLL
print('defining loss function:')
local normHist = histClasses / histClasses:sum()
local classWeights = torch.Tensor(#classes):fill(1)
for i = 1, #classes do
   if histClasses[i] < 1 or i == 1 then -- ignore unlabeled
      classWeights[i] = 0
   else
      classWeights[i] = 1 / (torch.log(1.2 + normHist[i]))
   end
end

loss = cudnn.SpatialCrossEntropyCriterion(classWeights)

loss:cuda()
----------------------------------------------------------------------
print '==> here is the model:'
print(model)


-- return package:
return {
   model = model,
   loss = loss,
}

