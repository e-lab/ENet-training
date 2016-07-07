----------------------------------------------------------------------
-- Main script for training a model for semantic segmentation
--
-- Abhishek Chaurasia, Eugenio Culurciello
-- Sangpil Kim, Adam Paszke
----------------------------------------------------------------------

require 'pl'
require 'nn'

----------------------------------------------------------------------
-- Local repo files
local opts = require 'opts'

-- Get the input arguments parsed and stored in opt
opt = opts.parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')

-- print('==> switching to CUDA')
require 'cudnn'
require 'cunn'
--print(cutorch.getDeviceProperties(opt.devid))
cutorch.setDevice(opt.devid)
print("Folder created at " .. opt.save)
os.execute('mkdir -p ' .. opt.save)

----------------------------------------------------------------------
print '==> load modules'
local data, chunks, ft
if opt.dataset == 'cv' then
   data  = require 'data/loadCamVid'
elseif opt.dataset == 'cs' then
   data = require 'data/loadCityscape'
elseif opt.dataset == 'su' then
   data = require 'data/loadSUN'
else
   error ("Dataset loader not found. (Available options are: cv/cs/su")
end

print 'saving opt as txt and t7'
local filename = paths.concat(opt.save,'opt.txt')
local file = io.open(filename, 'w')
for i,v in pairs(opt) do
    file:write(tostring(i)..' : '..tostring(v)..'\n')
end
file:close()
torch.save(path.join(opt.save,'opt.t7'),opt)

----------------------------------------------------------------------
print '==> training!'
local epoch = 1

t = paths.dofile(opt.model)

local train = require 'train'
local test  = require 'test'
while epoch < opt.maxepoch do
   local trainConf, model, loss = train(data.trainData, opt.dataClasses, epoch)
   test(data.testData, opt.dataClasses, epoch, trainConf, model, loss )
   trainConf = nil
   collectgarbage()
   epoch = epoch + 1
end
