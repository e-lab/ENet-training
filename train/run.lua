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
    for i = 1 , 8 do
        local cacheDir = paths.concat(opt.cachepath, 'sun')
        local cachePath = paths.concat(cacheDir, tostring(i)..'data.t7')
        local histPath = paths.concat(cacheDir, 'from_'..tostring(i)..'th_hist.txt')
        if  opt.cachepath ~= "none" and paths.filep(cachePath) or paths.filep(histPath)then
            IDX = i
        end
    end
    IDX = 1
    print('SUN loading data')
    ft = require 'data/loadSun'
    chunks = ft()
    print(chunks)
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
-- epoch tracker
if string.len(opt.startfrom) > 1 then
     local util = assert(require '/misc/util','no /misc/util')
     epoch = util.setOpt()
end

t = paths.dofile(opt.model)

if opt.dataset ~= 'su' then
   local train = require 'train'
   local test  = require 'test'
   while epoch < opt.maxepoch do
      local trainConf, model, loss = train(data.trainData, opt.dataclasses, epoch)
      test(data.testData, opt.dataclasses, epoch, trainConf, model, loss )
      trainConf = nil
      collectgarbage()
      epoch = epoch + 1
   end
elseif opt.dataset == 'su' then
   local util , train, test, trainConf, model, loss, epoch, train, test
   IDX = 1
   epoch = 1
   train = require 'train'
   test  = require 'test'
   while epoch < opt.maxepoch do
      for i = 1,8 do
         require 'xlua'
         xlua.progress(i,8)
         IDX = i
         if i > 1 or epoch ~= 1 then chunks = nil chunks = ft() end
         print('Current epoch: '..tostring(epoch))
         if IDX < 5 then
            print '==> training!'
            trainConf, model, loss = train(chunks.trainData, opt.dataclasses, epoch)
            collectgarbage()
         else
            print '==> evaluating'
            test(chunks.testData, opt.dataclasses, epoch, trainConf, model, loss )
            collectgarbage()
         end
      end
      epoch = epoch + 1
   end
else
    error('opt.dataset miss match')
end
