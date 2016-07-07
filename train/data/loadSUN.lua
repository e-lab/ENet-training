----------------------------------------------------------------------
-- indoor SUN data loader,
-- Abhishek, SangPil Kim, Eugenio Culurciello,
-- March 2016
----------------------------------------------------------------------

require 'image'   -- to visualize the dataset

torch.setdefaulttensortype('torch.FloatTensor')

-- Getting the height and width of images
--                                  |----  Train  ----|    |----   Test   ----|
local indexTable  = torch.Tensor({1860, 2790, 4478, 5050, 6974, 8047, 9748, 10335})
local heightTable = torch.Tensor({530,  427,  441,  531,  530,  427,  441,  531})
local widthTable  = torch.Tensor({730,  561,  591,  681,  730,  561,  591,  681})
local imgHeight   = opt.imHeight
local imgWidth    = torch.floor(torch.min(
                    torch.cdiv(imgHeight*widthTable, heightTable)))
local labelHeight = opt.labelHeight
local labelWidth  = torch.floor(torch.min(
                    torch.cdiv(labelHeight*widthTable, heightTable)))

local N = indexTable[8]
local trsize = indexTable[4]
local tesize = N - trsize

-- Class definition
local classes = {'unlabeled', 'wall',   'floor',      'cabinet',        'bed',
                 'chair',     'sofa',   'table',      'door',           'window',
                 'bookshelf', 'picture','counter',    'blinds',         'desk',
                 'shelves',   'curtain','dresser',    'pillow',         'mirror',
                 'floor_mat', 'clothes','ceiling',    'books',          'fridge',
                 'tv',        'paper',  'towel',      'shower_curtain', 'box',
                 'whiteboard','person', 'night_stand','toilet',         'sink',
                 'lamp',      'bathtub','bag'}

local conClasses = {'wall',      'floor',  'cabinet',    'bed',
                    'chair',     'sofa',   'table',      'door',           'window',
                    'bookshelf', 'picture','counter',    'blinds',         'desk',
                    'shelves',   'curtain','dresser',    'pillow',         'mirror',
                    'floor_mat', 'clothes','ceiling',    'books',          'fridge',
                    'tv',        'paper',  'towel',      'shower_curtain', 'box',
                    'whiteboard','person', 'night_stand','toilet',         'sink',
                    'lamp',      'bathtub','bag'}

print '==> remapping classes'
--0 is unlabeled and there are 894 classes in the original dataset
local classMap = {}
for i=0, #classes-1 do
   classMap[i] = i+1
end
local histClasses = torch.Tensor(#classes):zero()

print('==> number of classes: ' .. #classes)
print('classes are:')
print(classes)

--------------------------------------------------------------------------------
print '==> loading SUN RGBD dataset'
local trainData, testData
local loadedFromCache = false
paths.mkdir(paths.concat(opt.cachepath, 'sun'))
local sunCachePath = paths.concat(opt.cachepath, 'sun', 'data.t7')

if opt.cachepath ~= "none" and paths.filep(sunCachePath) then
   local dataCache = torch.load(sunCachePath)
   trainData = dataCache.trainData
   testData = dataCache.testData
   histClasses = dataCache.histClasses
   loadedFromCache = true
   dataCache = nil
   collectgarbage()
else
   -- initialize data structures:
   trainData = {
      data = torch.FloatTensor(trsize, opt.channels, opt.imHeight, opt.imWidth),
      labels = torch.FloatTensor(trsize, opt.labelHeight, opt.labelWidth),
      preverror = 1e10, -- a really huge number
      size = function() return trsize end
   }

   testData = {
      data = torch.FloatTensor(tesize, opt.channels, opt.imHeight, opt.imWidth),
      labels = torch.FloatTensor(tesize, opt.labelHeight, opt.labelWidth),
      preverror = 1e10, -- a really huge number
      size = function() return tesize end
   }

   print('==> loading training files')

   local dpathRoot = opt.datapath

   assert(paths.dirp(dpathRoot), 'No training folder found at: ' .. dpathRoot)
   --load training images and labels:
   local trc = 1
   local tec = 1
   for IDX = 1, 8 do
      local imageBatch = torch.load(dpathRoot .. '/Images/tensorImgs' .. tostring(IDX) .. '.t7')
      local labelBatch = torch.load(dpathRoot .. '/Labels/tensorLabels' .. tostring(IDX) .. '.t7')

      local maxN = imageBatch:size(1)

      local aspectRatio = widthTable[IDX]/heightTable[IDX]
      local batchImgWidth   = torch.floor(imgHeight * aspectRatio)
      local batchLabelWidth = torch.floor(labelHeight * aspectRatio)
      for n = 1, maxN do
         -- display progress
         xlua.progress(trc + tec - 1, trsize + tesize)

         -- load training images:
         local dataTemp = image.scale(imageBatch[n], batchImgWidth, opt.imHeight)
         local inputData = image.crop(dataTemp, 'c', opt.imWidth, opt.imHeight)
         if IDX <= 4 then
            trainData.data[trc] = inputData
         else
            testData.data[tec] = inputData
         end

         --load training labels:
         dataTemp = image.scale(labelBatch[n], batchLabelWidth, opt.labelHeight, 'simple')
         local inputLabel = image.crop(dataTemp, 'c', opt.labelWidth, opt.labelHeight)

         inputLabel:apply(function(x) return classMap[x] end)

         if IDX <= 4 then
            -- Syntax: histc(data, bins, min, max)
            histClasses = histClasses + torch.histc(inputLabel, #classes, 1, #classes)
            trainData.labels[trc] = inputLabel
            trc = trc + 1
         else
            testData.labels[tec] = inputLabel
            tec = tec + 1
         end
         collectgarbage()
      end
   end
   collectgarbage()
end

if opt.cachepath ~= "none" and not loadedFromCache then
    print('==> saving data to cache: ' .. sunCachePath)
    local dataCache = {
       trainData = trainData,
       testData = testData,
       histClasses = histClasses,
       testHistClasses = testHistClasses
    }
    torch.save(sunCachePath, dataCache)
    dataCache = nil
    collectgarbage()
end

print '==> preprocessing data: normalize each feature (channel) globally'
--local mean = {}
--local std = {}
--for i = 1, opt.channels do
--   -- normalize each channel globally:
--   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
--   std[i] = trainData.data[{ {},i,{},{} }]:std()
--   trainData.data[{ {},i,{},{} }]:add(-mean[i])
--   trainData.data[{ {},i,{},{} }]:div(std[i])
--
--   testData.data[{ {},i,{},{} }]:add(-mean[i])
--   testData.data[{ {},i,{},{} }]:div(std[i])
--end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i = 1, opt.channels do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('training data, '..i..'-channel, mean: ' .. trainMean)
   print('training data, '..i..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..i..'-channel, mean: ' .. testMean)
   print('test data, '..i..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------
local classes_td = {[1] = 'classes,targets\n'}
for _,cat in pairs(classes) do
   table.insert(classes_td, cat .. ',1\n')
end

local file = io.open(paths.concat(opt.save, 'categories.txt'), 'w')
file:write(table.concat(classes_td))
file:close()

-- Exports
opt.dataClasses = classes
opt.dataconClasses = conClasses
opt.datahistClasses = histClasses

return {
   trainData = trainData,
   testData = testData,
   mean = trainMean,
   std = trainStd
}
