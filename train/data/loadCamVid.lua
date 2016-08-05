----------------------------------------------------------------------
-- CamVid data loader,
-- Abhishek Chaurasia,
-- May 2016
----------------------------------------------------------------------

require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

local trainFile = opt.datapath .. '/train.txt'
local testFile = opt.datapath .. '/test.txt'

----------------------------------------------------------------------

local classes = {'Void', 'Sky', 'Building', 'Column-Pole',
                 'Road', 'Sidewalk', 'Tree', 'Sign-Symbol',
                 'Fence', 'Car', 'Pedestrian', 'Bicyclist'}

local conClasses = {'Sky', 'Building', 'Column-Pole',
                 'Road', 'Sidewalk', 'Tree', 'Sign-Symbol',
                 'Fence', 'Car', 'Pedestrian', 'Bicyclist'}

print('==> number of classes: ' .. #classes ..', classes: ', classes)

----------------------------------------------------------------------
-- saving training histogram of classes
local histClasses = torch.Tensor(#classes):fill(0)

local trainData
local testData

-- Function to read txt file and return image and ground truth path
function getPath(filepath)
   print("Extracting file names from: " .. filepath)
   local file = io.open(filepath, 'r')
   local imgPath = {}
   local gtPath = {}
   file:read()    -- throw away first line
   local fline = file:read()
   while fline ~= nil do
      local col1, col2 = fline:match("([^,]+),([^,]+)")
      col1 = opt.datapath .. col1
      col2 = opt.datapath .. col2
      table.insert(imgPath, col1)
      table.insert(gtPath, col2)
      fline = file:read()
   end
   return imgPath, gtPath
end

----------------------------------------------------------------------
-- Main section
local loadedFromCache = false
local cacheDir = paths.concat(opt.cachepath, 'camVid')
local camvidCachePath = paths.concat(opt.cachepath, 'camVid', 'data.t7')

if not paths.dirp(cacheDir) then paths.mkdir(cacheDir) end

if opt.cachepath ~= "none" and paths.filep(camvidCachePath) then
   print('Loading cache data')
   local dataCache = torch.load(camvidCachePath)
   assert(dataCache.trainData ~= nil, 'No trainData')
   assert(dataCache.testData ~= nil, 'No testData')
   trainData = dataCache.trainData
   testData = dataCache.testData
   histClasses = dataCache.histClasses
   loadedFromCache = true
   dataCache = nil
   collectgarbage()
else
   ----------------------------------------------------------------------
   -- Acquire image and ground truth paths for training set
   local imgPath, gtPath = getPath(trainFile)

   -- initialize data structures:
   trainData = {
      data = torch.FloatTensor(#imgPath, 3, opt.imHeight , opt.imWidth),
      labels = torch.FloatTensor(#imgPath, opt.labelHeight , opt.labelWidth),
      preverror = 1e10, -- a really huge number
      size = function() return trainData.data:size(1) end
   }

   print "==> Loading traning data"
   for i = 1, #imgPath do
      -- load original image
      local rawImg = image.load(imgPath[i])

      if (opt.imHeight == rawImg:size(2)) and
         (opt.imWidth == rawImg:size(3)) then
         trainData.data[i] = rawImg
      else
         trainData.data[i] = image.scale(rawImg, opt.imWidth, opt.imHeight)
      end

      -- load corresponding ground truth
      rawImg = image.load(gtPath[i], 1, 'byte'):squeeze():float() + 2
      local mask = rawImg:eq(13):float()
      rawImg = rawImg - mask * #classes

      if (opt.labelHeight == rawImg:size(2)) and
         (opt.labelWidth == rawImg:size(3)) then
         trainData.labels[i] = rawImg
      else
         trainData.labels[i] = image.scale(rawImg, opt.labelWidth, opt.labelHeight, 'simple')
      end
      histClasses = histClasses + torch.histc(trainData.labels[i], #classes, 1, #classes)
      xlua.progress(i, #imgPath)
      collectgarbage()
   end

   ----------------------------------------------------------------------
   -- Acquire image and ground truth paths for testing set
   imgPath, gtPath = getPath(testFile)

   testData = {
      data = torch.FloatTensor(#imgPath, 3, opt.imHeight , opt.imWidth),
      labels = torch.FloatTensor(#imgPath, opt.labelHeight , opt.labelWidth),
      preverror = 1e10, -- a really huge number
      size = function() return testData.data:size(1) end
   }

   print "\n==> Loading testing data"
   for i = 1, #imgPath do
      -- load original image
      local rawImg = image.load(imgPath[i])

      if (opt.imHeight == rawImg:size(2)) and
         (opt.imWidth == rawImg:size(3)) then
         testData.data[i] = rawImg
      else
         testData.data[i] = image.scale(rawImg, opt.imWidth, opt.imHeight)
      end

      -- load corresponding ground truth
      rawImg = image.load(gtPath[i], 1, 'byte'):squeeze():float() + 2
      local mask = rawImg:eq(13):float()
      rawImg = rawImg - mask * #classes

      if (opt.labelHeight == rawImg:size(2)) and
         (opt.labelWidth == rawImg:size(3)) then
         testData.labels[i] = rawImg
      else
         testData.labels[i] = image.scale(rawImg, opt.labelWidth, opt.labelHeight, 'simple')
      end

      --for j = 0, 11 do
      --   local mask = rawImg:eq(j)
      --   local layerImg = mask:maskedSelect(mask)
      --   image.display(layerImg)
      --   io.read()
      --end
      xlua.progress(i, #imgPath)
      collectgarbage()
   end

   collectgarbage()
end

----------------------------------------------------------------------
if opt.cachepath ~= "none" and not loadedFromCache then
    print('==> saving data to cache: ' .. camvidCachePath)
    local dataCache = {
       trainData = trainData,
       testData = testData,
       histClasses = histClasses,
    }
    torch.save(camvidCachePath, dataCache)
    dataCache = nil
    collectgarbage()
end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i = 1, opt.channels do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('training data, channel-'.. i ..', mean: ' .. trainMean)
   print('training data, channel-'.. i ..', standard deviation: ' .. trainStd)

   print('test data, channel-'.. i ..', mean: ' .. testMean)
   print('test data, channel-'.. i ..', standard deviation: ' .. testStd)
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
