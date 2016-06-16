----------------------------------------------------------------------
-- indoor SUN data loader,
-- SangPil Kim, Eugenio Culurciello,
-- March 2016
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')
local sizeFileName, histFileName, histLogger, sizeLogger
local cacheDir = paths.concat(opt.cachepath, 'sun')

sizeFileName = paths.concat(cacheDir, 'from_'..IDX..'th_size.txt')
histFileName = paths.concat(cacheDir, 'from_'..IDX..'th_hist.txt')
sizeLogger = optim.Logger(sizeFileName)
histLogger = optim.Logger(histFileName)
---------------------------------------------------------------------
function ft ()
    --dumy classes if becuase of the error whil calclating the histogram it will be niled when it saved
    classes = {'unlabed','wall', 'floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books','fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag'}
    --print(#classes)
    conClasses = {'wall', 'floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books','fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink','lamp','bathtub','bag'}
    opt.channels = {'r','g','b'}
    classMap = {}
    --------------------------------------------------------------------------------
    print '--> remapping classes'
    --0 is unlabeled and there are 894 classes in the original dataset
    for i=0, #classes-1 do
        classMap[i] = i+1
    end
    --print(classMap)
    -- set size of labels

    -- saving training histogram of classes
    histClasses = torch.Tensor(#classes+1):zero()
    --print('==> number of classes: ' .. #classes ..', classes: ', classes)

    --------------------------------------------------------------------------------
    print '==> loading indoor dataset'
    --local trainData, testData, tmpO
    local loadedFromCache = false
    local cachePath, imgs, labels
    local testIDX = 5
    local labelNum , labelWidth, labelHeight
    local dataSetSize
    if opt.trainMode == 1 then
        cachePath = paths.concat(cacheDir, IDX..'dataMode1.t7')
    else
        cachePath = paths.concat(cacheDir, IDX..'data.t7')
    end
    --create dir if not exist
    if not paths.dirp(cacheDir) then paths.mkdir(cacheDir) end
    --allocate from cache if exists
    if opt.cachepath ~= "none" and paths.filep(cachePath) then
       print('Loading '..tostring(IDX)..'th cache data')
       local dataCache = torch.load(cachePath)
       if IDX < testIDX then
           assert(dataCache.trainData ~= nil, 'No trainData')
           trainData = dataCache.trainData
           labelNum = dataCache.trainData.labels:size(1)
           labelWidth = dataCache.trainData.labels:size(2)
           labelHeight = dataCache.trainData.labels:size(3)
       else
           assert(dataCache.testData ~= nil, 'No testData')
           testData = dataCache.testData
           labelNum = dataCache.testData.labels:size(1)
           labelWidth = dataCache.testData.labels:size(2)
           labelHeight = dataCache.testData.labels:size(3)
       end
       histClasses = dataCache.histClasses

       loadedFromCache = true
       dataCache = nil

       collectgarbage()
    else
        -- create new data and save as cache
        -- tmpData for tmporary store images and labels
        local tmpData = {data = {},  labels = {}}

        print('==> loading training files');

        local dpath = paths.concat(opt.datapath, 'Images/')
        local lpath = paths.concat(opt.datapath,'Labels/')

        assert(paths.dirp(dpath), 'No training folder found at: ' .. opt.datapath)
        --load training images and labels:
        local c, tc, lct, cardinality = 1, 1, 1, 0
        k = paths.concat(lpath,'label.t7')
        imgs = torch.load(paths.concat(dpath,'tensorImgs'..IDX..'.t7'))
        labels = torch.load(paths.concat(lpath,'tensorLabels'..IDX..'.t7'))
        print(labels:max())
        print(labels:size())
        originHeight = labels:size(2)
        originWidth  = labels:size(3)
        local crop = false
        local cropWidth = originHeight * 1.28
        if cropWidth < originWidth then crop = true end
        opt.imWidth  = opt.imHeight * 1.28

        --print(opt.imHeight)
        --print(opt.imWidth)
        for i=1, labels:size(1) do
              cardinality = labels:size(1)
              -- process each image
              --print(img_path)
              --load training images:
              local tmp = imgs[i]
              local sum = tmp:sum()
              if sum ~= 0 then
                 if crop == true then
                    tmp = image.crop(tmp,'c', cropWidth, originHeight)
                 end
                 tmp = image.scale(tmp, opt.imWidth,opt.imHeight, 'simple')
                 tmpData.data[tc] = tmp
                 tc = tc + 1
              end
           local labelIn = labels[lct]
           -- print(labelIn:max())
           -- print(labelIn:type())
           if crop == true then
              labelIn = image.crop(labelIn,'c', cropWidth, originHeight)
           end
            labelFile = image.scale(labelIn,opt.labelWidth, opt.labelHeight,'simple')
            labelFile:apply(function(x) return classMap[x] end)
            histClasses = histClasses + torch.histc(labelFile:float(),#classes+1,1,#classes+1)
            print(histClasses)
            print(labelFile:size())
            print(#classes)
            print(c)
            -- convert to int and write to data structure:
            if labelFile:sum() ~= 0 then
                tmpData.labels[lct] = labelFile
                lct = lct + 1
            end
            c = c + 1
        end
        assert(#tmpData.data == #tmpData.labels ,'label and data size differnt')
        histClasses = histClasses:sub(1,#classes)
        print(classes)
        print 'Reduced histclasses'
        print(histClasses)
        print ('Total number of valid images :'..#tmpData.data)
        -- Normalize each channel, and store mean/std
        -- per channel. These values are important, as they are part of
        -- the trainable parameters. At test time, test data will be normalized
        -- using these values.

        print('==> creating caches with 8 imgs and labels chunks');
        xlua.progress(IDX,8)
        local tosize = #tmpData.data
        local shuffle = torch.randperm(tosize)
        if IDX < testIDX then
            trainData = {
               data = torch.FloatTensor(tosize, opt.channels, opt.imHeight, opt.imWidth),
               labels = torch.FloatTensor(tosize, opt.labelHeight, opt.labelWidth),
               size = function() return tosize end,
               preverror = 1e10, -- a really huge number
            }
            print 'assigning training images'
            for i = 1, tosize do
                trainData.data[i]   = tmpData.data[shuffle[i]]
                trainData.labels[i] = tmpData.labels[shuffle[i]]
            end
        else
            testData = {
               data = torch.FloatTensor(tosize, opt.channels, opt.imHeight, opt.imWidth),
               labels = torch.FloatTensor(tosize, opt.labelHeight, opt.labelWidth),
               size = function() return tosize end,
               preverror = 1e10, -- a really huge number
            }
            print 'assigning test images'
            for i = 1, tosize  do
                testData.data[i]   = tmpData.data[shuffle[i]]
                testData.labels[i] = tmpData.labels[shuffle[i]]
            end
        end
        tmpData = nil
        print '==> preprocessing data: normalize each feature (channel) globally'
        if IDX == 1 then
           opt.idxmean = {}
           opt.idxstd = {}
        end
         local mean = {}
         local std = {}
        if IDX < testIDX then
            -- normalize each channel globally:
            for i = 1, opt.channels do
               mean[i] = trainData.data[{ {},i,{},{} }]:mean()
               std[i] = trainData.data[{ {},i,{},{} }]:std()
               trainData.data[{ {},i,{},{} }]:add(-mean[i])
               trainData.data[{ {},i,{},{} }]:div(std[i])
            end
          opt.idxmean[IDX] = mean
          opt.idxstd[IDX] = mean
        else
           for i = 1, opt.channels do
               mean[i] = testData.data[{ {},i,{},{} }]:mean()
               std[i] = testData.data[{ {},i,{},{} }]:std()
              testData.data[{ {},i,{},{} }]:add(-mean[i])
              testData.data[{ {},i,{},{} }]:div(std[i])
           end
        end
        collectgarbage()
    end
    collectgarbage()
    if opt.cachepath ~= "none" and not loadedFromCache then
        print('==> saving data to cache: ' .. cachePath)
        local dataCache
        if IDX < testIDX then
           dataCache = {
              trainData = trainData,
              classes = classes,
              histClasses = histClasses,
           }
        else
           dataCache = {
              testData = testData,
              classes = classes,
              histClasses = histClasses,
           }
        end
        torch.save(cachePath, dataCache)
        if IDX < testIDX then
            dataSetSize = {trainData.labels:size(1),trainData.labels:size(2),trainData.labels:size(3)}
        else
            dataSetSize = {testData.labels:size(1),testData.labels:size(2),testData.labels:size(3)}
        end
        print('==> saving size info to txt: ' .. sizeFileName)
        print('==> saving hist info to txt: ' .. histFileName)
        sizeLogger:add{'SUN '..tostring(IDX)..'th dataset\n num:'..tostring(dataSetSize[1])..
        '\n height : '..tostring(dataSetSize[2])..'\n width : '..tostring(dataSetSize[3])}
        histLogger:add{IDX..'th histClasses\n'..tostring(histClasses)..'\n'}
        dataSetSize = nil
        dataCache = nil
        collectgarbage()
    end


    local classes_td = {[1] = 'classes,targets\n'}
    for _,cat in pairs(classes) do
       table.insert(classes_td, cat .. ',1\n')
    end

    os.execute('mkdir -p ' .. opt.save)
    local file = io.open(paths.concat(opt.save, 'categories.txt'), 'w')
    file:write(table.concat(classes_td))
    file:close()

    -- Exports
    opt.dataclasses = classes
    opt.dataconClasses = conClasses
    opt.datahistClasses = histClasses
    if IDX < testIDX then
       return {
          trainData = trainData,
          mean = trainMean,
          std = trainStd
       }
   else
       return {
          testData = testData,
          mean = trainMean,
          std = trainStd
       }
   end
end
return ft
