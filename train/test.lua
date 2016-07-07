----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data.
--
-- Written by  : Abhishek Chaurasia, Eugenio Culurcielo
-- Dated       : January 2016
-- Last updated: June, 2016
----------------------------------------------------------------------

require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
-- Logger:
errorLogger = optim.Logger(paths.concat(opt.save, 'error.log'))
coTotalLogger = optim.Logger(paths.concat(opt.save, 'confusionTotal.log'))
coAveraLogger = optim.Logger(paths.concat(opt.save, 'confusionAvera.log'))
coUnionLogger = optim.Logger(paths.concat(opt.save, 'confusionUnion.log'))

print '==> defining test procedure'
local teconfusion, filename

if opt.dataconClasses then
   teconfusion = optim.ConfusionMatrix(opt.dataconClasses)
else
   teconfusion = optim.ConfusionMatrix(opt.dataClasses)
end

-- Batch test:
local x = torch.Tensor(opt.batchSize, opt.channels, opt.imHeight, opt.imWidth)
local yt = torch.Tensor(opt.batchSize, opt.labelHeight, opt.labelWidth)
x = x:cuda()
yt = yt:cuda()

-- test function
function test(testData, classes, epoch, trainConf, model, loss )
   ----------------------------------------------------------------------
   -- local vars
   local time = sys.clock()
   -- total loss error
   local err = 0
   local totalerr = 0

   -- This matrix records the current confusion across classes

   model:evaluate()

   -- test over test data
   print('==> Testing:')
   for t = 1, testData:size(), opt.batchSize do
      -- disp progress
      xlua.progress(t, testData:size())

      -- batch fits?
      if (t + opt.batchSize - 1) > testData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         x[idx] = testData.data[i]
         yt[idx] = testData.labels[i]
         idx = idx + 1
      end

      -- test sample
      local y = model:forward(x)

      err = loss:forward(y,yt)
      if opt.noConfusion == 'tes' or opt.noConfusion == 'all' then
         local y = y:transpose(2, 4):transpose(2, 3)
         y = y:reshape(y:numel()/y:size(4), #classes):sub(1, -1, 2, #opt.dataClasses)
         local _, predictions = y:max(2)
         predictions = predictions:view(-1)
         local k = yt:view(-1)
         if opt.dataconClasses then k = k - 1 end
         teconfusion:batchAdd(predictions, k)
      end

      totalerr = totalerr + err
      collectgarbage()
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')


   -- print average error in train dataset
   totalerr = totalerr / (testData:size() / opt.batchSize)
   print('Test Error: ', totalerr )
   -- save/log current net
   errorLogger:add{['Training error'] = trainError,
                   ['Testing error'] = totalerr}
   if opt.plot then
      errorLogger:style{['Training error'] = '-',
      ['Testing error'] = '-'}
      errorLogger:plot()
   end
   if totalerr < testData.preverror then
      filename = paths.concat(opt.save, 'model-best.net')
      print('==> saving model to '..filename)

      torch.save(filename, model:clearState():get(1))
      -- update to min error:
      if opt.noConfusion == 'tes' or opt.noConfusion == 'all' then
         coTotalLogger:add{['confusion total accuracy'] = teconfusion.totalValid * 100 }
         coAveraLogger:add{['confusion average accuracy'] = teconfusion.averageValid * 100 }
         coUnionLogger:add{['confusion union accuracy'] = teconfusion.averageValid * 100 }

         filename = paths.concat(opt.save,'confusion-'..epoch..'.t7')
         print('==> saving confusion to '..filename)
         torch.save(filename,teconfusion)

         filename = paths.concat(opt.save, 'confusionMatrix-best.txt')
         print('==> saving confusion matrix to ' .. filename)
         local file = io.open(filename, 'w')
         file:write("--------------------------------------------------------------------------------\n")
         file:write("Training:\n")
         file:write("================================================================================\n")
         file:write(tostring(trainConf))
         file:write("\n--------------------------------------------------------------------------------\n")
         file:write("Testing:\n")
         file:write("================================================================================\n")
         file:write(tostring(teconfusion))
         file:write("\n--------------------------------------------------------------------------------")
         file:close()
      end
      filename = paths.concat(opt.save, 'best-number.txt')
      local file = io.open(filename, 'w')
      file:write("----------------------------------------\n")
      file:write("Best test error: ")
      file:write(tostring(totalerr))
      file:write(", in epoch: ")
      file:write(tostring(epoch))
      file:write("\n----------------------------------------\n")
      file:close()
      if totalerr < testData.preverror then testData.preverror = totalerr end
   end
   -- cudnn.convert(model, nn)
   local filename = paths.concat(opt.save, 'model-'..epoch..'.net')
   --os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model:clearState():get(1))
   if opt.noConfusion == 'tes' or opt.noConfusion == 'all' then
      -- update to min error:
      filename = paths.concat(opt.save, 'confusionMatrix-' .. epoch .. '.txt')
      print('==> saving confusion matrix to ' .. filename)
      local file = io.open(filename, 'w')
      file:write("--------------------------------------------------------------------------------\n")
      file:write("Training:\n")
      file:write("================================================================================\n")
      file:write(tostring(trainConf))
      file:write("\n--------------------------------------------------------------------------------\n")
      file:write("Testing:\n")
      file:write("================================================================================\n")
      file:write(tostring(teconfusion))
      file:write("\n--------------------------------------------------------------------------------")
      file:close()
      filename = paths.concat(opt.save, 'confusion-'..epoch..'.t7')
      print('==> saving test confusion object to '..filename)
      torch.save(filename,teconfusion)
      --resetting confusionMatrix
      trainConf:zero()
      teconfusion:zero()
   end
   if totalerr < testData.preverror then testData.preverror = totalerr end

   print('\n') -- separate epochs

   collectgarbage()
end

-- Export:
return test
