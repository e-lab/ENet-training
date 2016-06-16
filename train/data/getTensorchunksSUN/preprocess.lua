-- Made by Sangpil Kim
-- Jun 2016
require 'nn'
require 'xlua'
require 'paths'
require 'image'
local m = require 'matio'
local pl = require('pl.import_into')()

torch.setdefaulttensortype('torch.FloatTensor')

function lines_from(file)
  lines = {}
  for line in io.lines(file) do
    lines[#lines + 1] = line
  end
  return lines
end

-- creating files
if not paths.dirp(paths.cwd()..'/imgs') and not paths.dirp(paths.cwd()..'/labels') then
   print('create imgs and labels folder')
   paths.mkdir('imgs')
   paths.mkdir('labels')
else
   error('imgs and labels folder alredy exist')
end

local height, width, conti, contl, ipath, lfile, ifile
local chi , chl, inTemp, batch = 3 , 1, 0, 0
local sum, cn, max = 0, 0, 8

-- load indexTable and paths
local indexTable = { {1860, 530, 730},
               {2790, 427, 561},
               {4478, 441, 591},
               {5050, 531, 681},
               {6974, 530, 730},
               {8047, 427, 561},
               {9748, 441, 591},
               {10335,531, 681}
               }

local paths = lines_from('sunImgPath.tsv')

for i = 1, max do

   -- get dim width height from index
   print(tostring(i)..'th chunk out of 8 chunks')
   height = indexTable[i][2]
   width  = indexTable[i][3]
   sum = sum + batch
   batch  = indexTable[i][1] - sum

   --init container
   conti = torch.LongTensor(batch,chi,height,width)
   contl = torch.LongTensor(batch,chl,height,width)

   -- fill in container
   for j = sum + 1 , sum + batch do

      -- get label file path
      ipath = paths[j]..'/image/'

      -- get image file path
      for _, f in ipairs(pl.dir.getallfiles(ipath, '*.jpg')) do
         ifile = f
      end

      lfile = paths[j]..'/seg.mat'

      -- load images and label
      conti[i] = image.load(ifile)
      contl[i] = m.load(lfile).seglabel
      cn = cn + 1
   end

   xlua.progress(10335,cn)

   -- save files
   print('saving '..tostring(i)..'th chunk imgs')
   torch.save('imgs/tensorImgs'..tostring(i)..'.t7',conti:float())
   print('saving '..tostring(i)..'th chunk labels')
   torch.save('labels/tensorLabels'..tostring(i)..'.t7',conti:float())
   print('Done')

end

