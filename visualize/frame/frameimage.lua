local frame = {}
torch.setdefaulttensortype('torch.FloatTensor')

--[[
   opt fields
      is          eye size
      input       filename
      batch       batch size

   source fields
      w           image width
      h           image height

--]]
function frame:init(opt, source)

   local fi = require('fastimage')
   local batch = opt.batch
   local filenames
   local first = true
   local resolutions = nil

   fi.init(opt.input, opt.batch, 0, 0, 0.5)
   source.img, filenames = fi.load(nil)
   if source.img then
      source.w = source.img:size(4)
      source.origw = source.w
      source.h = source.img:size(3)
      source.origh = source.h
   end
   if batch == 1 and source.img then
      source.img = source.img[1]
      filenames = filenames[1]
   end

   frame.forward = function(img, res_idx)
      if first then
         first = false
         return source.img, filenames
      else
         source.img, filenames = fi.load(img, res_idx)
         if batch == 1 and source.img then
            source.img = source.img[1]
            filenames = filenames[1]
         end
         return source.img, filenames
      end
   end

end

return frame
