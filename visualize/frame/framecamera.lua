local frame = {}
local sys = assert(require('sys'))
torch.setdefaulttensortype('torch.FloatTensor')

local function prep_lua_camera(opt, source)

   assert(require('camera'))
   local cam = image.Camera {
      idx      = opt.cam,
      width    = source.w,
      height   = source.h,
   }
   frame.forward = function(img)
      return cam:forward()
   end
   source.cam = cam

end

local function prep_libvideo_decoder_camera(opt, source)

   local cam = require('libvideo_decoder')
   if not cam.capture('/dev/video'..opt.cam, source.w, source.h, source.fps, 3) then
      error('cam.capture failed')
   end

   if opt.spatial == 1 then -- In spatial mode 1 we can already get the scaled image
      source.w = opt.is * source.w / source.h
      source.h = opt.is
      framefunc = cam.frame_resized
   else
      framefunc = cam.frame_rgb
   end
   local img_tmp = torch.FloatTensor(3, source.h, source.w)

   -- set frame forward function
   frame.forward = function(img)
      if not framefunc(img_tmp) then
         return nil
      end
      return img_tmp
   end
   source.cam = cam

end

--[[
   opt fields
      is          eye size
      input       cam0, cam1...
      spatial     spatial mode 0, 1 or 2

   source fields
      w           image width
      h           image height
--]]
function frame:init(opt, source)

   opt.cam = tonumber(string.sub(opt.input,4,-1)) -- opt.input is in the format cam0
   if sys.OS == 'macos' then
      prep_lua_camera(opt, source)
   else
      prep_libvideo_decoder_camera(opt, source)
   end

end

return frame
