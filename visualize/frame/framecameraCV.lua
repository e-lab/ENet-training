local frame = {}
local sys = assert(require('sys'))
torch.setdefaulttensortype('torch.FloatTensor')

local function prep_lua_camera(opt, source)

   local cv = require 'cv'
   require 'cv.highgui'
   require 'cv.videoio'
   require 'cv.imgproc'

   local cap               -- openCV videoCapture object
--------------------------------------------------------------------------------
   -- check if input is a camera device or a video file
   local camID = tonumber(string.sub(opt.input,4,-1))
   cap = cv.VideoCapture{camID}
   if not cap:isOpened() then
      print("Failed to open the default camera")
      os.exit(-1)
   end
   local res = {HVGA  = {w= 320, h= 240},
                QHD   = {w= 640, h= 360},
                VGA   = {w= 640, h= 480},
                FWVGA = {w= 854, h= 480},
                HD    = {w=1280, h= 720},
                FHD   = {w=1920, h=1080}}

   cap:set{cv.CAP_PROP_FRAME_WIDTH, res[opt.camRes].w}
   cap:set{cv.CAP_PROP_FRAME_HEIGHT, res[opt.camRes].h}
   -- cap:set{cv.CAP_PROP_FPS, opt.fps}
   print("Accessing camera (" .. camID .. ")")
   print("Camera resolution set to: " .. res[opt.camRes].h .. " x " .. res[opt.camRes].w)
   -- capture the first frame
   local _, frameCV = cap:read{frameCV}

   -- get the resolution of the frame
   local height = frameCV:size(1) * opt.ratio
   local width = frameCV:size(2) * opt.ratio
   local frameRes = {height, width}

   frame.forward = function(img)
      -- read next frame
      cap:read{frameCV}

      frameCV = cv.flip{src = frameCV, flipCode = 1}
      -- Go from BGR=>RGB and then from hxwx3 => 3xhxw
      cv.cvtColor{frameCV, frameCV, cv.COLOR_BGR2RGB}
      local frameTH = frameCV:transpose(1,3):transpose(2,3):clone():float()/255
      return(frameTH)
   end
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
