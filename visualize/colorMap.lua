--------------------------------------------------------------------------------
-- Returns the color map for a given set of classes.
--
-- If you want to add more classes then modifiy section "Colors => Classes"
--
-- Written by: Abhishek Chaurasia, Sangpil Kim
-- Date      : March, 2016
--------------------------------------------------------------------------------

local colorMap = {}

-- Map color names to a shorter name
local red = 'red'
local gre = 'green'
local blu = 'blue'

local mag = 'magenta'
local yel = 'yellow'
local cya = 'cyan'

local gra = 'gray'
local whi = 'white'
local bla = 'black'

local lbl = 'lemonBlue'
local bro = 'brown'
local neg = 'neonGreen'
local pin = 'pink'
local pur = 'purple'
local kha = 'khaki'
local gol = 'gold'

-- Create color palette for all the defined colors
local colorPalette = {[red] = {1.0, 0.0, 0.0},
                      [gre] = {0.0, 1.0, 0.0},
                      [blu] = {0.0, 0.0, 1.0},
                      [mag] = {1.0, 0.0, 1.0},
                      [yel] = {1.0, 1.0, 0.0},
                      [cya] = {0.0, 1.0, 1.0},
                      [gra] = {0.3, 0.3, 0.3},
                      [whi] = {1.0, 1.0, 1.0},
                      [bla] = {0.0, 0.0, 0.0},
                      [lbl] = {30/255, 144/255,  255/255},
                      [bro] = {139/255, 69/255,   19/255},
                      [neg] = {202/255, 255/255, 112/255},
                      [pin] = {255/255, 20/255,  147/255},
                      [pur] = {128/255, 0/255,   128/255},
                      [kha] = {240/255, 230/255, 140/255},
                      [gol] = {255/255, 215/255,   0/255}}

-- Default color is chosen as black
local defaultColor = colorPalette[bla]

local function prepDrivingColors(classes)
   local colors = {}

   local mapping = {
      Unlabeled     = colorPalette[bla],

      EgoVehicle    = colorPalette[blu],

      Road          = colorPalette[gra],
      Sidewalk      = colorPalette[gra],
      Ground        = colorPalette[gra],
      Parking       = colorPalette[gra],
      RailTrack     = colorPalette[gra],

      Person        = colorPalette[mag],
      Rider         = colorPalette[mag],
      Pedestrian    = colorPalette[mag],

      Car           = colorPalette[red],
      Bus           = colorPalette[red],
      Truck         = colorPalette[red],
      Bicycle       = colorPalette[red],
      Motorcycle    = colorPalette[red],
      Trailer       = colorPalette[red],
      Train         = colorPalette[red],

      Building      = colorPalette[yel],
      Fence         = colorPalette[yel],
      Wall          = colorPalette[yel],

      Vegetation    = colorPalette[gre],
      Tree          = colorPalette[gre],
      Terrain       = colorPalette[gre],
      Plant         = colorPalette[gre],

      Pole          = colorPalette[cya],
      TrafficSign   = colorPalette[cya],
      TrafficLight  = colorPalette[cya],

      Sky           = colorPalette[whi],
   }

   for i,class in ipairs(classes) do
      colors[i] = mapping[class] or defaultColor
   end

   colorMap.getColors = function()
      return colors
   end
end

local function prepCamVidColors(classes)
   local colors = {}

   -- Assign default colors to all the classes
   for i = 1, #classes do
      table.insert(colors, defaultColor)
   end

   -- Colors => Classes
   -- Assign specific color to respective classes
   for i = 1, #classes do
      if classes[i] == 'Misc' then
         colors[i] = colorPalette[pur]
      elseif classes[i] == 'Building' then
         colors[i] = colorPalette[kha]
      elseif classes[i] == 'Bicyclist' then
         colors[i] = colorPalette[gre]
      elseif classes[i] == 'Car' then
         colors[i] = colorPalette[blu]
      elseif classes[i] == 'CarLuggagePram' then
         colors[i] = colorPalette[mag]
      elseif classes[i] == 'Pedestrian' then
         colors[i] = colorPalette[yel]
      elseif classes[i] == 'Pole' then
         colors[i] = colorPalette[cya]
      elseif classes[i] == 'Fence' then
         colors[i] = colorPalette[gra]
      elseif classes[i] == 'LameMkgs' then
         colors[i] = colorPalette[lbl]
      elseif classes[i] == 'MiscText' then
         colors[i] = colorPalette[red]
      elseif classes[i] == 'OtherMoving' then
         colors[i] = colorPalette[whi]
      elseif classes[i] == 'Road' then
         colors[i] = colorPalette[bro]
      elseif classes[i] == 'Sidewalk' then
         colors[i] = colorPalette[neg]
      elseif classes[i] == 'SignSymbol' then
         colors[i] = colorPalette[pin]
      elseif classes[i] == 'Sky' then
         colors[i] = colorPalette[whi]
      elseif classes[i] == 'Tree' then
         colors[i] = colorPalette[gre]
      elseif classes[i] == 'TruckBus' then
         colors[i] = colorPalette[blu]
      elseif classes[i] == 'Void' then
         colors[i] = colorPalette[pur]
      end
   end
   colorMap.getColors = function()
      return colors
   end
end

local function prepIndoorColors(classes)
   local colors = {}

   local mapping = {
      unlabeled      = colorPalette[bla],
      ceiling        = colorPalette[red],
      wall           = colorPalette[whi],
      floor          = colorPalette[gre],

      door           = colorPalette[blu],
      window         = colorPalette[blu],

      picture        = colorPalette[mag],
      whiteboard     = colorPalette[mag],
      bookshelf      = colorPalette[mag],
      shelves        = colorPalette[mag],
      books          = colorPalette[mag],

      cabinet        = colorPalette[bro],
      table          = colorPalette[bro],
      dresser        = colorPalette[bro],
      desk           = colorPalette[bro],
      counter        = colorPalette[bro],

      chair          = colorPalette[yel],
      sofa           = colorPalette[yel],
      bed            = colorPalette[yel],

      blinds         = colorPalette[pur],
      curtain        = colorPalette[pur],
      shower_curtain = colorPalette[pur],
      clothes        = colorPalette[pur],
      pillow         = colorPalette[pur],

      person         = colorPalette[cya],

      floor_mat      = colorPalette[gra],
      bag            = colorPalette[gra],
      box            = colorPalette[gra],
      fridge         = colorPalette[gra],
      mirror         = colorPalette[gra],
      tv             = colorPalette[gra],
      paper          = colorPalette[gra],
      night_stand    = colorPalette[gra],
      lamp           = colorPalette[gra],

      towel          = colorPalette[bla],
      toilet         = colorPalette[bla],
      sink           = colorPalette[bla],
      bathtub        = colorPalette[bla],
   }

   for i,class in ipairs(classes) do
      colors[i] = mapping[class] or defaultColor
   end

   colorMap.getColors = function()
      return colors
   end
end

function colorMap:init(opt, classes)
   if opt.dataset == 'su' then
      prepIndoorColors(classes)
   elseif opt.dataset == 'cs'then
      prepDrivingColors(classes)
   elseif opt.dataset == 'cv'then
      prepCamVidColors(classes)
   else
      prepDrivingColors(classes)
   end
end

return colorMap
