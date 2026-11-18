-- Transmitter switch (RCn_OPTION = 300) between the real GPS and our GPS_INPUT fixes.
-- Both changes have to move together: source set 1 asks for GPS velocity, which our GPS cannot give.
local SWITCH_OPTION = 300
local HIGH = 2
local MS_PER_CHECK = 100
local MAV_SEVERITY_WARNING = 4
local REAL_GPS = 0
local VISION_GPS = 1

local switch = nil
local last = -1

function update()
  if switch == nil then
    switch = rc:find_channel_for_option(SWITCH_OPTION)
  end
  local pos = switch and switch:get_aux_switch_pos() or 0
  if pos ~= last then
    last = pos
    if pos == HIGH then
      param:set('GPS_PRIMARY', VISION_GPS)
      ahrs:set_posvelyaw_source_set(VISION_GPS)
      gcs:send_text(MAV_SEVERITY_WARNING, 'Vision GPS: ON')
    else
      param:set('GPS_PRIMARY', REAL_GPS)
      ahrs:set_posvelyaw_source_set(REAL_GPS)
      gcs:send_text(MAV_SEVERITY_WARNING, 'Vision GPS: OFF')
    end
  end
  return update, MS_PER_CHECK
end

return update()
