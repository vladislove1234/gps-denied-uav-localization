# Feeding our visual position into ArduPilot as a GPS (October 2026)

How the Raspberry Pi 5 running the localiser can hand its fixes to an ArduPilot flight controller, so the vehicle can either
**log them next to the real GPS** or **fly on them** (Loiter, Auto missions). Everything below was checked against the
ArduPilot `master` source (`libraries/AP_GPS/AP_GPS_MAV.cpp`, `AP_GPS.cpp`, `AP_NavEKF3/*`, `AP_Arming/*`), not only the wiki,
because the wiki is silent on the details that matter here. Written assuming a multicopter (ArduCopter 4.6+). For a plane, see
"Gotchas" at the end.

---

## 1. The recommendation in one paragraph

Declare a **second GPS of type "MAVLink"** (`GPS2_TYPE = 14`). The Pi sends each accepted fix as a MAVLink `GPS_INPUT`
message with `gps_id = 1`. The real GPS stays GPS 1 and stays primary, so by default the flight controller only **logs** our
fixes: the dataflash `.bin` then holds the real GPS and ours side by side (`GPS` messages with `I = 0` and `I = 1`), which is
exactly the ground-truth comparison we need. To **fly on** our fixes, a switch on the transmitter (a 20-line Lua script)
sets `GPS_PRIMARY = 1` and changes the EKF to a source set that takes position from the GPS but no velocity. From then on the
EKF sees only our position, so the vehicle is effectively GPS-denied, yet the real GPS keeps recording the truth. Flipping the switch
back returns to the real GPS. A truly GPS-less flight is the same setup with GPS 1 unplugged.

```
 camera ──► Raspberry Pi 5 ─────────────────────── UART / MAVLink2 ──────────► flight controller (ArduPilot)
            segmentation + correlator                                          GPS 1: real receiver  (truth)
            ▲ prior, yaw, altitude, IMU velocity ◄── ATTITUDE, GLOBAL_POSITION_INT, LOCAL_POSITION_NED
            └─ fix ──────────────────────────────────── GPS_INPUT (gps_id=1) ─► GPS 2: "MAVLink" (ours)
                                                                               EKF3 uses GPS_PRIMARY
```

Why this rather than the alternatives:

| Route | How | Good | Bad |
|---|---|---|---|
| **`GPS_INPUT` as GPS 2** (chosen) | `GPS2_TYPE=14`, Pi sends lat/lon | native lat/lon, missions/fences/RTL work unchanged; real GPS logged in the same `.bin` | arming checks want ~5 Hz; must not send zero velocity (section 4) |
| `GPS_INPUT` as GPS 1 | `GPS1_TYPE=14`, no real GPS | simplest | no ground truth in the log, no fallback; only for the final GPS-less demo |
| External Nav (`VISION_POSITION_ESTIMATE`) | `VISO_TYPE=1`, `EK3_SRC2_POSXY=6`, RC switch option 90 | built-in source switching; per-message timestamp; logged as `VISP` | local NED metres, so the Pi must convert lat/lon using the EKF origin; health needs a message every 300 ms; made for VIO/mocap, not absolute fixes |
| Lua script reads our message and logs it | `NAMED_VALUE_FLOAT` + `logger:write` | zero risk | log only, can never fly on it |

The ExtNav route is a valid fallback if `GPS_INPUT` gives trouble. Its parameter recipe is the ArduPilot
"GPS / Non-GPS Transitions" page (SRC1 = GPS, SRC2 = ExtNav, `RC8_OPTION = 90`).

---

## 2. What ArduPilot actually does with a `GPS_INPUT` message

From `AP_GPS_MAV::handle_msg`:

- **`gps_id` must equal the instance number.** GPS 1 is `0`, GPS 2 is `1`. Any other value is silently dropped.
- **`ignore_flags` really ignores, it does not fill in.** If the horizontal-velocity flag is set, the driver simply does not
  touch the stored velocity, which stays **zero**. If the EKF is told to use GPS velocity, it then fuses "the drone is standing
  still" at every fix. This is the single most dangerous detail (section 4 handles it).
- `fix_type` is copied as is. Send `3` (3D fix). Anything below 3 and the EKF ignores the message (`"Waiting for 3D fix"`).
- `satellites_visible` matters: the EKF multiplies the position noise by 1.4 at 5 satellites and by 2 at 4 or fewer. Send `10`.
- `horiz_accuracy` (m) is used. The EKF takes the larger of it and `EK3_POSNE_M_NSE`, and keeps it as a **decaying maximum with
  a 5 s time constant**. So one fix sent with "30 m" keeps the EKF distrustful for several seconds after it.
- `time_week` / `time_week_ms` are only used for jitter statistics and the system clock. The EKF times the fix as
  "**arrival time minus `GPS2_DELAY_MS`**", and the default lag for this driver is 200 ms.
- If no message arrives for **4 s**, GPS 2 drops to "no GPS".

From `AP_GPS::is_healthy` and the arming code:

- A GPS counts as **healthy only if its average interval is under 215 ms**, i.e. about 5 Hz.
- Pre-arm checks require **every configured GPS** to have a 3D fix, be healthy, and lie within 50 m of the others. In Copter
  these checks run **only if the arming mode needs position** (Loiter, Auto, PosHold, or a fence is on). Arming in AltHold or
  Stabilize skips them.
- After arming, nothing checks the 5 Hz rate. The EKF fuses a 2 Hz GPS without complaint.

From `AP_GPS::update_primary` and EKF3:

- `GPS_AUTO_SWITCH` defaults to **1 = "use best"**: the receiver with the better fix, or the same fix and more satellites, wins. With
  our 10 fake satellites, **ours could take over by itself**. Set `GPS_AUTO_SWITCH = 0` (always use `GPS_PRIMARY`).
- `GPS_PRIMARY` is read on every update, so changing it in flight switches receivers immediately.
- When the receiver feeding the EKF changes, EKF3 **hard-resets its position to the new receiver's position**, with no
  innovation gate. The first fix after switching must therefore be a good one. Switch only after ours has agreed with the real
  GPS for a while.
- After the switch, normal EKF gating applies: a fix more than `EK3_POS_I_GATE` (default 5 σ) off is rejected. If fixes keep
  being rejected and the disagreement exceeds `EK3_GLITCH_RAD` (25 m), the EKF eventually resets to the GPS anyway. With a
  correct lock after a long drift that is what we want. With a wrong lock it is not, so the Pi's own gate stays essential.

---

## 3. Latency: the one thing to engineer, not just configure

Measured on the Pi 5: segmentation about 235 ms per frame, plus a correction of 150–300 ms, so a fix is ready **0.4–0.55 s
after the photo was taken**. The EKF can only look back as far as its buffer: both `GPS*_DELAY_MS` and `VISO_DELAY_MS` are capped at
**250 ms**, and anything older is clamped to the oldest buffered moment.

**Worked example.** The drone flies at 10 m/s. A photo is taken at t = 0 and the fix is sent at t = 0.5 s. With
`GPS2_DELAY_MS = 250` the EKF believes the fix describes t = 0.25 s, but the drone was already 2.5 m further along by then. So
every fix carries a 2.5 m along-track error. Our matcher's own error is about 6 m, so this is not negligible, and it grows with speed.

**Fix: carry the fix forward to the moment it is sent.** The flight controller's short-term motion is very accurate even when
its absolute position is not. The Pi keeps a 2 s buffer of `LOCAL_POSITION_NED` velocity (requested at 20–50 Hz). When a fix
for capture time *t_c* is ready at *t_send*, it integrates that velocity from *t_c* to *t_send*, adds the displacement to the
fix, and sends it with `GPS2_DELAY_MS` set to the link latency only (about 30–50 ms). Integrate the **velocity**, not position
differences: the EKF position can jump when an earlier fix is fused, the velocity can't. In the example the drone moved 5.0 m
between capture and send, so the fix is shifted 5.0 m along the track before sending, and the EKF's timing is right again.

This needs every photo stamped in **flight-controller time**:

- Use the camera driver's sensor timestamp (`picamera2` metadata `SensorTimestamp`), not the time Python received the frame.
- Map the Pi clock to the FC clock with MAVLink `TIMESYNC`. ArduPilot answers it, and the offset follows from the round trip.
- Interpolate `ATTITUDE` (50 Hz) and `GLOBAL_POSITION_INT` to *t_c* for the yaw, altitude and prior that go into
  `Corrector.correct(view, prior, yaw_deg, altitude_m)`.

The same timing chain is needed by the ExtNav route, so it isn't a reason to pick one route over the other.

---

## 4. Parameters

### Link (Pi 5 UART on GPIO 14/15 → FC TELEM2)

| Param | Value | Note |
|---|---|---|
| `SERIAL2_PROTOCOL` | 2 | MAVLink2 |
| `SERIAL2_BAUD` | 921 | 921600; drop to 115 if the wiring is long |
| Pi side | `dtparam=uart0` in `config.txt`, device `/dev/ttyAMA0` | disable the serial console |

### Phase 1: shadow logging (real GPS flies, ours is logged)

| Param | Value | Why |
|---|---|---|
| `GPS2_TYPE` | 14 | MAVLink GPS on instance 2 |
| `GPS_AUTO_SWITCH` | 0 | never switch away from `GPS_PRIMARY` by itself |
| `GPS_PRIMARY` | 0 | the real GPS feeds the EKF |
| `GPS2_DELAY_MS` | 50 | link latency only, after forward propagation (section 3) |
| `EK3_SRC1_*` | defaults (POSXY 3, VELXY 3, POSZ 1, VELZ 3, YAW 1) | normal GPS flight |

Nothing else changes, and the EKF never uses GPS 2. Arm in AltHold. Arming in Loiter would also require GPS 2 to be at 5 Hz,
3D, and within 50 m of GPS 1 while still on the ground, where the localiser can't work.

### Phase 2: fly on our fixes, real GPS still recording

Add a second EKF source set that takes position from the GPS but **no velocity**, because our `GPS_INPUT` carries none:

| Param | Value | Why |
|---|---|---|
| `EK3_SRC2_POSXY` | 3 | GPS position (the primary GPS, which will be ours) |
| `EK3_SRC2_VELXY` | 0 | **none**: otherwise the EKF fuses the zero velocity from section 2 |
| `EK3_SRC2_POSZ` | 1 | barometer |
| `EK3_SRC2_VELZ` | 0 | none |
| `EK3_SRC2_YAW` | 1 | compass |
| `EK3_SRC_OPTIONS` | 0 | don't fuse all velocities |
| `RC9_OPTION` | 300 | "Scripting1", a 2-position switch read by the Lua script |
| `SCR_ENABLE` | 1 | Lua on |

The Lua script (`APM/scripts/vision-gps-switch.lua`) does two things together when the switch moves:

```lua
local SWITCH = rc:find_channel_for_option(300)
local last = -1
function update()
  local pos = SWITCH and SWITCH:get_aux_switch_pos() or 0
  if pos ~= last then
    last = pos
    if pos == 2 then                       -- high: fly on vision
      param:set('GPS_PRIMARY', 1)
      ahrs:set_posvelyaw_source_set(1)     -- EK3_SRC2_*
      gcs:send_text(4, 'Vision GPS: ON')
    else                                   -- low: real GPS
      param:set('GPS_PRIMARY', 0)
      ahrs:set_posvelyaw_source_set(0)     -- EK3_SRC1_*
      gcs:send_text(4, 'Vision GPS: OFF')
    end
  end
  return update, 100
end
return update()
```

Both changes must happen together. If only `GPS_PRIMARY` changes, source set 1 still asks for GPS velocity and gets ours (zero).

Safety nets for this phase: a geofence, `FS_EKF_ACTION` left on, a pilot who can take over in AltHold (no position needed),
moderate speed (≤ 5 m/s at first), and the altitude where the localiser was validated (about 40 m on the 2025 flight).

### Phase 3: no real GPS at all (optional demo)

`GPS1_TYPE = 0`, `GPS2_TYPE = 14`, `GPS_PRIMARY = 1`, source set 2 as the default. The EKF has no origin until the first fix, so
arm in AltHold, climb, let the Pi send its first fix (the prior then comes from the known take-off point), and only then
switch to Loiter/Auto. Home can be set from the GCS ("Set Home Here" / `SET_HOME_POSITION`). Phase 2 already gives a GPS-denied
EKF with ground truth, so this phase adds little for the thesis.

---

## 5. Building one `GPS_INPUT` from one `Fix`

`uavloc.simulation.correctors.Fix` already has what is needed:

| `GPS_INPUT` field | Value |
|---|---|
| `gps_id` | 1 |
| `fix_type` | 3 |
| `lat`, `lon` | `Fix.position` after forward propagation, × 1e7 |
| `alt` | `GLOBAL_POSITION_INT.alt` at send time (AMSL, m), so the GPS altitude is consistent with the baro |
| `hdop` | 1.0 (passes `GPS_HDOP_GOOD`) |
| `horiz_accuracy` | the long axis of `Fix.uncertainty_m`, floored at about 3 m |
| `satellites_visible` | 10 |
| `ignore_flags` | VDOP 4 + VEL_HORIZ 8 + VEL_VERT 16 + SPEED_ACC 32 + VERT_ACC 128 = **188** |
| `time_week`, `time_week_ms` | from the Unix time in the FC's `SYSTEM_TIME` (the Pi has no clock in the field) |

**Only send a fix when the corrector accepts it** (`Fix.applied`). A rejected frame sends nothing, so the EKF coasts on the IMU
until the next good fix. Don't resend the last fix to keep up 5 Hz in flight: the EKF would treat each copy as new, independent
evidence and become over-confident. The 5 Hz rule only matters for arming, and arming happens in AltHold.

Minimal sender (pymavlink). The real one belongs in `uavloc/` once the bridge exists:

```python
GPS_EPOCH_UNIX_S = 315_964_800
GPS_LEAP_S = 18
SECONDS_PER_WEEK = 604_800
IGNORE_FLAGS = 4 | 8 | 16 | 32 | 128

def send_fix(master, lat_deg, lon_deg, alt_amsl_m, accuracy_m, unix_s):
    gps_s = unix_s - GPS_EPOCH_UNIX_S + GPS_LEAP_S
    master.mav.gps_input_send(
        0, 1, IGNORE_FLAGS,
        int(gps_s % SECONDS_PER_WEEK * 1000), int(gps_s // SECONDS_PER_WEEK),
        3, int(lat_deg * 1e7), int(lon_deg * 1e7), alt_amsl_m,
        1.0, 1.0, 0, 0, 0, 0, accuracy_m, 0, 10)
```

The Pi also requests its inputs once at start-up with `SET_MESSAGE_INTERVAL`: `ATTITUDE` 50 Hz, `LOCAL_POSITION_NED` 20 Hz,
`GLOBAL_POSITION_INT` 20 Hz, `SYSTEM_TIME` 1 Hz, `DISTANCE_SENSOR` if a rangefinder is fitted.

---

## 6. Test ladder

| Step | Where | Pass criterion |
|---|---|---|
| 0a. SITL, fake corrector | Mac: `sim_vehicle.py -v ArduCopter`, `GPS2_TYPE=14`; the bridge with a corrector that returns the simulator's true position + 6 m noise + 0.5 s delay | GPS 2 shows in the `.bin`; with the switch on, Auto mission flown on GPS 2 only; latency compensation removes the along-track bias |
| 0b. SITL, switch-over | same, flip the Lua switch mid-mission, then disable the simulated GPS 1 | no EKF failsafe; jump at switch ≈ the fix error |
| 0c. Bench | Pi + FC on the desk, real camera pointed at a printed map | link, `TIMESYNC` offset stable, `GPS` I=1 in the log |
| 1. Shadow flight | field, real GPS primary | our track vs GPS 1 in the same `.bin`: error over time, accepted share, latency histogram |
| 2. Switched flight | field, short legs in Loiter, then a small Auto mission | position hold, mission completed, error vs GPS 1 |
| 3. GPS-less (optional) | GPS 1 removed | as 2, without the truth |

Step 1 is also **data collection**: the Pi stores every frame with its FC timestamp, the interpolated attitude and prior, and the
`Fix`. That is a new real-flight set for `data/flights/` that can be replayed offline through `uavloc.simulation` with any corrector.

---

## 7. Gotchas

- **Parameter names changed in 4.6**: `GPS_TYPE` / `GPS_TYPE2` / `GPS_DELAY_MS` / `GPS_DELAY_MS2` became `GPS1_TYPE` /
  `GPS2_TYPE` / `GPS1_DELAY_MS` / `GPS2_DELAY_MS`. Check the firmware version on the FC.
- **`GPS_AUTO_SWITCH = 2` (blend)** would average ours into the real GPS. Never use it here.
- **The heading the correlator needs comes from the same compass the EKF uses.** Experiment 003 found per-pass heading
  errors of up to 26° on the 2025 flight. Either keep the heading search in the corrector, or calibrate the compass on the new
  airframe and compare `ATTITUDE.yaw` with the GPS course in step 1.
- **Altitude**: 003 found the footprint at about 0.6× what the logged altitude implies. Fix the camera intrinsics or the
  altitude source (rangefinder) before step 2. Don't leave it to the scale search.
- **Feedback loop**: once flying on our fixes, the prior the Pi reads back is built from our own previous fixes. A wrong accepted
  fix moves the next search window. The EKF gate and the Pi gate are the only protection, and look-alikes along tracks (025)
  are exactly what both gates miss. A particle filter over the flight (008) is the real fix.
- **Planes**: Plane uses the GPS ground speed and course (for example in its DCM fallback), and a `GPS_INPUT` without velocity
  leaves them at zero. A fixed-wing build would need to send a velocity, or use the ExtNav route.
- **Telemetry sharing**: if the GCS radio is on another port, nothing else is needed. If the Pi must relay the GCS link, run
  `mavlink-router` on the Pi rather than opening the serial port from two programs.

## Sources

- ArduPilot source, `master`, read October 2026: `libraries/AP_GPS/AP_GPS_MAV.cpp`, `AP_GPS.cpp` (`is_healthy`,
  `update_primary`, `get_lag`), `AP_GPS_Params.cpp`, `AP_NavEKF3/AP_NavEKF3_Measurements.cpp` (`readGpsData`,
  `writeExtNavData`), `AP_NavEKF3_core.cpp` (250 ms delay cap), `AP_NavEKF3_PosVelFusion.cpp` (reset on receiver change),
  `AP_Arming/AP_Arming.cpp`, `ArduCopter/AP_Arming_Copter.cpp`, `AP_VisualOdom/*`, `GCS_MAVLink/GCS_Common.cpp`.
- ArduPilot wiki: [GPS / Non-GPS Transitions](https://ardupilot.org/copter/docs/common-non-gps-to-gps.html),
  [VIO tracking camera](https://ardupilot.org/copter/docs/common-vio-tracking-camera.html),
  [MAVProxy GPSInput module](https://ardupilot.org/mavproxy/docs/modules/GPSInput.html).
- Forum: [GPS_MAV without velocity breaks ground course](https://discuss.ardupilot.org/t/incorrect-handling-of-ground-course-using-gps-mav-as-a-gps-backend/118158),
  [Pi sending VISION_POSITION_ESTIMATE with a Lua GPS fallback](https://discuss.ardupilot.org/t/testing-gps-denied-navigation/131724),
  [GPS-denied navigation for long outdoor missions](https://discuss.ardupilot.org/t/gps-denied-navigation-for-long-20km-outdoor-uav-missions/95277).
