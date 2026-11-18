# long-route-planner

**Status:** failed · **Question:** Does the 027.12 filter (particles + head 023.3) hold position over a 30 km
planner route through villages, and does taking fixes only where the planner expects them beat a fix every 5 s?

## The question

Fly a ~30 km route the route planner chose through a chain of villages. Compare two fix schedules on the same
INS realisations: `every-5s` (027-034) and `planner` (attempt a fix only where the planner's blind map, read at
the estimate, is below 0.5, at most one per 50 m of estimated travel). It fails if the filter loses the track
(error growing like INS-only) on the planner's own best route.

## Verdict

**Failed, on both questions.** No flight holds position over 30 km. All 12 end 2.3–3.9 km off, against
2.7–4.4 km for INS alone (`runs/fly/flights.csv`). Every flight that survives the first few km (8 of 12) loses
lock between 11.4 and 14.3 km (last time under 20 m 11.4–14.0 km, first time over 50 m 12.1–14.3 km), at
about the same flight time (~1,150–1,430 s) on both routes, at places up to 2 km apart. The cause is
the filter's motion model, not the ground. The simulated INS velocity error random-walks
(0.02 m/s/√s), so its drift speed grows from 0.1–0.6 m/s at 150 s (the regime 027.12 was tuned on:
sim-flights routes ≤ 2.8 km) to 0.65–1.1 m/s at 1,360 s and 1.3–2 m/s later. The particle filter takes
INS increments as true plus 2 m/√s of noise, so it falls behind until the truth leaves the 100 m-capped
search window, and it never recovers. Gating fixes on the planner's blind map does not help: each arm lost
2 of 6 flights for good early on, the planner arm sooner (at 0.5 and 2.0 km, against 1.8 and 5.4 km), and
it was never ahead. The every-5s
filter got useful updates on ground the planner calls blind. The planner's best route (37.1 km, 31 %
blind) did not outlast the shortest (31.1 km, 70 % blind).

Next step this points at: a filter that estimates INS velocity error (`028`'s `ins-error` arm models a
constant bias, not a random walk). Run it on the long route before any further route-planning work.

| Route | Arm | mean end error | mean error | time within 10 m | fixes | updates |
|---|---|---|---|---|---|---|
| best | every-5s | 3,558 m | 1,176 m | 21 % | 740 | 290 |
| best | planner | 3,729 m | 1,301 m | 17 % | 222 | 168 |
| shortest | every-5s | 2,687 m | 892 m | 17 % | 621 | 190 |
| shortest | planner | 2,864 m | 1,007 m | 14 % | 145 | 100 |

Means over 3 INS seeds (`runs/fly/results.json`). INS-only end error: best 4,016 m, shortest 3,055 m.

## What I ran

- **Corridor.** OSM `place` nodes (Overpass, 2026-11-03) in 49.3-50.1 N, 23.8-25.6 E. The best 30 km
  east-west line (most villages within 600 m) is lat 49.962, 24.78-25.20 E, east of Busk (Lviv oblast). The
  15 via-points are the shortest village-to-village path along it (legs <= 4.5 km): 30.8 km straight-line.
  A single start-end plan would cover ~93 km², over the app's 25 km² cap, so each leg is planned on its own.
  That is how an operator would add via-points.
- `plan.ipynb`: `uavloc.planning.plan` per leg, as the app does (Esri z18, `lraspp-satellite.onnx`,
  `SelfSimilarity`, 40 m, 10 m/s, sim-flights INS). `best` = the alternative with the least blind ground,
  `shortest` = the shortest alternative. Writes `runs/plan/routes.json` and the per-leg blind maps.
- `fly.ipynb`: the route is cut into 2 km chunks. Each chunk gets an Esri map raster (= onboard map) and a
  Google z18 drone raster, phase-correlated onto Esri, both with a 300 m margin. All flights advance together
  chunk by chunk and swap `estimator.correlator`. Filter = 027.12 (2000 particles, T 0.5, Q 2.0, head
  023.3, mobilenet LR-ASPP). 3 INS seeds, shared by both arms.
- Equipment change: `uavloc.simulation.sequence.fly` is now a thin wrapper over `FlightRun` (stepped flight)
  with a `FixSchedule` (`uavloc/simulation/schedule.py`: `EveryInterval`, `WhereLocalizable`). Checked
  bit-identical to the old `fly` on two sim-flights test routes (same fix steps, decisions, errors to 0.0 m).

## Evidence

- `figures/plan.png`: the 14 per-leg blind maps, the best and shortest routes, the 15 villages.
- `figures/error-along-route.png`: error along each route, 3 seeds per arm, planner-blind ground shaded.
  Error stays at 2–40 m with recoveries until 11.4–14.3 km on every surviving flight, then grows like INS.
- `figures/fix-locations.png`: where each arm attempts fixes on the best route (seed 0).
- `figures/videos/` (`video.ipynb`, rendered from the saved tracks): `best-every-5s-1-full.mp4` and
  `best-planner-1-full.mp4` (the whole 37 km route, 2 s of flight per frame, 93 s) and
  `best-every-5s-1-loss.mp4` (12–15.5 km, every pose, 35 s). Panels: the map following the drone with the
  true, corrected and INS tracks; the drone view (Google); its segmentation; error against distance.
- 4 km dry run (`runs/fly-dry/`) is identical to the first 4 km of the full run (max difference 0.0 m).
- Google→Esri co-registration: 2.0–4.5 m shifts applied in 26 of 35 chunks. 9 chunks were left
  unshifted (weak peak): best route 16–22 and 30–37 km, shortest 14–16, 18–20 and 26–31 km. All are at
  or after the point where each route's flights lost lock.

Not verified: whether real ArduPilot EKF velocity error grows like this INS model. The loss point near
1,400 s is a property of the simulated IMU, not a measured one.
## Cost

Planner: ~35 min for 14 legs (~50 km² of Esri z18 segmented + self-similarity), plus two restarts after
network errors (bug-172: tile retry). Flights: 70 min for 12 flights (6 at a time, one 2 km map chunk in
memory), CPU shared with four `034` jobs. ~3–4k Google tiles.
