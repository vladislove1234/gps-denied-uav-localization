# How we match a drone mask to a satellite mask: the methods we tried (October 2026)

A learning note on what was tried in experiments 024, 025 and 003, and the matcher they all start from: how each method works,
with small worked examples, why it did or didn't help, and where the code lives.
Numbers are from each experiment's `runs.jsonl`; the per-experiment `report.md` has the full tables.

---

## 1. The problem in one paragraph

The drone has lost GPS. Its inertial system still gives a **prior**: "I am roughly here", but that guess has drifted 5–15 m.
The camera looks down. A segmentation model turns the photo into a **class map**: for every pixel, how likely it is to be
background, building, track, water, woody or field. The same model has turned the satellite image of the area into a class map
too. We cut a satellite **patch** around the prior, about 1.7× the size of the drone's view, and slide the drone's class map over
it. At every offset we compute a **score** for how well the two agree. The grid of scores is the **heatmap**. The offset with the
best score is the correction.

So everything below is an answer to one of three questions:

1. **How do we score one offset?** (the matching function: section 2, method A)
2. **When do we trust the best offset?** (the gate: method B)
3. **What if the drone's heading or altitude is wrong, so the two maps don't line up at any offset?** (pose search: method C)

```
satellite patch (around the prior)          drone class map
+-------------------------------+           +-----------+
|      woody      track         |           | track     |
|   ......////////....          |   slide   |   ////    |
|   bldg  ////    woody         |  <------  | bldg      |
|        ////                   |           +-----------+
+-------------------------------+
        heatmap: one score per offset; the peak = where the drone probably is
```

---

## 2. The starting point: weighted per-class NCC (`007.1`, 5.861 m @10 m injected)

Code: `uavloc/matching/correlator.py` (`class_ncc`, `weighted_ncc_heatmap`, `decide`).

### How it scores one offset

It works class by class:

1. Take only the **building** channel of the drone map, and the building channel of the satellite window under it.
2. Compare their patterns with **normalised cross-correlation (NCC)**. NCC asks: where the drone has more building than its own
   average, does the satellite also have more building than *its* average? It gives +1 for a perfect pattern match, 0 for
   unrelated, and −1 for opposite. Because it subtracts the averages, a window that is "all building" doesn't win automatically.
   Only the *shape* of the building pattern counts.
3. Do the same for track, woody and the other classes.
4. Average the per-class results with **hand-set weights**: background 0, building 1, track 1, woody 0.5, field 0, water 0.5.

**Worked example.** At some offset the per-class NCCs are building 0.8, track 0.6, woody 0.1.
The weighted mean is (1·0.8 + 1·0.6 + 0.5·0.1) / (1 + 1 + 0.5) = 1.45 / 2.5 = **0.58**.
At another offset the same three values are 0.2, 0.3 and 0.9, which gives (0.2 + 0.3 + 0.45) / 2.5 = **0.38**.
The first offset wins: buildings and tracks agree there, and those are weighted more than trees.

### From scores to a decision

- **z-score.** Every heatmap is rescaled to mean 0 and spread 1, so a peak of "4" means "4 spreads above this heatmap's average".
- **argmax.** Move the drone to the best cell.
- Optional extras (007, 022): a **temperature** turns z-scores into probabilities. A **prior** says big moves are less likely.
  A **gate** refuses the move when the probability cloud is wide (`s_max`) or the drone sees too little of the weighted classes (`c_min`).

### Its hand-tuned constants, and why they bother us

| Constant | What it does | Problem |
|---|---|---|
| class weights 0/1/1/0.5/0/0.5 | how much each class counts | set by hand, never tuned |
| `MIN_PRESENT_SHARE` 1 % | a class takes part only if ≥ 1 % of drone pixels are that class | arbitrary cut-off |
| `MIN_WINDOW_STD` 0.05 | ignore satellite windows that are almost uniform in a class | arbitrary cut-off |
| temperature | sharpness of the probability cloud | best value is 0.25 in one village and 2.0 in another |
| gate `s_max`, `c_min` | when to refuse a move | tuned per village; doesn't transfer |

The temperature problem has a simple cause. The z-score rescales **each** heatmap to the same spread. A heatmap from a crisp
village with clear buildings and a heatmap from a uniform forest end up looking equally "confident", so the absolute evidence is
lost and has to be re-tuned per place.

**Blind frames.** If the drone sees only background and field (both weight 0), nothing can be scored. The heatmap is flat and the
drone keeps its prior. That happens on about 25 % of draws in the village bench.

### The vocabulary used below

| Term | Meaning |
|---|---|
| injected error | we take the true position, move it 5 / 10 / 15 m in 8 directions, and call that the prior |
| mean error | average distance from the truth after correction (lower is better; doing nothing = the injected error) |
| harm | share of corrections that end > 2 m *further* from the truth than the prior was |
| accept / applied share | share of draws where the gate allowed the move |
| reachable | the truth lies inside the search window (at 10 m injected only ~55 % of draws are) |
| identity / self-match check | use the satellite map itself as the "drone" map. A correct method must find the exact spot. |
| LOPO | leave-one-place-out: tune on one village, test on the other, so tuning can't cheat |
| holdout | 10 extra villages never used for anything else |
| bootstrap CI | resample frames many times to get a range for "A − B"; if the whole range is below 0, A is reliably better |

---

## 3. Method A: learn which classes go together (experiment 024)

Code: `uavloc/matching/compatibility.py`. Idea from Chan et al., CVPR 2003 (medical image registration).

### Intuition

Weighted NCC only compares building with building and track with track. But the drone photo is new and the satellite photo is
years old, and the model makes systematic mistakes. A field on the old satellite image might be young woody today. A building
roof might be called background on one date. Instead of *guessing* weights, **count** from data how often drone class *i* sits on
satellite class *j* when the two are truly aligned, and compare that with chance.

### How the table is built (arm `024.2`, "PMI")

We have 46 villages with images from several dates, aligned to the centimetre. For thousands of aligned pairs:

1. Count how often each (drone class, satellite class) pair co-occurs at the **true** position.
2. Compare with how often it would co-occur by **chance**, i.e. if the two maps had nothing to do with each other.
3. The table entry is the log of that ratio. This is called **pointwise mutual information (PMI)**.

**Worked example.** Suppose 3 % of drone pixels are building and 4 % of satellite pixels are building. By pure chance a pixel would
be building on both sides 0.03 × 0.04 = 0.12 % of the time. On truly aligned pairs it happens 2 % of the time.
The ratio is 2 / 0.12 ≈ 17, and log(17) ≈ **+2.8**: "building on building" is strong evidence for alignment.
"Drone woody on satellite building" might occur *less* often than chance, giving a **negative** entry: evidence *against* the offset.

### How an offset is scored

For every pixel, look up the table entry for (drone class there, satellite class under it) and add them all up. The offset with the
biggest total wins. With soft probabilities this is six convolutions, the same cost as weighted NCC. The table replaces all three
hand constants: weights, `MIN_PRESENT_SHARE` and `MIN_WINDOW_STD`.

### Arm `024.3`: mutual information (MI), no table at all

At each offset, build the joint histogram of (drone class, satellite class) pairs **inside that window**, and measure how much
knowing the drone class tells you about the satellite class. That measure is MI. No training data and no constants are needed.
This is the classic way to register medical images from different scanners.

### What happened

| @10 m injected | weighted NCC | PMI table | MI |
|---|---|---|---|
| mean error | **5.861 m** | 6.831 m | 7.450 m |
| harm | 0.084 | 0.219 | 0.259 |
| holdout villages | **7.225 m** | 9.892 m | 9.577 m |
| self-match check (22 reachable draws, found within 1 cell) | 22 / 22 | 10 / 22 | 10 / 22 |

**Both were worse, and both failed the self-match check.** A matcher that can't find a perfect copy of the map isn't a fair test of
the idea, so 024 rules out *these two formulas*, not "learning class correspondences".

### Why they failed: the lessons

1. **Rare classes got huge weights.** The model almost never predicts field or water (≤ 0.06 % of pixels). Their table entries
   were set mostly by the tiny smoothing number added to avoid log(0), not by data, and came out at +4.5 to +5.3. The largest real
   entries (building, track) were only about +2.6. Because the score is a plain sum, any offset that collected a little stray
   field or water probability won.
   *Lesson: never let a class with almost no data carry a big weight. Zero it or shrink it toward 0.*
2. **MI drifts with the window's contents.** As the window moves, both the joint and the satellite marginal change, so near the
   truth the MI surface is almost flat (a median gap of 0.04 z between the truth and the winner).
3. **Trying to match "blind" frames hurts.** Weighted NCC keeps the prior on frames with no buildings or tracks (10.0 m). The new
   scores moved anyway and ended at 10.9 m and 14.3 m. In a scene of just fields and background, the peak is noise.
   *Lesson: refusing to move is sometimes the right answer.*
4. Cross-class terms such as "building next to field" reward any building near open ground, which helps look-alikes as much as the truth.

---

## 4. Method B: an a contrario gate (experiment 025)

Code: `uavloc/matching/acontrario.py`. Idea from Desolneux, Moisan and Morel (*a contrario* detection, "number of false alarms").

### The question a gate answers

The best offset always exists, because some cell has the highest score. The question is whether it's **real** or just the luckiest
cell among thousands. The 022 gate answered with thresholds tuned per village. A contrario tries to answer with **no tuning at all**.

### How it works, in plain steps

1. Take the drone's class map.
2. Score it against **200 places where it certainly isn't**: random patches from 46 other villages (arm `025.2`), or patches of the
   same map at least 60 m from the prior (arm `025.3`). This gives the **null**: how high scores get by pure luck.
3. Look at the real peak score. What fraction of the luck-scores reach it? Call that *p*.
4. We searched about 3,000 cells, so we gave luck about 3,000 chances. The **expected number of false alarms** is
   NFA = 3,000 × *p*.
5. If NFA < 1, meaning that by luck we'd expect less than one score this good in the whole search, accept the move. Otherwise keep the prior.

**Worked example.** The real peak is 0.42. Of all the cells in the 200 null patches, 0.001 % score ≥ 0.42, so *p* = 0.00001.
NFA = 3,000 × 0.00001 = 0.03. That is below 1, so the move is accepted.
If instead 0.1 % of null cells scored ≥ 0.42, NFA = 3,000 × 0.001 = 3: we'd expect three such peaks by luck, so the move is rejected.

The appeal: the threshold "1 expected false alarm" is the same everywhere. It adapts to each frame automatically, because each frame
brings its own null. A flat forest scene scores high against random forest too, so it gets rejected without anyone tuning anything.

### What happened

| @10 m injected | tuned static gate (022.2) | NFA, null from other villages | NFA, null from same map | bar |
|---|---|---|---|---|
| harm | **0.00625** | 0.034 | 0.028 | ≤ 0.00625 |
| applied | **0.556** | 0.513 | 0.525 | ≥ 0.556 |
| mean error | **5.999 m** | 6.679 m | 6.430 m | ≤ 5.999 m |

Missed on all three. No choice of the threshold would have saved it: at every threshold tried (0.01, 0.1, 1, 10), the tuned gate does better.

### Why it failed: the key insight

**NFA tests "better than random ground", but the dangerous mistakes are look-alikes, and look-alikes ARE better than random
ground.** A drone over a straight track also matches the same track 15 m further along, or a parallel tree line. Those wrong peaks
are far above the null, so NFA accepts them. It accepted about a third of the harmful moves.

What a gate really needs to detect is **ambiguity**: "is there a *second* place that looks just as good?". A null of random patches
can't see that by design.

Two more lessons:
- The difference between villages didn't go away. The untuned NFA still accepts ~0.30 of draws in bushcha and ~0.73 in birky,
  the same split as the tuned gate. So that difference comes from the **scenes** (bushcha has less unique structure), not from badly tuned thresholds.
- On the 10 unseen holdout villages, the untuned NFA was roughly as good as a gate tuned on other places
  (7.84 / 7.90 m vs 7.74 m, with lower harm). That's useful if gates ever have to work on new ground.

**Where this points:** look-alikes along a road can't be resolved from one frame. Over several frames they can, because the next
frames don't share the same look-alike. That is a job for a **particle filter over the flight** (experiment 008).

---

## 5. Method C: search the heading and the scale (experiment 003)

Code: `uavloc/matching/pose_search.py`, plus pose fields in `CorrelatorConfig`.

### Why it matters

The correlator slides the drone map left/right/up/down only. It trusts the drone's **logged heading** to rotate the satellite patch,
and the **logged altitude** to decide how many metres the drone view covers. On the 2025 real flight both were wrong:
- heading off by a constant per pass: −9°, **+26.5°, +25°**, +8°, −3°;
- footprint about **0.6×** what the logged altitude implies.

A track in a class map is a thin line. Rotated by 25°, it crosses the true track instead of lying on it, so it scores low
**everywhere**, and the argmax lands on whatever happens to overlap. No sliding can fix a rotation.

### How the search works

Try many poses and keep the best:

- **heading offsets**: −36° … +36° in 3° steps (25 values), added to the logged heading;
- **scales** (optional): 0.5, 0.6, …, 1.0 times the logged footprint;
- for every pose, run the normal sliding correlation, and keep the pose and offset with the highest **raw** score.
  Raw, not z-scored, because z-scores from different poses aren't comparable.

### The search order mattered (two amendments)

Trying all 25 × 6 = 150 poses at full detail is expensive, so the plan was coarse-to-fine. That turned out to be harder than it sounds:

1. **Greedy (the original plan):** choose the heading at scale 1.0, then the scale at that heading, then refine the heading ±3°.
   The self-match test, with the map rotated +20° and shrunk to 0.6, failed on 9 of 40 draws. When the first step picked −24°
   instead of −20° (because at the wrong scale the wrong heading looked better), the true pose was never tried, even though it
   scored higher. *Lesson: a coarse-to-fine search must keep the true answer reachable at every stage.*
2. **Joint coarse pass:** try every heading at every scale (150 poses), then refine around the best. It recovered the scale every
   time, but the 3° grid still skipped the true heading on 4 draws.
3. **Joint + refine around the top 3 coarse poses:** the true pose was then always evaluated and never lost. The 3 draws still
   wrong were cases where another pose 8–11 m along the *same straight track* scored a hair higher. That is the map itself being
   ambiguous, and no search can fix it.

### What happened (15 m injected, real 2025 flight, 44 frames)

| Run | Features | Search | Mean error |
|---|---|---|---|
| 002.2 (baseline) | class maps | none | 12.113 m |
| **003.1** | class maps | heading | **9.967 m** (better by 2.15 m, CI 0.55–3.64) |
| 003.2 | class maps | heading + scale | 9.977 m |
| 001.1 (replayed on this machine) | VGG11 features | none | 11.469 m |
| 003.3 | VGG11 features | heading | 12.537 m |
| 003.4 | VGG11 features | heading + scale | 15.039 m |

(In `runs.jsonl` registered 003.2 and 003.3 appear as logged 003.3 and 003.2, because runs are numbered in the order they finish.)

### What we learned

1. **Heading search works for class maps, and it finds the real heading.** The chosen headings match what the earlier
   re-registration PoC measured per pass, to within 1–5°. The biggest gains are on the passes where the log was off by 25°.
2. **For texture features (VGG11) the search makes things worse.** Texture looks about equally similar at many angles, so more
   hypotheses just mean more chances for a wrong peak. Lines in a class map only match when aligned, which is why the search
   helps them. *Lesson: a wider search only helps a score that is sharp in the searched dimension.*
3. **Scale really is ~0.6, but searching it per frame doesn't pay.** The joint search picked scale < 1 on every draw (median 0.6),
   confirming the altitude reading is off. Searching it adds no accuracy (9.977 vs 9.967 m) and hurts at 5–10 m.
   A single constant altitude correction is the cheaper fix.
4. **At 5 m injected, every method still makes things worse than doing nothing** (best 7.5 m vs 5 m prior). Small errors still need a gate.

---

## 6. Summary

| Experiment | Question | Answer | One-line reason |
|---|---|---|---|
| 024 | Can data-learned class compatibility (or MI) replace hand weights? | **No** (these formulas) | rare classes got huge weights; moving on blind frames hurt |
| 025 | Can an untuned a contrario rule replace the tuned gate? | **No** | it detects "better than random", but errors come from look-alikes |
| 003 | Does searching heading/scale help? | **Yes for heading** (−2.15 m) | rotated lines don't match at any shift; scale adds nothing |

The big picture:
- **Per-frame tricks have hit a wall on ambiguity.** Better scores (024) and better gates (022, 025) can't tell two look-alike
  places apart from one frame. Sequences can (008).
- **Fix the geometry before the score.** The largest single gain came from getting the heading right (003), not from a cleverer comparison.
- **Every new matcher must pass the self-match check first.** It caught the broken formulas in 024 and the search-order bug in 003.

---

## 7. Methods from the literature we did *not* try yet

From the research in `~/deep-research-output/semantic-mask-matching/`:

| Method | Idea in one line | Why it might help | Status |
|---|---|---|---|
| Distance transform / chamfer | score = how far each drone boundary pixel is from the nearest same-class satellite boundary | partial credit for near-misses, a wide basin | best as the particle-filter likelihood (008) |
| Fourier–Mellin phase correlation | rotation and scale become simple shifts in the frequency domain, found in one step | closed-form heading + scale | needs the two images to overlap almost fully; ours don't |
| Object constellations (crater-style) | match triangles of building centres by their shape | ignores heading/scale errors by construction | an upgrade for 020 |
| Learned matcher (OrienterNet-style) | a small network scores offsets, trained so its output is a calibrated probability | removes temperature and weights | that is 023 |
| Particle filter over the flight | keep many position hypotheses and update them frame by frame | look-alikes get outvoted over time | 008, the next natural step |

## 8. Where things live

- Matcher: `uavloc/matching/correlator.py`. New scoring modes plug in via `register_matching(...)`.
- Method A: `uavloc/matching/compatibility.py`, `experiments/024-class-compatibility-matching/`
- Method B: `uavloc/matching/acontrario.py`, `experiments/025-a-contrario-gate/`
- Method C: `uavloc/matching/pose_search.py`, `experiments/003-pose-search/`
- Literature review: `~/deep-research-output/semantic-mask-matching/phase6_report/report.md`
