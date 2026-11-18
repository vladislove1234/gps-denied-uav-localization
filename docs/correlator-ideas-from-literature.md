# New ways to match a drone mask to a satellite mask: ideas from the literature (October 2026)

A follow-up to `matching-methods-explained.md`. That note covers what we tried (weighted NCC, PMI/MI in 024, the a contrario gate in 025,
pose search in 003). This one collects what we have **not** tried, from five literature sweeps run on 2026-10-30:

| Sweep | Where it looked | Notes file |
|---|---|---|
| A | template matching and distribution distances (computer vision) | `A_dense_scores.md` |
| B | object-, point- and graph-level matching (star trackers, astronomy, GIS, robotics) | `B_object_topology.md` |
| C | map matching in robotics, terrain navigation, planetary landing | `C_robotics_terrain.md` |
| D | label-map registration in medical imaging and remote sensing | `D_label_registration.md` |
| E | limitations and future work in the papers we hold in `refs/`, plus 4 new 2025–2026 papers | `E_refs_future_work.md` |

All five files are in `~/deep-research-output/correlator-alternatives/`. Each method entry there has its citation, cost estimate, the authors'
stated limitations and an experiment idea. The earlier sweep (`~/deep-research-output/semantic-mask-matching/`) covered chamfer,
MI/PMI, Fourier–Mellin, crater constellations, shape context, ICP/NDT, OrienterNet-style learned matchers and NFA. Those are not repeated here.

**How far to trust this.** The sweeps were done by cheaper agents. "Not used for UAV localisation" means their searches found nothing, not
that nothing exists. Several papers were read only as abstracts, and the notes mark those `[unverified]`. Every CPU cost below is an
estimate, not a measurement. No number here is ours: our only number is the 5.861 m baseline from `runs.jsonl` (007.1, 10 m injected).

---

## 1. What we already have under other names

Several "new" ideas turned out to be things the correlator already does. Don't run them again as new experiments.

| Literature idea | Where it comes from | What we already have |
|---|---|---|
| Covariance of the whole score surface; correct only along the well-constrained axis | Olson 2009 (correlative scan matching, the "long hallway" ellipse); Zhang, Kaess & Singh 2016 (degeneracy, "solution remapping") | `posterior()` covariance + the 007 `gate="full"` **across-only** move in `decide()` |
| Ratio of the second-best minimum to the best as a false-fix signal | TERCOM false-fix work (Uslu 2026 thesis) | `second_peak_ratio()`, used by the 022 risk gate |
| Co-occurrence statistics of pixel values | CoTM (Kat et al. CVPR 2018) | the PMI table of 024 (failed) |
| Bayesian "better than a random field" verification | astrometry.net (Lang et al. 2010) | the a contrario NFA gate of 025 (failed) |
| Signed distance maps / label distance maps | SDF registration, Rackerseder MICCAI 2018 | the chamfer family, already surveyed |
| Extra map layers beyond buildings | BRM future work, Terlizzi future work | we already match building + track + woody |

A correction to sweep A: it guessed that our weighted NCC might "mask the image first", the error ITK's documentation warns about.
It doesn't. `class_ncc` computes a plain NCC per class, and the weights are applied to the finished NCC maps (`correlator.py:250-291`).
Masked NCC (idea A1 below) is still worth trying, but as a new score, not a bug fix.

---

## 2. Which idea targets which failure

Our six failure modes, from `matching-methods-explained.md` and the 021/022/025 reports:

1. **Ridge**: a single straight track; every shift along it scores the same.
2. **Look-alikes**: rows of houses, parallel tree lines.
3. **Blind frames**: ~25 % of draws show only background, field or uniform woody.
4. **Rare classes**: tiny-support classes get huge weights (killed PMI).
5. **Heading/scale**: the real flight is off by up to 25° and ~0.6× in scale.
6. **Gate doesn't transfer**: thresholds tuned on one village fail on another.

| Idea (section) | 1 ridge | 2 look-alike | 3 blind | 4 rare | 5 pose | 6 gate | UAV use found? |
|---|---|---|---|---|---|---|---|
| A1 masked NCC + unknown/contradiction | | | ✔ | ✔ | | | no |
| A2 confusion-matrix weights | | ✔ | | ✔ | | | once (thermal, Lights Out 2026) |
| A3 diversity-penalised score (DDIS, QATM) | ✔ | ✔ | ✔ | | | | no |
| A4 blurred / multi-scale class maps | | | | | ✔ | | no |
| B sub-template consensus | ✔ | ✔ | ✔ | | | ✔ | no |
| C1 track junctions, bends and ends | ✔✔ | ✔ | | | ✔ | | roads to OSM yes; dirt-track skeletons no |
| C2 consistency graph / 4th-object check | | ✔ | ✔ | | | ✔ | no (cars to aerial map only) |
| C3 point-set mixture with outlier mass | | | ✔ | ✔ | ✔ | | no |
| D Gauss–Newton refinement on soft maps | ✔ | | ✔ | | ✔ | | yes on CNN features (Goforth), not on class maps |
| E exact branch-and-bound pose search | | | ✔ | | ✔ | | no |
| F offline self-similarity map + structure-count gate | ✔ | ✔ | ✔ | | | ✔ | no |
| G learned matcher on synthetic label maps | | ✔ | | ✔ | | | no |

---

## 3. The ideas

### A. New ways to score one offset

These replace or wrap `weighted_ncc_heatmap` and register through `register_matching(...)` (`correlator.py:44`).

#### A1. Masked NCC with "unknown" and "contradiction" cells

Sources: Padfield, *Masked object registration in the Fourier domain*, IEEE TIP 2012 (in scikit-image and ITK). Brett et al.,
cost-function masking, NeuroImage 2001. The scoring rule is from SASGeo (Trukhina & Vashkelis, arXiv 2607.07737, 2026, synthetic only).

**Idea.** Today every drone cell takes part in every class's NCC, and the class weights decide afterwards how much each class counts.
Masked NCC lets each **cell** opt out instead. The mean, variance and correlation are computed only over cells valid on both sides, and
it is still exact and FFT-fast. SASGeo adds a three-way rule per cell:

- **agree** (drone confident building, satellite confident building): adds evidence;
- **contradict** (drone confident building, satellite confident *not* building): subtracts evidence;
- **unknown** (either side unsure, or a class that says nothing about position, like field): contributes **zero**.

**Worked example.** A frame shows one house and a patch of field. Weighted NCC has field at weight 0, so the field cells drop out.
But they still enter the building channel's mean and spread, as "not building". With the mask, the field cells are simply not there,
so the building channel is scored only on the cells that carry information. On a frame that is 95 % field, the valid-cell count is tiny.
That count is an honest blind-frame signal in cells, not a tuned z-score.

**Why it might work here.** It removes the hand constants `MIN_PRESENT_SHARE` and `MIN_WINDOW_STD`, and the rare-class problem of 024.
A class nobody predicts is unknown, not +5 evidence.

**Risk.** With very few valid cells the NCC gets noisy, so offsets near the patch border with small overlaps can win. It needs a
minimum-overlap rule.

**Experiment.** Masked per-class NCC, valid cells = class probability above 0.5 on either side or the segmenter's entropy below a cut.
Bar: below 5.861 m at 10 m injected, *and* still below it when all four class weights are set equal (proving the mask, not the
weights, does the work). Pass the self-match check first.

#### A2. Class weights from the segmenter's confusion matrix

Source: Allen & Greeff, *Lights Out*, arXiv 2604.26201 (2026). Thermal UAV at night against a semantic map, real flights, translation
grid search ±30 m around the navigation prior. That is the same structure as ours. Their ablation reports 12.1 % lower mean horizontal error with
confusion-aware weighting (paper's number; GPU, 0.95 s per update).

**Idea.** Instead of "drone building matches satellite building only", let a drone building cell also partly match satellite
background, by how often the segmenter confuses the two on the validation set. Merged and missed buildings then enter the score as
known confusion rates instead of as disagreement.

**How this differs from 024's PMI.** PMI was counted from cross-date pairs and blew up on classes with almost no data. A confusion
matrix comes from the segmenter's own validation set, where every class has labelled pixels, and its rows sum to 1, so no entry can reach +5.

**Experiment.** Weight matrix = row-normalised validation confusion matrix of the 007.1 segmenter, used in place of the identity in
per-class scoring. Bar 5.861 m. Cheap: same cost as weighted NCC.

#### A3. Scores that punish "this matches many places" (DDIS, QATM)

Sources: Talmi, Mechrez & Zelnik-Manor, *Template Matching with Deformable Diversity Similarity*, CVPR 2017 (arXiv 1612.02190).
Cheng et al., *QATM: Quality-Aware Template Matching*, CVPR 2019 (arXiv 1903.07254). Both come from object tracking. No aerial or UAV
use found.

**Idea.** NCC asks "how well do the two windows agree?". These scores also ask "is that agreement **unique**?". QATM scores each small
patch of the drone map by how many satellite patches it matches equally well. A patch that matches exactly one place scores 1. One that
matches N places scores 1/N. DDIS does the same from the other side: if many window patches all pick the same drone patch as their best
match, that drone patch counts exponentially less.

**Worked example.** A 3×3-cell piece of straight track matches the satellite track at 20 positions along it, so it scores 1/20. A piece
holding the track's bend matches one position and scores 1. The bend now outweighs twenty pieces of plain track, which is exactly what
should break a ridge. On a uniform woody frame every patch matches everywhere, so every score is close to 0 and the frame looks blind
by construction.

**Risks.** Both were built for RGB photos. With only 6 classes per cell, nearest neighbours tie a lot, so patches of 3×3 cells (54
numbers) are needed. Cost is the main unknown. QATM over 3×3-pooled maps is about 2.4 million patch comparisons per frame (sweep A's
estimate), which is plausible on the Pi. DDIS needs a nearest-neighbour search and probably numba.

**Experiment.** QATM on 3×3-pooled 6-class maps. Bar 5.861 m, plus a check that blind-frame heatmaps come out flatter than NCC's.
Of everything in this note, this is the most "not used in UAV localisation" idea with a direct mechanism against failure modes 1–3.

#### A4. Blur the class maps before matching (distribution fields, multi-scale)

Source: Sevilla-Lara & Learned-Miller, *Distribution Fields for Tracking*, CVPR 2012 (abstract only read).

**Idea.** Our maps are already per-pixel class distributions. Blurring each class layer by 1–3 cells before NCC gives a smoother score
that tolerates a few degrees of heading error, a slightly wrong scale, and merged buildings. Then refine at full sharpness inside the
winning basin.

**Expectation.** Doesn't touch the ridge or look-alikes and loses some precision. It is almost free, though, and is the cheapest test of
"does a smoother score let us drop part of 003's heading search?".

### B. Split the drone map into pieces and let the pieces vote

Sources: the Mars 2020 Lander Vision System (Johnson et al., NASA; coarse mode 15 landmarks over 3 images, fine mode up to 150 small
templates per image, then outlier rejection before the filter). AROSICS (Scheffler et al., Remote Sensing 2017; satellite
co-registration with a grid of local shifts and false-tie-point removal). Vargas-Muñoz et al., ISPRS 2019 (arXiv 1901.08190; one shift per
cluster of OSM buildings, solved with a smoothness MRF). None of them used for UAV-to-satellite matching.

**Idea.** Instead of one 68×120 template, cut it into e.g. a 3×4 grid of tiles (or into one tile per building cluster and track
segment). Correlate each tile on its own, then keep the shift that most tiles agree on.

**Worked example.** A frame has a straight track across the top and two houses in the bottom right. The track tiles each give a
ridge: they agree on the across-track position and scatter along the track. The house tiles give a sharp peak, which pins the
along-track position. The median of the agreeing tiles lands where both agree. If the satellite is from 2018 and one house was built
since, its tile finds nothing and is outvoted rather than dragging the whole score. That is how remote sensing handles changed areas:
they are outvoted, not detected.

**Why it is promising.** It is the only idea here that handles failure modes 1, 2, 3 and cross-date change in one mechanism. Its gate
("at least k tiles agree within 2 m") is a count of independent votes, not a z-score, so it has a chance to transfer between villages.

**Risk.** A 46×26 m footprint split into 12 tiles leaves little structure per tile, and many tiles will be blind. That is fine: they abstain.

**Experiment.** 3×4 tiles, per-tile weighted NCC argmax, accept the median shift only if ≥ k tiles agree within 2 m, else keep the
prior. Bar 5.861 m. A second, separate claim: the agreeing-tile count predicts harm better than the 022 gate on the held-out villages.
This is an experiment on top of the 007.1 machinery, a few dozen small FFTs per frame.

### C. Object-level matching beyond "centre of each building"

He & Wu 2025 used distances and angles from the image centre to object centroids. Experiment 020 plans translation voting over
building-centroid pairs. The sweeps found these ways to go further.

#### C1. Use the track's shape, not just its pixels: junctions, bends and ends

Sources: Costea & Leordeanu, BMVC 2016 (arXiv 1605.08323; roads + intersections matched to OSM, error ≤ 2.5 m in ≥ 90 % of cases in
two European cities). Li et al. 2019 (arXiv 1906.12174, IEEE TGRS; pairs of intersections with their tangents). In GIS, road-network
conflation (Lei & Lei 2024).

**Idea.** Skeletonise the track mask on both sides. The points where the skeleton is not a plain line, namely junctions (T, Y, X), bends
and dead ends, become anchor points, each tagged with the bearings of its branches. Match anchors within the prior window.

**Why this targets the ridge.** A ridge exists because a straight track has no along-track feature. A bend or a fork **is** an
along-track feature. Li et al. say the same thing in their conclusion: their method "require[s] there are enough intersections", and
they propose accumulating intersections over a sequence. Matching dirt-track skeletons (rather than paved roads against OSM) at
our scale: not found.

**Risk.** Many of our frames contain one straight track and no node at all. On those this method abstains, and 008's particle filter
is the answer. Skeletons of soft, ragged masks grow spurs that need pruning.

**Experiment.** Not a full matcher. Run it on the ridge frames only, as a second opinion on the along-track coordinate. Bar: lower
along-track error on the ridge subset than the 007.1 argmax.

#### C2. Check that matched objects are consistent with each other

Sources: CLIPPER (Lusk, Fathian & How, ICRA 2021, arXiv 2011.10202), TEASER++ (Yang, Shi & Carlone, arXiv 2001.07715), Ankenbauer
et al. 2022 (arXiv 2209.14426; cars matched to an aerial object map). The pyramid star-identification algorithm (Mortari et al.,
*Navigation* 2004) adds the verification step.

**Idea.** Every (drone object, satellite object) pairing is a candidate. Two candidates are compatible if the distance between the two
drone objects equals the distance between the two satellite objects. The largest group of mutually compatible candidates is the
answer. If no group of 3 or more exists, abstain. The star-tracker version adds one rule: a triangle match is trusted only if a
**fourth** object lands where it predicts.

**For us** this is an upgrade of 020's acceptance rule rather than a new matcher: clique size instead of a vote count, and the
fourth-object check instead of a tuned threshold.

**Risks, in the authors' own words.** Ankenbauer: "point-centroid representations may not suffice for large objects such as buildings",
"repetitive object layouts can produce many geometrically consistent false matches", and they needed ≥ 20 objects per registration. We have 2–6.
So this is only useful after objects are accumulated over several frames (Terlizzi & Fathian 2026 do exactly that, with Delaunay triangles and
per-building shape: area, elongation, rectangularity, convexity).

#### C3. Treat buildings as blurred points with an "unmatched" share

Sources: Coherent Point Drift (Myronenko & Song, TPAMI 2010), GMM registration (Jian & Vemuri, TPAMI 2011).

**Idea.** Each building centroid becomes a Gaussian blob weighted by its area. Matching maximises overlap, with an explicit outlier
share for buildings that exist on only one side. The blob width starts wide (a wide basin for 10–15 m errors) and is narrowed step by step.

**Honest caveat from sweep B.** With a fixed blob width this is close to weighted NCC on blurred building masks. The new parts are the
outlier share and the narrowing schedule. Low priority, and only behind 020.

### D. Refine the winner and measure how well it is pinned

Sources: Goforth & Lucey, ICRA 2019 (in `refs/`; inverse-compositional Lucas–Kanade on CNN features, < 8 m on a real flight),
Zhao et al. CVPR 2021 (Deep Lucas–Kanade). Zhang et al. ICRA 2016 for the degeneracy reading.

**Idea.** Take today's argmax as the starting point and run a few Gauss–Newton steps on the blurred soft class maps, solving for a
sub-cell shift (and optionally yaw). At the end, the 2×2 curvature of the cost tells how well each direction is pinned.

**What is new compared with our posterior ellipse.** The 007 ellipse comes from turning z-scores into probabilities with a temperature.
That temperature was 0.25 in one village and 2.0 in another, and that is why the gate doesn't transfer. The curvature at the optimum is
measured directly on the class maps, in metres, with no temperature at all.

**Risk.** Lucas–Kanade only converges from close by. Hinzmann & Siegwart (arXiv 2008.04619) saw plain ICLK get *worse* than its start
across years (13.88 → 17.29 after 50 iterations), which they blame on infrastructure change. Use it only as a refinement, never as the search.

**Experiment.** NCC argmax, then 2-DoF Gauss–Newton on σ = 2 then 1 cell blurred maps. Two claims: error below 5.861 m, and rejecting
the lowest-curvature quartile removes more than half of the > 10 m errors in a village it was not tuned on.

### E. Search heading and offset exactly instead of by brute force

Sources: Hess et al., *Real-Time Loop Closure in 2D LIDAR SLAM*, ICRA 2016 (Cartographer's branch-and-bound). Olson, *M3RSM*, ICRA 2015.

**Idea.** Build a pyramid of the satellite class map where each coarse cell holds the **best** value of the fine cells below it. A whole
block of offsets can then be ruled out at once if even its optimistic score can't beat the best found so far. The result is provably
the same as brute force, much faster, and a refusal is cheap: if nothing can beat the minimum score, stop.

**For us.** 003 showed heading search is worth −2.15 m but costs ~0.8 s per draw for 36 poses. This could make a 1° heading step fit the
Pi budget.

**Catch.** The bound needs a score that is a plain sum over cells. NCC divides by the window's spread, which breaks the guarantee, so
the score would have to be re-expressed (e.g. a zero-mean weighted sum). The Cartographer paper is also a warning for failure mode 6:
its tuned minimum score gave 77.3 % precision on one dataset and 97–99 % on others.

### F. Decide before the flight where matching can work

Sources: terrain-referenced navigation, where this is standard. Roughness and uniqueness measures in underwater TRN (KAIST thesis 2019,
details unverified). The Cramér–Rao bound for terrain navigation (Bergman & Ljung, CDC 1997). Map-quality prediction for visual
localisation (Merzic et al., ICRA 2017, tested on a fixed-wing UAV with a keypoint map). The gate side is from Lights Out (edge-point
count) and RIM (arXiv 2607.20116; "retrieval margins, correspondence count and spatial coverage" as reliability).

**Idea.** The satellite map is on board before take-off. For each position, correlate the satellite patch **with itself** at all shifts
within the search window. If the self-correlation has one sharp peak, the place is unique. If it has a ridge, the place is a road. If
it has several peaks, there are look-alikes. If it is flat, it is blind. This needs no drone data and no tuning on flight data. It
turns failure modes 1, 2 and 3 into a map you can look at.

**Worked example.** Over the bushcha forest the self-correlation is flat, so any correction there is suspect before we fly. Over the
birky church square it has one sharp peak, so corrections there can be trusted. 022 found the same split village by village (bushcha
accepts ~0.30 of draws, birky ~0.73). This map would predict it per position, before flying.

**Two cheap companions:**
- **Structure count on the drone side** (Lights Out): the number of class-boundary cells in the drone map. In their data, keeping frames above
  8,000 edge points kept ~90 % of frames and cut 2D RMSE from 2.18 to 2.05 m (their numbers). It is a count, not a score, so it may
  transfer between villages.
- **Map age** (Jurevicius et al. 2019, in `refs/`): they say the likelihood sharpness "can be decreased ... if the maps are known to be up-to-date".
  Tie the gate or the temperature to the satellite capture date instead of the village.

**Experiment.** Self-correlation peak sharpness (in metres) at the true position, against per-draw error of 007.1. Bar: Spearman
correlation of at least 0.5 on the holdout villages, and better than the heatmap's own peak z. This needs no new matcher. It is also
what the cerebrum notes recommended on 2026-10-22 ("offline satellite self-similarity as an untuned ambiguity map"), and it has not been built.

### G. If we return to a learned matcher: train on synthetic label maps

Sources: SynthMorph (Hoffmann et al., IEEE TMI 2022, arXiv 2004.10282). Label-driven registration (Hu et al. ISBI 2018). Zampieri et
al. ECCV 2018 (coarse-to-fine displacement chain on map vs aerial).

**Idea.** 023's learned head was limited by training data (46 villages). SynthMorph trains registration networks only on randomly
generated label maps, so appearance never matters, only shape. For us: generate random village-like class maps, simulate our
segmenter's errors on them (~29 % merged and ~25 % missed buildings, thin tracks dropped), shift them, and train the matcher on unlimited pairs.

**Experiment.** Pretrain 023's head on synthetic pairs only. Claim: it comes within 1 m of the same head trained on real pairs. If so,
fine-tuning on real pairs is the next run.

---

## 4. What the authors of our `refs/` papers say should be improved

From sweep E. Quotes and page or section numbers are in `E_refs_future_work.md`. Entries for He 2025, Sui 2022, Nassar 2018, Qiu 2024 and
Cabrera 2021 rely on our own notes, because the full texts couldn't be read.

| Their proposal | Who says it | Who implemented it | For us |
|---|---|---|---|
| Accumulate evidence over a sequence for single-road ambiguity | Li 2019 ("accumulate the observation of intersections over image sequences"); same symptom in BRM, SWA-PF, Goforth (25.00 m on a low-texture gravel pit vs 7.06 m on a village) | nobody in this set | **008** particle filter; C1 and C2 also only pay off over several frames |
| A reliability score from structure, not from a score threshold | Lights Out (edge count); RIM (margin + correspondence count + coverage); SWA-PF names feature-poor views as its main failure | Lights Out | F |
| Make likelihood sharpness depend on map age | Jurevicius 2019; RIM and AeroMap3D ("map-validity modeling"); SASGeo (map-age decay) | nobody | F |
| Keep several peaks instead of committing to top-1 | AnyVisLoc ("when the Top1 image is not the ground truth, the subsequent matching algorithm will fail"); AeroMap3D (a coherent wrong consensus wins) | nobody | B, 008 |
| Confusion-aware class weighting | Lights Out (12.1 % lower mean error in its ablation) | Lights Out | A2 |
| "A matching measure that is invariant to [season, time, weather]" | Kinnari 2021 (their stated open problem). They also pick a score by how much the true-pose and random-pose score distributions overlap | nobody | use their overlap criterion to compare candidate scores without per-village tuning |
| Automatic selection of which frames to use as templates | Goforth 2019, future work | nobody | the structure count in F |
| Building-only maps are fragile when footprints are missing; add stable layers | Terlizzi 2026, Choi (BRM) 2020 | we already use track and woody | nothing new |

The new 2025–2026 papers found by sweep E are SASGeo (synthetic), Lights Out (real, thermal) and two that were read only as abstracts.
The closest relative of our pipeline is Jin, Ren & Chen, *J. Appl. Remote Sens.* 19(1):014503, 2025. They segment both images and
template-match the label maps with a correlation coefficient. Only the abstract was read. **It is the first paper to read in full**, to see how
they weight classes and whether they reject one-class frames.

---

## 5. Looked at and not recommended

| Method | Why not |
|---|---|
| Persistent homology of masks | doesn't depend on position, so it can't localise |
| Hu / Zernike / Fourier shape descriptors per building | rural houses are near-rectangles and look alike; merging changes the shape. Area, elongation and orientation carry the same information |
| Gromov–Wasserstein | needs many objects; with 3 objects every set matches. Only patents apply it to images |
| Star-tracker grid and triangle keys (Padgett, Liebe) | keys built from nearest neighbours break when a neighbour is merged or missed (29 % / 25 % for us) |
| Best-Buddies Similarity | same idea as DDIS but much slower (paper: 35 s average vs 0.86 s for DDIS) |
| Dice / Tversky as the score | ranks offsets almost like cosine, which is already tried |
| Local self-similarity descriptors (Shechtman & Irani) | helps across sensors; our two inputs are already the same kind of map |
| Multi-channel Demons, label fusion | deformable machinery for a 2–3-parameter problem; reduces to per-class scores |
| Scan Context / semantic scan context | global place recognition around a centre point, the same family as He & Wu 2025 |
| Exact 2D Earth mover's distance per offset | a linear program at each of ~4,300 offsets; far too slow |

---

## 6. Suggested order

Ordered by cost to test and by how directly each targets the failures that dominate our error. Each needs a proposal under `/exp-new`,
the 5.861 m bar at 10 m injected, and the self-match check before any real-data run.

| # | Idea | Why this order | Cost to test |
|---|---|---|---|
| 1 | **F: offline self-similarity map + structure-count gate** | no matcher change; tests on existing 007.1 draws whether ambiguity is predictable before flight. If yes, every later idea gets a better gate for free | small: one notebook over the existing benches |
| 2 | **A1 + A2: masked NCC with unknown/contradiction cells, confusion-matrix weights** | smallest change to the score; directly removes the hand constants and the rare-class trap | small: one new `register_matching` mode |
| 3 | **B: sub-template consensus** | the one mechanism that covers ridge, look-alikes, blind frames and changed buildings, with a gate that counts votes | medium |
| 4 | **A3: QATM-style uniqueness score** | the most novel for UAV localisation; directly punishes ridges and look-alikes. Pi cost is the open question | medium |
| 5 | **C1 + C2 on top of 020** | run 020 first (it is planned, not run); then add track anchors for ridge frames and the consistency/4th-object check | medium, after 020 |
| 6 | **D: Gauss–Newton refinement + curvature gate** | sub-cell accuracy and a temperature-free confidence; refinement only | small |
| later | E (branch-and-bound search), G (synthetic training) | engineering for speed, and a route back to 023 | — |

The literature is consistent on one point: no single-frame score fixes a long straight track or a row of identical houses. Ideas 1–6
make the single frame better at saying "I can't tell". Telling the places apart is the job of the sequence (008).

---

## 7. Sources

Full citations, links, and which ones are unverified are in the five notes files. The main ones:

- Talmi et al., DDIS, CVPR 2017, https://arxiv.org/abs/1612.02190
- Cheng et al., QATM, CVPR 2019, https://arxiv.org/abs/1903.07254
- Padfield, Masked object registration in the Fourier domain, IEEE TIP 21(5), 2012
- Allen & Greeff, Lights Out, 2026, https://arxiv.org/abs/2604.26201 (abstract checked: thermal, confusion-aware weighting, RMSE₂D 2.18 m)
- Trukhina & Vashkelis, SASGeo, 2026, https://arxiv.org/abs/2607.07737
- Johnson et al., Mars 2020 Lander Vision System; JPL TRN page https://www-robotics.jpl.nasa.gov/what-we-do/flight-projects/mars-2020-rover/terrain-relative-navigation/
- Scheffler et al., AROSICS, Remote Sensing 9(7):676, 2017
- Vargas-Muñoz et al., Correcting rural building annotations in OSM, ISPRS JPRS 147, 2019, https://arxiv.org/abs/1901.08190
- Costea & Leordeanu, BMVC 2016, https://arxiv.org/abs/1605.08323
- Li et al., Road-network-based rapid geolocalization, 2019, https://arxiv.org/abs/1906.12174
- Lusk, Fathian & How, CLIPPER, ICRA 2021, https://arxiv.org/abs/2011.10202
- Ankenbauer, Fathian & How, View-invariant localization using semantic objects, 2022, https://arxiv.org/abs/2209.14426
- Terlizzi & Fathian, IROS 2026, https://arxiv.org/abs/2609.28225 (in `refs/`)
- Mortari et al., The pyramid star identification technique, Navigation 51(3), 2004
- Myronenko & Song, Coherent Point Drift, TPAMI 2010, https://arxiv.org/abs/0905.2635
- Goforth & Lucey, ICRA 2019 (in `refs/`); Hinzmann & Siegwart, https://arxiv.org/abs/2008.04619
- Zhang, Kaess & Singh, On degeneracy of optimization-based state estimation, ICRA 2016
- Olson, Real-time correlative scan matching, ICRA 2009; Olson, M3RSM, ICRA 2015
- Hess et al., Real-time loop closure in 2D LIDAR SLAM, ICRA 2016
- Bergman & Ljung, Point-mass filter and Cramér–Rao bound for terrain-aided navigation, CDC 1997
- Merzic et al., Map quality evaluation for visual localization, ICRA 2017
- Hoffmann et al., SynthMorph, IEEE TMI 2022, https://arxiv.org/abs/2004.10282
- Jin, Ren & Chen, J. Appl. Remote Sens. 19(1):014503, 2025, DOI 10.1117/1.JRS.19.014503 (abstract only)
