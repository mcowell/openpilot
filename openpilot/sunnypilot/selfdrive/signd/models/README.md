# signd models (proof of concept)

- `sign_detector.onnx`: YOLO11n sign detector at 416x416, two classes (0 speed limit sign, 1 road work sign), trained
  with Ultralytics (AGPL-3.0). Training data includes Mapillary imagery (CC BY-SA 4.0). Not upstream-clean; to be replaced
  by a clean detector.
- `sign_detector_local.onnx`: the same detector weights exported at 160x160, for re-finding a tracked sign.
- `school_classifier.onnx` (optional): school zone check on 40x60 crops of the area above a speed sign (none / school
  pentagon), trained on Mapillary school zone photos (CC BY-SA 4.0, eye-checked) shrunk to 3X sizes, and negatives above
  ordinary speed signs from Mapillary and own drives.
- `worker_classifier.onnx` (optional): work zone check on 44x44 crops of a detected sign (other / road work sign: Ontario
  TC-2 orange diamond with a person digging, flagger signs), trained on Mapillary roadworks photos (CC BY-SA 4.0) and
  eye-labelled own-drive crops cut from raw camera frames as signd sees them. 44, not 40: the 3X compiler fails on 32,
  40 and 48 inputs.
- `sign_classifier.onnx`: value classifier on 64x96 panel+tab crops (10..130 km/h step 5, reject, end of zone), trained on synthetic signs,
  own drives and Mapillary crops (CC BY-SA 4.0).

Negatives (classifier v7, 2026-11-04): licence plates were read as signs at red lights, and a Highway 416 crown shield
as 40. The classifier now also trains on licence plates found in own drives while stopped or slow (OWLv2, reviewed by
eye), plates signd actually misread, and synthetic King's Highway shields, all as 'reject'. On held-out drives
(incl. the 2026-11-04 drive): 35/36 signs read (was 33), no plate, shield or end-of-zone accepts (was one plate). A
detector retrained with the same plates was tried and not shipped: it found more signs but read a dusk "50 AREA ENDS"
as 50 and lost another sign.

Classifier v9 (2026-11-10): retrained with signs from the 2026-11-08/09 drives (416, city, dusk), reviewed by eye.
Yellow advisory signs, "ahead" signs (up arrow), green guide signs and route shields on those drives are trained as
'reject'. Crops OCR could not read are no longer trained as 'reject' (v8 did, and learned to reject real dusk signs).
Every accepted limit over all 28 drives was checked by eye: 200 correct (v7: 179), on held-out drives 58 (v7: 52) of
64 signs found by any version; the only wrong accept is an "ahead" 80 on a Highway 416 off-ramp, which v7 also makes
(v7 additionally accepted an "ahead" 60 and a county road shield as 15). Detector unchanged.

Classifier v10 (2026-11-12): retrained with ~2x the Mapillary signs (Ottawa, 416/417/401 corridors; OCR-verified,
10.8k) and synthetic signs with condition plates (times/days/school/community safety zone) and TRUCKS plates as
'reject'. Replayed over all 29 drives with the filter's plausibility rules: 204 correct accepts graded before (v9: 200)
plus 2 new ones checked by eye, and the same single wrong accept (the "ahead" 80 on the Highway 416 work zone). It reads
a "MAXIMUM 40 ... ENDS" sign as end of zone where v9 accepted 40; it misses a night 80 and a 40 with plates below
that v9 reads. Without the plausibility rules it reads county road markers as 15/45 (as v9 seeds 0 and 2 do):
Mapillary-capped and synthetic county marker retrains did not fix that.

School check v1 (2026-11-23): own drives (never trained on) 15/15 school sign crops found, 0 false; Mapillary held-out
129/129 found, 10/563 false (a yellow curve + advisory plate, dark night photos). The filter needs 2 school readings of
the held limit's value (p >= 0.6) within 4 s. Replayed over the two school drives (Greenbank 50, 4a 40): the plate went on
at each school sign and off ~500 m later.

Detector v7 + classifier v12 (2026-11-23): the black NCC parkway signs (white on black, Kichi Zibi Mikan). The detector
retrained on all labelled drives (rounds 11-14) plus hand-boxed black sign frames now finds them; the classifier adds 551
eye-checked crops of 11 black signs (drives 36/50 trained, drive 52 held out) and three misreads as training samples (a
no-U-turn sign and a stacked 100/60 as reject, a school zone 60 read as 40). Colour-inverted white signs as extra
training data were tried and dropped: they added misreads of distant white signs (60 as 80). Replayed over all 63
drives and checked by eye: no wrong limit changes (v10: none); signs shown 90% on held-out drives (v10: 86%); the 9
black sign sightings all shown (v10: none); 17 school plates, all at real school signs. Classifier seeds still differ a
lot (two siblings of the shipped one made 3 wrong changes each): pick by the full replay, never by validation metrics.

Detector v9 + work zone check v3 (2026-11-26): the detector gains a road work sign class (Mapillary roadworks windows and
own-drive frames with the whole orange diamond boxed), fine-tuned from detector v7 so its speed sign features stay (v8,
trained from scratch, lost speed signs with plates below, read a 50 as 60 and turned a red store sign into a work zone).
Replayed over all 66 drives and checked by eye: 30 work zones (v7 with the work zone check: 11), 29 at worker signs and
one at an orange lane shift sign in a construction zone; no wrong speed limits; vs v7 about 10 signs gained (night,
40 AREA) and 8 lost (other 40 AREA signs, two night signs), ~15 short 40 flickers at "40 AREA ... ENDS" signs gone.
Orange temporary limits are no longer read (never trained on). On the 3X: detector 72 ms (416) / 14 ms (160), work zone
check 5 ms, about 1% of signd's GPU jobs.
