# Whole-cortex localisation — and why the obvious reading is wrong

Generator: `scripts/analyze_cortex_localisation.py`. Cached: `docs/cortex_localisation_l23.json`.
Run 2026-08-16, layer 23, eight subjects. **EXPLORATORY**, uncorrected.

Follows [[stream-localisation]], which could only test visual cortex. The searchlight in fact
covers the whole HCP-MMP1 parcellation — all 102,533 labelled voxels are valid centres, ~75,000 of
them outside the visual streams — so the frontoparietal question was answerable from data already
on disk.

## The raw numbers, which look decisive and are misleading

Primary contrast, visual streams versus the rest of labelled cortex:

```
inside visual streams   +0.004602
outside (rest of ctx)   +0.001822
outside minus inside    -0.002780   p = 0.0078  (8/8 subjects)
```

Top parcels are **TPOJ2, MST, TPOJ3, FST, MT, V4t, FFC, TPOJ1** — the motion complex and
temporo-parieto-occipital junction. Canonical frontal parcels sit at **median rank 113 of 180**,
with a mean delta 0.63x the all-parcel mean.

Read naively that says: the J advantage is a visual-cortex effect, notably weaker in prefrontal
cortex, and the workspace account is refuted. **That reading is wrong**, and it is the reading this
analysis would have produced without the next section.

## The SNR control, which inverts it

NSD is a *visual* experiment. Visual cortex carries far more signal, so **every** alignment measure
is larger there. Across the 180 parcels:

```
Pearson(J-raw delta, raw alignment level) = +0.798
```

The absolute advantage is largely proportional to how well raw features align in a parcel at all.
The spatial gradient is mostly a **signal-quality** gradient, not a representational one.

Normalising each parcel's advantage by its own raw alignment level:

```
relative advantage   frontal parcels  +0.0944   (n=18)
                     all others       +0.0774   (n=162)
                     Mann-Whitney      p = 0.065
```

Frontal cortex is **not** deficient. If anything its relative advantage is marginally *higher*.

## What the result actually is

**The J advantage is close to a uniform multiplicative gain of roughly 8% on whatever alignment a
region already has, everywhere in cortex.** 94 of 180 parcels reach uncorrected p < 0.05, against
9 expected by chance, so the breadth is real even though individual parcels should not be claimed.

For the workspace account this is still not supportive, but for a different reason than the raw
numbers suggested. A workspace account predicts the advantage **concentrates** somewhere. It does
not concentrate anywhere — not in visual cortex once signal is accounted for, and not in
frontoparietal cortex either. A uniform proportional gain looks like a property of the
representation as a whole, not of a specific network.

## Method note worth keeping

This is the second time tonight an analysis produced a confident answer that a control reversed.
Any spatial claim about an effect measured on NSD must be normalised by regional signal level
before it is interpreted, because the experiment's own visual bias will otherwise manufacture a
sensory-to-association gradient — or destroy one — regardless of the representation being tested.
