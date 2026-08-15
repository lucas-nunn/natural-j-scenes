# Verified: brain and model RDMs index the same conditions in the same order

Status: **VERIFIED 2026-08-15**, empirically and by reading the chain. Pinned by
`tests/test_condition_alignment.py`.

## Why this is the axiom that matters most

A permutation between the averaged-beta column order and the model RDM row order **raises
nothing**. RSA would simply correlate mismatched conditions and return a small, plausible,
publishable-looking effect. It is indistinguishable by inspection from a real weak result — which
is exactly the kind of result this project reports. So it had to be checked, not assumed.

## The chain, as verified

`get_subject_conditions(nsd_dir, subj, 10, keep_only_3repeats=True)` returns three arrays. Measured
for subj01:

| return value | shape | sorted | unique | used for |
|---|---|---|---|---|
| `conditions` | (7500,) | no | no | unused here |
| `sampled_conditions` | (2505,) | no | no | **betas** (835 x 3 repeats) |
| `subject_conditions` | (835,) | **yes** | **yes** | **model RDMs** |

`subject_conditions == np.unique(sampled_conditions)` -> True.

The two sides are wired through different return values, so alignment is not obvious:

1. **Model side.** `prepare_conditions` takes `subject_conditions` and *asserts*
   `np.array_equal(conditions, np.unique(conditions))` before storing
   `subj*_condition_ids.npy`. `_subject_condition_ids` re-checks sorted+unique on load.
   `prepare_grouped_rdms` then orders rows by that array. So model RDM order = sorted NSD ID.
2. **Brain side.** `_compute_betas_average` receives the *unsorted, repeated* `sampled_conditions`
   and builds its column map from `np.unique(...)`. **`np.unique` sorts**, so betas column order =
   sorted NSD ID.
3. Both sides therefore agree, and the locked `choices` array indexes both consistently.

Empirical confirmation — averaged betas last axis vs stored condition count:

```
subj01  (81, 104, 83, 835)  == 835   sorted+unique ids
subj02  (82, 106, 84, 835)  == 835   sorted+unique ids
subj08  (80, 103, 78, 835)  == 835   sorted+unique ids
```

## The fragility that was fixed

Correctness rested on an **implicit** property of `np.unique`. Replacing it with a
first-appearance unique (`pd.unique`, `dict.fromkeys`, a set comprehension) would permute the
brain side against the model side, silently, with every test still green.

Extracted to `nsd_adapter.condition_column_index()` with the contract stated in its docstring, and
pinned by five tests including one that explicitly asserts sorted order beats first-appearance
order for a case where the two differ.

## Not yet checked

The MPNet reference RDM is **copied** from the external MPNet tree rather than recomputed, so its
condition order is inherited, not verified here. It agrees numerically with the historical run to
four decimals, which is strong indirect evidence, but the ordering itself is unverified. See
[[hardening-backlog]].
