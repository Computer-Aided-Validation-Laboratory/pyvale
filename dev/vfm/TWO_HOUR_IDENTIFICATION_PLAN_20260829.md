# Two-hour notched-EBW identification follow-up

## Decision to target

Decide whether BF6 should be the normal endpoint with BF7 admitted only by a
guarded selector, and determine which scale/regime/sensitivity components can
make that BF6-to-BF7 decision under WDBN1-calibrated noise.

## Work programme

1. Complete the eight missing BF7 native-DOF projection states. Do not compute
   BF8 in this run.
2. Merge BF5-BF7 results and evaluate rankings within BF6-BF7, not only over
   the pooled state library.
3. Test the smallest evidence vector:
   - raw fine-scale EGI at yield onset;
   - one fine/mid-scale projected or yield-unique term in developed/late
     plasticity;
   - broad EGI/FRE as a non-regression guard.
4. Check redundancy across EGI7/15/29/57 in physical-scale order. Retain a
   middle scale only when it improves BF6-to-BF7 pairwise decisions beyond the
   fine and broad terms.
5. Produce a concise decision table for every seed: map-error change,
   mechanical-objective change, component decisions, and whether BF7 should
   have been accepted without access to truth.

## Interpretation

- BF7 is useful only if a truth-free evidence rule accepts most genuinely
  improving BF7 states and rejects degrading ones under calibrated noise.
- Strong pooled correlation is insufficient; BF6-BF7 paired accuracy is the
  primary result.
- BF8 remains a later negative-control test for the frozen stopping rule. It is
  not required to choose the next objective candidate.
- Do not launch further full identification variants until this offline result
  identifies a compact selector worth replicating.
