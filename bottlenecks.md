  1. Flow Slot Serialization (1 slot/cycle)                                                                                                           
  - Branch phase: 2 vselect operations per group
  - Wrap phase: 2 vselect operations per group                                                                                                        
  - Total: 4 flow ops per group per round = 128 flow ops total                                                                                      
  - With only 1 flow slot, these cannot be parallelized
  - Current seq assignment gives each vselect its own seq (1, 2), forcing 2 cycles each for branch and wrap

  ---
  2. Scatter-Gather Loads in Rounds 1-15 (2 slots/cycle)
  - 16 individual element loads per group per round
  - At 2 loads/cycle = 8 cycles of load time per group
  - 15 rounds × 2 groups × 8 cycles = 240 cycles minimum just for gather loads
  - Loads are sequenced across 8 different seq values, limiting overlap opportunities

  ---
  3. Sequential Seq Dependencies Within Groups
  - The scheduler enforces strict (round, phase, seq) ordering within each group
  - A's phase 1 (hash) cannot start until A's phase 0 (gather) completes all seq values
  - This prevents A's hash from overlapping with A's gather loads, even though they use different engines (valu vs load)

  ---
  4. Underutilized Valu Slots During Hash
  - Hash part 1: uses 4 of 6 valu slots (2 slots wasted)
  - Hash part 2: uses 2 of 6 valu slots (4 slots wasted)
  - 12 sequential hash bundles per group, but neither A nor B can use the spare valu slots due to phase dependencies

  ---
  5. Branch/Wrap Valu Ops Don't Overlap With Other Group's Flow
  - Branch valu (6 ops) fills the valu slot, then branch flow uses flow slot
  - During A's branch flow cycles, B's valu slots could theoretically be used, but seq ordering limits this

  ---
  Summary: ~34 cycles per pair-round, with the primary constraints being:
  - Flow operations are fundamentally serial (1 slot)
  - Scatter-gather loads are slow (2 slots for 16 loads)
  - Seq ordering prevents optimal cross-phase/cross-group packing