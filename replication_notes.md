# Replication Remarks

The Python leabra library strive to be quantitatively equivalent to the Leabra implementation of the emergent software.
Below are useful remarks about some differences and where in emergent some of the behavior of Leabra
are to be found.

## Float precision

`emergent` is in coded with float precision, while Python natively supports doubles. This difference
means that the quantitative equivalence does not go beyond the 1e-7 precision (relative to the values considered).
Moreover, rounding differences add up in an iterative process such as neural networks, and may for
instant produce significantly different asymptotes.

## Code correspondance

* Hard clamping behavior (forcing activities) is found in `LeabraUnitSpec::Compute_HardClamp()`.
* `Connection.learning_rule()` and `Connection.apply_dwt()` from `LeabraConSpec::C_Compute_Weights_CtLeabraXCAL()`.
* `Unit.avg_l_lrn` from `LeabraUnitSpec::GetLrn()`
