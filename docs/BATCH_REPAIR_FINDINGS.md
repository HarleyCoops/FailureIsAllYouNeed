# Batch repair: negative result and limits of the inference

Source: KimiBrainSurgery commit `2b417b1092eb8c61ba1d9989cf182d014c96ae81`.
The saved sweep has 18 configurations and seven mined failure states. None fixed
all seven while preserving the 64-game gate mean. No v5 was promoted; v4 remains
the published checkpoint. The gate baseline here is 12.09375, distinct from the
1,000-game v4 development mean 12.182 and fresh mean 12.185.

| Candidate | Fixed states | Probe flips | Gate return |
| --- | ---: | ---: | ---: |
| v4 incumbent | 0/7 | 0 | 12.09375 |
| W3 exact, lambda 100 | 5/7 | 5,561 | 2.40625 |
| W3 soft, left only, lambda 100 | 1/7 | 174 | 12.09375 |
| W2, lambda 1,000, cap 0.5 | 5/7 | 1,750 | 7.3125 |

This supports a measured interference problem for the tested updates. It does
not prove linear inseparability, exhausted capacity, or that additional training
will solve the failures. Large edit norms are not an infeasibility certificate.

The W3 implementation groups states by saving action, solves each selected-column
equality system separately, then applies all column edits. Its constraints compare
against each state's original wrong action, not every competing action after all
updates. Exact satisfaction of such an equality is weaker than making a saving
action the final argmax. The W2 method also uses local linearization and finite
trust-region steps. These restrictions matter when interpreting solver failure.

One candidate fixes one state at unchanged gate mean (174 probe flips). Thus
"every additional fixed constraint costs more than a point" is not supported by
the recorded table. That candidate is an improvement in fixed-count at equal
return, but fails this experiment's stricter all-seven-fixed promotion rule and
changes probe decisions. No candidate meets that rule; this is the defensible
negative result.

The batch miner explicitly includes eight v4 fresh-set loss seeds in the 30000
range as well as old-set losses. They are now development examples for subsequent
repairs. The earlier frozen v4 fresh result remains valid as a historical result;
a future v5 requires a new untouched audit set. Re-running the same paired set
would be a regression check, not a fresh generalization test.

Brief gradient fine-tuning is a reasonable experiment, not a guaranteed escape
from these constraints. It should use a correction set plus preservation examples,
compare to the same frozen v4 checkpoint, pass a declared gate, and then receive
one untouched final audit. No such training was launched in this synchronization.
