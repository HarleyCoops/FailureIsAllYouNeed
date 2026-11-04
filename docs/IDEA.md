# Failure as a constraint on parameter repair

The useful intuition is to begin with the outcome we want to change and work
backward through the decisions that produced it. A failed trajectory contains
more diagnostic information than an aggregate reward alone: which observations
were available, which actions were chosen, and when recovery became impossible.

The research question is whether that evidence can guide a small parameter edit
whose effects survive independent evaluation.

## An inverse problem, with multiple possible answers

Write a policy as `pi_theta`, its weights as `theta`, and a failed trajectory as
`tau`. We want an edit `delta` for which the policy `pi_(theta + delta)` changes
the relevant outcome. A possible research objective is:

```text
find delta that minimizes:
    targeted_failure_loss(theta + delta; tau)
    + alpha * edit_size(delta)
    + beta * preservation_loss(theta + delta; reference_cases)

subject to:
    a bounded, valid parameter edit
    a predefined independent acceptance test
```

This is a statement of the intended problem, not the optimizer implemented in
the published run. That run used LLM-proposed assignments and empirical game
comparisons. It did not solve this objective analytically.

The inverse is generally underdetermined. Different parameters can compensate
for one another, hidden features can be distributed, and a sampled action can
fail even when it is not the most probable action. An observation may also omit
information needed to avoid an outcome. A failure therefore supplies constraints
and hypotheses; it is not a uniquely decodable weight-update instruction.

## Relationship to gradient descent

Backpropagation already moves from an error signal to parameter sensitivities.
The distinction here is how the update is proposed and validated: explicit
diagnosis, sparse weight assignments, practice feedback, and an independent
acceptance gate, without computing gradients in this experiment.

Weights can reflect structure learned during training. Their values do not
uniquely reveal the sequence of gradients or updates that produced them. Tensor
dimensions identify the architecture, not the model's training history. Our
starting Breakout controller was handwritten, so no training history existed to
recover in this case.

“Working backward from failure” is a useful description of the reasoning task.
“Inverting gradient descent” would claim an algorithm and evidence we do not have.

## From a plausible edit to a causal claim

Three questions need separate answers:

1. **Did the edit change the intended behavior?** Replay the same trajectory with
   controlled initial conditions and randomness. Where possible, compare a
   frozen-prefix intervention with a full episode from reset.
2. **Did performance improve elsewhere?** Evaluate a frozen candidate against the
   incumbent on separate matched cases. Report regressions as well as averages.
3. **Did these parameters specifically account for the change?** Compare component
   ablations, alternate edits, and controls. A successful combined patch is not
   sufficient to attribute its full effect to an individual parameter.

The published experiment answers the second question for one simulator and seed
distribution. It records a negative answer to the first question for the
diagnostic miss. It has not performed the ablations needed for the third.

## What a repeated repair loop would need

An accepted edit becomes the new incumbent. Subsequent failures should be mined
from that incumbent. A rejected candidate leaves the incumbent intact. Repeated
search requires separating development feedback from future reserved evaluations;
reusing a fixed gate adaptively can overfit that gate.

A gate can enforce non-decreasing measured average return on the cases it checks.
It cannot guarantee non-decreasing performance on every possible input. Our final
audit makes this concrete: 671 games improved, 243 regressed, and 110 tied despite
a strong positive average gain.

For language models, the definition of a failure and the preservation cases
would be different. Linguistic validity, dialect, alternative valid answers,
and judgments by qualified speakers matter. The current game result supplies
no evidence yet that a Cree or Dakota model can be repaired this way.
