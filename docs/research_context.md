# Research context

## Question

How much faster can a robot execute the same task with continuous cubic B-spline
actions before tracking error, collisions, actuator saturation, or task failure become
unacceptable?

## Comparison

`action_chunk` predicts absolute joint targets at the policy rate and holds discrete
commands between updates. `bspline_action` represents a local sequence with spline
control points and evaluates a continuous curve at the controller rate. Both methods
receive paired task layouts, the same controller, the same nominal path, and the same
speedup request.

The educational `stop_and_go` planner explains continuity but is not the scientific
baseline.

## Evidence policy

A configured speedup is an input, not a result. The benchmark reports achieved
completion time only for successful MuJoCo rollouts and reports failure rate separately.
Every aggregate must trace to `rollouts.json` and retained experiment bundles.

The milestone requires multiple speeds and paired seeds. A one-seed pilot can validate
the pipeline but cannot support a research conclusion.

## Relation to B-spline Policy

This is a small, reproduction-inspired study of the action-representation and temporal
scaling mechanism in the linked B-spline Policy paper. It does not reproduce the
paper's large policies, datasets, real-robot evaluation, Diffusion Policy, or ACT
experiments. The initial learned study uses matched compact state-based regressors.
The implemented closed-loop smoke test currently fails the release success gate. See
[`learned_policy_status.md`](learned_policy_status.md) for measured diagnostics. Learned
results are therefore not part of the headline v0.3 conclusion.

## Scope boundary

RGB, metric depth, and simulator segmentation are retained for replay and debugging.
They are not policy inputs in the core milestone. Guidance flow and warped noise are
excluded so the central B-spline action claim remains measurable and auditable.
