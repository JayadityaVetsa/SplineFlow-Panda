# Architecture

```text
YAML config -> paired task seeds -> action representation -> IK/controller -> MuJoCo
                         |                                      |
                         +-------- experiment bundle <----------+
                                      |
                         metrics + RGB/depth/segmentation
                                      |
                           benchmark report + dashboard
```

The dashboard reads saved artifacts and launches CLI subprocesses. Planning,
simulation, collision geometry, evaluation, dataset generation, and policy training
remain tested package code.
