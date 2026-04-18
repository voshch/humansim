# Benchmark configs

Relative benchmarks: run two parameter sets (reference vs candidate) across agent-count stages and compare tick-time distributions.

Consumed by `ros2 run arena_humansim benchmark` (entry point: `arena_humansim.utils.benchmark:main`).

## Config shape

```yaml
dt: 0.05              # sim timestep
ticks: 1000           # ticks per round
warmup: 50            # discarded leading ticks
rounds: 3             # repetitions per (config, stage) pair

stages:
  - label: open space
    agents: [10, 25, 50, 100]
  - label: 10x10 maze
    agents: [10, 25, 50, 100]
    walls: maze(10)

reference:
  label: dijkstra
  params_file: pkg://arena_humansim/config/default.yaml
  params:
    global_planner: dijkstra

candidate:
  label: astar
  params_file: pkg://arena_humansim/config/default.yaml
  params:
    global_planner: astar
```

Each run uses `reference.params_file` (or `candidate.params_file`) as the sim base config and overlays `params:` on top.

- `stages[].agents` — agent counts to sweep. Each becomes its own measurement row.
- `stages[].walls` — optional wall layout. Literals: `maze(N)` for an N×N maze; or a list of segments `[[[x1,y1],[x2,y2]], ...]`.
- `params_file` supports `pkg://<pkg>/<path>` to resolve package-share paths.
- `params:` is an inline override dict. Also settable on the command line via `--ref-params k=v,k2=v2`.

## CLI overrides

Any config value can be overridden at invocation:

```bash
ros2 run arena_humansim benchmark astar_vs_dijkstra \
  --agents 10,50,200 --ticks 500 --rounds 5 \
  --profile --csv results.csv
```

Bare names (`astar_vs_dijkstra`) resolve to `config/benchmark/<name>.yaml` in the package share. Paths and `.yaml` are used verbatim.

## Included configs

| File | Purpose |
|---|---|
| `default.yaml` | Self-comparison (config vs itself). Sanity check: RTF spread should be within round-to-round noise. |
| `astar_vs_dijkstra.yaml` | Global planner sweep across open space and a 10×10 maze. Canonical example for a stage-with-walls run. |

## Adding a benchmark

1. Drop a yaml here.
2. Point `reference` and `candidate` at the params file(s) you want to compare; they can be the same file with different `params:` overlays.
3. Run `ros2 run arena_humansim benchmark <name>` — no rebuild needed if `install/` is the devel install.

## Output

Printed table per stage: median, p95, p99, RTF per agent-count. With `--profile`, per-phase means and p95s; with `--csv`, every tick time as a row.

RTF note: the master-mode sim is the time authority — `rtf` is a ceiling. Benchmarks report actual per-tick wall cost, which is the signal that matters for comparison.
