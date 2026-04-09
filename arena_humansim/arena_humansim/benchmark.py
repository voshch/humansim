"""Relative benchmark: compare two configurations at various agent counts."""

import argparse
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import RemoveAgents, SpawnAgents
from geometry_msgs.msg import Pose2D, Vector3
from rclpy.node import Node


@dataclass
class RunResult:
    n_agents: int
    dt: float
    tick_times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.tick_times_ms))

    @property
    def std_ms(self) -> float:
        return float(np.std(self.tick_times_ms))

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.tick_times_ms, 95))

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.tick_times_ms, 99))

    @property
    def rtf(self) -> float:
        return (self.dt * 1000.0) / self.mean_ms if self.mean_ms > 0 else float("inf")


@dataclass
class IncrementalResult:
    dt: float
    spawn_interval: int
    total_ticks: int
    ticks: list[dict] = field(default_factory=list)

    def by_agent_count(self) -> dict[int, RunResult]:
        groups: dict[int, list[float]] = {}
        for t in self.ticks:
            groups.setdefault(t["n_agents"], []).append(t["tick_ms"])
        return {n: RunResult(n_agents=n, dt=self.dt, tick_times_ms=times) for n, times in sorted(groups.items())}


class BenchmarkDriver(Node):
    def __init__(self):
        super().__init__("benchmark_driver")
        self._spawn_cli = self.create_client(SpawnAgents, "spawn_agents")
        self._remove_cli = self.create_client(RemoveAgents, "remove_agents")

        self._tick_times: list[float] = []
        self._last_recv: float | None = None
        self._collecting = False
        self._target_ticks = 0
        self._done_event = threading.Event()

        self._sub = self.create_subscription(
            AgentStatesMsg,
            "agent_states",
            self._on_agent_states,
            10,
        )

    def _on_agent_states(self, msg: AgentStatesMsg):
        now = time.perf_counter()
        if not self._collecting:
            self._last_recv = now
            return
        if self._last_recv is not None:
            self._tick_times.append((now - self._last_recv) * 1000.0)
        self._last_recv = now
        if len(self._tick_times) >= self._target_ticks:
            self._collecting = False
            self._done_event.set()

    def wait_for_services(self, timeout: float = 10.0):
        for cli in (self._spawn_cli, self._remove_cli):
            if not cli.wait_for_service(timeout_sec=timeout):
                raise TimeoutError(f"Service {cli.srv_name} not available")

    def spawn_random_agents(self, n: int, seed: int = 42, radius: float = 20.0):
        rng = np.random.default_rng(seed)
        req = SpawnAgents.Request()
        for i in range(n):
            angle = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0, radius)
            x, y = r * math.cos(angle), r * math.sin(angle)
            goal_angle = rng.uniform(0, 2 * math.pi)
            gx, gy = radius * math.cos(goal_angle), radius * math.sin(goal_angle)

            msg = AgentStateMsg()
            msg.agent_id = 0
            msg.pose = Pose2D(x=x, y=y, theta=angle)
            msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
            msg.desired_velocity = float(rng.normal(1.3, 0.15))
            msg.radius = 0.0
            msg.agent_type = "adult"
            wp = WaypointMsg()
            wp.pose = Pose2D(x=gx, y=gy, theta=0.0)
            msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)
            req.agents.append(msg)

        resp = self._call_sync(self._spawn_cli, req, timeout=30.0)
        if resp is None or not resp.success:
            raise RuntimeError(f"Spawn failed: {resp}")
        return resp.spawned_ids

    def _call_sync(self, client, request, timeout=10.0):
        future = client.call_async(request)
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout=timeout):
            raise TimeoutError(f"Service call to {client.srv_name} timed out")
        return future.result()

    def remove_all(self, ids: Iterable[int]):
        req = RemoveAgents.Request()
        req.agent_ids = list(ids)
        self._call_sync(self._remove_cli, req, timeout=10.0)

    def collect_ticks(self, n_ticks: int, warmup: int = 50) -> list[float]:
        self._tick_times = []
        self._last_recv = None
        self._target_ticks = warmup + n_ticks
        self._done_event.clear()
        self._collecting = True

        timeout = (warmup + n_ticks) * 1.0
        self._done_event.wait(timeout=timeout)

        if len(self._tick_times) < warmup + n_ticks:
            raise RuntimeError(f"Only collected {len(self._tick_times)}/{warmup + n_ticks} ticks")
        return self._tick_times[warmup:]

    def collect_incremental(
        self,
        total_ticks: int,
        spawn_interval: int,
        seed: int = 42,
        radius: float = 20.0,
        warmup: int = 20,
    ) -> IncrementalResult:
        rng = np.random.default_rng(seed)
        all_ids: list[int] = []
        result = IncrementalResult(dt=0.0, spawn_interval=spawn_interval, total_ticks=total_ticks)

        self._tick_times = []
        self._last_recv = None
        self._target_ticks = warmup
        self._done_event.clear()
        self._collecting = True
        self._done_event.wait(timeout=warmup * 1.0)
        self._tick_times = []

        for tick in range(total_ticks):
            if tick > 0 and tick % spawn_interval == 0:
                ids = self._spawn_one_random(rng, radius)
                all_ids.extend(ids)

            self._tick_times = []
            self._last_recv = None
            self._target_ticks = 1
            self._done_event.clear()
            self._collecting = True
            self._done_event.wait(timeout=2.0)

            if self._tick_times:
                result.ticks.append({"tick": tick, "n_agents": len(all_ids), "tick_ms": self._tick_times[0]})

        return all_ids, result

    def _spawn_one_random(self, rng: np.random.Generator, radius: float = 20.0) -> list[int]:
        angle = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0, radius)
        x, y = r * math.cos(angle), r * math.sin(angle)
        goal_angle = rng.uniform(0, 2 * math.pi)
        gx, gy = radius * math.cos(goal_angle), radius * math.sin(goal_angle)

        msg = AgentStateMsg()
        msg.agent_id = 0
        msg.pose = Pose2D(x=x, y=y, theta=angle)
        msg.velocity = Vector3(x=0.0, y=0.0, z=0.0)
        msg.desired_velocity = float(rng.normal(1.3, 0.15))
        msg.radius = 0.0
        msg.agent_type = "adult"
        wp = WaypointMsg()
        wp.pose = Pose2D(x=gx, y=gy, theta=0.0)
        msg.waypoints = WaypointsMsg(points=[wp], mode=WaypointsMsg.MODE_ONCE)

        req = SpawnAgents.Request()
        req.agents.append(msg)
        resp = self._call_sync(self._spawn_cli, req, timeout=10.0)
        if resp is None or not resp.success:
            raise RuntimeError(f"Spawn failed: {resp}")
        return list(resp.spawned_ids)


def launch_humansim(
    params_file: str | None,
    extra_params: dict,
    dt: float,
    profile: bool = False,
    profile_interval: int = 100,
) -> subprocess.Popen:
    cmd = [
        "ros2",
        "run",
        "arena_humansim",
        "arena_humansim_node",
    ]
    if profile:
        cmd.extend(["--profile", "--profile-interval", str(profile_interval)])
    cmd.extend([
        "--ros-args",
        "-p",
        "mode:=benchmark",
        "-p",
        "publish_markers:=0",
        "-p",
        f"dt:={dt}",
    ])
    if params_file:
        cmd.extend(["--params-file", params_file])
    for k, v in extra_params.items():
        cmd.extend(["-p", f"{k}:={v}"])
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL if not profile else None,
        stderr=subprocess.PIPE if not profile else None,
        start_new_session=True,
    )
    return proc


def kill_proc(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGINT)
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait()


def run_single(
    driver: BenchmarkDriver,
    params_file: str | None,
    extra_params: dict,
    n_agents: int,
    dt: float,
    n_ticks: int,
    warmup: int,
    seed: int,
    executor,
    profile: bool = False,
    profile_interval: int = 100,
) -> RunResult:
    proc = launch_humansim(params_file, extra_params, dt, profile=profile, profile_interval=profile_interval)
    try:
        time.sleep(1.5)
        driver.wait_for_services(timeout=10.0)
        ids = driver.spawn_random_agents(n_agents, seed=seed)
        tick_times = driver.collect_ticks(n_ticks, warmup=warmup)
        result = RunResult(n_agents=n_agents, dt=dt, tick_times_ms=tick_times)
        driver.remove_all(ids)
    finally:
        kill_proc(proc)
    time.sleep(0.5)
    return result


def run_incremental(
    driver: BenchmarkDriver,
    params_file: str | None,
    extra_params: dict,
    dt: float,
    total_ticks: int,
    spawn_interval: int,
    warmup: int,
    seed: int,
    executor,
    profile: bool = False,
    profile_interval: int = 100,
) -> IncrementalResult:
    proc = launch_humansim(params_file, extra_params, dt, profile=profile, profile_interval=profile_interval)
    try:
        time.sleep(1.5)
        driver.wait_for_services(timeout=10.0)
        ids, result = driver.collect_incremental(
            total_ticks=total_ticks,
            spawn_interval=spawn_interval,
            seed=seed,
            warmup=warmup,
        )
        result.dt = dt
        driver.remove_all(ids)
    finally:
        kill_proc(proc)
    time.sleep(0.5)
    return result


def print_incremental_results(label: str, result: IncrementalResult):
    by_count = result.by_agent_count()
    if not by_count:
        print(f"  {label}: no data collected")
        return

    def ms(v):
        return f"{v:.2f}ms"

    W = 10
    counts = list(by_count.keys())
    header = f"{'agents':>12}" + "".join(f"{n:>{W}}" for n in counts)

    print(f"  {label} (incremental: 1 agent every {result.spawn_interval} ticks, {result.total_ticks} total)")
    print(header)
    print(f"{'mean':>12}" + "".join(f"{ms(r.mean_ms):>{W}}" for r in by_count.values()))
    print(f"{'p95':>12}" + "".join(f"{ms(r.p95_ms):>{W}}" for r in by_count.values()))
    print(f"{'p99':>12}" + "".join(f"{ms(r.p99_ms):>{W}}" for r in by_count.values()))
    print(f"{'rtf':>12}" + "".join(f"{r.rtf:>{W}.1f}" for r in by_count.values()))
    print(f"{'samples':>12}" + "".join(f"{len(r.tick_times_ms):>{W}}" for r in by_count.values()))
    print()


def merge_rounds(round_results: Sequence[RunResult]) -> RunResult:
    all_times = []
    for r in round_results:
        all_times.extend(r.tick_times_ms)
    return RunResult(
        n_agents=round_results[0].n_agents,
        dt=round_results[0].dt,
        tick_times_ms=all_times,
    )


def print_results(
    cand_label: str,
    cand_results: Sequence[RunResult],
    ref_label: str,
    ref_results: Sequence[RunResult],
):
    def ms(v):
        return f"{v:.2f}ms"

    W = 10
    agents = [r.n_agents for r in cand_results]
    header = f"{'':>12}" + "".join(f"{n:>{W}}" for n in agents)

    for label, results in [(cand_label, cand_results), (ref_label, ref_results)]:
        print()
        print(f"  {label}")
        print(header)
        print(f"{'mean':>12}" + "".join(f"{ms(r.mean_ms):>{W}}" for r in results))
        print(f"{'p95':>12}" + "".join(f"{ms(r.p95_ms):>{W}}" for r in results))
        print(f"{'p99':>12}" + "".join(f"{ms(r.p99_ms):>{W}}" for r in results))
        print(f"{'rtf':>12}" + "".join(f"{r.rtf:>{W}.1f}" for r in results))

    print()
    print("  comparison (candidate vs reference)")
    print(header)
    ratios = []
    speedups = []
    for rc, rr in zip(cand_results, ref_results):
        ratios.append(rc.mean_ms / rr.mean_ms if rr.mean_ms > 0 else float("inf"))
        speedups.append((rr.mean_ms - rc.mean_ms) / rr.mean_ms * 100 if rr.mean_ms > 0 else 0)
    print(f"{'ratio':>12}" + "".join(f"{r:>{W}.3f}" for r in ratios))
    print(f"{'speedup':>12}" + "".join(f"{f'+{s:.1f}%':>{W}}" if s >= 0 else f"{f'{s:.1f}%':>{W}}" for s in speedups))
    print()


def parse_params(raw: str | None) -> dict:
    if not raw:
        return {}
    params = {}
    for pair in raw.split(","):
        k, v = pair.split("=", 1)
        params[k.strip()] = v.strip()
    return params


def load_benchmark_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_config_path(path: str | None) -> str | None:
    if path is None:
        return None
    if path.startswith("pkg://"):
        rest = path[len("pkg://") :]
        pkg, _, rel = rest.partition("/")
        return get_package_share_directory(pkg) + "/" + rel
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Relative benchmark: compare two arena_humansim configurations",
    )
    parser.add_argument("config", nargs="?", default=None, metavar="BENCHMARK_YAML", help="Benchmark config YAML (default: config/benchmark/default.yaml)")
    parser.add_argument("--ref", default=None, metavar="FILE", help="Reference params YAML file")
    parser.add_argument("--candidate", default=None, metavar="FILE", help="Candidate params YAML file")
    parser.add_argument("--ref-params", default=None, help="Extra reference params as key=val,key=val")
    parser.add_argument("--candidate-params", default=None, help="Extra candidate params as key=val,key=val")
    parser.add_argument("--ref-label", default=None)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--agents", default=None, help="Comma-separated agent counts")
    parser.add_argument("--dt", type=float, default=None, help="Simulation timestep in seconds (default: 0.05)")
    parser.add_argument("--ticks", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None, help="Rounds per (config, agent_count) pair (default: 3)")
    parser.add_argument("--incremental", action="store_true", help="Spawn 1 agent every --spawn-interval ticks for --ticks total")
    parser.add_argument("--spawn-interval", type=int, default=None, help="Ticks between spawns in incremental mode (default: 5)")
    parser.add_argument("--profile", action="store_true", help="Enable per-phase tick profiling inside the sim node")
    parser.add_argument("--profile-interval", type=int, default=100, help="Ticks between profile log dumps (default: 100)")
    args = parser.parse_args()

    cfg = {}
    config_path = args.config
    if config_path is None and args.ref is None and args.candidate is None:
        pkg_share = get_package_share_directory("arena_humansim")
        config_path = pkg_share + "/config/benchmark/default.yaml"
    if config_path is not None:
        cfg = load_benchmark_config(config_path)

    ref_label = args.ref_label or cfg.get("reference", {}).get("label", "reference")
    cand_label = args.candidate_label or cfg.get("candidate", {}).get("label", "candidate")
    ref_file = resolve_config_path(args.ref or cfg.get("reference", {}).get("params_file"))
    cand_file = resolve_config_path(args.candidate or cfg.get("candidate", {}).get("params_file"))
    ref_extra = parse_params(args.ref_params) or cfg.get("reference", {}).get("params", {})
    cand_extra = parse_params(args.candidate_params) or cfg.get("candidate", {}).get("params", {})
    agent_counts = [int(x) for x in args.agents.split(",")] if args.agents else cfg.get("agents", [10, 25, 50, 100, 200])
    dt = args.dt or cfg.get("dt", 0.05)
    n_ticks = args.ticks or cfg.get("ticks", 200)
    warmup = args.warmup or cfg.get("warmup", 50)
    rounds = args.rounds or cfg.get("rounds", 3)
    spawn_interval = args.spawn_interval or cfg.get("spawn_interval", 5)
    cfg_seed = cfg.get("seed")
    if args.seed is not None:
        seed = args.seed
    elif cfg_seed is not None:
        seed = cfg_seed
    else:
        seed = int.from_bytes(os.urandom(4), "little")

    print(f"Benchmark: {cand_label} vs {ref_label}")
    print(f"  agents: {agent_counts}  dt: {dt}  ticks: {n_ticks}  warmup: {warmup}")
    print(f"  rounds: {rounds}  seed: {seed}")
    print()

    rclpy.init()
    driver = BenchmarkDriver()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(driver)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if args.incremental:
            max_agents = n_ticks // spawn_interval
            print(f"Incremental mode: 1 agent every {spawn_interval} ticks, {n_ticks} total ticks, up to {max_agents} agents")
            print()
            for label, params_file, extra in [
                (cand_label, cand_file, cand_extra),
                (ref_label, ref_file, ref_extra),
            ]:
                print(f"{label}: running incremental...", flush=True)
                result = run_incremental(
                    driver, params_file, extra, dt, n_ticks, spawn_interval, warmup, seed, executor,
                    profile=args.profile, profile_interval=args.profile_interval,
                )
                print_incremental_results(label, result)
            executor.shutdown()
            driver.destroy_node()
            rclpy.shutdown()
            return

        cand_results = []
        ref_results = []
        for n_agents in agent_counts:
            cand_rounds = []
            ref_rounds = []
            for label, params_file, extra, round_list in [
                (cand_label, cand_file, cand_extra, cand_rounds),
                (ref_label, ref_file, ref_extra, ref_rounds),
            ]:
                print(f"{label}: {n_agents} agents", flush=True)
                for r in range(rounds):
                    print(f"  [{r + 1}/{rounds}] running...", end="", flush=True)
                    result = run_single(
                        driver,
                        params_file,
                        extra,
                        n_agents,
                        dt,
                        n_ticks,
                        warmup,
                        seed,
                        executor,
                        profile=args.profile,
                        profile_interval=args.profile_interval,
                    )
                    round_list.append(result)
                    print(
                        f"\r  [{r + 1}/{rounds}] mean={result.mean_ms:.2f}ms  p95={result.p95_ms:.2f}ms  rtf={result.rtf:.2f}",
                        flush=True,
                    )
            cand_results.append(merge_rounds(cand_rounds))
            ref_results.append(merge_rounds(ref_rounds))
        print_results(cand_label, cand_results, ref_label, ref_results)
    finally:
        executor.shutdown()
        driver.destroy_node()
        rclpy.shutdown()
