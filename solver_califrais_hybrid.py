#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Califrais VRPTW (JuliaEvaluator-aligned) — Hybrid Solver (best of Code1 + Code2 + recommendations)

Primary objective:
- MINIMIZE total cost across instances, while ALWAYS passing juliaEvaluator feasibility + cost.

Hard alignment to juliaEvaluator/src/eval.jl + instance.jl:
- Travel time:
    δ = manhattan_distances[i+1, j+1]
    τ = δ / speed
    γ(t) = Σ_{n=0..3} alpha[n]*cos(n*ω*t) + beta[n]*sin(n*ω*t)
    ω = 2π / (24*60)      (from constants.jl)
    travel_time = τ * γ(t)
  NOTE: juliaEvaluator does NOT use parking_time in travel time.
- Feasibility:
    Simulate only from depot through orders (NO return-to-depot feasibility check)
    waiting allowed; arrival time is service start; violation if time > window_end + 1e-5
    capacity checked per route
    each order visited exactly once
- Costs:
    rental: sum rental per route
    fuel: includes return to depot; uses Manhattan distances
    radius: diameter among orders in the route (no depot) * radius_cost / 2

Algorithmic hybrid:
- Numpy-packed instance (fast distances, arrays)
- Multi-start construction (several modes)
- ALNS with:
    * adaptive destroy weights (scored operators)
    * strict partition integrity maintained during the search (no “lost orders”)
    * greedy regret-k repair
    * strong intensification: relocate/swap/2-opt* + route elimination
- Route pool + Set Partitioning polishing (OR-Tools CP-SAT) when available

Usage examples:
  python solve_califrais_hybrid.py --vehicles vehicles.csv --instance instance_01.csv --out solution_01.csv --time_limit 60
  python solve_califrais_hybrid.py --vehicles vehicles.csv --instance_dir ./instances --out_dir ./solutions --time_limit 240

Notes:
- This solver assumes (as juliaEvaluator does) that order IDs in each instance are 1..N (contiguous).
  If not, juliaEvaluator itself would mis-index instance.orders[id].
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Iterable

import numpy as np

# Optional polishing (set partitioning)
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except Exception:
    ORTOOLS_AVAILABLE = False

# -----------------------------
# Julia evaluator constants
# -----------------------------
EARTH_R = 6.371e6
JULIA_TOL = 1e-5

# Matches juliaEvaluator/src/constants.jl:
# const T = 24 * 60
# const ω = 2 * π / T
OMEGA = 2.0 * math.pi / (24.0 * 60.0)


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class VehicleFamily:
    family: int
    max_capacity: float
    rental_cost: float
    fuel_cost: float
    radius_cost: float
    speed: float
    parking_time: float  # present in CSV but NOT USED in Julia travel time
    alpha: Tuple[float, float, float, float]  # fourier_cos_0..3
    beta: Tuple[float, float, float, float]   # fourier_sin_0..3


@dataclass(frozen=True)
class PackedInstance:
    n_orders: int              # number of customers (excluding depot)
    depot_id: int              # always 0
    # For Julia alignment: order IDs must be 1..n_orders, depot is 0.
    x: np.ndarray              # shape (n_orders+1,)
    y: np.ndarray              # shape (n_orders+1,)
    weight: np.ndarray         # shape (n_orders+1,)
    tw_start: np.ndarray       # shape (n_orders+1,)
    tw_end: np.ndarray         # shape (n_orders+1,)
    service: np.ndarray        # shape (n_orders+1,)
    dM: np.ndarray             # shape (n_orders+1, n_orders+1)
    dE: np.ndarray             # shape (n_orders+1, n_orders+1)


@dataclass
class Route:
    family: int                # Julia "vehicle_index" == family id
    orders: List[int]          # customer IDs (1..N), depot excluded
    load: float = 0.0
    feasible: bool = True
    cost: float = 0.0
    rental: float = 0.0
    fuel: float = 0.0
    radius: float = 0.0
    dist_manh: float = 0.0
    half_diam: float = 0.0


@dataclass
class Solution:
    routes: List[Route]
    unassigned: Set[int] = field(default_factory=set)
    cost: float = 0.0


# -----------------------------
# CSV parsing (Julia-aligned indexing)
# -----------------------------
def read_vehicles(path: str) -> Dict[int, VehicleFamily]:
    fams: Dict[int, VehicleFamily] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = [
            "family", "max_capacity", "rental_cost", "fuel_cost", "radius_cost",
            "speed", "parking_time",
            "fourier_cos_0", "fourier_cos_1", "fourier_cos_2", "fourier_cos_3",
            "fourier_sin_0", "fourier_sin_1", "fourier_sin_2", "fourier_sin_3",
        ]
        for col in required:
            if col not in r.fieldnames:
                raise ValueError(f"vehicles.csv missing column: {col}")

        for row in r:
            fam = int(row["family"])
            alpha = (
                float(row["fourier_cos_0"]),
                float(row["fourier_cos_1"]),
                float(row["fourier_cos_2"]),
                float(row["fourier_cos_3"]),
            )
            beta = (
                float(row["fourier_sin_0"]),
                float(row["fourier_sin_1"]),
                float(row["fourier_sin_2"]),
                float(row["fourier_sin_3"]),
            )
            fams[fam] = VehicleFamily(
                family=fam,
                max_capacity=float(row["max_capacity"]),
                rental_cost=float(row["rental_cost"]),
                fuel_cost=float(row["fuel_cost"]),
                radius_cost=float(row["radius_cost"]),
                speed=float(row["speed"]),
                parking_time=float(row["parking_time"]),
                alpha=alpha,
                beta=beta,
            )
    if not fams:
        raise ValueError("vehicles.csv: no vehicle families read.")
    return fams


def read_instance_to_packed(path: str) -> PackedInstance:
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    depot_rows = [r for r in rows if int(r["id"]) == 0]
    if len(depot_rows) != 1:
        raise ValueError("instance.csv must contain exactly one depot row with id=0.")
    depot = depot_rows[0]
    lat0 = float(depot["latitude"])
    lon0 = float(depot["longitude"])
    cos_lat0 = math.cos(math.radians(lat0))
    factor = 2.0 * math.pi / 360.0

    def to_xy(lat: float, lon: float) -> Tuple[float, float]:
        # Matches juliaEvaluator/src/instance.jl manhattan_distance/euclidean_distance
        dy = EARTH_R * factor * (lat - lat0)
        dx = EARTH_R * cos_lat0 * factor * (lon - lon0)
        return dx, dy

    # Read orders
    order_rows = [r for r in rows if int(r["id"]) != 0]
    order_ids = [int(r["id"]) for r in order_rows]
    n_orders = len(order_ids)

    # Julia evaluator indexes instance.orders by id: instance.orders[id]
    # This implies IDs are 1..n_orders contiguous.
    expected = set(range(1, n_orders + 1))
    got = set(order_ids)
    if got != expected:
        # In practice, evaluator would break. Raise loudly.
        missing = sorted(list(expected - got))[:20]
        extra = sorted(list(got - expected))[:20]
        raise ValueError(
            "Order IDs must be exactly 1..N to match juliaEvaluator indexing. "
            f"Missing (preview): {missing}; Extra (preview): {extra}"
        )

    # Allocate arrays indexed by id (0..n_orders)
    x = np.zeros(n_orders + 1, dtype=np.float64)
    y = np.zeros(n_orders + 1, dtype=np.float64)
    weight = np.zeros(n_orders + 1, dtype=np.float64)
    tw_start = np.zeros(n_orders + 1, dtype=np.float64)
    tw_end = np.zeros(n_orders + 1, dtype=np.float64)
    service = np.zeros(n_orders + 1, dtype=np.float64)

    # Depot
    xd, yd = to_xy(lat0, lon0)
    x[0], y[0] = xd, yd  # should be (0,0)
    weight[0] = 0.0
    tw_start[0] = 0.0
    tw_end[0] = 0.0
    service[0] = float(depot["delivery_duration"]) if depot.get("delivery_duration") else 0.0

    # Orders
    for r in order_rows:
        oid = int(r["id"])
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        xi, yi = to_xy(lat, lon)
        x[oid] = xi
        y[oid] = yi
        weight[oid] = float(r["order_weight"])
        tw_start[oid] = float(r["window_start"])
        tw_end[oid] = float(r["window_end"])
        service[oid] = float(r["delivery_duration"])

    # Distances (as Julia builds them)
    dx = x.reshape(-1, 1) - x.reshape(1, -1)
    dy = y.reshape(-1, 1) - y.reshape(1, -1)
    dM = np.abs(dx) + np.abs(dy)
    dE = np.hypot(dx, dy)
    np.fill_diagonal(dM, 0.0)
    np.fill_diagonal(dE, 0.0)

    return PackedInstance(
        n_orders=n_orders,
        depot_id=0,
        x=x, y=y,
        weight=weight,
        tw_start=tw_start,
        tw_end=tw_end,
        service=service,
        dM=dM, dE=dE,
    )


# -----------------------------
# Julia-aligned travel time + feasibility + costs
# -----------------------------
def gamma_julia(vf: VehicleFamily, t: float) -> float:
    # Matches eval.jl exactly:
    # γ = sum( cos(n*ω*t)*alpha[n] + sin(n*ω*t)*beta[n] for n=0..3 )
    a0, a1, a2, a3 = vf.alpha
    b0, b1, b2, b3 = vf.beta
    w = OMEGA
    # n=0: cos(0)=1, sin(0)=0 => contributes a0
    # We still compute in the same form for clarity.
    val = (
        a0 * math.cos(0.0) + b0 * math.sin(0.0) +
        a1 * math.cos(1.0 * w * t) + b1 * math.sin(1.0 * w * t) +
        a2 * math.cos(2.0 * w * t) + b2 * math.sin(2.0 * w * t) +
        a3 * math.cos(3.0 * w * t) + b3 * math.sin(3.0 * w * t)
    )
    return val


def travel_time(i: int, j: int, t: float, vf: VehicleFamily, inst: PackedInstance) -> float:
    # compute_travel_time in eval.jl:
    # δ = instance.manhattan_distances[i+1, j+1]
    # τ = δ / v.speed
    # return τ * γ
    base = float(inst.dM[i, j]) / vf.speed
    return base * gamma_julia(vf, t)


def simulate_feasible(seq: List[int], fam: int, inst: PackedInstance, families: Dict[int, VehicleFamily]) -> Tuple[bool, float]:
    # eval.jl is_feasible: simulate only depot -> orders (no return leg check)
    vf = families[fam]
    t = 0.0
    cur = inst.depot_id
    for oid in seq:
        t += travel_time(cur, oid, t, vf, inst)
        if t < inst.tw_start[oid]:
            t = float(inst.tw_start[oid])
        if t > float(inst.tw_end[oid]) + JULIA_TOL:
            return False, t
        t += float(inst.service[oid])
        cur = oid
    return True, t


def fuel_distance(seq: List[int], inst: PackedInstance) -> float:
    # eval.jl fuel_cost includes return to depot
    if not seq:
        return 0.0
    dist = float(inst.dM[inst.depot_id, seq[0]])
    for a, b in zip(seq, seq[1:]):
        dist += float(inst.dM[a, b])
    dist += float(inst.dM[seq[-1], inst.depot_id])
    return dist


def half_diameter_exact(seq: List[int], inst: PackedInstance) -> float:
    # eval.jl radius_cost: diameter among orders; total += diameter * radius_cost / 2
    m = len(seq)
    if m <= 1:
        return 0.0
    diam = 0.0
    dE = inst.dE
    for i in range(m):
        ai = seq[i]
        row = dE[ai]
        for j in range(i + 1, m):
            d = float(row[seq[j]])
            if d > diam:
                diam = d
    return 0.5 * diam


def evaluate_route(seq: List[int], fam: int, inst: PackedInstance, families: Dict[int, VehicleFamily]) -> Optional[Route]:
    # seq: list of order IDs (no depot)
    if not seq:
        return None
    vf = families[fam]

    load = float(inst.weight[seq].sum())
    if load > vf.max_capacity + 1e-12:
        return None

    feas, _ = simulate_feasible(seq, fam, inst, families)
    if not feas:
        return None

    dist = fuel_distance(seq, inst)
    hd = half_diameter_exact(seq, inst)
    cost_rental = vf.rental_cost
    cost_fuel = vf.fuel_cost * dist
    cost_radius = vf.radius_cost * hd
    total = cost_rental + cost_fuel + cost_radius

    return Route(
        family=fam,
        orders=seq[:],
        load=load,
        feasible=True,
        cost=total,
        rental=cost_rental,
        fuel=cost_fuel,
        radius=cost_radius,
        dist_manh=dist,
        half_diam=hd,
    )


def choose_best_family(seq: List[int], inst: PackedInstance, families: Dict[int, VehicleFamily]) -> Optional[Route]:
    # Evaluate all families that can carry the load; pick min cost feasible route
    if not seq:
        return None
    load = float(inst.weight[seq].sum())
    best: Optional[Route] = None
    for fam, vf in families.items():
        if load > vf.max_capacity + 1e-12:
            continue
        r = evaluate_route(seq, fam, inst, families)
        if r is None:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


# -----------------------------
# Solution integrity (partition) — ALWAYS enforced
# -----------------------------
def normalize_solution(sol: Solution) -> None:
    sol.routes = [r for r in sol.routes if r.orders]
    sol.cost = sum(r.cost for r in sol.routes)


def check_partition(sol: Solution, n_orders: int) -> Tuple[bool, List[int], List[int], List[int]]:
    """
    Returns:
      ok, missing, duplicated, invalid_ids
    """
    cnt = [0] * (n_orders + 1)
    invalid: List[int] = []
    for r in sol.routes:
        seen = set()
        for v in r.orders:
            if v < 1 or v > n_orders:
                invalid.append(v)
                continue
            if v in seen:
                # internal dup
                cnt[v] += 2
            else:
                cnt[v] += 1
                seen.add(v)

    missing = [i for i in range(1, n_orders + 1) if cnt[i] == 0]
    duplicated = [i for i in range(1, n_orders + 1) if cnt[i] > 1]
    ok = (not invalid) and (not missing) and (not duplicated) and (len(sol.unassigned) == 0)
    return ok, missing, duplicated, invalid


def enforce_partition_and_repair(sol: Solution,
                                inst: PackedInstance,
                                families: Dict[int, VehicleFamily],
                                rng: random.Random,
                                regret_k: int = 3,
                                max_passes: int = 3) -> bool:
    """
    Ensure that:
      - every order 1..N appears exactly once across routes
      - sol.unassigned is empty at end
    If duplicates exist: keep the first occurrence, unassign the rest.
    If missing: add to unassigned.
    Then repair by regret insertion.
    """
    N = inst.n_orders

    # Step 1: remove internal duplicates per route
    for r in sol.routes:
        seen = set()
        new_seq = []
        for v in r.orders:
            if v in seen:
                sol.unassigned.add(v)
            else:
                seen.add(v)
                new_seq.append(v)
        r.orders = new_seq

    # Step 2: remove cross-route duplicates
    owner: Dict[int, int] = {}
    for ridx, r in enumerate(sol.routes):
        new_seq = []
        for v in r.orders:
            if v not in owner:
                owner[v] = ridx
                new_seq.append(v)
            else:
                sol.unassigned.add(v)
        r.orders = new_seq

    # Step 3: add missing
    for v in range(1, N + 1):
        if v not in owner:
            sol.unassigned.add(v)

    # Step 4: re-evaluate routes (family might no longer be best after deletions)
    new_routes: List[Route] = []
    for r in sol.routes:
        if not r.orders:
            continue
        best = choose_best_family(r.orders, inst, families)
        if best is None:
            # if route becomes infeasible, dump its customers to unassigned
            for v in r.orders:
                sol.unassigned.add(v)
            continue
        new_routes.append(best)
    sol.routes = new_routes
    normalize_solution(sol)

    # Step 5: repair
    for _ in range(max_passes):
        if not sol.unassigned:
            ok, missing, duplicated, invalid = check_partition(sol, N)
            return ok
        ok = regret_repair(sol, inst, families, rng, regret_k=regret_k)
        if not ok:
            break

    ok, missing, duplicated, invalid = check_partition(sol, N)
    return ok


# -----------------------------
# Construction / Repair (regret insertion)
# -----------------------------
def candidate_positions(m: int, rng: random.Random, max_positions: int) -> List[int]:
    """
    For route length m, positions are 0..m (insert before index pos).
    If too many, sample a subset but always include ends.
    """
    positions = list(range(m + 1))
    if m + 1 <= max_positions:
        return positions
    rng.shuffle(positions)
    # Keep some random positions + ends
    keep = positions[:max_positions - 2] + [0, m]
    keep = sorted(set(keep))
    return keep


def regret_repair(sol: Solution,
                  inst: PackedInstance,
                  families: Dict[int, VehicleFamily],
                  rng: random.Random,
                  regret_k: int = 3,
                  max_positions_per_route: int = 18) -> bool:
    """
    Insert all unassigned using regret-k.
    Returns True if it managed to empty unassigned, else False.
    """
    while sol.unassigned:
        best_pick = None  # (regret, best_cost, oid, ridx_or_none, best_route)

        # randomize candidate order for diversification
        cand_list = list(sol.unassigned)
        rng.shuffle(cand_list)

        for oid in cand_list:
            options: List[Tuple[float, Optional[int], Route]] = []

            # Try insert into existing routes
            for ridx, r in enumerate(sol.routes):
                cur_seq = r.orders
                # fast load pruning by family is not safe (family can change), so don't over-prune
                pos_list = candidate_positions(len(cur_seq), rng, max_positions=max_positions_per_route)
                best_local: Optional[Route] = None
                best_local_cost = float("inf")

                for pos in pos_list:
                    cand_seq = cur_seq[:pos] + [oid] + cur_seq[pos:]
                    cand_route = choose_best_family(cand_seq, inst, families)
                    if cand_route is None:
                        continue
                    if cand_route.cost < best_local_cost:
                        best_local = cand_route
                        best_local_cost = cand_route.cost

                if best_local is not None:
                    options.append((best_local.cost, ridx, best_local))

            # New singleton route
            singleton = choose_best_family([oid], inst, families)
            if singleton is not None:
                options.append((singleton.cost, None, singleton))

            if not options:
                continue

            options.sort(key=lambda x: x[0])
            best0 = options[0]
            kth = options[min(regret_k - 1, len(options) - 1)]
            regret = kth[0] - best0[0]

            # Choose by max regret, tie-break by lowest best cost
            if best_pick is None or (regret > best_pick[0]) or (regret == best_pick[0] and best0[0] < best_pick[1]):
                best_pick = (regret, best0[0], oid, best0[1], best0[2])

        if best_pick is None:
            # As a last resort, try to add a singleton for an arbitrary unassigned.
            oid = sol.unassigned.pop()
            singleton = choose_best_family([oid], inst, families)
            if singleton is None:
                return False
            sol.routes.append(singleton)
            normalize_solution(sol)
            continue

        _, _, oid, ridx, best_route = best_pick
        sol.unassigned.remove(oid)
        if ridx is None:
            sol.routes.append(best_route)
        else:
            sol.routes[ridx] = best_route
        normalize_solution(sol)

    return True


def detect_outliers(orders: List[int], inst: PackedInstance, q: float = 0.97) -> Set[int]:
    dists = [(i, float(inst.dE[inst.depot_id, i])) for i in orders]
    dists.sort(key=lambda x: x[1])
    if not dists:
        return set()
    idx = int(q * (len(dists) - 1))
    thr = dists[idx][1]
    return {i for i, d in dists if d >= thr}


def order_difficulty(oid: int, inst: PackedInstance) -> float:
    slack = max(1.0, float(inst.tw_end[oid] - inst.tw_start[oid]))
    return (1.0 / slack) + (1.0 / max(1.0, float(inst.tw_end[oid]))) + 1e-6 * float(inst.service[oid])


def build_initial_solution(inst: PackedInstance,
                           families: Dict[int, VehicleFamily],
                           seed: int,
                           mode: str = "difficulty_first") -> Solution:
    rng = random.Random(seed)
    all_orders = list(range(1, inst.n_orders + 1))
    remaining: Set[int] = set(all_orders)
    routes: List[Route] = []

    # Seed outliers as singletons (helps radius / feasibility)
    outliers = detect_outliers(all_orders, inst, q=0.97)
    for oid in sorted(outliers):
        if oid not in remaining:
            continue
        rr = choose_best_family([oid], inst, families)
        if rr is not None:
            routes.append(rr)
            remaining.remove(oid)

    if mode == "difficulty_first":
        seq = sorted(list(remaining), key=lambda i: (-order_difficulty(i, inst), -float(inst.weight[i])))
    elif mode == "time_tight_first":
        seq = sorted(list(remaining), key=lambda i: (float(inst.tw_end[i]), float(inst.tw_start[i]), -float(inst.weight[i])))
    elif mode == "sweep":
        x0, y0 = float(inst.x[0]), float(inst.y[0])
        seq = sorted(list(remaining), key=lambda i: math.atan2(float(inst.y[i]) - y0, float(inst.x[i]) - x0))
    else:
        seq = list(remaining)
        rng.shuffle(seq)

    # Greedy cheapest insertion (with limited positions)
    for oid in seq:
        if oid not in remaining:
            continue

        best_move = None  # (delta, ridx, new_route)
        for ridx, r in enumerate(routes):
            cur_seq = r.orders
            pos_list = candidate_positions(len(cur_seq), rng, max_positions=16 if len(cur_seq) > 30 else 64)
            for pos in pos_list:
                cand_seq = cur_seq[:pos] + [oid] + cur_seq[pos:]
                nr = choose_best_family(cand_seq, inst, families)
                if nr is None:
                    continue
                delta = nr.cost - r.cost
                if best_move is None or delta < best_move[0]:
                    best_move = (delta, ridx, nr)

        if best_move is not None:
            _, ridx, nr = best_move
            routes[ridx] = nr
            remaining.remove(oid)
            continue

        # Open new route
        rr = choose_best_family([oid], inst, families)
        if rr is not None:
            routes.append(rr)
            remaining.remove(oid)

    sol = Solution(routes=routes, unassigned=set(remaining))
    normalize_solution(sol)

    if sol.unassigned:
        regret_repair(sol, inst, families, rng, regret_k=3)

    # Final enforce
    enforce_partition_and_repair(sol, inst, families, rng, regret_k=3)
    normalize_solution(sol)
    return sol


# -----------------------------
# Local search operators (Julia-aligned evaluation)
# -----------------------------
def relocate_move(sol: Solution,
                  inst: PackedInstance,
                  families: Dict[int, VehicleFamily],
                  rng: random.Random,
                  tries: int = 250,
                  max_positions: int = 14) -> bool:
    if len(sol.routes) < 1:
        return False

    for _ in range(tries):
        ra = rng.randrange(len(sol.routes))
        rb = rng.randrange(len(sol.routes))
        if ra == rb:
            continue

        A = sol.routes[ra]
        B = sol.routes[rb]
        if len(A.orders) <= 1:
            continue

        idx_a = rng.randrange(len(A.orders))
        v = A.orders[idx_a]

        if v in B.orders:
            continue

        newA_seq = A.orders[:idx_a] + A.orders[idx_a + 1:]
        if not newA_seq:
            # would eliminate route A; still allowed
            newA = None
        else:
            newA = choose_best_family(newA_seq, inst, families)
            if newA is None:
                continue

        bestB: Optional[Route] = None
        best_pair_cost = float("inf")

        pos_list = candidate_positions(len(B.orders), rng, max_positions=max_positions)
        for pos in pos_list:
            candB_seq = B.orders[:pos] + [v] + B.orders[pos:]
            candB = choose_best_family(candB_seq, inst, families)
            if candB is None:
                continue

            costA = 0.0 if newA is None else newA.cost
            pair_cost = costA + candB.cost
            if pair_cost < best_pair_cost:
                best_pair_cost = pair_cost
                bestB = candB

        if bestB is None:
            continue

        old = A.cost + B.cost
        if best_pair_cost + 1e-9 < old:
            # Apply
            if newA is None:
                # remove route A; careful with indices
                keep = [sol.routes[i] for i in range(len(sol.routes)) if i != ra and i != rb]
                # rb shifts if ra < rb
                keep.append(bestB)
                sol.routes = keep
            else:
                sol.routes[ra] = newA
                sol.routes[rb] = bestB
                sol.routes = [r for r in sol.routes if r.orders]
            normalize_solution(sol)
            return True

    return False


def swap_move(sol: Solution,
              inst: PackedInstance,
              families: Dict[int, VehicleFamily],
              rng: random.Random,
              tries: int = 220,
              max_positions: int = 12) -> bool:
    if len(sol.routes) < 2:
        return False

    for _ in range(tries):
        ra, rb = rng.sample(range(len(sol.routes)), 2)
        A = sol.routes[ra]
        B = sol.routes[rb]
        if len(A.orders) == 0 or len(B.orders) == 0:
            continue

        ia = rng.randrange(len(A.orders))
        ib = rng.randrange(len(B.orders))
        va = A.orders[ia]
        vb = B.orders[ib]
        if va == vb:
            continue
        if vb in A.orders and A.orders[ia] != vb:
            continue
        if va in B.orders and B.orders[ib] != va:
            continue

        newA_seq = A.orders[:]
        newB_seq = B.orders[:]
        newA_seq[ia] = vb
        newB_seq[ib] = va

        newA = choose_best_family(newA_seq, inst, families)
        newB = choose_best_family(newB_seq, inst, families)
        if newA is None or newB is None:
            continue

        old = A.cost + B.cost
        new = newA.cost + newB.cost
        if new + 1e-9 < old:
            sol.routes[ra] = newA
            sol.routes[rb] = newB
            normalize_solution(sol)
            return True

    return False


def two_opt_star(sol: Solution,
                 inst: PackedInstance,
                 families: Dict[int, VehicleFamily],
                 rng: random.Random,
                 tries: int = 160) -> bool:
    """
    2-opt* between two routes: exchange tails.
    Often reduces rental count / improves structure.
    """
    if len(sol.routes) < 2:
        return False

    for _ in range(tries):
        ra, rb = rng.sample(range(len(sol.routes)), 2)
        A = sol.routes[ra]
        B = sol.routes[rb]
        if len(A.orders) <= 1 or len(B.orders) <= 1:
            continue

        # avoid overlap duplicates (shouldn't exist if partition maintained)
        setA = set(A.orders)
        setB = set(B.orders)
        if setA & setB:
            continue

        cutA = rng.randrange(1, len(A.orders))   # cut after at least 1
        cutB = rng.randrange(1, len(B.orders))

        newA_seq = A.orders[:cutA] + B.orders[cutB:]
        newB_seq = B.orders[:cutB] + A.orders[cutA:]

        if not newA_seq or not newB_seq:
            continue

        newA = choose_best_family(newA_seq, inst, families)
        newB = choose_best_family(newB_seq, inst, families)
        if newA is None or newB is None:
            continue

        old = A.cost + B.cost
        new = newA.cost + newB.cost
        if new + 1e-9 < old:
            sol.routes[ra] = newA
            sol.routes[rb] = newB
            normalize_solution(sol)
            return True

    return False


def intensify(sol: Solution,
              inst: PackedInstance,
              families: Dict[int, VehicleFamily],
              rng: random.Random,
              rounds: int = 180) -> None:
    """
    Local search loop: try elimination + relocate/swap/2opt* until no progress or budget consumed.
    """
    for it in range(rounds):
        if it % 35 == 0:
            if attempt_route_elimination(sol, inst, families, rng, max_attempts=8):
                continue
        if relocate_move(sol, inst, families, rng, tries=80):
            continue
        if swap_move(sol, inst, families, rng, tries=60):
            continue
        if two_opt_star(sol, inst, families, rng, tries=40):
            continue
        break


# -----------------------------
# Route elimination (rental savings)
# -----------------------------
def attempt_route_elimination(sol: Solution,
                             inst: PackedInstance,
                             families: Dict[int, VehicleFamily],
                             rng: random.Random,
                             max_attempts: int = 25) -> bool:
    if len(sol.routes) <= 1:
        return False

    # Prefer short routes first (easier to eliminate)
    idxs = list(range(len(sol.routes)))
    idxs.sort(key=lambda i: len(sol.routes[i].orders))

    base_cost = sol.cost
    attempts = 0
    for ridx in idxs:
        if attempts >= max_attempts:
            break
        attempts += 1

        victim = sol.routes[ridx]
        victims = victim.orders[:]
        if not victims:
            continue

        # Candidate without victim
        cand_routes = [r for i, r in enumerate(sol.routes) if i != ridx]
        cand = Solution(routes=[r for r in cand_routes], unassigned=set(victims), cost=0.0)
        normalize_solution(cand)

        # Repair
        ok = regret_repair(cand, inst, families, rng, regret_k=3)
        if not ok or cand.unassigned:
            continue

        intensify(cand, inst, families, rng, rounds=120)
        normalize_solution(cand)

        if cand.cost + 1e-9 < base_cost:
            sol.routes = cand.routes
            sol.unassigned = set()
            sol.cost = cand.cost
            return True

    return False


# -----------------------------
# ALNS destroy operators (partition-safe)
# -----------------------------
def all_assigned_orders(sol: Solution) -> List[int]:
    out = []
    for r in sol.routes:
        out.extend(r.orders)
    return out


def destroy_random(sol: Solution, rng: random.Random, q: int) -> List[int]:
    cust = list(set(all_assigned_orders(sol)))
    rng.shuffle(cust)
    return cust[:q]


def destroy_route_removal(sol: Solution, rng: random.Random) -> List[int]:
    if not sol.routes:
        return []
    candidates = sorted(sol.routes, key=lambda r: (len(r.orders), r.cost))
    victim = rng.choice(candidates[:min(8, len(candidates))])
    return victim.orders[:]


def destroy_deadline_focus(sol: Solution, inst: PackedInstance, rng: random.Random, q: int) -> List[int]:
    cust = list(set(all_assigned_orders(sol)))
    if not cust:
        return []
    scored = []
    for oid in cust:
        slack = max(1.0, float(inst.tw_end[oid] - inst.tw_start[oid]))
        score = (1.0 / slack) + (1.0 / max(1.0, float(inst.tw_end[oid])))
        scored.append((score, oid))
    scored.sort(reverse=True)
    removed = [oid for _, oid in scored[:q]]
    rng.shuffle(removed)
    return removed


def destroy_worst_manhattan_proxy(sol: Solution, inst: PackedInstance, rng: random.Random, q: int) -> List[int]:
    scored = []
    dM = inst.dM
    depot = inst.depot_id
    for r in sol.routes:
        seq = r.orders
        if len(seq) <= 1:
            continue
        full = [depot] + seq + [depot]
        for k in range(1, len(full) - 1):
            i = full[k]
            prev = full[k - 1]
            nxt = full[k + 1]
            contrib = float(dM[prev, i] + dM[i, nxt] - dM[prev, nxt])
            scored.append((contrib, i))
    if not scored:
        return destroy_random(sol, rng, q)
    scored.sort(reverse=True)
    removed = []
    seen = set()
    for _, oid in scored:
        if oid not in seen:
            seen.add(oid)
            removed.append(oid)
        if len(removed) >= q:
            break
    rng.shuffle(removed)
    return removed


def remove_orders(sol: Solution,
                 inst: PackedInstance,
                 families: Dict[int, VehicleFamily],
                 to_remove: List[int]) -> None:
    """
    Partition-safe removal:
    - remove those orders from routes
    - any route that becomes infeasible during rebuild sends its remaining orders to unassigned
    """
    rem = set(to_remove)
    new_routes: List[Route] = []
    for r in sol.routes:
        seq = [i for i in r.orders if i not in rem]
        if not seq:
            continue
        nr = choose_best_family(seq, inst, families)
        if nr is None:
            # dump remaining customers
            sol.unassigned |= set(seq)
            continue
        new_routes.append(nr)

    sol.routes = new_routes
    sol.unassigned |= rem
    normalize_solution(sol)


# -----------------------------
# Route pool + set partitioning polishing (OR-Tools)
# -----------------------------
class RoutePool:
    def __init__(self, max_size: int = 140000):
        self.max_size = max_size
        self._routes: Dict[Tuple[int, Tuple[int, ...]], Route] = {}

    def add_route(self, r: Route) -> None:
        if not r.orders:
            return
        key = (r.family, tuple(r.orders))
        ex = self._routes.get(key)
        if ex is None or r.cost < ex.cost:
            self._routes[key] = r
            if len(self._routes) > self.max_size:
                self._evict_some()

    def add_solution(self, sol: Solution) -> None:
        for r in sol.routes:
            self.add_route(r)

    def _evict_some(self) -> None:
        # Keep cheapest 90%
        items = sorted(self._routes.items(), key=lambda kv: kv[1].cost)
        keep = int(0.90 * self.max_size)
        self._routes = dict(items[:keep])

    def routes(self) -> List[Route]:
        return list(self._routes.values())


def polish_set_partitioning(pool_routes: List[Route],
                            n_orders: int,
                            time_limit_s: float = 25.0) -> Optional[Solution]:
    if not ORTOOLS_AVAILABLE:
        return None
    if not pool_routes:
        return None

    all_orders = list(range(1, n_orders + 1))
    order_index = {oid: k for k, oid in enumerate(all_orders)}
    m = len(pool_routes)

    # Build cover lists
    route_covers: List[List[int]] = []
    for r in pool_routes:
        cov = []
        ok = True
        seen = set()
        for oid in r.orders:
            if oid not in order_index or oid in seen:
                ok = False
                break
            seen.add(oid)
            cov.append(order_index[oid])
        route_covers.append(cov if ok and cov else [])

    routes_by_order = [[] for _ in range(n_orders)]
    for ri, cov in enumerate(route_covers):
        for idx in cov:
            routes_by_order[idx].append(ri)

    for idx in range(n_orders):
        if not routes_by_order[idx]:
            return None

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"r_{i}") for i in range(m)]
    for idx in range(n_orders):
        model.Add(sum(x[ri] for ri in routes_by_order[idx]) == 1)

    scale = 10**6
    model.Minimize(sum(int(round(pool_routes[i].cost * scale)) * x[i] for i in range(m)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = [pool_routes[i] for i in range(m) if solver.Value(x[i]) == 1]
    sol = Solution(routes=[r for r in chosen], unassigned=set(), cost=0.0)
    normalize_solution(sol)
    ok, _, _, _ = check_partition(sol, n_orders)
    if not ok:
        return None
    return sol


# -----------------------------
# Adaptive ALNS (scored destroy operators) + SA acceptance
# -----------------------------
def alns_solve(inst: PackedInstance,
              families: Dict[int, VehicleFamily],
              seed: int,
              time_limit_s: float,
              multistart: int = 8,
              polish_every: int = 45,
              polish_time_s: float = 12.0,
              pool_max: int = 140000) -> Solution:
    rng = random.Random(seed)
    pool = RoutePool(max_size=pool_max)

    # Multi-start initial solutions
    modes = ["difficulty_first", "time_tight_first", "sweep", "random"]
    best: Optional[Solution] = None

    for ms in range(multistart):
        s = seed + 1000 * ms + 17
        mode = modes[ms % len(modes)]
        sol0 = build_initial_solution(inst, families, seed=s, mode=mode)

        # quick intensification
        intensify(sol0, inst, families, rng=random.Random(s + 1), rounds=150)
        enforce_partition_and_repair(sol0, inst, families, rng=random.Random(s + 2), regret_k=3)
        normalize_solution(sol0)

        pool.add_solution(sol0)
        if best is None or sol0.cost < best.cost:
            best = sol0

    assert best is not None
    current = Solution(routes=[r for r in best.routes], unassigned=set(), cost=best.cost)

    # Destroy operators with adaptive weights
    destroy_ops = ["random", "worst", "deadline", "route_removal"]
    weights = {op: 1.0 for op in destroy_ops}
    scores = {op: 0.0 for op in destroy_ops}
    uses = {op: 0 for op in destroy_ops}

    # SA temperature: scale with cost (stable across instances)
    temperature = max(1.0, 0.02 * current.cost)
    cooling = 0.9992

    t0 = time.time()
    it = 0

    while (time.time() - t0) < time_limit_s:
        it += 1

        # roulette wheel selection
        total_w = sum(weights.values())
        pick = rng.random() * total_w
        acc = 0.0
        op = destroy_ops[0]
        for name in destroy_ops:
            acc += weights[name]
            if pick <= acc:
                op = name
                break

        # clone current
        cand = Solution(routes=[r for r in current.routes], unassigned=set(current.unassigned), cost=current.cost)

        # choose destruction size proportional to size (but bounded)
        n_assigned = sum(len(r.orders) for r in cand.routes)
        if n_assigned <= 0:
            # should never happen
            enforce_partition_and_repair(cand, inst, families, rng, regret_k=3)
            normalize_solution(cand)
            current = cand
            continue

        # q tuned for 17..500 orders
        q_min = 4
        q_max = max(10, int(0.06 * n_assigned))
        q = rng.randint(q_min, min(q_max, max(6, n_assigned // 2)))

        if op == "random":
            removed = destroy_random(cand, rng, q)
        elif op == "worst":
            removed = destroy_worst_manhattan_proxy(cand, inst, rng, q)
        elif op == "deadline":
            removed = destroy_deadline_focus(cand, inst, rng, q)
        else:
            removed = destroy_route_removal(cand, rng)

        if removed:
            remove_orders(cand, inst, families, removed)

        # Repair
        ok = regret_repair(cand, inst, families, rng, regret_k=3)
        if not ok:
            # hard fail -> treat as reject
            uses[op] += 1
            scores[op] += 0.0
            continue

        # Intensify
        intensify(cand, inst, families, rng, rounds=160)

        # Enforce partition safety (no missing/duplicate), then re-evaluate cost
        if not enforce_partition_and_repair(cand, inst, families, rng, regret_k=3):
            uses[op] += 1
            scores[op] += 0.0
            continue
        normalize_solution(cand)

        old_current_cost = current.cost
        delta = cand.cost - old_current_cost

        # SA acceptance
        accept = False
        if delta <= 0.0:
            accept = True
        else:
            prob = math.exp(-delta / max(1e-9, temperature))
            if rng.random() < prob:
                accept = True

        uses[op] += 1
        if accept:
            current = cand
            pool.add_solution(current)

            # Scoring scheme requested:
            # 0 if reject, 1 if accepted, 5 if improves current, 10 if improves best
            if best is None or current.cost + 1e-9 < best.cost:
                best = Solution(routes=[r for r in current.routes], unassigned=set(), cost=current.cost)
                pool.add_solution(best)
                scores[op] += 10.0
            elif current.cost + 1e-9 < old_current_cost:
                scores[op] += 5.0
            else:
                scores[op] += 1.0
        else:
            scores[op] += 0.0

        # cooling
        temperature *= cooling
        if temperature < 1e-6:
            temperature = 1e-6

        # Update weights periodically
        if it % 40 == 0:
            for name in destroy_ops:
                if uses[name] > 0:
                    avg = scores[name] / uses[name]
                    # EMA update; keep lower bound to preserve diversification
                    weights[name] = 0.6 * weights[name] + 0.4 * max(0.05, avg)
                    scores[name] = 0.0
                    uses[name] = 0

        # Optional SP polish occasionally
        if ORTOOLS_AVAILABLE and (it % polish_every == 0):
            pool_routes = pool.routes()
            pool_routes.sort(key=lambda r: r.cost)

            # Cap for practicality (instances up to 500 orders)
            # Keep balanced by length to preserve structural diversity
            capped: List[Route] = []
            seen_len: Dict[int, int] = {}
            for r in pool_routes:
                L = len(r.orders)
                if L == 0:
                    continue
                seen_len[L] = seen_len.get(L, 0) + 1
                if seen_len[L] <= 2500:
                    capped.append(r)
                if len(capped) >= min(90000, len(pool_routes)):
                    break

            polished = polish_set_partitioning(capped, inst.n_orders, time_limit_s=polish_time_s)
            if polished is not None and polished.cost + 1e-9 < best.cost:
                best = polished
                current = Solution(routes=[r for r in polished.routes], unassigned=set(), cost=polished.cost)

    assert best is not None

    # Final cleanup: re-pick best family per route (defensive)
    final_routes: List[Route] = []
    for r in best.routes:
        rr = choose_best_family(r.orders, inst, families)
        if rr is None:
            # should not happen; fallback to singleton rebuild later
            continue
        final_routes.append(rr)

    final = Solution(routes=final_routes, unassigned=set())
    normalize_solution(final)

    # Final enforce + “safety singletons”
    ok = enforce_partition_and_repair(final, inst, families, rng, regret_k=3)
    if not ok:
        # As absolute last resort, add singletons for missing.
        assigned = set()
        for r in final.routes:
            assigned.update(r.orders)
        missing = [i for i in range(1, inst.n_orders + 1) if i not in assigned]
        for oid in missing:
            rr = choose_best_family([oid], inst, families)
            if rr is not None:
                final.routes.append(rr)
        final.unassigned = set()
        normalize_solution(final)
        enforce_partition_and_repair(final, inst, families, rng, regret_k=3)
        normalize_solution(final)

    return final


# -----------------------------
# Output
# -----------------------------
def write_routes_csv(path: str, sol: Solution) -> None:
    max_len = max((len(r.orders) for r in sol.routes), default=0)
    header = ["family"] + [f"order_{k}" for k in range(1, max_len + 1)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in sol.routes:
            row = [r.family] + r.orders + [""] * (max_len - len(r.orders))
            w.writerow(row)


def list_instances(instance_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(instance_dir):
        if fn.lower().endswith(".csv") and "instance" in fn.lower():
            files.append(os.path.join(instance_dir, fn))
    files.sort()
    return files


# -----------------------------
# Budget allocation across instances
# -----------------------------
def per_instance_time_budget(total_budget_s: float, n_orders: int, n_orders_min: int = 16, n_orders_max: int = 500) -> float:
    """
    Allocate more time to larger instances while keeping small ones cheap.
    You can tune this. Uses smooth scaling ~ n^1.6.
    """
    n = max(n_orders_min, min(n_orders_max, n_orders))
    # weights
    w = (n ** 1.6)
    w_min = (n_orders_min ** 1.6)
    w_max = (n_orders_max ** 1.6)
    frac = (w - w_min) / max(1e-9, (w_max - w_min))
    # ensure at least 10% of equal-share, at most 2.5x equal-share
    return total_budget_s * (0.25 + 1.50 * frac)


# -----------------------------
# Main
# -----------------------------
def solve_one(vehicles_path: str,
              instance_path: str,
              out_path: str,
              seed: int,
              time_limit_s: float,
              multistart: int,
              polish_every: int,
              polish_time_s: float) -> Tuple[Solution, float]:
    families = read_vehicles(vehicles_path)
    inst = read_instance_to_packed(instance_path)

    t0 = time.time()
    sol = alns_solve(
        inst=inst,
        families=families,
        seed=seed,
        time_limit_s=time_limit_s,
        multistart=multistart,
        polish_every=polish_every,
        polish_time_s=polish_time_s,
    )
    elapsed = time.time() - t0

    # Hard validity check against partition + feasibility
    ok, missing, duplicated, invalid = check_partition(sol, inst.n_orders)
    if not ok:
        raise RuntimeError(f"Internal error: invalid solution partition. missing={missing[:10]} dup={duplicated[:10]} invalid={invalid[:10]} unassigned={len(sol.unassigned)}")

    # Validate route feasibility + capacity explicitly
    for r in sol.routes:
        if r.orders:
            rr = evaluate_route(r.orders, r.family, inst, families)
            if rr is None:
                raise RuntimeError("Internal error: route infeasible at final check.")
    write_routes_csv(out_path, sol)
    return sol, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", required=True)
    ap.add_argument("--instance", default=None)
    ap.add_argument("--instance_dir", default=None)
    ap.add_argument("--out", default="routes.csv")
    ap.add_argument("--out_dir", default="solutions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_limit", type=float, default=180.0, help="Time budget (seconds). If instance_dir is used, budget is split with size-weighting.")
    ap.add_argument("--multistart", type=int, default=8)
    ap.add_argument("--polish_every", type=int, default=45)
    ap.add_argument("--polish_time", type=float, default=12.0)
    args = ap.parse_args()

    if (args.instance is None) == (args.instance_dir is None):
        raise SystemExit("Provide exactly one of --instance or --instance_dir")

    if args.instance:
        sol, elapsed = solve_one(
            vehicles_path=args.vehicles,
            instance_path=args.instance,
            out_path=args.out,
            seed=args.seed,
            time_limit_s=args.time_limit,
            multistart=args.multistart,
            polish_every=args.polish_every,
            polish_time_s=args.polish_time,
        )
        print(f"Solved {args.instance}")
        print(f"Routes: {len(sol.routes)} | Cost: {sol.cost:.2f} € | Time: {elapsed:.1f} s | OR-Tools: {ORTOOLS_AVAILABLE}")
        return

    # Directory mode: solve all instances; allocate time by size
    os.makedirs(args.out_dir, exist_ok=True)
    inst_paths = list_instances(args.instance_dir)
    if not inst_paths:
        raise SystemExit(f"No instance CSV files found in {args.instance_dir}")

    # Pre-read to get sizes for budgets
    packed_list = []
    for p in inst_paths:
        inst = read_instance_to_packed(p)
        packed_list.append((p, inst.n_orders))

    # Compute per-instance budgets and normalize to total
    raw_budgets = [per_instance_time_budget(args.time_limit, n) for _, n in packed_list]
    s_raw = sum(raw_budgets)
    budgets = [args.time_limit * (b / s_raw) for b in raw_budgets]

    total_cost = 0.0
    total_time = 0.0

    for k, ((inst_path, n), budget) in enumerate(zip(packed_list, budgets)):
        out_path = os.path.join(args.out_dir, os.path.basename(inst_path).replace(".csv", "_routes.csv"))
        sol, elapsed = solve_one(
            vehicles_path=args.vehicles,
            instance_path=inst_path,
            out_path=out_path,
            seed=args.seed + 10000 * k,
            time_limit_s=budget,
            multistart=max(4, args.multistart),   # keep some diversity
            polish_every=args.polish_every,
            polish_time_s=args.polish_time,
        )
        total_cost += sol.cost
        total_time += elapsed
        print(f"{os.path.basename(inst_path)} (n={n}) -> routes={len(sol.routes)} cost={sol.cost:.2f}€ time={elapsed:.1f}s budget={budget:.1f}s")

    print(f"TOTAL over {len(inst_paths)} instances: {total_cost:.2f} € | Total time: {total_time:.1f} s | OR-Tools: {ORTOOLS_AVAILABLE}")


if __name__ == "__main__":
    main()
