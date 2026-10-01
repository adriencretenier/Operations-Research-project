# Vehicle Routing for Urban Food Delivery — Operations Research Project

**Yasmine Boubou, Adrien Creténier & Ilyas Sabri**
Operations Research course project, École des Ponts ParisTech — **January 2026**

The problem comes from the **KIRO 2025** inter-school operations research challenge, built on a real need of **Califrais**, a food distribution company operating in the Paris region.

---

## The problem

Califrais must deliver hundreds of orders across the Paris region from a single depot, using vehicles rented from a heterogeneous fleet. The aim is to build a set of delivery routes that minimises the total daily cost.

Each order has a weight, a delivery time window and a service duration. Each vehicle family has its own capacity, speed, parking time and cost structure.

The problem is a **Vehicle Routing Problem with Time Windows (VRPTW)**, which is NP-hard. Three features make it harder than the textbook version:

- **Time-dependent travel times.** Travel times vary over the day according to a truncated Fourier series that models traffic, such as rush hour on the périphérique.
- **Heterogeneous fleet.** Vehicle families differ in capacity, speed, rental cost and fuel cost.
- **Non-additive cost.** Each route pays a penalty on its squared Euclidean diameter, which favours geographically compact routes. This cost cannot be broken down arc by arc.

The objective is the sum of rental, fuel (based on Manhattan distance) and radius costs, over 10 instances of increasing size.

## Contents

| File | Description |
|---|---|
| `subject.pdf` | Project subject (KIRO 2025 / École des Ponts course) |
| `report.pdf` | Our 4-page report: mathematical modelling and algorithm |
| `solver.py` | Heuristic solver (~1,500 lines of Python) |
| `LICENSE` | MIT License (applies to the code) |

## Part 1 — Mathematical modelling

The report answers four modelling questions on a simplified version of the problem: homogeneous fleet, constant travel times, fuel cost only.

1. **Three-index MILP formulation** with variables $x_{ij}^k$, including flow conservation, capacity, and time windows. Subtours are eliminated with MTZ-type constraints.
2. **Two-index MILP formulation** with variables $x_{ij}$, using rank-based and load-based Miller–Tucker–Zemlin constraints.
3. **Dynamic programming for the single-route TSPTW.** The state is (set of visited customers, last customer), and the recurrence computes the earliest feasible arrival time.
4. **Lower bound on the number of vehicles.** It is the maximum of two bounds:
   - a capacity bound, $\lceil \sum_i w_i / Q \rceil$;
   - a time bound, given by the size of a clique in an *incompatibility graph*. Two orders are linked when no vehicle can serve them consecutively in either order.

## Part 2 — Solution algorithm

Exact solvers cannot handle instances of several hundred orders, so we use an **Adaptive Large Neighborhood Search (ALNS)** metaheuristic combined with exact polishing.

- **Preprocessing.** Coordinates are projected to metric coordinates. Manhattan and Euclidean distance matrices are precomputed. The Fourier traffic profile $\gamma_f(t)$ is tabulated, then linearly interpolated.
- **Feasibility check.** Each route is simulated forward with time-dependent travel times, waiting allowed, and checks against the time windows.
- **Initial solution.** Routes are seeded with the orders farthest from the depot, then filled by **regret-k insertion**. Hard-to-place orders are inserted first.
- **ALNS loop:**
  - Four destroy operators: Random, Worst, Related and Radius. Radius targets peripheral customers to reduce the diameter cost.
  - Repair by regret-k reinsertion.
  - Local search with relocate, swap and 2-opt* moves.
  - Simulated-annealing acceptance.
  - Operator weights adapted according to past performance.
- **Route elimination.** The rental cost dominates, so the solver regularly tries to empty short routes and reinsert their customers elsewhere.
- **Set-partitioning polishing.** High-quality routes found during the search are stored in a pool. Periodically, **CP-SAT (Google OR-Tools)** selects the minimum-cost subset of routes that covers each order exactly once. This recombines good routes coming from different solutions.

## Results

| Instance | Time limit (s) | Total cost (€) |
|---|---|---|
| 1 | 120 | 452.49 |
| 2 | 150 | 775.76 |
| 3 | 500 | 1,330.30 |
| 4 | 700 | 2,034.89 |
| 5 | 1,000 | 2,623.61 |
| 6 | 2,000 | 3,252.84 |
| 7 | 3,000 | 4,443.54 |
| 8 | 4,000 | 6,032.85 |
| 9 | 5,000 | 7,420.59 |
| 10 | 10,000 | 8,921.80 |
| **Total** | | **37,288.67** |

Shorter runs already give competitive costs. For example, instance 10 reaches €9,708 in 1,500 s. The full parameter settings are given in the report.

## Usage

```bash
python solver.py --vehicles vehicles.csv --instance instance_XX.csv --out routes_XX.csv \
    --time_limit T --seed S --starts K --time_step dt --polish_every E --polish_time Tp
```

| Argument | Meaning |
|---|---|
| `--time_limit` | Maximum running time (s) |
| `--seed` | Random seed |
| `--starts` | Number of multi-start initial solutions |
| `--time_step` | Discretisation step of the traffic profile $\gamma_f(t)$ (s) |
| `--polish_every` | ALNS iterations between two CP-SAT polishing steps |
| `--polish_time` | Time limit for each polishing step (s) |

The input and output formats (`vehicles.csv`, `instance_XX.csv`, `routes_XX.csv`) are described in `subject.pdf`.

**Requirements:** Python 3, NumPy, Google OR-Tools.

## License

The code is released under the MIT License. The project subject (`subject.pdf`) was written by the KIRO 2025 organisers and the course teaching team. It is included for context only and is not covered by this license.

## Acknowledgements

We thank Califrais for providing the problem and data, the KIRO 2025 organisers, and the École des Ponts teaching team for the course and the challenge design.
