# Vehicle Routing for Urban Food Delivery — Operations Research Project

**Yasmine Boubou, Adrien Creténier & Ilyas Sabri**
Operations Research course project, École des Ponts ParisTech — **January 2026**

The problem comes from the **KIRO 2025** inter-school operations research challenge. It is based on a real need of **Califrais**, a food distribution company operating in the Paris region.

---

## The problem

Califrais must deliver hundreds of orders across the Paris region from a single depot, using vehicles rented from a heterogeneous fleet. The goal is to build a set of delivery routes that minimises the total daily cost.

Each order has a weight, a delivery time window and a service duration. Each vehicle family has its own capacity, speed, parking time and cost structure.

The problem is a **Vehicle Routing Problem with Time Windows (VRPTW)**, which is NP-hard. Three features make it harder than the textbook version:

- **Time-dependent travel times.** Travel times vary over the day according to a truncated Fourier series modelling traffic (e.g. rush hour on the périphérique).
- **Heterogeneous fleet.** Vehicle families differ in capacity, speed, rental cost and fuel cost.
- **Non-additive cost.** Each route pays a penalty on its squared Euclidean diameter, which favours geographically compact routes. This cost cannot be broken down arc by arc.

The objective is the sum of rental, fuel (based on Manhattan distance) and radius costs, over 10 instances of increasing size.

## Repository contents

| Path | Description |
|---|---|
| `Operations_Research_Project.pdf` | Project subject (KIRO 2025 / École des Ponts course) |
| `Operations_Research_Project_Report.pdf` | Our 4-page report: mathematical modelling and algorithm |
| `solver_califrais_hybrid.py` | Heuristic solver (~1,500 lines of Python) |
| `instances/` | Input data: `vehicles.csv` (vehicle families) and the 10 instance files |
| `solutions/` | Our solutions: `solution_01.csv` to `solution_10.csv` |
| `LICENSE` | MIT License (applies to the code) |

## Part 1 — Mathematical modelling

The report answers four modelling questions on a simplified version of the problem: homogeneous fleet, constant travel times, fuel cost only.

1. **Three-index MILP formulation** with variables $x_{ij}^k$. It includes flow conservation, capacity and time windows. Subtours are eliminated with MTZ-type constraints.
2. **Two-index MILP formulation** with variables $x_{ij}$, using rank-based and load-based Miller–Tucker–Zemlin constraints.
3. **Dynamic programming for the single-route TSPTW.** The state is (set of visited customers, last customer). The recurrence computes the earliest feasible arrival time.
4. **Lower bound on the number of vehicles.** The bound is the maximum of two quantities:
   - a capacity bound, $\lceil \sum_i w_i / Q \rceil$;
   - a time bound: the size of a clique in an *incompatibility graph*, where two orders are linked when no vehicle can serve them consecutively in either order.

## Part 2 — Solution algorithm

Exact solvers cannot handle instances with several hundred orders. We therefore use an **Adaptive Large Neighborhood Search (ALNS)** metaheuristic, combined with an exact polishing step.

- **Preprocessing.** Coordinates are projected to metric coordinates. Manhattan and Euclidean distance matrices are precomputed. The Fourier traffic profile $\gamma_f(t)$ is tabulated, then linearly interpolated.
- **Feasibility check.** Each route is simulated forward with time-dependent travel times. Waiting is allowed, and every arrival is checked against its time window.
- **Initial solution.** Routes are seeded with the orders farthest from the depot, then filled by **regret-k insertion**, so that hard-to-place orders are inserted first.
- **ALNS loop.**
  - Four destroy operators: Random, Worst, Related, and Radius (which targets peripheral customers to reduce the diameter cost).
  - Repair by regret-k reinsertion.
  - Local search with relocate, swap and 2-opt* moves.
  - Simulated-annealing acceptance.
  - Operator weights adapted according to past performance.
- **Route elimination.** Rental cost dominates, so the solver regularly tries to empty short routes and reinsert their customers elsewhere.
- **Set-partitioning polishing.** High-quality routes found during the search are stored in a pool. Periodically, **CP-SAT (Google OR-Tools)** selects the minimum-cost subset of routes covering each order exactly once. This recombines good routes coming from different solutions.

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

Shorter runs already give competitive costs: for example, instance 10 reaches €9,708 in 1,500 s. The full parameter settings are given in the report. The corresponding solution files are in `solutions/`.

## Usage

Example for instance 1, with the parameters used in the report:

```bash
python solver_califrais_hybrid.py \
    --vehicles instances/vehicles.csv \
    --instance instances/instance_001.csv \
    --out solutions/solution_01.csv \
    --time_limit 120 --seed 123 --starts 4 --time_step 60 \
    --polish_every 40 --polish_time 10
```

| Argument | Meaning |
|---|---|
| `--time_limit` | Maximum running time (s) |
| `--seed` | Random seed |
| `--starts` | Number of multi-start initial solutions |
| `--time_step` | Discretisation step of the traffic profile $\gamma_f(t)$ (s) |
| `--polish_every` | ALNS iterations between two CP-SAT polishing steps |
| `--polish_time` | Time limit for each polishing step (s) |

The input and output formats are described in `Operations_Research_Project.pdf`.

### Requirements

- Python 3.7+
- NumPy
- Google OR-Tools (`pip install ortools`). This dependency is optional: without it, the solver runs the ALNS alone and skips the set-partitioning polishing step, which usually gives higher costs.

## Use of AI tools

As allowed by the course rules, large language models were used to help generate the code. The modelling work, the choice and design of the algorithm, the parameter tuning and the analysis presented in the report are our own.

## License

The code is released under the MIT License. The project subject (`Operations_Research_Project.pdf`) and the instance data were written and provided by the KIRO 2025 organisers, the course teaching team and Califrais. They are included for context and reproducibility only, and are not covered by this license.

## Acknowledgements

We thank Califrais for providing the problem and data, the KIRO 2025 organisers, and the École des Ponts teaching team for the course and the challenge design.
