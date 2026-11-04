"""Energy and delivery for one frozen schedule. Positions are not recomputed."""
from __future__ import annotations

from wt.ocean_observation_m0.flight import energy_limit_j, hover_power_w, usable_energy_j
from wt.ocean_observation_m0.parameters import BUOY_IDLE_W, COMPUTE_W, EVALUATION_HORIZON_S, GROUND_IDLE_W
from wt.ocean_observation_m0.radio import rx_power_w, tx_power_w

CONCLUSIONS = (
    "direct is enough",
    "fixed relay is enough",
    "motion has a conditional gain",
    "the current contract is still infeasible",
)


def _ground_s(uav: dict) -> float:
    cursor = 0.0
    ground = 0.0
    for window in uav.get("windows", []):
        launch = float(window["launch_s"])
        if launch > cursor:
            ground += max(0.0, min(launch, EVALUATION_HORIZON_S) - min(cursor, EVALUATION_HORIZON_S))
        cursor = max(cursor, float(window["landing_s"]))
    if cursor < EVALUATION_HORIZON_S:
        ground += EVALUATION_HORIZON_S - max(cursor, 0.0)
    return ground


def _uav_energy(uav: dict) -> dict:
    hover = hover_power_w()
    vertical = uav["vertical_s"]
    vertical_s = sum(float(value) for value in vertical.values())
    propulsion = sum(float(leg["propulsion_J"]) for leg in uav["legs"]) + hover * vertical_s
    compute = sum(float(leg["compute_J"]) for leg in uav["legs"]) + COMPUTE_W * vertical_s
    radio = rx_power_w() * float(uav["radio_s"]["rx"]) + tx_power_w() * float(uav["radio_s"]["tx"])
    ground = GROUND_IDLE_W * _ground_s(uav)
    flight = sum(float(leg["propulsion_J"]) + float(leg["compute_J"]) for leg in uav["legs"] if leg["role"] != "return")
    flight += (hover + COMPUTE_W) * float(vertical["takeoff"])
    returned = sum(float(leg["propulsion_J"]) + float(leg["compute_J"]) for leg in uav["legs"] if leg["role"] == "return")
    returned += (hover + COMPUTE_W) * float(vertical["landing"])
    total = propulsion + compute + radio + ground
    return {
        "flight_J": flight,
        "return_J": returned,
        "ground_J": ground,
        "radio_J": radio,
        "propulsion_J": propulsion,
        "compute_J": compute,
        "energy_J": total,
        "within_reserve": bool(uav["track_feasible"]) and total <= energy_limit_j(),
    }


def _delivery(products: list[dict]) -> dict:
    delivered = [row for row in products if row["delivered"]]
    count = len(products)
    delays = [float(row["delay_s"]) for row in delivered]
    times = [float(row["delivery_s"]) for row in delivered]
    return {
        "product_count": count,
        "delivered_count": len(delivered),
        "undelivered_count": count - len(delivered),
        "full_delivery_fraction": 0.0 if count == 0 else len(delivered) / count,
        "first_delivery_s": None if not times else min(times),
        "latest_delivery_s": None if not times else max(times),
        "mean_delay_s": None if not delays else sum(delays) / len(delays),
    }


def score_method(plan: dict) -> dict:
    delivery = _delivery(plan["products"])
    nodes = {}
    uav_energies = []
    for index, uav in enumerate(plan["uavs"]):
        energy = _uav_energy(uav)
        uav_energies.append(energy)
        nodes[f"uav_{index}"] = energy["energy_J"]
    buoy_idle = BUOY_IDLE_W * EVALUATION_HORIZON_S
    for index, tx_s in enumerate(plan["buoy_tx_s"]):
        nodes[f"buoy_{index}"] = buoy_idle + tx_power_w() * float(tx_s)
    nodes["shore"] = tx_power_w() * float(plan["shore_tx_s"]) + rx_power_w() * float(plan["shore_rx_s"])
    acceptable = (
        bool(plan["track_feasible"])
        and delivery["undelivered_count"] == 0
        and all(item["within_reserve"] for item in uav_energies)
    )
    return {
        "method": plan["method"],
        **delivery,
        "track_feasible": bool(plan["track_feasible"]),
        "energy_within_reserve": all(item["within_reserve"] for item in uav_energies),
        "acceptable": acceptable,
        "nodes_J": nodes,
        "uavs": uav_energies,
        "system_energy_J": sum(nodes.values()),
    }


def conclude(rows: list[dict]) -> str:
    """One label for the six registered contexts. Simpler methods stay preferred when they finish the same task."""
    by_context: dict[tuple, dict] = {}
    for row in rows:
        by_context.setdefault((row["geometry"], row["wind"]), {})[row["method"]] = row
    if all(methods["direct"]["acceptable"] for methods in by_context.values()):
        return CONCLUSIONS[0]
    if all(methods["fixed"]["acceptable"] for methods in by_context.values()):
        return CONCLUSIONS[1]
    motion_gain = False
    for methods in by_context.values():
        if methods["moving"]["acceptable"] and not methods["direct"]["acceptable"] and not methods["fixed"]["acceptable"]:
            motion_gain = True
        if not methods["direct"]["acceptable"] and not methods["fixed"]["acceptable"] and not methods["moving"]["acceptable"]:
            return CONCLUSIONS[3]
    if motion_gain:
        return CONCLUSIONS[2]
    return CONCLUSIONS[3]


def comparison_table(schedule: dict) -> dict:
    rows = []
    for context in schedule["contexts"]:
        for wind in context["winds"]:
            for method in ("direct", "fixed", "moving"):
                scored = score_method(wind["methods"][method])
                scored["geometry"] = context["geometry"]
                scored["wind"] = wind["wind"]
                rows.append(scored)
    if len(rows) != 18:
        raise RuntimeError(f"expected 18 method rows, found {len(rows)}")
    return {
        "usable_energy_J": usable_energy_j(),
        "energy_limit_J": energy_limit_j(),
        "hover_W": hover_power_w(),
        "rows": rows,
        "conclusion": conclude(rows),
    }
