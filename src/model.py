#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль построения ЛП-модели оптимизации суточного диспетчерского графика.
Задача: min c^T x при Ax <= b, p_min <= x <= p_max.
"""

import json
from typing import Dict, Any, Tuple, List, Optional
import numpy as np


class DispatchLPModel:
    """
    Класс для построения линейной модели распределения генерации (Economic Dispatch).
    Переменные решения: p_{i, t} - мощность станции i в час t (МВт).
    Индекс переменной: j = t * n_stations + i.
    """

    def __init__(self, data_path: str = "data/input_data.json"):
        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.stations = self.raw_data["stations"]
        self.n_stations = len(self.stations)
        self.base_demand = np.array(self.raw_data["demand_mwh"], dtype=np.float64)
        self.n_hours = len(self.base_demand)
        self.system_limits = self.raw_data["metadata"]["system_limits"]

        self.station_names = [s["name"] for s in self.stations]
        self.costs_per_mwh = np.array([s["marginal_cost"] for s in self.stations], dtype=np.float64)
        self.p_min_nominal = np.array([s["p_min"] for s in self.stations], dtype=np.float64)
        self.p_max_nominal = np.array([s["p_max"] for s in self.stations], dtype=np.float64)
        self.ramp_up = np.array([s["ramp_up"] for s in self.stations], dtype=np.float64)
        self.ramp_down = np.array([s["ramp_down"] for s in self.stations], dtype=np.float64)
        self.co2_rates = np.array([s["co2_rate"] for s in self.stations], dtype=np.float64)

    def build_lp(
        self,
        demand_modifier: float = 1.0,
        capacity_modifiers: Optional[Dict[int, float]] = None,
        hydro_limit: Optional[float] = None,
        coal_limit: Optional[float] = None,
        co2_limit: Optional[float] = None,
        custom_demand: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Формирует векторы и матрицы для солвера HiGHS:
        min c^T x
        Ax <= b
        lb <= x <= ub
        """
        if custom_demand is not None:
            demand = custom_demand.copy()
            n_hours = len(demand)
        else:
            demand = self.base_demand * demand_modifier
            n_hours = self.n_hours

        n_vars = self.n_stations * n_hours

        # 1. Вектор целевой функции c (руб. / МВт*ч)
        c = np.tile(self.costs_per_mwh, n_hours)

        # 2. Границы переменных (lb, ub)
        p_max = self.p_max_nominal.copy()
        p_min = self.p_min_nominal.copy()
        if capacity_modifiers:
            for st_id, mod in capacity_modifiers.items():
                p_max[st_id - 1] *= mod
                p_min[st_id - 1] = min(p_min[st_id - 1], p_max[st_id - 1])

        bounds = []
        var_names = []
        for t in range(n_hours):
            for i in range(self.n_stations):
                bounds.append((float(p_min[i]), float(p_max[i])))
                var_names.append(f"p_{self.stations[i]['type']}_h{t+1}")

        # 3. Ограничения неравенств A_ub * x <= b_ub
        A_ub_list = []
        b_ub_list = []
        constraint_names = []

        # 3.1. Баланс спроса в каждый час: sum_i p_{i,t} >= D_t  <=>  -sum_i p_{i,t} <= -D_t
        for t in range(n_hours):
            row = np.zeros(n_vars, dtype=np.float64)
            for i in range(self.n_stations):
                row[t * self.n_stations + i] = -1.0
            A_ub_list.append(row)
            b_ub_list.append(-float(demand[t]))
            constraint_names.append(f"Demand_Balance_Hour_{t+1}")

        # 3.2. Ограничения на скорость набора мощности (Ramp Up): p_{i,t} - p_{i,t-1} <= R_i^up
        for t in range(1, n_hours):
            for i in range(self.n_stations):
                row = np.zeros(n_vars, dtype=np.float64)
                row[t * self.n_stations + i] = 1.0
                row[(t - 1) * self.n_stations + i] = -1.0
                A_ub_list.append(row)
                b_ub_list.append(float(self.ramp_up[i]))
                constraint_names.append(f"Ramp_Up_{self.stations[i]['type']}_h{t}->h{t+1}")

        # 3.3. Ограничения на скорость сброса мощности (Ramp Down): p_{i,t-1} - p_{i,t} <= R_i^down
        for t in range(1, n_hours):
            for i in range(self.n_stations):
                row = np.zeros(n_vars, dtype=np.float64)
                row[(t - 1) * self.n_stations + i] = 1.0
                row[t * self.n_stations + i] = -1.0
                A_ub_list.append(row)
                b_ub_list.append(float(self.ramp_down[i]))
                constraint_names.append(f"Ramp_Down_{self.stations[i]['type']}_h{t}->h{t+1}")

        # 3.4. Суточный лимит гидроресурса ГЭС (id=4, индекс 3): sum_t p_{4,t} <= HydroLimit
        h_lim = hydro_limit if hydro_limit is not None else self.system_limits["hydro_daily_energy_limit_mwh"]
        row = np.zeros(n_vars, dtype=np.float64)
        for t in range(n_hours):
            row[t * self.n_stations + 3] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(float(h_lim))
        constraint_names.append("Hydro_Daily_Energy_Limit")

        # 3.5. Суточный лимит расхода топлива угольной станции ГРЭС (id=2, индекс 1)
        coal_lim = coal_limit if coal_limit is not None else self.system_limits["coal_daily_fuel_limit_mwh"]
        row = np.zeros(n_vars, dtype=np.float64)
        for t in range(n_hours):
            row[t * self.n_stations + 1] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(float(coal_lim))
        constraint_names.append("Coal_Daily_Fuel_Limit")

        # 3.6. Суточный экологический лимит выбросов CO2: sum_{t, i} e_i * p_{i,t} <= CO2_Limit
        emiss_lim = co2_limit if co2_limit is not None else self.system_limits["max_daily_co2_emission_tons"]
        row = np.zeros(n_vars, dtype=np.float64)
        for t in range(n_hours):
            for i in range(self.n_stations):
                row[t * self.n_stations + i] = float(self.co2_rates[i])
        A_ub_list.append(row)
        b_ub_list.append(float(emiss_lim))
        constraint_names.append("CO2_Daily_Emission_Limit")

        A_ub = np.array(A_ub_list, dtype=np.float64)
        b_ub = np.array(b_ub_list, dtype=np.float64)

        return {
            "c": c,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "bounds": bounds,
            "var_names": var_names,
            "constraint_names": constraint_names,
            "n_vars": n_vars,
            "n_constraints": len(b_ub),
            "n_hours": n_hours,
            "n_stations": self.n_stations,
            "demand": demand,
            "stations": self.stations
        }

