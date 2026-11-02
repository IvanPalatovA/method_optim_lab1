#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Задание со звёздочкой (*):
Сравнительный бенчмарк: симплекс-метод (highs-ds) против метода внутренней точки (highs-ipm)
на 10 сгенерированных задачах растущей размерности (от 6 ч до 336 ч / 2 недели).
Построение графиков времени и числа итераций.
"""

import os
import sys
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import DispatchLPModel
from src.solver import HighsSolver


class BenchmarkRunner:
    """
    Класс проведения сравнительного бенчмарка Simplex vs IPM.
    """

    def __init__(self, model_builder: DispatchLPModel):
        self.mb = model_builder

    def run_benchmark(self, horizons: List[int] = None) -> pd.DataFrame:
        """
        Запуск бенчмарка на 10 размерностях.
        """
        if horizons is None:
            # 10 горизонтов планирования: 6 ч, 12 ч, 24 ч, 48 ч (2 сут), 72 ч (3 сут),
            # 96 ч (4 сут), 120 ч (5 сут), 168 ч (1 нед), 240 ч (10 сут), 336 ч (2 нед)
            horizons = [6, 12, 24, 48, 72, 96, 120, 168, 240, 336]

        print("\n" + "=" * 70)
        print("ЗАПУСК БЕНЧМАРКА: SIMPLEX (highs-ds) VS INTERIOR POINT (highs-ipm) (*)")
        print(f"Горизонты планирования ({len(horizons)} задач): {horizons}")
        print("=" * 70)

        results = []

        # Базовый суточный профиль спроса (24 ч) для периодического повторения
        base_demand_24 = self.mb.base_demand
        rng = np.random.default_rng(10116)

        for h in horizons:
            # Генерируем спрос длины h
            repeats = int(np.ceil(h / 24))
            extended_demand = np.tile(base_demand_24, repeats)[:h]
            # Добавляем небольшой шум для реалистичности каждого дня
            extended_demand = extended_demand + rng.normal(0, 15.0, size=h)
            extended_demand = np.maximum( extended_demand, 800.0)

            # Пропорционально масштабируем суточные лимиты под горизонт h
            h_ratio = h / 24.0
            hydro_lim = self.mb.system_limits["hydro_daily_energy_limit_mwh"] * h_ratio
            coal_lim = self.mb.system_limits["coal_daily_fuel_limit_mwh"] * h_ratio
            co2_lim = self.mb.system_limits["max_daily_co2_emission_tons"] * h_ratio

            lp_model = self.mb.build_lp(
                custom_demand=extended_demand,
                hydro_limit=hydro_lim,
                coal_limit=coal_lim,
                co2_limit=co2_lim
            )

            n_vars = lp_model["n_vars"]
            n_constraints = lp_model["n_constraints"]

            # 1. Запуск Simplex (highs-ds)
            solver_simplex = HighsSolver(method="highs-ds", presolve=True)
            res_simplex = solver_simplex.solve(lp_model)

            # 2. Запуск Interior Point (highs-ipm)
            solver_ipm = HighsSolver(method="highs-ipm", presolve=True)
            res_ipm = solver_ipm.solve(lp_model)

            t_simplex_ms = res_simplex["solve_time_sec"] * 1000.0
            t_ipm_ms = res_ipm["solve_time_sec"] * 1000.0
            nit_simplex = res_simplex["nit"]
            nit_ipm = res_ipm["nit"]

            fun_diff = abs(res_simplex["fun"] - res_ipm["fun"]) if (res_simplex["success"] and res_ipm["success"]) else np.nan

            print(f"Горизонт T={h:3d} ч | Переменных={n_vars:4d} | Ограничений={n_constraints:4d} | "
                  f"Simplex: {t_simplex_ms:6.2f} мс ({nit_simplex:3d} ит.) | "
                  f"IPM: {t_ipm_ms:6.2f} мс ({nit_ipm:2d} ит.) | Разница цели: {fun_diff:.2e}")

            results.append({
                "horizon_hours": h,
                "n_vars": n_vars,
                "n_constraints": n_constraints,
                "simplex_time_ms": t_simplex_ms,
                "ipm_time_ms": t_ipm_ms,
                "simplex_iterations": nit_simplex,
                "ipm_iterations": nit_ipm,
                "simplex_fun": res_simplex["fun"],
                "ipm_fun": res_ipm["fun"],
                "fun_abs_diff": fun_diff,
                "status_simplex": res_simplex["status"],
                "status_ipm": res_ipm["status"]
            })

        df_bench = pd.DataFrame(results)
        df_bench.to_csv("output/benchmark_results.csv", index=False, encoding="utf-8-sig")

        # Построение графиков
        self._plot_benchmark(df_bench)

        return df_bench

    def _plot_benchmark(self, df: pd.DataFrame):
        """
        Построение графиков времени решения и числа итераций.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        x_vars = df["n_vars"]

        # 1. График времени выполнения
        ax1.plot(x_vars, df["simplex_time_ms"], marker="o", linewidth=2.2, color="#e67e22", label="HiGHS Dual Simplex")
        ax1.plot(x_vars, df["ipm_time_ms"], marker="s", linewidth=2.2, color="#2980b9", label="HiGHS Interior Point (IPM)")
        ax1.set_xlabel("Число переменных задачи (N = 5 станций × T часов)", fontsize=11)
        ax1.set_ylabel("Время решения, миллисекунды (мс)", fontsize=11)
        ax1.set_title("Сравнение времени решения: Simplex vs IPM", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(fontsize=10)

        # 2. График числа итераций
        ax2.plot(x_vars, df["simplex_iterations"], marker="o", linewidth=2.2, color="#e67e22", label="Итерации Simplex")
        ax2.plot(x_vars, df["ipm_iterations"], marker="s", linewidth=2.2, color="#2980b9", label="Итерации IPM (внутренняя точка)")
        ax2.set_xlabel("Число переменных задачи (N = 5 станций × T часов)", fontsize=11)
        ax2.set_ylabel("Количество итераций", fontsize=11)
        ax2.set_title("Сравнение числа итераций: Simplex vs IPM", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig("output/plots/benchmark_simplex_ipm.png", dpi=200)
        plt.close()
        print("  Сохранен график: output/plots/benchmark_simplex_ipm.png")


if __name__ == "__main__":
    mb = DispatchLPModel()
    bench = BenchmarkRunner(mb)
    bench.run_benchmark()
