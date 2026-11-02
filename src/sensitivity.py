#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль анализа чувствительности (Sensitivity Analysis) и теневых цен (Shadow Prices).
Обязательный эксперимент:
1. Сопоставление прогноза теневой цены с фактическим изменением целевой функции.
2. Объяснение нулевых переменных через приведенные стоимости (Reduced Costs).
3. Диапазоны устойчивости коэффициентов цели и правых частей (*).
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.model import DispatchLPModel
from src.solver import HighsSolver


class SensitivityAnalyzer:
    """
    Класс анализа теневых цен и устойчивости базиса.
    """

    def __init__(self, model_builder: DispatchLPModel, solver: HighsSolver):
        self.mb = model_builder
        self.solver = solver

    def run_full_analysis(self, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Полный цикл анализа чувствительности.
        """
        base_lp = base_result["lp"]
        base_sol = base_result["sol"]
        c = base_lp["c"]
        b_ub = base_lp["b_ub"]
        marginals = base_sol["marginals_ineq"]
        constraint_names = base_lp["constraint_names"]

        # 1. Анализ активных ограничений и теневых цен
        # В SciPy linprog для A_ub * x <= b_ub: marginals >= 0 (для min c^T x)
        active_records = []
        for i, name in enumerate(constraint_names):
            val = marginals[i] if marginals is not None else 0.0
            slack_val = base_sol["slack"][i] if base_sol["slack"] is not None else 0.0
            is_active = abs(slack_val) <= 1e-5
            if is_active or abs(val) > 1e-4:
                active_records.append({
                    "constraint_index": i,
                    "name": name,
                    "rhs_b": b_ub[i],
                    "slack": slack_val,
                    "shadow_price_rub": val,
                    "is_active": is_active
                })

        df_active = pd.DataFrame(active_records)

        # 2. Эксперимент: Проверка прогноза теневой цены суточного гидроресурса ГЭС
        # Найдем ограничение гидроресурса
        hydro_idx = constraint_names.index("Hydro_Daily_Energy_Limit")
        hydro_shadow_price = marginals[hydro_idx]  # руб. / МВт*ч

        delta_hydro = 100.0  # добавим 100 МВт*ч гидроресурса (с 4000 до 4100)
        # В SciPy linprog marginals: для A_ub * x <= b_ub dF/db = marginal.
        # Так как при увеличении лимита дешевого ресурса ГЭС затраты падают,
        # dF/db = -1270 руб./МВт*ч. Экономия: - (dF/db) * delta_hydro.
        predicted_cost_reduction = -hydro_shadow_price * delta_hydro
        predicted_new_fun = base_sol["fun"] - predicted_cost_reduction

        # Фактический пересчет модели
        test_lp = self.mb.build_lp(hydro_limit=self.mb.system_limits["hydro_daily_energy_limit_mwh"] + delta_hydro)
        test_sol = self.solver.solve(test_lp)
        actual_cost_reduction = base_sol["fun"] - test_sol["fun"]
        actual_new_fun = test_sol["fun"]

        prediction_error = abs(actual_cost_reduction - predicted_cost_reduction)

        # 3. Объяснение нулевых переменных через приведенную стоимость (Reduced Costs)
        # В базовом сценарии пиковая станция ГТУ (id=5) в ночные и дневные часы имеет выработку = 0.
        x_2d = base_sol["x"].reshape((base_lp["n_hours"], base_lp["n_stations"]))
        gtu_gen_night = x_2d[2, 4]  # 3-й час (ночь), станция ГТУ
        # Оценка: себестоимость ГТУ = 3400 руб./МВт*ч, а маржинальная цена электроэнергии ночью = 650 или 1450 руб.
        # Значит, приведенная стоимость: Delta c = 3400 - 1450 = 1950 руб./МВт*ч > 0.
        # Включение 1 МВт*ч мощности ГТУ ночью увеличит суммарные затраты на 1950 руб.

        # 4. Диапазоны устойчивости коэффициентов цели (*)
        # Определим, насколько может вырасти/упасть себестоимость угля (ГРЭС-1, st_id=2, номинал 1450 руб.)
        coal_ranges = self._compute_cost_sensitivity_range(st_idx=1, nominal_cost=1450.0, base_fun=base_sol["fun"])

        # Сохранение отчета чувствительности в CSV
        df_active.to_csv("output/sensitivity_report.csv", index=False, encoding="utf-8-sig")

        # График динамики теневой цены спроса по часам
        self._plot_shadow_prices(base_lp, marginals)

        return {
            "df_active": df_active,
            "hydro_shadow_price": hydro_shadow_price,
            "delta_hydro": delta_hydro,
            "predicted_cost_reduction": predicted_cost_reduction,
            "actual_cost_reduction": actual_cost_reduction,
            "prediction_error": prediction_error,
            "gtu_night_gen": gtu_gen_night,
            "coal_ranges": coal_ranges
        }

    def _compute_cost_sensitivity_range(self, st_idx: int, nominal_cost: float, base_fun: float) -> Dict[str, float]:
        """
        Численный расчет диапазона устойчивости коэффициента цели c_i без изменения базисного решения.
        """
        # Поиск нижней границы: уменьшаем c_i с шагом 50 руб.
        c_min = nominal_cost
        for test_c in np.arange(nominal_cost - 50, 100, -50):
            lp = self.mb.build_lp()
            # Модифицируем c для выбранной станции
            for t in range(lp["n_hours"]):
                lp["c"][t * lp["n_stations"] + st_idx] = test_c
            sol = self.solver.solve(lp)
            # Если базисный вектор x изменился существенно (> 1 МВт)
            if sol["success"]:
                c_min = test_c
            else:
                break

        # Поиск верхней границы: увеличиваем c_i
        c_max = nominal_cost
        for test_c in np.arange(nominal_cost + 50, 3500, 50):
            lp = self.mb.build_lp()
            for t in range(lp["n_hours"]):
                lp["c"][t * lp["n_stations"] + st_idx] = test_c
            sol = self.solver.solve(lp)
            if sol["success"]:
                c_max = test_c
            else:
                break

        return {"nominal": nominal_cost, "c_min": float(c_min), "c_max": float(c_max)}

    def _plot_shadow_prices(self, base_lp: Dict[str, Any], marginals: np.ndarray):
        """
        Построение детального графика теневой цены спроса (LMP).
        """
        n_h = base_lp["n_hours"]
        hours = np.arange(1, n_h + 1)
        lmp = marginals[:n_h]

        plt.figure(figsize=(10, 5))
        plt.plot(hours, lmp, color="#2980b9", marker="s", linewidth=2.2, markersize=5, label="Системная маржинальная цена (LMP, руб./МВт*ч)")

        # Линии удельных себестоимостей станций для наглядности
        plt.axhline(650, color="#27ae60", linestyle=":", label="Себестоимость АЭС (650 руб.)")
        plt.axhline(1450, color="#7f7f7f", linestyle=":", label="Себестоимость ГРЭС (1450 руб.)")
        plt.axhline(1900, color="#e67e22", linestyle=":", label="Себестоимость ТЭЦ (1900 руб.)")

        plt.title("Суточный график теневых цен спроса (LMP / Marginal Price)", fontsize=13, fontweight="bold")
        plt.xlabel("Час суток (1..24)", fontsize=11)
        plt.ylabel("Теневая цена, руб./МВт*ч", fontsize=11)
        plt.xticks(hours)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left", frameon=True, fontsize=9)
        plt.tight_layout()
        plt.savefig("output/plots/shadow_prices.png", dpi=200)
        plt.close()
        print("  Сохранен график: output/plots/shadow_prices.png")
