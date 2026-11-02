#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль сценарного анализа:
1. Базовый сценарий (номинальный режим)
2. Дефицитный сценарий (ограничение ресурса / рост спроса)
3. Стрессовый сценарий (одновременное ухудшение -> infeasible)
4. Эластичный фильтр (Elastic Filter) для восстановления допустимости (*)
5. Построение графиков диспетчеризации и сравнения сценариев
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.model import DispatchLPModel
from src.solver import HighsSolver
from src.verification import SolutionVerifier


class ScenarioManager:
    """
    Управление сценарными расчётами и визуализацией.
    """

    def __init__(self, model_builder: DispatchLPModel, solver: HighsSolver):
        self.mb = model_builder
        self.solver = solver

    def run_all_scenarios(self) -> Dict[str, Any]:
        """
        Выполняет расчет всех трех сценариев + эластичный фильтр.
        """
        print("\n>>> Запуск Базового сценария...")
        base_lp = self.mb.build_lp()
        base_sol = self.solver.solve(base_lp)
        base_verif = SolutionVerifier.verify(
            x=base_sol["x"],
            c=base_lp["c"],
            A_ub=base_lp["A_ub"],
            b_ub=base_lp["b_ub"],
            bounds=base_lp["bounds"],
            reported_fun=base_sol["fun"],
            constraint_names=base_lp["constraint_names"]
        )

        print("\n>>> Запуск Дефицитного сценария...")
        # Уменьшение мощности базовой АЭС на 20% (плановый останов одного блока, st_id=1, mod=0.8)
        # и одновременный рост спроса на 10%
        deficit_lp = self.mb.build_lp(
            demand_modifier=1.10,
            capacity_modifiers={1: 0.80}
        )
        deficit_sol = self.solver.solve(deficit_lp)
        deficit_verif = SolutionVerifier.verify(
            x=deficit_sol["x"],
            c=deficit_lp["c"],
            A_ub=deficit_lp["A_ub"],
            b_ub=deficit_lp["b_ub"],
            bounds=deficit_lp["bounds"],
            reported_fun=deficit_sol["fun"],
            constraint_names=deficit_lp["constraint_names"]
        )

        print("\n>>> Запуск Стрессового сценария...")
        # Двойное резкое ухудшение:
        # 1. Аномальная волна холода: рост спроса на 30%
        # 2. Ограничение газовых станций на 50% (ТЭЦ st_id=3 и ГТУ st_id=5)
        # 3. Засуха: гидроресурс ГЭС снижен на 60% (до 1600 МВт*ч)
        # 4. Авария на АЭС (st_id=1, mod=0.6)
        stress_lp = self.mb.build_lp(
            demand_modifier=1.30,
            capacity_modifiers={1: 0.60, 3: 0.50, 5: 0.50},
            hydro_limit=1600.0
        )
        stress_sol = self.solver.solve(stress_lp)

        # Диагностика несовместимости стрессового сценария
        stress_diagnosis = self._diagnose_infeasibility(stress_lp)

        print("\n>>> Запуск Эластичного фильтра (Elastic Filter) для стрессового сценария (*)...")
        elastic_result = self._run_elastic_filter(stress_lp)

        # Сводная таблица сценариев
        summary_rows = [
            {
                "Сценарий": "1. Базовый (номинал)",
                "Статус HiGHS": base_sol["status"],
                "Интерпретация": "OPTIMAL (допустим и оптимален)",
                "Стоимость генерации, руб.": f"{base_sol['fun']:,.2f}",
                "Выработка суммарная, МВт*ч": f"{base_sol['x'].sum():,.1f}",
                "Средняя себестоимость, руб./МВт*ч": f"{base_sol['fun'] / base_sol['x'].sum():.2f}",
                "Итераций HiGHS": base_sol["nit"],
                "Время решения, мс": f"{base_sol['solve_time_sec'] * 1000:.2f}"
            },
            {
                "Сценарий": "2. Дефицитный (+10% спрос, -20% АЭС)",
                "Статус HiGHS": deficit_sol["status"],
                "Интерпретация": "OPTIMAL (допустим и оптимален)",
                "Стоимость генерации, руб.": f"{deficit_sol['fun']:,.2f}",
                "Выработка суммарная, МВт*ч": f"{deficit_sol['x'].sum():,.1f}",
                "Средняя себестоимость, руб./МВт*ч": f"{deficit_sol['fun'] / deficit_sol['x'].sum():.2f}",
                "Итераций HiGHS": deficit_sol["nit"],
                "Время решения, мс": f"{deficit_sol['solve_time_sec'] * 1000:.2f}"
            },
            {
                "Сценарий": "3. Стрессовый (+30% спрос, аварии)",
                "Статус HiGHS": stress_sol["status"],
                "Интерпретация": "INFEASIBLE (дефицит мощности)",
                "Стоимость генерации, руб.": "Н/Д (несовместима)",
                "Выработка суммарная, МВт*ч": "Н/Д",
                "Средняя себестоимость, руб./МВт*ч": "Н/Д",
                "Итераций HiGHS": stress_sol["nit"],
                "Время решения, мс": f"{stress_sol['solve_time_sec'] * 1000:.2f}"
            },
            {
                "Сценарий": "4. Стрессовый + Elastic Filter (*)",
                "Статус HiGHS": elastic_result["sol"]["status"],
                "Интерпретация": f"OPTIMAL (дефицит компенсирован: {elastic_result['total_curtailment']:,.1f} МВт*ч)",
                "Стоимость генерации, руб.": f"{elastic_result['gen_cost']:,.2f}",
                "Выработка суммарная, МВт*ч": f"{elastic_result['total_gen']:,.1f}",
                "Средняя себестоимость, руб./МВт*ч": f"{elastic_result['gen_cost'] / elastic_result['total_gen']:.2f}",
                "Итераций HiGHS": elastic_result["sol"]["nit"],
                "Время решения, мс": f"{elastic_result['sol']['solve_time_sec'] * 1000:.2f}"
            }
        ]

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv("output/scenarios_comparison.csv", index=False, encoding="utf-8-sig")

        # Сохранение базового решения в CSV (по часам и станциям)
        x_base_2d = base_sol["x"].reshape((base_lp["n_hours"], base_lp["n_stations"]))
        base_df_dict = {"hour": list(range(1, base_lp["n_hours"] + 1)), "demand_mwh": base_lp["demand"]}
        for i, st in enumerate(base_lp["stations"]):
            base_df_dict[f"{st['name']}_mw"] = x_base_2d[:, i]
        base_df_dict["total_generation_mw"] = x_base_2d.sum(axis=1)

        # Добавим маржинальные цены (LMP) по часам
        # В SciPy linprog marginals для A_ub * x <= b_ub: теневые цены для первых 24 ограничений спроса
        base_df_dict["lmp_rub_per_mwh"] = base_sol["marginals_ineq"][:base_lp["n_hours"]]

        df_base = pd.DataFrame(base_df_dict)
        df_base.to_csv("output/solution_base.csv", index=False, encoding="utf-8-sig")

        # Построение графиков
        self._plot_base_dispatch(df_base, base_lp)
        self._plot_scenarios_comparison(df_base, deficit_sol, deficit_lp, elastic_result)

        return {
            "base": {"sol": base_sol, "lp": base_lp, "verif": base_verif, "df": df_base},
            "deficit": {"sol": deficit_sol, "lp": deficit_lp, "verif": deficit_verif},
            "stress": {"sol": stress_sol, "lp": stress_lp, "diagnosis": stress_diagnosis},
            "elastic": elastic_result,
            "summary_df": df_summary
        }

    def _diagnose_infeasibility(self, stress_lp: Dict[str, Any]) -> Dict[str, Any]:
        """
        Диагностика конфликтующих ограничений при статусе INFEASIBLE.
        Сравнивает суммарный пиковый спрос с максимальной суммарной располагаемой мощностью.
        """
        demand = stress_lp["demand"]
        # Извлекаем ub для станций
        bounds = stress_lp["bounds"]
        n_st = stress_lp["n_stations"]
        n_h = stress_lp["n_hours"]

        p_max_per_hour = np.zeros(n_h)
        for t in range(n_h):
            p_max_per_hour[t] = sum(bounds[t * n_st + i][1] for i in range(n_st))

        deficit_per_hour = np.maximum(0.0, demand - p_max_per_hour)
        conflicting_hours = np.where(deficit_per_hour > 0)[0] + 1

        diagnosis = {
            "max_demand_mw": float(demand.max()),
            "max_system_capacity_mw": float(p_max_per_hour.max()),
            "conflicting_hours": conflicting_hours.tolist(),
            "max_hourly_deficit_mw": float(deficit_per_hour.max()),
            "total_unmet_energy_mwh": float(deficit_per_hour.sum())
        }
        return diagnosis

    def _run_elastic_filter(self, stress_lp: Dict[str, Any]) -> Dict[str, Any]:
        """
        Задание со звёздочкой (*): Эластичный фильтр (Elastic Filter).
        Вводит переменные срезки нагрузки s_t >= 0 (недоотпуск электроэнергии)
        с штрафной стоимостью VOLL = 50 000 руб./МВт*ч.
        min c^T p + VOLL * sum_t s_t
        при sum_i p_{i,t} + s_t >= D_t
        """
        n_h = stress_lp["n_hours"]
        n_st = stress_lp["n_stations"]
        n_orig_vars = stress_lp["n_vars"]  # n_h * n_st
        n_total_vars = n_orig_vars + n_h   # добавляем n_h переменных s_t

        voll_penalty = 50000.0  # руб./МВт*ч (Value of Lost Load)

        c_elastic = np.concatenate([stress_lp["c"], np.full(n_h, voll_penalty)])

        # Границы: для p_{i,t} - исходные, для s_t - [0, inf)
        bounds_elastic = list(stress_lp["bounds"]) + [(0.0, None) for _ in range(n_h)]

        # Матрица A_ub: расширяем столбцы для s_t
        # Для ограничений спроса: -sum_i p_{i,t} - s_t <= -D_t
        A_orig = stress_lp["A_ub"]
        A_elastic = np.zeros((len(stress_lp["b_ub"]), n_total_vars), dtype=np.float64)
        A_elastic[:, :n_orig_vars] = A_orig

        for t in range(n_h):
            # s_t имеет индекс n_orig_vars + t
            A_elastic[t, n_orig_vars + t] = -1.0

        elastic_model = {
            "c": c_elastic,
            "A_ub": A_elastic,
            "b_ub": stress_lp["b_ub"],
            "bounds": bounds_elastic
        }

        sol = self.solver.solve(elastic_model)
        x_all = sol["x"]
        p_opt = x_all[:n_orig_vars]
        s_opt = x_all[n_orig_vars:]

        gen_cost = float(np.dot(stress_lp["c"], p_opt))
        penalty_cost = float(voll_penalty * np.sum(s_opt))

        return {
            "sol": sol,
            "p_opt": p_opt,
            "curtailment_s": s_opt,
            "total_curtailment": float(np.sum(s_opt)),
            "total_gen": float(np.sum(p_opt)),
            "gen_cost": gen_cost,
            "penalty_cost": penalty_cost,
            "full_cost": sol["fun"]
        }

    def _plot_base_dispatch(self, df_base: pd.DataFrame, base_lp: Dict[str, Any]):
        """
        Построение графика суточного покрытия спроса в базовом сценарии.
        """
        hours = df_base["hour"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

        colors = ["#2b5c8f", "#7f7f7f", "#e67e22", "#27ae60", "#e74c3c"]
        station_cols = [f"{st['name']}_mw" for st in base_lp["stations"]]
        labels = [st["name"] for st in base_lp["stations"]]

        # Стековый график генерации
        gen_data = [df_base[c] for c in station_cols]
        ax1.stackplot(hours, gen_data, labels=labels, colors=colors, alpha=0.85)
        ax1.plot(hours, df_base["demand_mwh"], color="black", linewidth=2.5, linestyle="--", label="Спрос D_t (МВт)")

        ax1.set_ylabel("Электрическая мощность, МВт", fontsize=12)
        ax1.set_title("Суточный оптимальный диспетчерский график (Базовый сценарий)", fontsize=14, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", frameon=True, fontsize=10)

        # График теневой цены электроэнергии (LMP)
        ax2.plot(hours, df_base["lmp_rub_per_mwh"], color="#8e44ad", linewidth=2.0, marker="o", markersize=4, label="Теневая цена спроса (LMP, руб./МВт*ч)")
        ax2.set_xlabel("Час суток (1..24)", fontsize=12)
        ax2.set_ylabel("Теневая цена, руб./МВт*ч", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left", frameon=True, fontsize=10)

        plt.xticks(np.arange(1, 25, 1))
        plt.tight_layout()
        plt.savefig("output/plots/dispatch_base.png", dpi=200)
        plt.close()
        print("  Сохранен график: output/plots/dispatch_base.png")

    def _plot_scenarios_comparison(self, df_base: pd.DataFrame, deficit_sol: Dict[str, Any], deficit_lp: Dict[str, Any], elastic_result: Dict[str, Any]):
        """
        Сравнительный график генерации по сценариям.
        """
        hours = df_base["hour"]
        x_def_2d = deficit_sol["x"].reshape((deficit_lp["n_hours"], deficit_lp["n_stations"]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 1. Сравнение структуры выработки (суммарно за сутки)
        st_names = [s["name"] for s in self.mb.stations]
        base_tot = df_base[[f"{st['name']}_mw" for st in self.mb.stations]].sum().values
        def_tot = x_def_2d.sum(axis=0)

        x_indices = np.arange(len(st_names))
        width = 0.35

        ax1.bar(x_indices - width/2, base_tot, width, label="Базовый сценарий", color="#3498db")
        ax1.bar(x_indices + width/2, def_tot, width, label="Дефицитный (+10% спрос, -20% АЭС)", color="#e67e22")

        ax1.set_ylabel("Суточная выработка, МВт*ч", fontsize=11)
        ax1.set_title("Суточная выработка станций по сценариям", fontsize=12, fontweight="bold")
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels([s["name"].split()[0] for s in self.mb.stations], rotation=15)
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.6)

        # 2. Недоотпуск в стрессовом сценарии (Elastic filter)
        curtailment = elastic_result["curtailment_s"]
        ax2.bar(hours, curtailment, color="#c0392b", alpha=0.8, label="Срезка нагрузки (дефицит мощности, МВт)")
        ax2.set_xlabel("Час суток", fontsize=11)
        ax2.set_ylabel("Непокрытый дефицит, МВт", fontsize=11)
        ax2.set_title("Стрессовый сценарий: профиль дефицита (Elastic Filter)", fontsize=12, fontweight="bold")
        ax2.set_xticks(np.arange(1, 25, 2))
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig("output/plots/scenarios_compare.png", dpi=200)
        plt.close()
        print("  Сохранен график: output/plots/scenarios_compare.png")

