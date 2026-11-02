#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Задание со звёздочкой (*):
Двойственная интерпретация модели диспетчеризации (Duality Analysis):
1. Аналитическое и матричное построение двойственной задачи.
2. Решение прямой и двойственной задач независимыми вызовами HiGHS.
3. Проверка сильной теоремы двойственности (c^T x* == b^T y*).
4. Проверка условий дополняющей нежёсткости (Complementary Slackness).
5. Экономическая интерпретация двойственных переменных (LMP, теневые цены ресурсов, рента мощности).
"""

import os
import sys
from typing import Dict, Any, Tuple
import numpy as np
import scipy.optimize as opt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import DispatchLPModel
from src.solver import HighsSolver


class DualityAnalyzer:
    """
    Класс для построения, решения и анализа двойственной задачи.
    """

    TOLERANCE = 1e-7

    def __init__(self, model_builder: DispatchLPModel):
        self.mb = model_builder

    def analyze_duality(self) -> Dict[str, Any]:
        """
        Решает прямую и двойственную задачу и проверяет теоремы двойственности.
        """
        print("\n" + "=" * 70)
        print("ДВОЙСТВЕННАЯ ИНТЕРПРЕТАЦИЯ И ПРОВЕРКА СИЛЬНОЙ ДВОЙСТВЕННОСТИ (*)")
        print("=" * 70)

        # 1. Построение прямой задачи в матричной форме
        # min c^T x
        # s.t. A_all x <= b_all
        #      x >= 0
        lp = self.mb.build_lp()
        c = lp["c"]
        A_orig = lp["A_ub"]
        b_orig = lp["b_ub"]
        bounds = lp["bounds"]
        n_vars = lp["n_vars"]
        m_orig = len(b_orig)

        l_vec = np.array([b[0] for b in bounds], dtype=np.float64)
        u_vec = np.array([b[1] for b in bounds], dtype=np.float64)

        # Объединённая матрица ограничений: A_all * x <= b_all
        # Блоки:
        # 1. A_orig * x <= b_orig (m_orig строк)
        # 2. I * x <= u_vec (n_vars строк)
        # 3. -I * x <= -l_vec (n_vars строк)
        A_all = np.vstack([
            A_orig,
            np.eye(n_vars),
            -np.eye(n_vars)
        ])
        b_all = np.concatenate([
            b_orig,
            u_vec,
            -l_vec
        ])
        m_all = len(b_all)

        # 2. Решение прямой задачи (Primal)
        primal_res = opt.linprog(c=c, A_ub=A_all, b_ub=b_all, bounds=(0, None), method="highs-ds")
        x_star = primal_res.x
        f_primal = primal_res.fun

        # 3. Построение двойственной задачи (Dual)
        # Исходная: min c^T x при A_all x <= b_all, x >= 0
        # Эквивалентна: min c^T x при (-A_all) x >= (-b_all), x >= 0
        # Двойственная: max (-b_all)^T y при (-A_all)^T y <= c, y >= 0
        # В терминах функции min: min b_all^T y при (-A_all)^T y <= c, y >= 0
        # Значение максимума = -min b_all^T y
        c_dual_min = b_all
        A_dual = -A_all.T
        b_dual = c
        bounds_dual = [(0.0, None) for _ in range(m_all)]

        dual_res = opt.linprog(c=c_dual_min, A_ub=A_dual, b_ub=b_dual, bounds=bounds_dual, method="highs-ds")
        y_star = dual_res.x
        f_dual = -dual_res.fun  # восстанавливаем исходное max (-b_all)^T y = - (min b_all^T y)

        # 4. Проверка сильной двойственности: f_primal == f_dual
        duality_gap = abs(f_primal - f_dual)
        rel_gap = duality_gap / (abs(f_primal) + 1e-12)
        strong_duality_holds = duality_gap <= 1e-5 or rel_gap <= 1e-7

        # 5. Проверка условий дополняющей нежёсткости:
        # Прямая невязка (slack): s_primal = b_all - A_all * x* >= 0
        primal_slack = b_all - np.dot(A_all, x_star)
        cs_primal_products = y_star * primal_slack
        max_cs_primal_violation = float(np.max(np.abs(cs_primal_products)))

        # Двойная невязка: s_dual = c - (-A_all^T * y*) = c + A_all^T y* >= 0
        dual_slack = b_dual - np.dot(A_dual, y_star)
        cs_dual_products = x_star * dual_slack
        max_cs_dual_violation = float(np.max(np.abs(cs_dual_products)))

        cs_holds = (max_cs_primal_violation <= 1e-4) and (max_cs_dual_violation <= 1e-4)

        # 6. Экономическая декомпозиция вектора y*
        y_ineq = y_star[:m_orig]
        y_upper = y_star[m_orig:m_orig + n_vars]
        y_lower = y_star[m_orig + n_vars:]

        print(f"1. ЦЕЛЕВЫЕ ФУНКЦИИ:")
        print(f"   - Прямая задача (min Затраты)  : {f_primal:,.2f} руб.")
        print(f"   - Двойственная (max Стоимость) : {f_dual:,.2f} руб.")
        print(f"   - Двойственный зазор (Gap)    : {duality_gap:.3e} руб. (отн: {rel_gap:.2e})")
        print(f"   - Сильная двойственность      : {'[PASS] ВЫПОЛНЯЕТСЯ' if strong_duality_holds else '[FAIL] НАРУШЕНА'}")

        print(f"\n2. ДОПОЛНЯЮЩАЯ НЕЖЁСТКОСТЬ (KKT / Complementary Slackness):")
        print(f"   - Макс. невязка y_i * s_i      : {max_cs_primal_violation:.3e}")
        print(f"   - Макс. невязка x_j * s_dual_j : {max_cs_dual_violation:.3e}")
        print(f"   - Статус дополняющей нежёстк. : {'[PASS] ВЫПОЛНЯЕТСЯ' if cs_holds else '[FAIL] НАРУШЕНА'}")

        print(f"\n3. ЭКОНОМИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ ДВОЙСТВЕННЫХ ПЕРЕМЕННЫХ:")
        print(f"   - y_ineq[0..23]: Теневые цены баланса спроса (LMP, маржинальная цена электроэнергии в каждый час)")
        print(f"     Диапазон цен: от {y_ineq[:24].min():.2f} руб./МВт*ч (ночь) до {y_ineq[:24].max():.2f} руб./МВт*ч (пик)")
        print(f"   - y_ineq[Hydro]: Ценность запаса воды ГЭС = {y_ineq[lp['constraint_names'].index('Hydro_Daily_Energy_Limit')]:.2f} руб./МВт*ч")
        print(f"   - y_upper: Квазирента мощности (scarcity rent) при работе станции на максимуме p_max")
        print(f"   - y_lower: Издержки технологического минимума (must-run cost) при фиксации на p_min")
        print("=" * 70)

        return {
            "f_primal": f_primal,
            "f_dual": f_dual,
            "duality_gap": duality_gap,
            "strong_duality_holds": strong_duality_holds,
            "max_cs_primal_violation": max_cs_primal_violation,
            "max_cs_dual_violation": max_cs_dual_violation,
            "cs_holds": cs_holds,
            "x_star": x_star,
            "y_star": y_star,
            "y_ineq": y_ineq,
            "y_upper": y_upper,
            "y_lower": y_lower
        }


if __name__ == "__main__":
    mb = DispatchLPModel()
    da = DualityAnalyzer(mb)
    da.analyze_duality()
