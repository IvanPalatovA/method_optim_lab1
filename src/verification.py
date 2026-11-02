#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль независимой проверки найденного решения на чистом NumPy (без обращения к солверу).
Точность проверки: 1e-7 (требование методических указаний).
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class SolutionVerifier:
    """
    Класс независимой верификации решения.
    """

    TOLERANCE = 1e-7

    @classmethod
    def verify(
        cls,
        x: np.ndarray,
        c: np.ndarray,
        A_ub: np.ndarray,
        b_ub: np.ndarray,
        bounds: List[Tuple[float, float]],
        reported_fun: float,
        constraint_names: List[str]
    ) -> Dict[str, Any]:
        """
        Независимая проверка допустимости и пересчёт целевой функции.
        """
        x = np.asarray(x, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        A_ub = np.asarray(A_ub, dtype=np.float64)
        b_ub = np.asarray(b_ub, dtype=np.float64)

        # 1. Пересчёт целевой функции на чистом NumPy
        recalculated_fun = float(np.dot(c, x))
        fun_abs_diff = abs(recalculated_fun - reported_fun)
        fun_rel_diff = fun_abs_diff / (abs(reported_fun) + 1e-12)
        fun_check_passed = fun_abs_diff <= cls.TOLERANCE or fun_rel_diff <= cls.TOLERANCE

        # 2. Проверка ограничений неравенств: Ax <= b <=> Ax - b <= 0
        lhs = np.dot(A_ub, x)
        slack = b_ub - lhs  # должно быть >= 0
        ineq_violations = np.maximum(0.0, -slack)  # величина нарушения
        max_ineq_violation = float(np.max(ineq_violations)) if len(ineq_violations) > 0 else 0.0
        ineq_check_passed = max_ineq_violation <= cls.TOLERANCE

        # 3. Проверка границ переменных: lb <= x <= ub
        lbs = np.array([b[0] for b in bounds], dtype=np.float64)
        ubs = np.array([b[1] for b in bounds], dtype=np.float64)

        lb_violations = np.maximum(0.0, lbs - x)
        ub_violations = np.maximum(0.0, x - ubs)

        max_lb_violation = float(np.max(lb_violations))
        max_ub_violation = float(np.max(ub_violations))
        bounds_check_passed = (max_lb_violation <= cls.TOLERANCE) and (max_ub_violation <= cls.TOLERANCE)

        # 4. Итоговый статус допустимости
        overall_passed = fun_check_passed and ineq_check_passed and bounds_check_passed

        # Поиск активных ограничений (slack <= tolerance)
        active_indices = np.where(np.abs(slack) <= 1e-5)[0]
        active_constraints = [constraint_names[i] for i in active_indices]

        # Формирование текстового отчета
        report_lines = [
            "=" * 70,
            "НЕЗАВИСИМАЯ ПРОВЕРКА РЕШЕНИЯ (NUMPY VERIFICATION REPORT)",
            f"Машинный допуск (TOLERANCE) : {cls.TOLERANCE:.1e}",
            "=" * 70,
            f"1. ЦЕЛЕВАЯ ФУНКЦИЯ:",
            f"   - Значение от солвера     : {reported_fun:,.6f} руб.",
            f"   - Независимый пересчёт   : {recalculated_fun:,.6f} руб.",
            f"   - Абсолютная невязка      : {fun_abs_diff:.3e} руб.",
            f"   - Относительная невязка   : {fun_rel_diff:.3e}",
            f"   - Статус проверки цели    : {'[PASS] СОШЛОСЬ' if fun_check_passed else '[FAIL] РАСХОЖДЕНИЕ'}",
            "",
            f"2. ОГРАНИЧЕНИЯ НЕРАВЕНСТВ (A_ub * x <= b_ub):",
            f"   - Число ограничений       : {len(b_ub)}",
            f"   - Максимальное нарушение  : {max_ineq_violation:.3e}",
            f"   - Число нарушенных (>1e-7): {np.sum(ineq_violations > cls.TOLERANCE)}",
            f"   - Статус проверки A*x<=b  : {'[PASS] ДОПУСТИМО' if ineq_check_passed else '[FAIL] НЕДОПУСТИМО'}",
            "",
            f"3. ГРАНИЦЫ ПЕРЕМЕННЫХ (lb <= x <= ub):",
            f"   - Макс. нарушение lb      : {max_lb_violation:.3e}",
            f"   - Макс. нарушение ub      : {max_ub_violation:.3e}",
            f"   - Статус проверки границ  : {'[PASS] ДОПУСТИМО' if bounds_check_passed else '[FAIL] НЕДОПУСТИМО'}",
            "",
            f"4. ИТОГОВЫЙ ВЕРДИКТ: {'РЕШЕНИЕ ПОЛНОСТЬЮ ДОПУСТИМО И КОРРЕКТНО [PASS]' if overall_passed else 'ОБНАРУЖЕНЫ НАРУШЕНИЯ [FAIL]'}",
            "=" * 70,
            f"Активных ограничений (насыщенных): {len(active_constraints)} из {len(b_ub)}",
            "Примеры активных ограничений:"
        ]
        for ac in active_constraints[:10]:
            report_lines.append(f"   * {ac}")
        if len(active_constraints) > 10:
            report_lines.append(f"   ... и ещё {len(active_constraints) - 10} ограничений")
        report_lines.append("=" * 70)

        report_text = "\n".join(report_lines)

        return {
            "passed": overall_passed,
            "fun_check_passed": fun_check_passed,
            "ineq_check_passed": ineq_check_passed,
            "bounds_check_passed": bounds_check_passed,
            "recalculated_fun": recalculated_fun,
            "fun_abs_diff": fun_abs_diff,
            "max_ineq_violation": max_ineq_violation,
            "max_bounds_violation": max(max_lb_violation, max_ub_violation),
            "active_constraints": active_constraints,
            "report_text": report_text
        }

