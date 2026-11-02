#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль решения ЛП-задачи промышленным решателем HiGHS (через SciPy linprog).
Поддерживает dual simplex (highs-ds), interior point (highs-ipm), автоматический выбор (highs).
"""

import time
from typing import Dict, Any, Optional
import numpy as np
import scipy.optimize as opt


class HighsSolver:
    """
    Класс-обёртка над солвером HiGHS.
    """

    STATUS_MEANINGS = {
        0: "OPTIMAL: Найдено оптимальное решение с гарантией глобального минимума.",
        1: "ITERATION_LIMIT: Превышен лимит итераций.",
        2: "INFEASIBLE: Задача не имеет допустимых решений (область ОДР пуста из-за конфликта ограничений).",
        3: "UNBOUNDED: Целевая функция неограничена снизу на области допустимых решений.",
        4: "NUMERICAL_DIFFICULTIES: Численные проблемы или потеря устойчивости."
    }

    def __init__(self, method: str = "highs", presolve: bool = True, time_limit: Optional[float] = None):
        self.method = method
        self.presolve = presolve
        self.time_limit = time_limit

    def solve(self, model_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Запуск решения ЛП-задачи.
        """
        c = model_dict["c"]
        A_ub = model_dict["A_ub"]
        b_ub = model_dict["b_ub"]
        bounds = model_dict["bounds"]

        options = {
            "presolve": self.presolve,
            "disp": False
        }
        if self.time_limit is not None:
            options["time_limit"] = self.time_limit

        t_start = time.perf_counter()
        res = opt.linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method=self.method,
            options=options
        )
        t_elapsed = time.perf_counter() - t_start

        # Извлечение результатов
        status_code = res.status
        is_success = res.success
        status_desc = self.STATUS_MEANINGS.get(status_code, res.message)

        x_opt = res.x if is_success else None
        fun_opt = res.fun if is_success else None
        slack = res.slack if hasattr(res, "slack") and res.slack is not None else None

        # Двойственные переменные (теневые цены)
        marginals_ineq = None
        marginals_bounds_lower = None
        marginals_bounds_upper = None

        if is_success and hasattr(res, "ineqlin") and res.ineqlin is not None:
            marginals_ineq = res.ineqlin.marginals

        if is_success and hasattr(res, "lower") and res.lower is not None:
            marginals_bounds_lower = res.lower.marginals
        if is_success and hasattr(res, "upper") and res.upper is not None:
            marginals_bounds_upper = res.upper.marginals

        # Лог работы солвера
        log_lines = [
            "=" * 60,
            f"HiGHS Solver Log [Method: {self.method}]",
            "=" * 60,
            f"Status Code     : {status_code} ({res.message})",
            f"Interpretation  : {status_desc}",
            f"Success         : {is_success}",
            f"Iterations      : {res.nit}",
            f"Solve Time      : {t_elapsed * 1000.0:.3f} ms",
        ]
        if is_success:
            log_lines.extend([
                f"Objective Value : {fun_opt:,.2f} руб.",
                f"Variables Count : {len(c)}",
                f"Constraints (ub): {len(b_ub)}",
            ])
        log_lines.append("=" * 60)
        solver_log_text = "\n".join(log_lines)

        return {
            "success": is_success,
            "status": status_code,
            "message": res.message,
            "interpretation": status_desc,
            "fun": fun_opt,
            "x": x_opt,
            "nit": res.nit,
            "solve_time_sec": t_elapsed,
            "slack": slack,
            "marginals_ineq": marginals_ineq,
            "marginals_lower": marginals_bounds_lower,
            "marginals_upper": marginals_bounds_upper,
            "log": solver_log_text
        }

