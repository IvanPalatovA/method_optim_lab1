#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Задание со звёздочкой (*):
Собственная реализация двухфазного симплекс-метода (Two-Phase Simplex Method)
для канонической задачи ЛП и полная сверка с промышленным солвером HiGHS.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import scipy.optimize as opt


class CustomTwoPhaseSimplex:
    """
    Собственная реализация двухфазного табличного симплекс-метода
    с правилом Бленда для разрешения спорных случаев.
    """

    TOLERANCE = 1e-9

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.iteration_log: List[Dict[str, Any]] = []

    def solve(
        self,
        c_orig: np.ndarray,
        A_ub: np.ndarray,
        b_ub: np.ndarray,
        var_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Решает задачу:
          min c^T x
          A_ub x <= b_ub
          x >= 0
        """
        c_orig = np.asarray(c_orig, dtype=np.float64)
        A_ub = np.asarray(A_ub, dtype=np.float64)
        b_ub = np.asarray(b_ub, dtype=np.float64)

        m = len(b_ub)
        n_orig = len(c_orig)

        if var_names is None:
            var_names = [f"x{i+1}" for i in range(n_orig)]

        # 1. Построение равенств: добавление слабых (slack) и избыточных (surplus) переменных
        n_slack = m
        A_eq = np.zeros((m, n_orig + n_slack), dtype=np.float64)
        A_eq[:, :n_orig] = A_ub
        b_eq = b_ub.copy()

        basis_entries = []
        art_needed = []

        all_names = list(var_names) + [f"s{i+1}" for i in range(m)]

        for i in range(m):
            if b_eq[i] < -self.TOLERANCE:
                # Ограничение вида a_i x <= -b_i  =>  -a_i x >= b_i  =>  -a_i x - s_i + a_art = b_i
                A_eq[i, :] *= -1.0
                b_eq[i] *= -1.0
                A_eq[i, n_orig + i] = -1.0  # surplus
                art_needed.append(i)
            else:
                # Ограничение вида a_i x <= b_i  =>  a_i x + s_i = b_i
                A_eq[i, n_orig + i] = 1.0   # slack
                basis_entries.append((i, n_orig + i))

        n_art = len(art_needed)
        if n_art > 0:
            A_full = np.zeros((m, n_orig + n_slack + n_art), dtype=np.float64)
            A_full[:, :n_orig + n_slack] = A_eq
            for k, row_i in enumerate(art_needed):
                art_col = n_orig + n_slack + k
                A_full[row_i, art_col] = 1.0
                basis_entries.append((row_i, art_col))
                all_names.append(f"art_{k+1}")
        else:
            A_full = A_eq

        basis_entries.sort(key=lambda item: item[0])
        basis_vars = [item[1] for item in basis_entries]

        # 2. ФАЗА I: Поиск допустимого базисного решения (если были искусственные переменные)
        if n_art > 0:
            if self.verbose:
                print("--- ФАЗА I: Поиск начального допустимого базисного решения ---")

            c_p1 = np.zeros(A_full.shape[1], dtype=np.float64)
            for k in range(n_art):
                c_p1[n_orig + n_slack + k] = 1.0

            T1 = np.zeros((m + 1, A_full.shape[1] + 1), dtype=np.float64)
            T1[:m, :-1] = A_full
            T1[:m, -1] = b_eq
            T1[-1, :-1] = c_p1

            # Исключаем базисные переменные из строки цели
            for r, b_col in enumerate(basis_vars):
                T1[-1, :] -= c_p1[b_col] * T1[r, :]

            # Симплекс-шаги Фазы I
            sol_p1 = self._run_tableau_loop(T1, basis_vars, all_names, phase=1)
            if sol_p1["status"] != "OPTIMAL" or abs(T1[-1, -1]) > 1e-6:
                return {
                    "success": False,
                    "status": "INFEASIBLE",
                    "message": "Задача не имеет допустимых решений (сумма искусственных переменных > 0)",
                    "nit": len(self.iteration_log),
                    "log": self.iteration_log
                }

            # Отбрасываем искусственные переменные
            A_p2 = T1[:m, :n_orig + n_slack]
            b_p2 = T1[:m, -1]
            all_names = all_names[:n_orig + n_slack]
        else:
            A_p2 = A_full
            b_p2 = b_eq

        # 3. ФАЗА II: Оптимизация исходной целевой функции
        if self.verbose:
            print("\n--- ФАЗА II: Оптимизация исходной целевой функции ---")

        c_p2 = np.zeros(n_orig + n_slack, dtype=np.float64)
        c_p2[:n_orig] = c_orig

        T2 = np.zeros((m + 1, n_orig + n_slack + 1), dtype=np.float64)
        T2[:m, :-1] = A_p2
        T2[:m, -1] = b_p2
        T2[-1, :-1] = c_p2

        # Исключаем базисные переменные из строки исходной цели
        for r, b_col in enumerate(basis_vars):
            T2[-1, :] -= c_p2[b_col] * T2[r, :]

        sol_p2 = self._run_tableau_loop(T2, basis_vars, all_names, phase=2)

        # Восстановление вектора решения
        x_full = np.zeros(n_orig + n_slack, dtype=np.float64)
        for r, b_col in enumerate(basis_vars):
            if b_col < len(x_full):
                x_full[b_col] = T2[r, -1]

        fun_val = -float(T2[-1, -1])
        x_orig = x_full[:n_orig]

        return {
            "success": sol_p2["status"] == "OPTIMAL",
            "status": sol_p2["status"],
            "fun": fun_val,
            "x": x_orig,
            "basis": basis_vars,
            "nit": len(self.iteration_log),
            "log": self.iteration_log,
            "tableau": T2
        }

    def _run_tableau_loop(
        self,
        tableau: np.ndarray,
        basis_vars: List[int],
        names: List[str],
        phase: int
    ) -> Dict[str, Any]:
        """
        Цикл симплекс-таблицы (минимизация).
        """
        m = len(basis_vars)
        n_cols = tableau.shape[1] - 1
        max_iters = 1000

        while len(self.iteration_log) < max_iters:
            # Строка цели: r_j = c_j - z_j.
            # Если все r_j >= -TOLERANCE, текущий базис оптимален!
            rc = tableau[-1, :n_cols]
            min_rc_val = np.min(rc)

            if min_rc_val >= -self.TOLERANCE:
                return {"status": "OPTIMAL"}

            candidates = np.where(rc < -self.TOLERANCE)[0]
            best_val = np.min(rc[candidates])
            ties = [idx for idx in candidates if abs(rc[idx] - best_val) < 1e-12]
            enter_col = min(ties)

            col = tableau[:m, enter_col]
            rhs = tableau[:m, -1]

            valid_ratios = []
            for i in range(m):
                if col[i] > self.TOLERANCE:
                    ratio = rhs[i] / col[i]
                    valid_ratios.append((ratio, basis_vars[i], i))

            if not valid_ratios:
                return {"status": "UNBOUNDED"}

            min_ratio = min(item[0] for item in valid_ratios)
            best_rows = [item for item in valid_ratios if abs(item[0] - min_ratio) < self.TOLERANCE]
            best_rows.sort(key=lambda item: item[1])
            leave_row = best_rows[0][2]
            leave_var = basis_vars[leave_row]

            step_record = {
                "phase": phase,
                "iteration": len(self.iteration_log) + 1,
                "entering_var": names[enter_col] if enter_col < len(names) else f"var_{enter_col}",
                "leaving_var": names[leave_var] if leave_var < len(names) else f"var_{leave_var}",
                "pivot_element": float(tableau[leave_row, enter_col]),
                "ratio": float(min_ratio),
                "current_z": -float(tableau[-1, -1])
            }
            self.iteration_log.append(step_record)

            if self.verbose:
                print(f"  Итерация {step_record['iteration']} (Фаза {phase}): Входит {step_record['entering_var']}, "
                      f"Выходит {step_record['leaving_var']}, Опорный элемент={step_record['pivot_element']:.4f}, z={step_record['current_z']:,.2f}")

            pivot = tableau[leave_row, enter_col]
            tableau[leave_row, :] /= pivot
            for r in range(m + 1):
                if r != leave_row:
                    factor = tableau[r, enter_col]
                    tableau[r, :] -= factor * tableau[leave_row, :]

            basis_vars[leave_row] = enter_col

        return {"status": "ITERATION_LIMIT"}


def verify_custom_simplex_vs_highs() -> Dict[str, Any]:
    """
    Верификационный эксперимент со звёздочкой (*):
    Сверка собственной реализации симплекс-метода с HiGHS на модельной задаче
    диспетчеризации двух электростанций на час пика:
      Станции: x1 (АЭС, c1 = 650), x2 (ГРЭС, c2 = 1450)
      Спрос: x1 + x2 >= 900  <=>  -x1 - x2 <= -900
      Мощности: x1 <= 800, x2 <= 600
      Экология: 0.85 * x2 <= 400
      Цель: min 650 x1 + 1450 x2
    """
    print("\n" + "=" * 70)
    print("ВЕРИФИКАЦИЯ СОБСТВЕННОГО СИМПЛЕКС-МЕТОДА И СВЕРКА С HiGHS (*)")
    print("=" * 70)

    c = np.array([650.0, 1450.0])
    A_ub = np.array([
        [-1.0, -1.0],  # -x1 - x2 <= -900
        [1.0,  0.0],   # x1 <= 800
        [0.0,  1.0],   # x2 <= 600
        [0.0,  0.85],  # 0.85 x2 <= 400
    ])
    b_ub = np.array([-900.0, 800.0, 600.0, 400.0])
    var_names = ["x_АЭС", "x_ГРЭС"]

    # 1. Решение собственной реализацией симплекс-метода
    custom_solver = CustomTwoPhaseSimplex(verbose=True)
    custom_res = custom_solver.solve(c, A_ub, b_ub, var_names=var_names)

    # 2. Решение промышленным солвером HiGHS
    bounds_highs = [(0.0, None), (0.0, None)]
    highs_res = opt.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_highs, method="highs-ds")

    fun_custom = custom_res["fun"]
    fun_highs = highs_res.fun
    fun_diff = abs(fun_custom - fun_highs)

    x_custom = custom_res["x"]
    x_highs = highs_res.x
    x_diff = float(np.max(np.abs(x_custom - x_highs)))

    is_matched = (fun_diff < 1e-6) and (x_diff < 1e-6)

    print("\nРЕЗУЛЬТАТЫ СВЕРКИ:")
    print(f"  Целевая функция (Custom Simplex) : {fun_custom:,.2f} руб.")
    print(f"  Целевая функция (HiGHS)          : {fun_highs:,.2f} руб.")
    print(f"  Разница значений цели            : {fun_diff:.2e} руб.")
    print(f"  Вектор решения x (Custom)        : {x_custom}")
    print(f"  Вектор решения x (HiGHS)         : {x_highs}")
    print(f"  Максимальное расхождение x       : {x_diff:.2e}")
    print(f"  Число итераций Custom Simplex    : {custom_res['nit']}")
    print(f"  Число итераций HiGHS             : {highs_res.nit}")
    print(f"  ИТОГОВЫЙ СТАТУС СВЕРКИ           : {'[PASS] ТОЧНОЕ СОВПАДЕНИЕ' if is_matched else '[FAIL] РАСХОЖДЕНИЕ'}")
    print("=" * 70)

    return {
        "matched": is_matched,
        "custom_fun": fun_custom,
        "highs_fun": fun_highs,
        "fun_diff": fun_diff,
        "x_custom": x_custom.tolist(),
        "x_highs": x_highs.tolist(),
        "custom_nit": custom_res["nit"],
        "highs_nit": highs_res.nit,
        "log": custom_res["log"]
    }


if __name__ == "__main__":
    verify_custom_simplex_vs_highs()
