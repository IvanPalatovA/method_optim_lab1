#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор исходных данных для Лабораторной работы № 1.
Вариант 6: "План выработки электроэнергии станциями: минимум затрат" (5 станций, горизонт 24 ч).

Индивидуальные параметры (по формуле из п. 1.5):
  Студент: Палатов Иван Александрович
  Учебная группа: Б24-517 (Q = 1)
  Порядковый номер по списку: N = 16
  Год: 2026 (Y = 26)
  Вариант: V = 1 + ((17*16 + 7*1 + 26) mod 20) = 1 + (305 mod 20) = 1 + 5 = 6
  Seed генерации: 10000 + 100*Q + N = 10000 + 100*1 + 16 = 10116
"""

import json
import csv
import numpy as np


def generate_variant_data(seed: int = 10116, output_json: str = "data/input_data.json", output_csv: str = "data/hourly_demand.csv"):
    rng = np.random.default_rng(seed)

    # 1. Станции энергосистемы (5 станций)
    # Единицы измерения:
    # - мощность p_min, p_max: МВт
    # - маневренность ramp_up, ramp_down: МВт/ч
    # - себестоимость marginal_cost: руб. / МВт*ч
    # - удельные выбросы co2_rate: т CO2 / МВт*ч
    stations = [
        {
            "id": 1,
            "name": "АЭС 'Восход'",
            "type": "nuclear",
            "marginal_cost": 650.0,
            "p_min": 400.0,
            "p_max": 800.0,
            "ramp_up": 40.0,
            "ramp_down": 40.0,
            "co2_rate": 0.0,
            "description": "Базовая атомная станция (низкая себестоимость, малая маневренность)"
        },
        {
            "id": 2,
            "name": "ГРЭС-1 (Уголь)",
            "type": "coal",
            "marginal_cost": 1450.0,
            "p_min": 150.0,
            "p_max": 600.0,
            "ramp_up": 100.0,
            "ramp_down": 100.0,
            "co2_rate": 0.85,
            "description": "Тепловая угольная станция (средняя стоимость, ограничение по топливу)"
        },
        {
            "id": 3,
            "name": "ТЭЦ-2 (Газ)",
            "type": "gas_cogen",
            "marginal_cost": 1900.0,
            "p_min": 100.0,
            "p_max": 500.0,
            "ramp_up": 150.0,
            "ramp_down": 150.0,
            "co2_rate": 0.42,
            "description": "Газовая теплоэлектроцентраль (высокая маневренность)"
        },
        {
            "id": 4,
            "name": "ГЭС 'Бурея'",
            "type": "hydro",
            "marginal_cost": 180.0,
            "p_min": 0.0,
            "p_max": 450.0,
            "ramp_up": 250.0,
            "ramp_down": 250.0,
            "co2_rate": 0.0,
            "description": "Гидроэлектростанция (минимальная себестоимость, ограничена суточным гидроресурсом)"
        },
        {
            "id": 5,
            "name": "ГТУ-Пиковая (Газ)",
            "type": "gas_turbine",
            "marginal_cost": 3400.0,
            "p_min": 0.0,
            "p_max": 350.0,
            "ramp_up": 300.0,
            "ramp_down": 300.0,
            "co2_rate": 0.55,
            "description": "Пиковая газотурбинная установка (быстрый пуск, высокая стоимость топлива)"
        }
    ]

    # 2. Суточный профиль спроса (24 часа, МВт*ч)
    # Реалистичный суточный график энергосистемы: ночной спад, утренний пик, дневной уровень, вечерний максимум
    hourly_factors = np.array([
        0.68, 0.64, 0.62, 0.63, 0.67, 0.76,
        0.92, 1.08, 1.15, 1.18, 1.14, 1.10,
        1.08, 1.06, 1.05, 1.07, 1.12, 1.25,
        1.32, 1.28, 1.20, 1.10, 0.95, 0.80
    ])
    base_mean_demand = 1500.0
    demand_noise = rng.normal(0, 20.0, size=24)
    demand = np.round(base_mean_demand * hourly_factors + demand_noise, 1)

    # 3. Системные и экологические ограничения
    system_limits = {
        "hydro_daily_energy_limit_mwh": 4000.0,     # Суточный гидроресурс ГЭС (МВт*ч)
        "coal_daily_fuel_limit_mwh": 11000.0,       # Суточный лимит расхода топлива ГРЭС (МВт*ч эквивалент)
        "max_daily_co2_emission_tons": 14000.0      # Экологический лимит выбросов CO2 (тонн в сутки)
    }

    metadata = {
        "student": "Палатов Иван Александрович",
        "group": "Б24-517",
        "group_index_Q": 1,
        "student_number_N": 16,
        "year_Y": 26,
        "variant": 6,
        "seed": seed,
        "n_stations": len(stations),
        "horizon_hours": len(demand),
        "total_installed_capacity_mw": sum(s["p_max"] for s in stations),
        "total_min_capacity_mw": sum(s["p_min"] for s in stations),
        "system_limits": system_limits
    }

    full_data = {
        "metadata": metadata,
        "stations": stations,
        "demand_mwh": demand.tolist()
    }

    # Сохраняем JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

    # Сохраняем CSV спроса
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hour", "demand_mwh"])
        for h, d in enumerate(demand, start=1):
            writer.writerow([h, d])

    print(f"Данные успешно сгенерированы (seed={seed}, вариант={metadata['variant']}).")
    print(f"  Сохранено в {output_json} и {output_csv}")
    print(f"  Суточный спрос: min={demand.min()} МВт, max={demand.max()} МВт, суммарно={demand.sum()} МВт*ч")


if __name__ == "__main__":
    generate_variant_data()

