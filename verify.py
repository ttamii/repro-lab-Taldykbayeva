# verify.py

import json
from pathlib import Path

from src import train


def run_verification():
    """
    Запускает эксперимент дважды и проверяет идентичность результатов.
    """
    print("--- Запуск №1 ---")
    train.main()

    # --- Чтение результатов первого запуска ---
    run_path = Path(f"runs/run_seed{train.SEED}.json")
    with open(run_path) as f:
        run_data_1 = json.load(f)

    loss1 = run_data_1["results"]["final_loss"]
    sha1 = run_data_1["artifacts"]["model_sha256"]
    print(f"Результаты №1: Loss={loss1:.8f}, SHA={sha1[:10]}...\n")

    # --- Запуск №2 ---
    print("--- Запуск №2 ---")
    train.main()

    # --- Чтение результатов второго запуска ---
    with open(run_path) as f:
        run_data_2 = json.load(f)

    loss2 = run_data_2["results"]["final_loss"]
    sha2 = run_data_2["artifacts"]["model_sha256"]
    print(f"Результаты №2: Loss={loss2:.8f}, SHA={sha2[:10]}...\n")

    # --- Сравнение результатов ---
    print("--- Сравнение ---")
    TOLERANCE = 1e-8
    # Сообщения в assert были укорочены для flake8
    assert abs(loss1 - loss2) < TOLERANCE, "ОШИБКА: loss не совпадает!"
    assert sha1 == sha2, "ОШИБКА: хэши SHA256 не совпадают!"

    print("✅ Воспроизводимость успешно подтверждена!")


if __name__ == "__main__":
    run_verification()
