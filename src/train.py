# src/train.py

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .seed_utils import set_seed

# --- Параметры из варианта 15 ---
SEED = 173
LEARNING_RATE = 0.0015
HIDDEN_NEURONS = 28
BATCH_SIZE = 128
EPOCHS = 5
MU0 = [-1.1, -0.5]
MU1 = [+0.9, +0.9]
SIGMA = 0.9


def get_git_commit_hash() -> str:
    """Получает хэш текущего коммита Git."""
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit_hash = "N/A"
    return commit_hash


def calculate_sha256(filepath: Path) -> str:
    """Вычисляет SHA256 хэш файла."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    """Основная функция для запуска эксперимента."""
    set_seed(SEED)

    # 2. Генерация данных
    # читает параметры из CLI; генерирует два гауссовых класса по вашему варианту
    # (в данном случае параметры захардкожены, а не из CLI)
    n_samples = 1000
    cov = [[SIGMA**2, 0], [0, SIGMA**2]]
    class0_data = np.random.multivariate_normal(MU0, cov, n_samples // 2)
    class1_data = np.random.multivariate_normal(MU1, cov, n_samples // 2)

    X = np.vstack((class0_data, class1_data)).astype(np.float32)
    y_combined = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    y = y_combined.astype(np.float32)
    y = y.reshape(-1, 1)

    X_tensor, y_tensor = torch.tensor(X), torch.tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)

    # 3. Создание DataLoader
    # DataLoader(..., shuffle=True, generator=make_generator(seed), num_workers=0)
    generator = torch.Generator().manual_seed(SEED)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    # 4. Определение модели и оптимизатора
    # модель: Linear -> ReLU -> Linear (hidden из варианта)
    model = nn.Sequential(
        nn.Linear(2, HIDDEN_NEURONS),
        nn.ReLU(),
        nn.Linear(HIDDEN_NEURONS, 1),
    )
    # оптимизатор и lr — из варианта
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    # 5. Цикл обучения
    for _ in range(EPOCHS):
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        final_loss = loss_fn(model(X_tensor), y_tensor).item()
    print(f"Обучение завершено. Финальная ошибка: {final_loss:.8f}")

    # 6. Сохранение артефактов
    # сохраняет model_*.pt и run_*.json (final_loss, версии, commit, sha256)
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    model_path = runs_dir / f"model_seed{SEED}.pt"
    torch.save(model.state_dict(), model_path)
    model_sha256 = calculate_sha256(model_path)

    run_data = {
        "params": {"seed": SEED, "lr": LEARNING_RATE, "epochs": EPOCHS},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "results": {"final_loss": final_loss},
        "artifacts": {"model_sha256": model_sha256},
        "git_commit": get_git_commit_hash(),
    }

    run_path = runs_dir / f"run_seed{SEED}.json"
    with open(run_path, "w") as f:
        json.dump(run_data, f, indent=2)
    print(f"Артефакты сохранены в папку: {runs_dir.resolve()}")


if __name__ == "__main__":
    main()
