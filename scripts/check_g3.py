import pandas as pd
from pathlib import Path

def check_rsl_loss():
    print("="*80)
    print("  G.3: ПРОВЕРКА КОЛОНОК RSL LOSS В results.csv")
    print("="*80)
    target_run = "e6p__lam-1e-4__s0"
    patterns = [
        Path(f"runs/detect/runs/{target_run}/results.csv"),
        Path(f"runs/{target_run}/results.csv"),
        Path(f"runs/detect/runs_fixed/{target_run}/results.csv"),
    ]
    
    csv_path = next((p for p in patterns if p.exists()), None)
    
    if not csv_path:
        print(f"[ОШИБКА] Файл results.csv для {target_run} не найден!")
        return

    df = pd.read_csv(csv_path)
    # Выводим сырые колонки до стрипа пробелов
    print(f"--- Оригинальные названия колонок в {csv_path} ---")
    print(list(df.columns))
    
    df.columns = [c.strip() for c in df.columns]
    rsl_cols = [c for c in df.columns if 'rsl' in c.lower() or 'tv' in c.lower()]
    print(f"\n--- Колонки, похожие на RSL/TV: {rsl_cols} ---")
    
    if rsl_cols:
        col = rsl_cols[0]
        print(f"\nЗначения '{col}' для первых 15 эпох:")
        for ep, val in zip(df['epoch'].head(15), df[col].head(15)):
            print(f"  Epoch {ep}: {val}")
    else:
        print("\n[ОШИБКА] Колонка с rsl/tv loss не найдена в results.csv!")

if __name__ == "__main__":
    check_rsl_loss()
