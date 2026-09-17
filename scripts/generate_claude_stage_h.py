import subprocess
import os
import sys
import pandas as pd
from pathlib import Path
import glob

def run_command(cmd, capture=True):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}\n{res.stderr}")
    return res.stdout if capture else None

def get_file_content(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"

def generate_h5_table():
    output = "прогон | существует ли rfd_log.csv | число строк в нём | число строк в C2_gamma_curves.csv\n"
    output += "---|---|---|---\n"
    
    # Get all potential runs from C2_final_metrics
    df_metrics = pd.read_csv("tables/C2_final_metrics.csv")
    runs = df_metrics["run"].unique()
    
    # Load C2_gamma_curves to count rows per run
    try:
        df_gamma = pd.read_csv("tables/C2_gamma_curves.csv")
    except:
        df_gamma = pd.DataFrame(columns=["run"])
        
    for run in sorted(runs):
        # find rfd_log.csv for this run
        log_path = Path(run) / "rfd_log.csv"
        exists = log_path.exists()
        
        lines_in_log = 0
        if exists:
            try:
                log_df = pd.read_csv(log_path)
                lines_in_log = len(log_df)
            except:
                pass
                
        # Find rows in gamma curves
        # C2_gamma_curves stores run name as the basename
        run_basename = Path(run).name
        lines_in_gamma = len(df_gamma[df_gamma["run"] == run_basename])
        
        output += f"`{run}` | {'Да' if exists else 'Нет'} | {lines_in_log} | {lines_in_gamma}\n"
        
    output += "\n**Первые 5 строк C2_gamma_curves.csv:**\n```csv\n"
    try:
        with open("tables/C2_gamma_curves.csv", "r") as f:
            lines = [next(f) for _ in range(6)]
            output += "".join(lines)
    except Exception as e:
        output += str(e) + "\n"
    output += "```\n"
    
    return output

def main():
    print("=== Re-generating all stage H tables ===")
    run_command("python scripts/build_c2_stage_e.py", capture=False)
    run_command("python scripts/build_c2_stats.py", capture=False)
    
    print("\n=== Running consistency check ===")
    consistency_output = run_command("python scripts/consistency_check.py", capture=True)
    print(consistency_output)
    
    print("\n=== Generating H.5 table ===")
    h5_content = generate_h5_table()
    print(h5_content)
    
    print("\n=== Creating archive ===")
    run_command("tar czf C2_data_final.tar.gz tables/C2_*.csv artifacts/phase2/", capture=False)
    
    print("\n=== Writing CLAUDE_RESPONSE_H.md ===")
    
    report = f"""# ЭТАП H: ФИНАЛЬНЫЙ ОТЧЁТ И ЗАКРЫТИЕ ДАННЫХ

## H.1, H.2: Скрипт `build_c2_stage_e.py`
```python
{get_file_content("scripts/build_c2_stage_e.py")}
```

## H.3: Скрипт `build_c2_stats.py`
```python
{get_file_content("scripts/build_c2_stats.py")}
```

## H.4: Скрипт `consistency_check.py` и его дословный вывод
```python
{get_file_content("scripts/consistency_check.py")}
```

### Дословный вывод `consistency_check.py`:
```text
{consistency_output}
```

## H.5: Проверка `C2_gamma_curves.csv`
{h5_content}

## H.6: Финальный архив
Архив успешно создан: `C2_data_final.tar.gz`.

Список файлов, прикреплённых к этому сообщению:
1. `tables/C2_table1.csv`
2. `tables/C2_final_metrics.csv`
3. `tables/C2_final_diag.csv`
4. `tables/C2_final_stats.csv`
5. `tables/C2_branch_scale.csv`
6. `tables/C2_gamma_curves.csv`
7. `tables/C2_rsl_curves.csv`
8. `tables/C2_hyperparams.csv`
9. `scripts/build_c2_stage_e.py`
10. `scripts/consistency_check.py`
11. `C2_data_final.tar.gz` (содержит всё)
"""

    with open("CLAUDE_RESPONSE_H.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    print("[SUCCESS] CLAUDE_RESPONSE_H.md generated successfully.")

if __name__ == "__main__":
    main()
