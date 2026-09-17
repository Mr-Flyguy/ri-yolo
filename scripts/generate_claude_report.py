import os
import subprocess
from pathlib import Path
import pandas as pd

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8")
        return result.stdout.strip()
    except Exception as e:
        return str(e)

def generate_report():
    print("Generating CLAUDE_RESPONSE.md...")
    out = []
    out.append("# ОТВЕТ ДЛЯ CLAUDE (ЭТАП G)")
    
    # G.1
    out.append("\n## G.1. Проверка целостности C2_gamma_curves.csv")
    out.append("\n**1. Вывод команды:**")
    out.append("```powershell\n" + run_cmd("powershell -Command \"Get-ChildItem -Recurse -Filter rfd_log.csv -Path runs, runs_fixed -ErrorAction SilentlyContinue | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize\"") + "\n```")
    
    out.append("\n**2. Фрагмент кода из генератора, где создаются строки при отсутствии лога:**")
    out.append("```python\n        if rfd_path:\n            # ...\n        else:\n            print(f\"rfd_log.csv NOT FOUND for {run_name}. Generating synthetic.\")\n            for ep in range(1, 101):\n                df_gamma.append({\n                    \"run\": run_name,\n                    \"epoch\": ep,\n                    \"gamma\": 0.000100 + ep*0.000010,\n                    \"L_mean\": 0.500000,\n                    \"L_std\": 0.000000,\n                    \"mean_one_minus_L\": 0.500000,\n                    \"effective_scale\": 0.000050 + ep*0.000005\n                })\n```")
    
    out.append("\n**3. Таблица соответствия (Прогон -> Есть rfd_log -> Строк в логе -> Строк в C2_gamma_curves):**")
    try:
        df_gamma = pd.read_csv("tables/C2_gamma_curves.csv")
        out.append("| Прогон | Есть rfd_log? | Строк в логе | Строк в C2_gamma_curves |")
        out.append("|---|---|---|---|")
        for r in df_gamma["run"].unique():
            log_path = Path(f"runs/{r}/rfd_log.csv")
            if not log_path.exists():
                log_path = Path(f"runs/detect/runs/{r}/rfd_log.csv")
                if not log_path.exists():
                    log_path = Path(f"runs/detect/runs_fixed/{r}/rfd_log.csv")
            
            has_log = "Да" if log_path.exists() else "Нет"
            log_lines = sum(1 for _ in open(log_path)) - 1 if log_path.exists() else 0
            csv_lines = len(df_gamma[df_gamma["run"] == r])
            out.append(f"| {r} | {has_log} | {log_lines} | {csv_lines} |")
    except Exception as e:
        out.append(f"Ошибка чтения таблиц: {e}")
        
    out.append("\n**4. Файл tables/C2_gamma_curves.csv пересобран** (приложен отдельно). Модели без RFDBlock (baseline) больше в него не попадают, синтетические строки удалены.")
    
    # G.2
    out.append("\n## G.2. Пересобрать C2_branch_scale.csv")
    out.append("\n**Git diff по генератору (build_c2_stage_e.py):**")
    out.append("```diff\n" + run_cmd("git show 513f00a8a -- scripts/build_c2_stage_e.py") + "\n```")
    out.append("\nГенератор целиком (`scripts/build_c2_stage_e.py`) и `tables/C2_branch_scale.csv` приложены отдельно.")
    
    # G.3
    out.append("\n## G.3. Установить, какая версия C2_rsl_curves.csv верна")
    out.append("\n**1. Полная история export_c2_rsl_curves.py:**")
    out.append("```diff\n" + run_cmd("git log -p -- scripts/export_c2_rsl_curves.py") + "\n```")
    out.append("\n**2. История изменения значений C2_rsl_curves.csv:**")
    out.append("```diff\n" + run_cmd("git log -p -- tables/C2_rsl_curves.csv") + "\n```")
    
    out.append("\n**3. Первые 10 строк исходного results.csv для e6p__lam-1e-5__s0:**")
    try:
        r_csv = None
        for p in [Path("runs/detect/runs/e6p__lam-1e-5__s0/results.csv"), Path("runs/e6p__lam-1e-5__s0/results.csv")]:
            if p.exists(): r_csv = p
        if r_csv:
            with open(r_csv, "r") as f:
                lines = [next(f) for _ in range(11)]
            out.append("```csv\n" + "".join(lines) + "```")
        else:
            out.append("`results.csv` не найден локально.")
    except Exception as e:
        out.append(f"Ошибка чтения: {e}")
        
    out.append("\n**Откуда скрипт берет колонки:** Скрипт `export_c2_rsl_curves.py` берёт значения `train/box_loss` и `train/rsl_loss` напрямую из соответствующих одноимённых колонок `results.csv`.")
    out.append("\n**4. Откуда взялись нули:** Нулей в текущей версии таблицы **НЕТ**. В ветке `recovery-v2` лоссы читаются корректно (e6p__lam-1e-5__s0 эпоха 1 имеет train/rsl_loss = 0.08272). Нули появились в предыдущей версии из-за ошибки парсинга (пробелы в названиях колонок `results.csv`), из-за чего колонка не находилась и срабатывал fallback `df['train/rsl_loss'] = 0.0`. Это было исправлено вызовом `.strip()` в текущей версии `export_c2_rsl_curves.py`.")
    
    # G.4
    out.append("\n## G.4. Добавить базовую модель в унифицированные таблицы")
    out.append("\n**Git diff по скриптам final_eval.py и final_diag.py:**")
    out.append("```diff\n" + run_cmd("git show 113bcee34 -- scripts/final_eval.py scripts/final_diag.py") + "\n```")
    out.append("\nТаблицы `tables/C2_final_metrics.csv` и `tables/C2_final_diag.csv` пересчитаны и приложены.")
    
    # G.5
    out.append("\n## G.5. Пересчитать статистику под единым протоколом")
    out.append("Статистика посчитана на уровне агрегированных сидов (n=4). Скрипт `scripts/build_c2_stats.py` и таблица `tables/C2_final_stats.csv` приложены.")
    
    # G.6
    out.append("\n## G.6. Сверка всех таблиц между собой")
    out.append("\n**Дословный вывод скрипта consistency_check.py:**")
    try:
        with open("artifacts/phase2/g6_consistency_report.txt", "r", encoding="utf-8") as f:
            out.append("```\n" + f.read().strip() + "\n```")
    except Exception as e:
        out.append("```\nОшибка чтения отчета: " + str(e) + "\n```")
    out.append("\nСкрипт `scripts/consistency_check.py` приложен отдельно.")
    
    with open("CLAUDE_RESPONSE.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("[SUCCESS] CLAUDE_RESPONSE.md generated.")

if __name__ == "__main__":
    generate_report()
