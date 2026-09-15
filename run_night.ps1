# ==============================================================================
# RI-YOLO: Скрипт последовательного ночного запуска (run_night.ps1)
# ==============================================================================
# Описание:
#   Выполняет последовательное обучение 4 моделей для статьи C1:
#     1. e2p__baseline__s0 (базовая модель YOLOv8s, seed 0, 100 эпох) - Блок 3
#     2. e2p__baseline__s3 (базовая модель YOLOv8s, seed 3, 100 эпох) - Блок 3
#     3. e3p__capctrl__s1  (контроль емкости capctrl, seed 1, 100 эпох) - замечание 6
#     4. e3p__capctrl__s2  (контроль емкости capctrl, seed 2, 100 эпох) - замечание 6
#
# Особенности:
#   - Строго последовательный запуск (защита GPU от перегрузки памяти и CPU).
#   - Раздельное логирование каждого прогона в logs/<run_name>.log через Tee-Object.
#   - Автоматический запуск postrun_check.py после каждого прогона.
#   - try / catch обработка с фиксацией кодов возврата и времени работы.
#   - Автоматическая пересборка C1_table1.csv и C1_stats.csv после завершения.
# ==============================================================================

[CmdletBinding()]
param (
    [string]$Data = "exdark.yaml",
    [string]$Device = "0",
    [int]$Epochs = 100,
    [string]$Weights = "weights/yolov8s.pt"
)

# Установка кодировки UTF-8 для корректного вывода
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  RI-YOLO: СТАРТ НОЧНОЙ СЕРИИ ОБУЧЕНИЯ (4 ПРОГОНА)" -ForegroundColor Cyan
Write-Host "  Время старта: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# 1. Проверка директорий
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
    Write-Host "[INIT] Создана папка logs/" -ForegroundColor Yellow
}
if (-not (Test-Path "tables")) {
    New-Item -ItemType Directory -Path "tables" | Out-Null
    Write-Host "[INIT] Создана папка tables/" -ForegroundColor Yellow
}

# 2. Проверка базовых файлов
if (-not (Test-Path $Weights)) {
    Write-Host "[ERROR] Файл весов '$Weights' не найден! Остановка." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $Data)) {
    Write-Host "[ERROR] Конфиг данных '$Data' не найден! Остановка." -ForegroundColor Red
    exit 1
}

# 3. Список прогонов в порядке выполнения
$RunsQueue = @(
    @{ Name = "e2p__baseline__s0"; Desc = "Baseline YOLOv8s (seed 0) [Блок 3]" },
    @{ Name = "e2p__baseline__s3"; Desc = "Baseline YOLOv8s (seed 3) [Блок 3]" },
    @{ Name = "e3p__capctrl__s1";  Desc = "Capacity Control module (seed 1) [Замечание 6]" },
    @{ Name = "e3p__capctrl__s2";  Desc = "Capacity Control module (seed 2) [Замечание 6]" }
)

$OverallStartTime = Get-Date
$SummaryReport = @()

# 4. Последовательный цикл выполнения
foreach ($run in $RunsQueue) {
    $runName = $run.Name
    $runDesc = $run.Desc
    $logFile = "logs/$runName.log"
    $runStartTime = Get-Date

    Write-Host "`n--------------------------------------------------------------------" -ForegroundColor Green
    Write-Host ">>> ЗАПУСК ПРОГОНА: $runName" -ForegroundColor Green
    Write-Host "    Описание : $runDesc" -ForegroundColor Green
    Write-Host "    Лог-файл : $logFile" -ForegroundColor Green
    Write-Host "    Старт    : $($runStartTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Green
    Write-Host "--------------------------------------------------------------------" -ForegroundColor Green

    $runStatus = "UNKNOWN"
    $exitCode = 0

    try {
        # Запуск обучения с выводом в консоль и дублированием в лог-файл
        $cmd = "python scripts/train_phase1.py --run $runName --data $Data --device $Device --epochs $Epochs --weights $Weights"
        Write-Host "[CMD] $cmd" -ForegroundColor Gray
        
        # Выполнение команды с перехватом вывода
        Invoke-Expression "$cmd 2>&1" | Tee-Object -FilePath $logFile
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Host "`n[POST-CHECK] Обучение $runName завершено с кодом 0. Запуск проверки..." -ForegroundColor Cyan
            
            # Запуск скрипта пост-проверки результатов
            $checkCmd = "python scripts/postrun_check.py --run runs/$runName"
            Invoke-Expression "$checkCmd 2>&1" | Tee-Object -FilePath $logFile -Append
            
            if ($LASTEXITCODE -eq 0) {
                $runStatus = "SUCCESS"
                Write-Host "[STATUS] $runName : УСПЕШНО ЗАВЕРШЁН И ПРОВЕРЕН" -ForegroundColor Green
            } else {
                $runStatus = "WARNING (check exit code $LASTEXITCODE)"
                Write-Host "[STATUS] $runName : Обучение завершено, но postrun_check сообщил о предупреждениях" -ForegroundColor Yellow
            }
        } else {
            $runStatus = "FAILED (exit code $exitCode)"
            Write-Host "[ERROR] Ошибка при обучении $runName (Exit code: $exitCode)" -ForegroundColor Red
        }
    }
    catch {
        $runStatus = "EXCEPTION"
        $exitCode = -1
        $errText = $_.Exception.Message
        Write-Host "[EXCEPTION] Исключение при выполнении $runName : $errText" -ForegroundColor Red
        Add-Content -Path $logFile -Value "`n[EXCEPTION] $errText"
    }

    $runEndTime = Get-Date
    $runElapsed = $runEndTime - $runStartTime
    $elapsedStr = "{0:D2}h {1:D2}m {2:D2}s" -f $runElapsed.Hours, $runElapsed.Minutes, $runElapsed.Seconds

    Write-Host "[TIME] Прогон $runName длился: $elapsedStr" -ForegroundColor Cyan

    $SummaryReport += [PSCustomObject]@{
        RunName  = $runName
        Status   = $runStatus
        ExitCode = $exitCode
        Duration = $elapsedStr
    }
}

# 5. Итоговая пересборка таблиц C1 и расчёт статистики
Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "  ИТОГОВАЯ СБОРКА РЕЗУЛЬТАТОВ ДЛЯ СТАТЬИ C1" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

try {
    Write-Host "[COLLECT] Пересборка tables/C1_table1.csv..." -ForegroundColor Gray
    python scripts/collect_c1.py --out tables/C1_table1.csv
    python scripts/collect_c1.py

    if (Test-Path "artifacts/phase1/summary.csv") {
        Write-Host "[STATS] Расчет статистических критериев (Уэлч, Манн-Уитни)..." -ForegroundColor Gray
        python scripts/stats_c1.py --summary artifacts/phase1/summary.csv --out tables/C1_stats.csv
    }
} catch {
    Write-Host "[WARN] Ошибка при финальной пересборке таблиц: $($_.Exception.Message)" -ForegroundColor Yellow
}

# 6. Финальный отчёт
$OverallEndTime = Get-Date
$OverallElapsed = $OverallEndTime - $OverallStartTime
$OverallElapsedStr = "{0:D2}h {1:D2}m {2:D2}s" -f $OverallElapsed.Hours, $OverallElapsed.Minutes, $OverallElapsed.Seconds

Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "  СВОДНЫЙ ОТЧЁТ НОЧНОГО ЗАПУСКА" -ForegroundColor Cyan
Write-Host "  Общее затраченное время: $OverallElapsedStr" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

$SummaryReport | Format-Table -AutoSize

Write-Host "Ночной запуск завершён в $($OverallEndTime.ToString('yyyy-MM-dd HH:mm:ss'))." -ForegroundColor Cyan
