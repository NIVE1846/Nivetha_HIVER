@echo off
REM run_all.bat — Full training pipeline for SpotifyCares support agent
REM Requires: archive/twcs/twcs.csv (download from Kaggle)
REM Runtime: ~10-15 minutes on 8 GB RAM laptop

echo === Phase 1: Dataset analysis ===
python src\analyze_data.py || goto :error

echo === Phase 2: Preprocessing ===
python src\preprocessing.py || goto :error

echo === Phase 2b: Conversation reconstruction ===
python src\conversation_builder.py || goto :error

echo === Phase 3: Build golden set ===
python src\build_golden_set.py || goto :error

echo === Phase 4a: Trivial baseline ===
cd src
python baseline_trivial.py || goto :error

echo === Phase 4b: TF-IDF baseline ===
python baseline_tfidf.py || goto :error

echo === Phase 5: Main classifier ===
python intent_classifier.py || goto :error

echo === Phase 6: Retrieval index ===
python retrieval.py || goto :error

echo === Phase 7: Threshold selection ===
python select_thresholds.py || goto :error
cd ..

echo === Phase 10: Pipeline smoke test ===
cd src && python pipeline.py && cd ..

echo === Phase 11: Evaluation ===
cd evaluation
python evaluate.py || goto :error
python judge.py || goto :error
cd ..

echo.
echo === All phases complete ===
echo Results: reports/results.md
echo Scores:  evaluation/judge_scores.csv
goto :eof

:error
echo.
echo ERROR: Phase failed. Check output above.
exit /b 1
