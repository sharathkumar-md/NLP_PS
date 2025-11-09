@echo off
echo ================================================================================
echo BERT Implementation - Creating Submission Package
echo ================================================================================
echo.

REM Create submission folder
if exist BERT_Submission rmdir /s /q BERT_Submission
mkdir BERT_Submission

echo Copying files...
echo.

REM Copy all required files
copy bert_model.py BERT_Submission\
copy config.py BERT_Submission\
copy dataset.py BERT_Submission\
copy demo.py BERT_Submission\
copy test_model.py BERT_Submission\
copy train.py BERT_Submission\
copy visualize.py BERT_Submission\
copy README.md BERT_Submission\
copy QUICKSTART.md BERT_Submission\
copy PROJECT_SUMMARY.md BERT_Submission\
copy SUBMISSION_CHECKLIST.md BERT_Submission\
copy requirements.txt BERT_Submission\
copy best_model.pt BERT_Submission\
copy training_history.json BERT_Submission\

echo.
echo ================================================================================
echo Submission package created successfully!
echo ================================================================================
echo.
echo Location: BERT_Submission\
echo.
echo Files included (14):
dir /b BERT_Submission
echo.
echo ================================================================================
echo READY FOR SUBMISSION!
echo ================================================================================
echo.
echo You can now:
echo   1. Zip the BERT_Submission folder, OR
echo   2. Upload files directly from BERT_Submission folder
echo.
pause
