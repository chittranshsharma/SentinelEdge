$env:PYTHONIOENCODING="utf-8"
py -3.11 01_generate_synthetic_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -3.11 02_feature_engineering.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -3.11 03_train_model.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -3.11 04_convert_to_tflite.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -3.11 05_validate_tflite.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
