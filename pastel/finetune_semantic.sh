mkdir -p logs
python semantic_fine_tuning.py fit --trainer.devices [0] --config configs/semantic_finetune.yaml >logs/semantic.txt 2>&1
