mkdir -p logs
python boundary_fine_tuning.py fit --trainer.devices [0] --config configs/boundary_finetune.yaml >logs/boundary.txt 2>&1
