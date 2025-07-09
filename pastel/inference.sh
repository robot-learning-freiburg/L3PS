mkdir -p logs
python inference.py test --trainer.devices [0,1,2,3,4,5,6,7] --config configs/panoptic_inference.yaml >logs/inference.txt 2>&1
