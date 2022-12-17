# The Pipeline
The following folders are needed
- `backend/predict/data/input`
- `backend/predict/data/output`
- `backend/predict/data/tmp`
\
\
**The folder `{root}/predict/data/input` must contain:**
- `communication.csv`
- `item.csv` 
- `relation.csv`   
- `relation_history.csv`
- `request.csv`

## (1) Main
The models are created by running `main.py`. This runs the function `run()`.

## Tensorboard
`tensorboard --logdir models/logs/tensorboard`