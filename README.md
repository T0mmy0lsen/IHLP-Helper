# The Pipeline
This explains exactly what to do to run the code. 
All steps are done in such a way that the code will check if files exists before trying to generate them. 
So fell free to run the steps multiple times if you get lost. 
If all hope is lost just make sure you have the following files in the {root}/model/input folder, and you can run everything from scratch.
\
\
**The folder `data` must contain:**
- `communication.csv`
- `item.csv` 
- `relation.csv`   
- `relation_history.csv`
- `request.csv`

## (1) Preprocess
This is run my simply running `main.py`