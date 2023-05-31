### Remove files from git
Did you commit some big files and made a mess?
````
java -jar bfg.jar --delete-files *.csv
````

Did you pull and made a local mess?
````
git merge --abort
````

Did you prefer your local changes?
````
git push origin --force
````


### Run the frontend
````
cd /frontend

npm install
npm run start
````

### Run the backend

````
cd /backend

python3 -m venv ./venv
venv\Scripts\Activate.ps1

python -m pip install django
python -m pip install numpy
python -m pip install pandas
python -m pip install scikit-learn
python -m pip install django-cors-headers
python -m pip install djangorestframework
python -m pip install mysqlclient
python -m pip install jsonfield
python -m pip install lxml
python -m pip install beautifulsoup4
python -m pip install tensorflow-gpu 

python ./manage.py runserver 0.0.0.0:8000
````
### Load the data locally
Place the data in /backend/notebooks/database (the files with Request.csv, etc.)
Call the API http://127.0.0.1:8000/ihlp/load
This will take some time to load. It's the best method I got to work.

### Admin page
Visit http://127.0.0.1:8000/admin/ with `-u tool -p ?`

### Errors
None.

### Following the code

Once the 

### How the application works

1. When a Request is received: users and time is predicted and saved to the database.
2. For all open Request with Responsibles: the workload is calculated.
3. The user may search in the Request that has a prediction.
4. The user is presented with suggestions for the Responsible given prediction and workload.