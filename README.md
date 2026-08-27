# Start the Crop Disease Analyzer

## First-time setup

Open PowerShell and run:

```bat
cd "%USERPROFILE%\Crop-Disease-Analyzer"
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
pip install -r ml\requirements.txt
```

Install the frontend dependencies:

```bat
cd frontend
npm install
```

## Start the project

### 1. Start MongoDB

Make sure MongoDB is running on `mongodb://localhost:27017`. If it is not running as a Windows service, use:

```bat
mongod
```

### 2. Seed the database and train the model

Open a second PowerShell window and run:

```bat
cd "%USERPROFILE%\Crop-Disease-Analyzer"
.venv\Scripts\activate
python database\seed.py

cd ml
python train.py
```

### 3. Start the backend

In the same terminal, run:

```bat
cd ..\backend
uvicorn app.main:app --reload
```

Keep this terminal open. The backend runs at http://127.0.0.1:8000.

### 4. Start the frontend

Open a third PowerShell window and run:

```bat
cd "%USERPROFILE%\Crop-Disease-Analyzer\frontend"
npm run dev
```

Open http://localhost:5173 in your browser.

### 5. Log in and analyze an image

Log in with:

- Email: `admin@cropanalyzer.com`
- Password: `Admin@123456`

Open **Analyze** and upload a crop leaf image.
