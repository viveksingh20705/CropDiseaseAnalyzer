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

Copy `backend\.env.example` to `backend\.env` and set your MongoDB URI and a
unique random JWT secret before starting the API. Never commit `backend\.env`.

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

## Deploy the frontend to GitHub Pages

The `Deploy frontend to GitHub Pages` workflow builds and publishes the frontend
when changes are pushed to `main`. In the repository settings, set
**Settings → Pages → Build and deployment → Source** to **GitHub Actions**.
The deployed frontend uses the existing Render API at
`https://cropdiseaseanalyzer-100.onrender.com`; configure that backend to allow
the GitHub Pages origin if its environment overrides the app defaults.

The API also needs valid MongoDB credentials and a trained model file on its
host for prediction requests. Do not commit credentials or model weights.
