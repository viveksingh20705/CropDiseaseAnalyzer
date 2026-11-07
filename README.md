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

Training also exports `ml\models\crop_disease_savedmodel`, the TensorFlow
SavedModel used by API inference. Keep this generated model directory with the
backend deployment.

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

After signing in, use the sidebar to open the dashboard, prediction history,
disease library, weather, crop guide, and account settings. The dashboard reads
your saved predictions; profile and password changes are authenticated API
operations. Weather uses precise browser location when available and falls back
to Gorakhpur when the device location is unavailable or too approximate. The
crop guide searches the disease information stored in MongoDB rather than
returning generated medical or agricultural advice.

## Deploy the frontend to GitHub Pages

The `Deploy frontend to GitHub Pages` workflow builds and publishes the frontend
when changes are pushed to `main`. In the repository settings, set
**Settings → Pages → Build and deployment → Source** to **GitHub Actions**.
The deployed frontend uses the existing Render API at
`https://cropdiseaseanalyzer-100.onrender.com`; configure that backend to allow
the GitHub Pages origin if its environment overrides the app defaults.

The API also needs valid MongoDB credentials and the exported SavedModel on its
host for prediction requests. Do not commit credentials.
