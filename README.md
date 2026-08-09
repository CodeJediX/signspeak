# SignSpeak

Sign-language recognition with webcam landmark tracking, speech output, and a
browser-based articulated 3D hand viewer.

## Run the viewer

From the repository root:

```powershell
python -m http.server 8000 --directory docs
```

Open <http://localhost:8000/sign_viewer.html>.

The viewer must be served over HTTP so the browser can load the local
MediaPipe model and vocabulary files. Click **Load model**, then **Start
camera**. The built-in landmark matcher works without a converted TensorFlow.js
classifier. For the trained browser classifier, convert the Keras model into
`docs/models_web/` with TensorFlow.js.

## Install Python dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The Python live recognizer uses:

```powershell
python src\recognize_live.py
```

The supplied MediaPipe task and model assets are kept outside normal Git
tracking when they are large. The browser viewer includes its generated
vocabulary export and local task asset when available.
