# Atomo AI – Face Recognition and Video Analytics Platform

Atomo AI is a real-time face recognition and video analytics platform for camera monitoring, person detection, face recognition, unknown-person clustering, event management, and camera configuration.

The system processes RTSP camera streams, detects faces, generates face embeddings, compares them with enrolled persons, clusters unknown faces, records recognition events, and provides a web dashboard.

## Current Architecture

- **Backend:** Node.js + Express
- **Frontend:** Static web application served by Express
- **Database:** SQLite using Node.js built-in `node:sqlite`
- **Face detection:** YuNet
- **Face recognition:** SFace
- **AI acceleration:** OpenCV TIM-VX / NPU when supported by the installed OpenCV build
- **Video streaming:** MediaMTX + RTSP
- **Face worker:** Python
- **Realtime communication:** WebSocket
- **Database format:** SQLite only — the old JSON database is no longer used

---

## Main Features

- User registration and login
- Role-based access control
- Camera registration and management
- RTSP camera stream support
- Live camera viewing
- Real-time person detection
- Real-time face detection
- Face recognition using YuNet and SFace
- Enrolled-person management
- Face embedding generation and storage
- Unknown-person detection
- Automatic unknown-face clustering
- Cluster review and person enrollment
- Detection event history
- Event images and recognition details
- Camera stream logs
- Live system metrics
- Alert configuration
- Device registration
- User and cluster role management
- Line-crossing configuration
- MediaMTX-based stream handling
- SQLite persistence

---

# System Requirements

## Operating System

Recommended:

- Ubuntu 22.04
- Ubuntu 24.04

## Required Software

- Git
- **Node.js 24 or newer**
- npm
- Python 3
- Python virtual environment
- FFmpeg
- FFprobe
- MediaMTX executable included in the project

For NPU acceleration, the system must have an OpenCV build that provides the required **TIM-VX / NPU** backend.

Check Node.js:

```bash
node --version
```

The project requires:

```text
v24.x or newer
```

---

# Installation

## 1. Clone the Repository

Clone the current repository and branch:

```bash
git clone -b agent/recognition-on-going https://github.com/JoshiMitesh/atomo-ai-mitesh.git
```

Enter the face-recognition project:

```bash
cd atomo-ai-mitesh/face_recognition
```

---

## 2. Install Ubuntu Packages

```bash
sudo apt update
```

```bash
sudo apt install -y \
git \
curl \
ffmpeg \
python3 \
python3-pip \
python3-venv \
build-essential \
libgl1 \
libglib2.0-0
```

Verify FFmpeg:

```bash
ffmpeg -version
```

Verify FFprobe:

```bash
ffprobe -version
```

---

## 3. Install Node.js 24+

The application uses Node.js's built-in `node:sqlite`, so **Node.js 24 or newer is required**.

For a fresh Ubuntu installation, install Node.js 24:

```bash
curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
sudo apt install -y nodejs
```

Verify:

```bash
node --version
npm --version
```

The Node.js version must be **24 or newer**.

---

## 4. Install Node.js Dependencies

From the `face_recognition` directory:

```bash
npm install
```

The current backend dependencies include:

- Express
- WebSocket
- CORS
- Multer
- ExcelJS

There is **no `better-sqlite3` native dependency** and no separate SQLite server is required.

---

## 5. Create the Python Virtual Environment

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Upgrade Python packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 6. Install Python Dependencies

Install the Python packages used by the face worker:

```bash
pip install numpy opencv-python-headless huggingface-hub
```

Verify:

```bash
python -c "import cv2, numpy, huggingface_hub; print('Python dependencies installed successfully')"
```

### NPU Note

The face worker is configured to use OpenCV **TIM-VX / NPU** for YuNet when the installed OpenCV build supports it.

You can check the available OpenCV build with:

```bash
python -c "import cv2; print(cv2.__version__); print('TIM-VX:', getattr(cv2.dnn, 'DNN_BACKEND_TIMVX', 'not available')); print('NPU:', getattr(cv2.dnn, 'DNN_TARGET_NPU', 'not available'))"
```

If the required TIM-VX/NPU backend is unavailable, the worker can fall back to CPU.

---

## 7. Prepare MediaMTX

The repository contains the MediaMTX executable:

```text
face_recognition/mediamtx
```

Give it execute permission:

```bash
chmod +x mediamtx
```

Verify:

```bash
./mediamtx --version
```

The binary must match the target system architecture.

Examples:

- x86-64 → amd64 MediaMTX binary
- ARM64 → arm64 MediaMTX binary
- ARM32 → armv7 MediaMTX binary

---

# Start Face Recognition

## First Startup

From the repository:

```bash
cd atomo-ai-mitesh/face_recognition
```

Activate the Python environment:

```bash
source venv/bin/activate
```

Start the application:

```bash
npm start
```

**Do not start MediaMTX or the Python face worker manually.** The Node.js server starts the required MediaMTX process and the Python face-recognition worker.

---

## Daily Start

After the initial installation, the normal startup is:

```bash
cd atomo-ai-mitesh/face_recognition
source venv/bin/activate
npm start
```

Or in one command:

```bash
cd atomo-ai-mitesh/face_recognition && source venv/bin/activate && npm start
```

The web application is available at:

```text
http://localhost:3000
```

---

## Open the Dashboard from Another Computer

Find the server IP:

```bash
hostname -I
```

Example:

```text
192.168.1.100
```

Open:

```text
http://192.168.1.100:3000
```

Replace the IP address with the actual IP of the face-recognition server.

---

# Database

The application now uses **SQLite as the only application database**.

Database file:

```text
face_recognition/data/face_recognition.db
```

The database is created automatically by `db.js`.

The database contains:

- Persons
- Person photos
- Face embeddings
- Recognition events
- Cameras
- Unknown-face clusters
- Cluster photos and embeddings
- Application settings
- Recognition threshold
- Recognition distance type
- Cluster counter

Face embeddings are stored as SQLite **BLOB** values.

### Important

The old file:

```text
data/database.json
```

is no longer used and has been removed from the project.

Do **not** recreate `database.json` manually.

Runtime data is intentionally ignored by Git through:

```text
face_recognition/.gitignore
```

---

# Database Reset

The project provides a database reset command:

```bash
npm run reset-db
```

This clears:

- Persons
- Person photos
- Recognition events
- Cameras
- Unknown clusters
- Cluster photos
- Application settings

It also removes runtime files from the uploads and crops directories.

**Warning:** This operation deletes application data. Use it only when a complete reset is required.

---

# Camera Configuration

The application supports RTSP cameras.

General RTSP format:

```text
rtsp://USERNAME:PASSWORD@CAMERA_IP:554/STREAM_PATH
```

Example:

```text
rtsp://admin:password@192.168.1.116:554/stream1
```

Add the RTSP URL through the camera management interface.

The camera must be reachable from the face-recognition server.

---

## Test an RTSP Camera with FFprobe

```bash
ffprobe \
-v error \
-rtsp_transport tcp \
-select_streams v:0 \
-show_entries stream=codec_name,width,height \
-of default=noprint_wrappers=1 \
"rtsp://USERNAME:PASSWORD@CAMERA_IP:554/STREAM_PATH"
```

---

## Test an RTSP Camera with FFmpeg

```bash
ffmpeg \
-hide_banner \
-loglevel warning \
-rtsp_transport tcp \
-i "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/STREAM_PATH" \
-t 10 \
-an \
-f null -
```

If the command processes video for approximately 10 seconds without an RTSP error, the stream is reachable.

---

# Application Ports

| Port | Purpose |
|---:|---|
| `3000` | Atomo AI web application and API |
| `8554` | MediaMTX RTSP |
| `8888` | MediaMTX HLS |
| `8889` | MediaMTX WebRTC/WHEP HTTP |
| `8189/UDP` | MediaMTX WebRTC media |

If Ubuntu UFW is enabled:

```bash
sudo ufw allow 3000/tcp
sudo ufw allow 8554/tcp
sudo ufw allow 8888/tcp
sudo ufw allow 8889/tcp
```

Check:

```bash
sudo ufw status
```

---

# Architecture Flow

```text
RTSP Camera
    |
    v
MediaMTX
    |
    v
Node.js Server
    |
    +----------------------+
    |                      |
    v                      v
Python Face Worker       Web Dashboard
    |
    +--> YuNet
    |      |
    |      +--> TIM-VX / NPU
    |
    +--> SFace
    |
    +--> Face Embeddings
    |
    +--> Recognition / Clustering
    |
    v
SQLite
    |
    +--> Persons
    +--> Photos + Embeddings
    +--> Events
    +--> Cameras
    +--> Clusters
    +--> Settings
```

---

# Running in the Background with PM2

PM2 is optional.

Install:

```bash
sudo npm install -g pm2
```

Start:

```cd atomo-ai-mitesh/face_recognition
source venv/bin/activate
pm2 start npm --name atomo-ai -- start
```

Check:

```bash
pm2 status
```

View logs:

```bash
pm2 logs atomo-ai
```

Restart:

```bash
pm2 restart atomo-ai
```

Stop:

```bash
pm2 stop atomo-ai
```

Remove:

```bash
pm2 delete atomo-ai
```

Enable startup after reboot:

```bash
pm2 save
pm2 startup
```

Run the `sudo ...` command printed by PM2, then:

```bash
pm2 save
```

---

# Troubleshooting

## Check Node.js Version

```bash
node --version
```

The project requires Node.js 24 or newer.

---

## Check SQLite

From `face_recognition`:

```bash
node -e "const db=require('./db.js'); console.log('SQLite DB OK'); console.log(db.getPersons())"
```

If successful, the SQLite database can be opened by the application.

Node.js may print an experimental warning for `node:sqlite`; this is a runtime warning and does not by itself indicate a database failure.

---

## Check JavaScript Syntax

```bash
node --check server.js
node --check db.js
```

---

## Check Python

```bash
source venv/bin/activate
python --version
python -c "import cv2, numpy; print('Python/OpenCV OK')"
```

---

## MediaMTX Permission Denied

```bash
chmod +x mediamtx
```

Then restart:

```bash
npm start
```

---

## MediaMTX Port Already in Use

Check:

```bash
sudo lsof -i :8554
```

Stop an old MediaMTX process if required:

```bash
pkill -f mediamtx
```

Then restart:

```bash
npm start
```

---

## Port 3000 Already in Use

Check:

```bash
sudo lsof -i :3000
```

You can run the application on another port:

```bash
PORT=3001 npm start
```

Then open:

```text
http://localhost:3001
```

---

## Python Module Not Found

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install numpy opencv-python-headless huggingface-hub
```

---

## RTSP Camera Not Connecting

Check:

1. Camera IP address
2. RTSP username
3. RTSP password
4. RTSP stream path
5. RTSP port
6. Network connectivity
7. Camera RTSP configuration
8. Camera codec compatibility
9. Firewall rules

Test connectivity:

```bash
ping CAMERA_IP
```

Then test the RTSP URL with FFprobe or FFmpeg.

---

## Check Running Processes

Node.js:

```bash
ps aux | grep node
```

MediaMTX:

```bash
ps aux | grep mediamtx
```

Python worker:

```bash
ps aux | grep face_worker
```

FFmpeg:

```bash
ps aux | grep ffmpeg
```

---

# Stop the Application

If running directly in the terminal:

```text
Ctrl + C
```

The Node.js server manages MediaMTX and the Python worker.

If using PM2:

```bash
pm2 stop atomo-ai
```

---

# Update the Project

Switch to the current branch:

```bash
cd atomo-ai-mitesh
git checkout agent/recognition-on-going
git pull origin agent/recognition-on-going
```

Install/update Node.js dependencies:

```bash
cd face_recognition
npm install
```

Activate the Python environment:

```bash
source venv/bin/activate
```

Update Python dependencies if required:

```bash
pip install --upgrade numpy opencv-python-headless huggingface-hub
```

Start the application:

```bash
npm start
```

If using PM2:

```bash
pm2 restart atomo-ai
```

---

# Quick Start

For an already-installed system:

```bash
cd atomo-ai-mitesh/face_recognition
source venv/bin/activate
npm start
```

Open:

```text
http://localhost:3000
```

For another computer on the same network:

```text
http://SERVER_IP:3000
```

---

# Quick Commands

### Start

```bash
cd atomo-ai-mitesh/face_recognition
source venv/bin/activate
npm start
```

### Stop

```text
Ctrl + C
```

### Reset SQLite Database

```bash
npm run reset-db
```

### PM2 Start

```bash
pm2 start npm --name atomo-ai -- start
```

### PM2 Logs

```bash
pm2 logs atomo-ai
```

### PM2 Restart

```bash
pm2 restart atomo-ai
```

### Application

```text
http://localhost:3000
```
