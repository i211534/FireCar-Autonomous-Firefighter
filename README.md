# FireCar

An edge-AI prototype that combines fire detection, visual obstacle mapping,
path planning, robot actuation, and a web monitoring dashboard on a Raspberry
Pi.

> [!WARNING]
> FireCar is an academic/prototype system. It is not a certified fire-detection
> or life-safety device and has not been validated for unattended operation in
> real environments.

<p align="center">
  <img
    src="https://github.com/i211534/FireCar-Autonomous-Firefighter/raw/master/image.jpeg"
    alt="FireCar autonomous firefighting robot"
    width="720"
  />
</p>

<p align="center">
  <a href="https://www.linkedin.com/posts/alyan-shahid-272a272bb_ai-robotics-computervision-activity-7327387840407887872-LPNF">
    Watch the project demo
  </a>
</p>

## Overview

FireCar explores an end-to-end perception-to-action pipeline for indoor fire
response:

1. A Raspberry Pi camera captures the environment.
2. Custom Ultralytics YOLO models detect fire, obstacles, and the robot.
3. OpenCV converts detections into a `50 × 50` occupancy grid.
4. A vehicle-aware A* planner generates a route to the detected fire.
5. Python sends movement and spray commands over UART.
6. A Flask API exposes camera, navigation, threshold, and fire-log data.
7. A Next.js dashboard provides monitoring and manual controls.

```text
Pi Camera
    │
    ▼
YOLO fire and object perception
    │
    ▼
OpenCV mask → 50×50 occupancy grid
    │
    ▼
A* path planner
    │
    ▼
UART / HC-05 → drive and spray controller
    │
    └──────── Flask API ──────── Next.js dashboard
```

## Implemented capabilities

- Fire detection with configurable confidence thresholds.
- Obstacle segmentation/detection and car localization.
- Fire-region exclusion from the obstacle map so the robot can approach the
  target.
- A* planning with obstacle inflation, boundary margins, turning penalties,
  and obstacle-clearance costs.
- Serial commands for forward, backward, left, right, stop, and spray actions.
- Dashboard pages for camera snapshots, occupancy-grid visualization,
  navigation controls, detection thresholds, and fire-event history.
- Fire-event persistence through Flask-SQLAlchemy.

## Technology stack

### Edge and robotics

- Raspberry Pi and Raspberry Pi Camera
- Python
- Picamera2
- UART serial communication at 9600 baud
- HC-05 Bluetooth serial module
- pySerial

### Computer vision and navigation

- Ultralytics YOLO
- PyTorch
- OpenCV
- NumPy
- Occupancy grids
- A* pathfinding

### Backend and frontend

- Flask and Flask-CORS
- Flask-SQLAlchemy
- Next.js 14 App Router
- React 18
- TypeScript 5
- Axios and Fetch API
- CSS Modules

## Repository structure

```text
.
├── Backend/
│   ├── src/
│   │   ├── api.py                 # Flask API and fire-log model
│   │   ├── camera.py              # Picamera2 capture
│   │   ├── fire_detection.py      # Fire-model inference
│   │   ├── object_detection.py    # Segmentation and occupancy-grid creation
│   │   ├── path.py                # Vehicle-aware A* planner
│   │   ├── main.py                # Autonomous control loop
│   │   ├── func.py                # Car position extraction
│   │   ├── carorig.py             # Serial communication experiment
│   │   └── debug_car.py           # Path visualization experiment
│   └── ultralytics-main/          # Vendored Ultralytics source
└── Frontend/
    ├── src/app/
    │   ├── camera/                # Camera and detection status
    │   ├── map/                   # Occupancy-grid and navigation UI
    │   ├── firelogs/              # Fire-event history
    │   └── components/            # Shared navigation
    └── package.json
```

## Getting started

Clone the repository:

```bash
git clone https://github.com/i211534/FireCar-Autonomous-Firefighter.git
cd FireCar-Autonomous-Firefighter
```

### Run the web dashboard

The frontend can be installed independently:

```bash
cd Frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

The layout references `src/app/fonts/GeistVF.woff` and
`src/app/fonts/GeistMonoVF.woff`, which are not currently checked in. Restore
those files or replace the local-font declarations before building the current
frontend.

The dashboard currently expects the Flask API at
`http://localhost:5001`. If the backend runs on a Raspberry Pi while the
dashboard is opened from another computer, update the API base URLs in the
frontend first.

### Backend prerequisites

The backend is intended to run on Raspberry Pi OS with:

- A supported Raspberry Pi camera configured for Picamera2.
- `/dev/serial0` enabled and connected to the robot controller.
- Python packages for Flask, Flask-CORS, Flask-SQLAlchemy, OpenCV, NumPy,
  PyTorch, Ultralytics, Picamera2, pySerial, Requests, and Matplotlib.
- Custom fire and obstacle/car YOLO model weights.
- A valid SQLAlchemy database URI.

The checked-in backend is **not yet reproducibly installable**. Before running
`Backend/src/main.py`, you must:

1. Replace the machine-specific model and label paths in `main.py`, `api.py`,
   and `camera.py`.
2. Supply the required custom `.pt` model files.
3. Configure `SQLALCHEMY_DATABASE_URI` in `api.py`.
4. Install the Python dependencies manually.
5. Verify the UART device, baud rate, and downstream command protocol.

The repository does not currently include a backend dependency lockfile,
environment template, model download process, or microcontroller firmware.

## API overview

The Flask service runs on port `5001` and exposes:

- `GET /raw_frame` — latest camera frame as JPEG.
- `GET /detect_fire` — fire detection and grid goal.
- `GET /detect_objects` — annotated object-detection image.
- `GET /get_occupancy_grid` — grid, path, goal, and car position.
- `GET /navigate_to_fire` — combined detection and planning sequence.
- `POST /set_goal` — update the manual navigation goal.
- `GET /thresholds` — current detection threshold profiles.
- `POST /thresholds/set/<name>` — select an object threshold profile.
- `POST /fire_thresholds/set/<name>` — select a fire threshold profile.
- `GET|POST /firelogs` — read or create fire-event records.

The `/api/car/start_following` and `/api/car/stop_following` endpoints currently
set an in-memory flag that is not consumed by the physical control loop.

## Current limitations

- Model files are not included and model paths are machine-specific.
- The SQLAlchemy database URI is empty.
- Python dependencies are not captured in a project-level manifest.
- The camera-derived grid is not calibrated to real-world distance.
- Motion uses timed, open-loop serial commands without odometry,
  acknowledgement, watchdog, or emergency-stop feedback.
- The repository does not include downstream microcontroller firmware,
  motor-driver wiring, or the spray-controller implementation.
- YOLO models are recreated inside inference functions, increasing latency.
- API requests and the autonomous loop can perform duplicate inference.
- Frontend API URLs are hardcoded to browser-local `localhost`.
- Actuator routes have no authentication, CORS is unrestricted, and Flask
  runs in debug mode.
- Application-level automated tests and deployment configuration are absent.

See the source before using any component in a physical environment.

## Roadmap

- [ ] Add a backend `pyproject.toml` or locked requirements file.
- [ ] Move model, database, serial, and API settings into environment-based
      configuration.
- [ ] Add a hardware-free simulation mode and recorded-frame fixtures.
- [ ] Load and warm up models once instead of per inference.
- [ ] Add planner, API, perception, and mocked serial-control tests.
- [ ] Calibrate image coordinates to a ground plane or integrate localization.
- [ ] Add command acknowledgements, a watchdog, emergency stop, and
      post-suppression verification.
- [ ] Connect dashboard start/stop controls to the physical control state.
- [ ] Upgrade the frontend to a supported Next.js release.
- [ ] Add production deployment and observability documentation.

## Contributing

Issues and pull requests are welcome. Please keep changes focused, document any
hardware assumptions, and include tests where practical. Safety-related changes
should explain failure behavior and how the robot returns to a stopped state.

## License and third-party software

This repository is shared for portfolio purposes. See [LICENSE](./LICENSE) for usage terms.

`Backend/ultralytics-main` contains vendored Ultralytics software under its own
AGPL-3.0 license. Review the Ultralytics license and the licensing of custom
model weights before redistribution or commercial deployment.

