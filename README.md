# FireCar

An edge-AI prototype that combines fire detection, visual obstacle mapping,
path planning, robot actuation, and a web monitoring dashboard on a Raspberry
Pi.

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

The dashboard currently expects the Flask API at
`http://localhost:5001`. If the backend runs on a Raspberry Pi while the
dashboard is opened from another computer, configure the frontend API base URL
for the Raspberry Pi host.

### Backend prerequisites

The backend is intended to run on Raspberry Pi OS with:

- A supported Raspberry Pi camera configured for Picamera2.
- `/dev/serial0` enabled and connected to the robot controller.
- Python packages for Flask, Flask-CORS, Flask-SQLAlchemy, OpenCV, NumPy,
  PyTorch, Ultralytics, Picamera2, pySerial, Requests, and Matplotlib.
- Custom fire and obstacle/car YOLO model weights.
- A valid SQLAlchemy database URI.

Configure the Raspberry Pi environment before running `Backend/src/main.py`:

1. Set the fire model, object model, and label paths in `main.py`, `api.py`,
   and `camera.py`.
2. Place the custom `.pt` model files at the configured locations.
3. Set `SQLALCHEMY_DATABASE_URI` in `api.py`.
4. Install the listed Python dependencies.
5. Verify the UART device, baud rate, and downstream command protocol.

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

## Future enhancements

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
hardware assumptions, and include tests where practical.

## License and third-party software

This repository is shared for portfolio purposes. See [LICENSE](./LICENSE) for usage terms.

`Backend/ultralytics-main` contains vendored Ultralytics software under its own
AGPL-3.0 license. Custom model weights and other dependencies remain subject to
their respective licenses.

