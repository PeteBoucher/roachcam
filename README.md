# RoachCam — Motion-sensitive CCTV using your MacBook camera

Quick script to capture snapshots (and optional short clips) when motion is detected. Tuned for small pests like roaches by allowing a low area threshold.

Setup

- Install Python 3.8+ and create a virtualenv (recommended).
- Install dependencies:

```bash
pip install -r requirements.txt
```

- On macOS, grant camera access in System Settings → Privacy & Security → Camera.

Run

```bash
python main.py --min-area 100 --cooldown 2 --save-dir captures --record-seconds 3
```

- `--min-area`: lower values make detection more sensitive to small motion (try 50–200 for insects).
- `--cooldown`: seconds between saved captures to avoid many duplicates.
- `--display`: add to see live window for tuning (press `q` to quit).
- `--record-seconds`: if >0, records a short MP4 clip when motion is detected.

Outputs

- Snapshots and clips are saved under the directory given by `--save-dir` (default `captures`).

Running continuously

For persistent monitoring you can run it in background with `nohup` or create a `launchd` service. Example:

```bash
nohup python main.py --min-area 120 --cooldown 3 --save-dir /path/to/store &
```

Notes

- This script uses the default camera (`0`). If you have multiple cameras, change the index in the source.
- Motion detection is done via a running-average background model; tune `--min-area` and `--cooldown` for your environment.
- Keep the laptop positioned and lit so the camera can see entry points; avoid direct strong backlight that may cause false triggers.

If you want, I can add a `launchd` plist and example systemd/cron instructions for background running.
