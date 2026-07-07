#!/usr/bin/env bash
set -e

echo "=== RoachCam Pi Setup ==="

# System packages (pre-compiled OpenCV is the only reliable option on Pi Zero armv6)
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y git python3-pip python3-numpy python3-opencv libcamera-apps

# picamera2 is pre-installed on Bullseye/Bookworm — fall back to apt if missing
if python3 -c "import picamera2" 2>/dev/null; then
    echo "picamera2 already available."
else
    echo "Installing picamera2..."
    sudo apt-get install -y python3-picamera2
fi

# Enable the camera interface (required on Pi Zero v1.3)
echo "Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Clone or update the repo
REPO_URL="https://github.com/PeteBoucher/roachcam"
if [ -f "main.py" ]; then
    echo "Already in roachcam directory — pulling latest..."
    git pull
    git checkout pi-camera
    REPO_DIR="$(pwd)"
elif [ -d "roachcam" ]; then
    echo "Updating existing roachcam directory..."
    cd roachcam
    git pull
    git checkout pi-camera
    REPO_DIR="$(pwd)"
    cd ..
else
    echo "Cloning roachcam (pi-camera branch)..."
    git clone --branch pi-camera "$REPO_URL"
    REPO_DIR="$(pwd)/roachcam"
fi

# Install systemd service
WHOAMI="$(whoami)"
echo "Installing systemd service (running as $WHOAMI)..."
sudo tee /etc/systemd/system/roachcam.service > /dev/null << UNIT
[Unit]
Description=RoachCam motion capture
After=network.target

[Service]
User=$WHOAMI
WorkingDirectory=$REPO_DIR
ExecStart=/usr/bin/python3 $REPO_DIR/main.py --rotate 180 --stream-port 8080
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable roachcam
echo "Service installed and enabled."

echo ""
echo "=== Setup complete! ==="
echo ""
echo "A reboot is required for the camera to activate:"
echo "  sudo reboot"
echo ""
echo "After reboot the service starts automatically. Useful commands:"
echo "  sudo systemctl status roachcam    # check it's running"
echo "  sudo journalctl -u roachcam -f    # live logs"
echo "  sudo systemctl restart roachcam   # restart after config changes"
echo ""
echo "Edit /etc/systemd/system/roachcam.service to change flags (e.g. --rotate 180),"
echo "then run: sudo systemctl daemon-reload && sudo systemctl restart roachcam"
