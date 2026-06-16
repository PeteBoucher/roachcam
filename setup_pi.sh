#!/usr/bin/env bash
set -e

echo "=== RoachCam Pi Setup ==="

# System packages (pre-compiled OpenCV is the only reliable option on Pi Zero armv6)
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y git python3-pip python3-opencv

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
elif [ -d "roachcam" ]; then
    echo "Updating existing roachcam directory..."
    cd roachcam
    git pull
    git checkout pi-camera
    cd ..
else
    echo "Cloning roachcam (pi-camera branch)..."
    git clone --branch pi-camera "$REPO_URL"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "A reboot is required for the camera to activate:"
echo "  sudo reboot"
echo ""
echo "After reboot, verify the camera works:"
echo "  libcamera-hello --timeout 2000"
echo ""
echo "Then start monitoring:"
echo "  cd roachcam && python3 main.py"
echo "  python3 main.py --process-every 3   # if CPU is struggling"
