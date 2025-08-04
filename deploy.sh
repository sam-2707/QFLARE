#!/bin/bash

# QFLARE Server Deployment Script
# For Ubuntu/Debian systems

echo "🚀 Deploying QFLARE Server..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv nginx

# Create application directory
sudo mkdir -p /opt/qflare
sudo chown $USER:$USER /opt/qflare

# Copy application files
cp -r . /opt/qflare/
cd /opt/qflare

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create systemd service
sudo tee /etc/systemd/system/qflare.service > /dev/null <<EOF
[Unit]
Description=QFLARE Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/qflare
Environment=PATH=/opt/qflare/venv/bin
ExecStart=/opt/qflare/venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Configure Nginx
sudo tee /etc/nginx/sites-available/qflare > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site and service
sudo ln -s /etc/nginx/sites-available/qflare /etc/nginx/sites-enabled/
sudo systemctl enable qflare
sudo systemctl start qflare
sudo systemctl restart nginx

echo "✅ QFLARE Server deployed successfully!"
echo "🌐 Access your server at: http://your-domain.com"
echo "📊 Dashboard: http://your-domain.com"
echo "🔧 API Docs: http://your-domain.com/docs" 