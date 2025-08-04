# 🚀 QFLARE Server Deployment Guide

Deploy your QFLARE server to make it accessible from anywhere on the internet.

## 🌐 **Quick Deployment Options**

### **Option 1: Railway (Recommended - Free)**
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your QFLARE repository
5. Railway will automatically detect and deploy your app
6. Get your public URL (e.g., `https://qflare-server.railway.app`)

### **Option 2: Render (Free)**
1. Go to [render.com](https://render.com)
2. Sign up and connect your GitHub
3. Click "New" → "Web Service"
4. Connect your QFLARE repository
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `uvicorn server.main:app --host 0.0.0.0 --port $PORT`
7. Deploy and get your public URL

### **Option 3: Heroku (Free Tier)**
1. Install Heroku CLI
2. Run these commands:
```bash
heroku create your-qflare-app
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```
3. Get your public URL: `https://your-qflare-app.herokuapp.com`

## 🐳 **Docker Deployment**

### **Local Docker**
```bash
# Build and run with Docker
docker build -t qflare-server .
docker run -p 8000:8000 qflare-server

# Or use Docker Compose
docker-compose up -d
```

### **Docker on VPS**
1. Install Docker on your VPS
2. Clone your repository
3. Run: `docker-compose up -d`
4. Access at `http://your-server-ip`

## ☁️ **VPS Deployment**

### **DigitalOcean/AWS/Google Cloud**

1. **Create a VPS** (Ubuntu 20.04+ recommended)
2. **SSH into your server**
3. **Run the deployment script**:
```bash
chmod +x deploy.sh
./deploy.sh
```

4. **Configure your domain** (optional):
   - Point your domain to your VPS IP
   - Update `/etc/nginx/sites-available/qflare` with your domain
   - Restart nginx: `sudo systemctl restart nginx`

### **Manual VPS Setup**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip nginx

# Clone your repository
git clone <your-repo-url>
cd QFLARE_Project_Structure

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## 🔧 **Environment Configuration**

### **Production Environment Variables**
Create a `.env` file:
```env
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# Database Configuration
DATABASE_URL=sqlite:///./qflare.db

# Security Configuration
SECURITY_LEVEL=3
KEY_ROTATION_INTERVAL=24

# For production databases (optional)
# DATABASE_URL=postgresql://user:password@localhost/qflare
```

### **SSL/HTTPS Setup**
1. **Get SSL certificate** (Let's Encrypt):
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

2. **Or use Cloudflare** (free SSL):
   - Add your domain to Cloudflare
   - Point DNS to your server
   - Enable "Always Use HTTPS"

## 📊 **Monitoring & Maintenance**

### **Health Checks**
- **Endpoint**: `/health`
- **Expected response**: `{"status": "healthy", ...}`

### **Logs**
```bash
# View application logs
sudo journalctl -u qflare -f

# View nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### **Backup Database**
```bash
# Backup SQLite database
cp qflare.db backup_$(date +%Y%m%d).db

# Or for PostgreSQL
pg_dump qflare > backup_$(date +%Y%m%d).sql
```

## 🔒 **Security Considerations**

### **Firewall Setup**
```bash
# Allow only necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### **Rate Limiting**
The server includes built-in rate limiting:
- API endpoints: 10-60 requests per minute
- Health checks: 60 requests per minute

### **SSL/TLS**
- Always use HTTPS in production
- Configure proper SSL certificates
- Enable HSTS headers

## 🚀 **Deployment Checklist**

- [ ] Choose deployment platform
- [ ] Set up environment variables
- [ ] Configure domain (if applicable)
- [ ] Set up SSL certificate
- [ ] Configure firewall
- [ ] Test all endpoints
- [ ] Set up monitoring
- [ ] Configure backups
- [ ] Test client connections

## 📱 **Client Connection**

Once deployed, other systems can connect using your public URL:

```python
import requests

# Replace with your deployed URL
SERVER_URL = "https://your-qflare-app.railway.app"

# Health check
response = requests.get(f"{SERVER_URL}/health")
print(response.json())

# Generate token
response = requests.post(f"{SERVER_URL}/api/generate_token", 
    json={"device_id": "test-device", "expiration_hours": 24})
print(response.json())
```

## 🆘 **Troubleshooting**

### **Common Issues**

1. **Port already in use**:
   ```bash
   sudo lsof -i :8000
   sudo kill -9 <PID>
   ```

2. **Permission denied**:
   ```bash
   sudo chown -R $USER:$USER /opt/qflare
   ```

3. **Database errors**:
   ```bash
   rm qflare.db
   # Restart server
   ```

4. **Nginx errors**:
   ```bash
   sudo nginx -t
   sudo systemctl restart nginx
   ```

### **Getting Help**
- Check logs: `sudo journalctl -u qflare -f`
- Test endpoints: `curl http://localhost:8000/health`
- Check nginx: `sudo nginx -t`

## 🎯 **Next Steps**

1. **Deploy to your chosen platform**
2. **Test all functionality**
3. **Configure monitoring**
4. **Share the URL with other systems**
5. **Monitor usage and performance**

Your QFLARE server will be accessible worldwide! 🌍 