# 🚀 Railway Deployment Guide

Deploy your QFLARE server to Railway in 5 minutes!

## 📋 **Prerequisites**
- GitHub account
- Railway account (free at [railway.app](https://railway.app))

## 🚀 **Quick Deployment Steps**

### **1. Prepare Your Repository**
Your repository is already ready with:
- ✅ `Procfile` - Tells Railway how to start the app
- ✅ `requirements.txt` - Python dependencies
- ✅ `server/start.py` - Simple start script

### **2. Deploy to Railway**

1. **Go to Railway**: Visit [railway.app](https://railway.app)
2. **Sign up**: Use your GitHub account
3. **Create New Project**: Click "New Project"
4. **Deploy from GitHub**: Select "Deploy from GitHub repo"
5. **Select Repository**: Choose your QFLARE repository
6. **Wait for Deployment**: Railway will automatically:
   - Install dependencies
   - Start your server
   - Provide a public URL

### **3. Get Your Public URL**
After deployment, Railway will give you a URL like:
```
https://qflare-server-production.up.railway.app
```

## 🔧 **Environment Variables (Optional)**

You can set these in Railway dashboard:

```env
HOST=0.0.0.0
PORT=8000
DATABASE_URL=sqlite:///./qflare.db
```

## 📱 **Test Your Deployment**

Once deployed, test these endpoints:

```bash
# Health check
curl https://your-app.railway.app/health

# Get devices
curl https://your-app.railway.app/api/devices

# Generate token
curl -X POST https://your-app.railway.app/api/generate_token \
  -H "Content-Type: application/json" \
  -d '{"device_id": "test-device", "expiration_hours": 24}'
```

## 🌐 **Access Your Dashboard**

- **Dashboard**: `https://your-app.railway.app/`
- **API Docs**: `https://your-app.railway.app/docs`
- **Health Check**: `https://your-app.railway.app/health`

## 🔗 **Share with Other Systems**

Other systems can now connect using your Railway URL:

```python
import requests

SERVER_URL = "https://your-app.railway.app"

# Health check
response = requests.get(f"{SERVER_URL}/health")
print(response.json())

# Generate token
response = requests.post(f"{SERVER_URL}/api/generate_token", 
    json={"device_id": "test-device", "expiration_hours": 24})
print(response.json())
```

## 🆘 **Troubleshooting**

### **Build Fails**
- Check Railway logs for errors
- Ensure all files are committed to GitHub
- Verify `requirements.txt` is correct

### **App Won't Start**
- Check the `Procfile` is correct
- Verify `server/start.py` exists
- Check environment variables

### **Database Issues**
- Railway provides persistent storage
- SQLite database will be created automatically

## ✅ **Success!**

Your QFLARE server is now:
- ✅ **Publicly accessible** from anywhere
- ✅ **HTTPS enabled** automatically
- ✅ **Always online** with Railway's infrastructure
- ✅ **Scalable** and reliable

Share your Railway URL with other systems to connect! 🌍 