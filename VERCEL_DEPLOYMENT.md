# 🚀 Vercel Deployment Guide

Deploy your QFLARE server to Vercel for lightning-fast, global deployment!

## 🌟 **Why Vercel?**

- ✅ **Free tier available**
- ✅ **Automatic HTTPS**
- ✅ **Global CDN**
- ✅ **Serverless functions**
- ✅ **GitHub integration**
- ✅ **Instant deployments**

## 📋 **Prerequisites**

- GitHub account
- Vercel account (free at [vercel.com](https://vercel.com))

## 🚀 **Quick Deployment Steps**

### **1. Prepare Your Repository**
Your repository is already ready with:
- ✅ `vercel.json` - Vercel configuration
- ✅ `server/vercel_app.py` - Vercel-optimized app
- ✅ `requirements-vercel.txt` - Compatible dependencies

### **2. Deploy to Vercel**

#### **Option A: Vercel Dashboard (Recommended)**
1. **Go to [vercel.com](https://vercel.com)**
2. **Sign up with GitHub**
3. **Click "New Project"**
4. **Import your QFLARE repository**
5. **Vercel will automatically detect it's a Python app**
6. **Click "Deploy"**

#### **Option B: Vercel CLI**
```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy from your project directory
vercel

# Follow the prompts
```

### **3. Get Your Public URL**
After deployment, Vercel will give you a URL like:
```
https://qflare-server.vercel.app
```

## 🔧 **Configuration**

### **Vercel Configuration (`vercel.json`)**
```json
{
  "version": 2,
  "builds": [
    {
      "src": "server/vercel_app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "server/vercel_app.py"
    }
  ]
}
```

### **Environment Variables (Optional)**
You can set these in Vercel dashboard:
```env
HOST=0.0.0.0
PORT=8000
```

## 📱 **Test Your Deployment**

Once deployed, test these endpoints:

```bash
# Health check
curl https://your-app.vercel.app/health

# Get devices
curl https://your-app.vercel.app/api/devices

# Generate token
curl -X POST https://your-app.vercel.app/api/generate_token \
  -H "Content-Type: application/json" \
  -d '{"device_id": "test-device", "expiration_hours": 24}'

# Enroll device
curl -X POST https://your-app.vercel.app/api/enroll \
  -H "Content-Type: application/json" \
  -d '{"device_id": "test-device", "enrollment_token": "your-token"}'
```

## 🌐 **Access Your API**

- **Main API**: `https://your-app.vercel.app/`
- **Health Check**: `https://your-app.vercel.app/health`
- **API Docs**: `https://your-app.vercel.app/docs`
- **Devices**: `https://your-app.vercel.app/api/devices`

## 🔗 **Share with Other Systems**

Other systems can connect using your Vercel URL:

```python
import requests

SERVER_URL = "https://your-app.vercel.app"

# Health check
response = requests.get(f"{SERVER_URL}/health")
print(response.json())

# Generate token
response = requests.post(f"{SERVER_URL}/api/generate_token", 
    json={"device_id": "test-device", "expiration_hours": 24})
print(response.json())

# Enroll device
response = requests.post(f"{SERVER_URL}/api/enroll", 
    json={"device_id": "test-device", "enrollment_token": "your-token"})
print(response.json())
```

## 🔐 **Vercel-Specific Features**

### **Serverless Functions**
- ✅ **Automatic scaling**
- ✅ **Pay-per-use pricing**
- ✅ **Global edge network**
- ✅ **Cold start optimization**

### **Security**
- ✅ **Automatic HTTPS**
- ✅ **DDoS protection**
- ✅ **Security headers**
- ✅ **Rate limiting**

### **Performance**
- ✅ **Global CDN**
- ✅ **Edge caching**
- ✅ **Instant deployments**
- ✅ **Automatic rollbacks**

## 🆘 **Troubleshooting**

### **Build Fails**
- Check Vercel logs for errors
- Ensure `vercel.json` is correct
- Verify `server/vercel_app.py` exists
- Check `requirements-vercel.txt` for compatibility

### **Function Timeout**
- Vercel has a 10-second timeout for free tier
- Upgrade to Pro for longer timeouts
- Optimize your code for serverless

### **Cold Starts**
- Vercel functions have cold starts
- Consider using Vercel Pro for better performance
- Optimize imports and dependencies

## 📊 **Monitoring**

### **Vercel Dashboard**
- **Function logs**: View in Vercel dashboard
- **Performance metrics**: Built-in analytics
- **Error tracking**: Automatic error reporting
- **Deployment history**: Track all deployments

### **Health Checks**
```bash
# Check if your app is running
curl https://your-app.vercel.app/health

# Expected response:
{
  "status": "healthy",
  "message": "QFLARE Server is operational",
  "version": "1.0.0",
  "deployment": "vercel",
  "device_count": 0
}
```

## 🎯 **Next Steps**

1. **Deploy to Vercel** using the steps above
2. **Test all endpoints** to ensure they work
3. **Set up custom domain** (optional)
4. **Configure environment variables** (optional)
5. **Share your Vercel URL** with other systems

## ✅ **Success!**

Your QFLARE server is now:
- ✅ **Deployed on Vercel**
- ✅ **Globally accessible**
- ✅ **HTTPS enabled**
- ✅ **Auto-scaling**
- ✅ **High performance**

Share your Vercel URL with other systems to connect! 🌍

## 🔄 **Updates**

To update your deployment:
1. **Push changes to GitHub**
2. **Vercel automatically redeploys**
3. **Zero downtime updates**

Your QFLARE server is now running on Vercel's global infrastructure! 🚀 