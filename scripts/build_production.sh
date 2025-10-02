#!/bin/bash
echo "🏗️  Building QFLARE Production Environment..."

echo ""
echo "📦 Building Frontend..."
cd "$(dirname "$0")/../frontend/qflare-ui"
npm run build
if [ $? -ne 0 ]; then
    echo "❌ Frontend build failed!"
    exit 1
fi

echo ""
echo "🐍 Installing Python Dependencies..."
cd "$(dirname "$0")/../server"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Python dependencies installation failed!"
    exit 1
fi

echo ""
echo "🐳 Building Docker Images..."
cd "$(dirname "$0")/.."
docker-compose -f docker-compose.prod.yml build
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo ""
echo "✅ Production build completed successfully!"
echo "🚀 Ready to deploy QFLARE!"
echo ""
echo "Next steps:"
echo "1. Run: docker-compose -f docker-compose.prod.yml up -d"
echo "2. Access: http://localhost:4000 (Frontend)"
echo "3. Access: http://localhost:8000 (Backend API)"
echo ""