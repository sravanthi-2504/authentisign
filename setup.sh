#!/bin/bash

# AI Signature Verification System - Setup Script
# This script automates the installation process

set -e

echo "=========================================="
echo "  AI Signature Verification System"
echo "  Automated Setup Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found${NC}"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js is not installed${NC}"
    echo "Please install Node.js 16 or higher"
    exit 1
fi

echo -e "${GREEN}✓ Node.js found${NC}"
echo ""

# Setup Backend
echo "=========================================="
echo "Setting up Backend..."
echo "=========================================="
cd backend

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✓ Backend setup complete${NC}"
echo ""

# Train Model
echo "=========================================="
echo "Checking ML Model..."
echo "=========================================="
cd ../model

if [ -f "signature_embedding_model.keras" ]; then
    echo -e "${GREEN}✓ Pre-trained model found — skipping training${NC}"
else
    echo -e "${YELLOW}No trained model found. Training now (15-30 min)...${NC}"
    python train_model.py
    echo -e "${GREEN}✓ Model training complete${NC}"
fi
echo ""

# Setup Frontend
echo "=========================================="
echo "Setting up Frontend..."
echo "=========================================="
cd ../frontend

echo "Installing npm dependencies..."
npm install

echo -e "${GREEN}✓ Frontend setup complete${NC}"
echo ""

# Create startup scripts
echo "=========================================="
echo "Creating startup scripts..."
echo "=========================================="

cd ..

# Backend startup script
cat > start-backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
python app.py
EOF

chmod +x start-backend.sh

# Frontend startup script
cat > start-frontend.sh << 'EOF'
#!/bin/bash
cd frontend
npm run dev
EOF

chmod +x start-frontend.sh

# Combined startup script
cat > start-all.sh << 'EOF'
#!/bin/bash

# Start backend in background
echo "Starting Backend..."
gnome-terminal -- bash -c "cd backend && source venv/bin/activate && python app.py; exec bash" 2>/dev/null || \
xterm -e "cd backend && source venv/bin/activate && python app.py" 2>/dev/null || \
osascript -e 'tell app "Terminal" to do script "cd '$(pwd)'/backend && source venv/bin/activate && python app.py"' 2>/dev/null || \
echo "Please start backend manually: ./start-backend.sh"

sleep 3

# Start frontend in background
echo "Starting Frontend..."
gnome-terminal -- bash -c "cd frontend && npm run dev; exec bash" 2>/dev/null || \
xterm -e "cd frontend && npm run dev" 2>/dev/null || \
osascript -e 'tell app "Terminal" to do script "cd '$(pwd)'/frontend && npm run dev"' 2>/dev/null || \
echo "Please start frontend manually: ./start-frontend.sh"

echo ""
echo "=========================================="
echo "  Application Starting..."
echo "=========================================="
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo "=========================================="
EOF

chmod +x start-all.sh

echo -e "${GREEN}✓ Startup scripts created${NC}"
echo ""

# Final instructions
echo "=========================================="
echo "  Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "To start the application:"
echo ""
echo "Option 1 - Start everything at once:"
echo "  ${GREEN}./start-all.sh${NC}"
echo ""
echo "Option 2 - Start separately:"
echo "  Terminal 1: ${GREEN}./start-backend.sh${NC}"
echo "  Terminal 2: ${GREEN}./start-frontend.sh${NC}"
echo ""
echo "Access the application:"
echo "  🌐 http://localhost:3000"
echo ""
echo "Default Login Credentials:"
echo "  📧 Email: bsonakshi@gmail.com"
echo "  🔑 Password: password123"
echo ""
echo "=========================================="