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
