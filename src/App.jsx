import React, { useState, useEffect } from 'react';
import { Shield, History, LogOut, CheckCircle2, AlertTriangle } from 'lucide-react';
import AuthPage from './pages/AuthPage';
import VerifySignature from "./pages/VerifySignature";
import VerificationHistory from './pages/VerificationHistory';
import './App.css';

function App() {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [currentPage, setCurrentPage] = useState('verify');
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Check if user is already logged in
        const token = localStorage.getItem('token');
        if (token) {
            verifyToken(token);
        } else {
            setLoading(false);
        }
    }, []);

    const verifyToken = async (token) => {
        try {
            const response = await fetch('http://localhost:5000/api/auth/verify', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                const data = await response.json();
                setUser(data.user);
                setIsAuthenticated(true);
            } else {
                localStorage.removeItem('token');
            }
        } catch (error) {
            console.error('Token verification failed:', error);
            localStorage.removeItem('token');
        } finally {
            setLoading(false);
        }
    };

    const handleLogin = (token, userData) => {
        localStorage.setItem('token', token);
        setUser(userData);
        setIsAuthenticated(true);
    };

    const handleLogout = () => {
        localStorage.removeItem('token');
        setUser(null);
        setIsAuthenticated(false);
        setCurrentPage('verify');
    };

    if (loading) {
        return (
            <div className="loading-screen">
                <div className="loading-spinner"></div>
                <p>Loading...</p>
            </div>
        );
    }

    if (!isAuthenticated) {
        return <AuthPage onLogin={handleLogin} />;
    }

    return (
        <div className="app-container">
            {/* Sidebar */}
            <aside className="sidebar">
                <div className="sidebar-header">
                    <Shield className="logo-icon" />
                    <div className="logo-text">
                        <h1>AI Signature</h1>
                        <p>Verification System</p>
                    </div>
                </div>

                <nav className="sidebar-nav">
                    <button
                        className={`nav-item ${currentPage === 'verify' ? 'active' : ''}`}
                        onClick={() => setCurrentPage('verify')}
                    >
                        <CheckCircle2 size={20} />
                        <span>Verify Signature</span>
                    </button>

                    <button
                        className={`nav-item ${currentPage === 'history' ? 'active' : ''}`}
                        onClick={() => setCurrentPage('history')}
                    >
                        <History size={20} />
                        <span>Verification History</span>
                    </button>
                </nav>

                <div className="sidebar-footer">
                    <div className="user-info">
                        <div className="user-avatar">
                            {user?.name?.charAt(0).toUpperCase() || 'U'}
                        </div>
                        <div className="user-details">
                            <p className="user-name">{user?.name || 'User'}</p>
                            <p className="user-email">{user?.email || ''}</p>
                        </div>
                    </div>
                    <button className="logout-btn" onClick={handleLogout}>
                        <LogOut size={18} />
                        <span>Logout</span>
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <main className="main-content">
                {currentPage === 'verify' && <VerifySignature />}
                {currentPage === 'history' && <VerificationHistory />}
            </main>
        </div>
    );
}

export default App;