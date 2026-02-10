import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import AuthPage from './pages/AuthPage';
import DashboardLayout from './pages/DashboardLayout';
import VerifySignature from './pages/VerifySignature';
import VerificationHistory from './pages/VerificationHistory';

function App() {
    return (
        <AuthProvider>
            <Router>
                <Routes>
                    {/* Public Route */}
                    <Route path="/" element={<AuthPage />} />

                    {/* Protected Routes */}
                    <Route
                        path="/dashboard"
                        element={
                            <ProtectedRoute>
                                <DashboardLayout />
                            </ProtectedRoute>
                        }
                    >
                        {/* Default redirect to verify page */}
                        <Route index element={<Navigate to="/dashboard/verify" replace />} />
                        <Route path="verify" element={<VerifySignature />} />
                        <Route path="history" element={<VerificationHistory />} />
                    </Route>

                    {/* Catch all - redirect to home */}
                    <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
            </Router>
        </AuthProvider>
    );
}

export default App;