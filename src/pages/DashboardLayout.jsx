import React from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ShieldCheck, History, LogOut, Menu, X } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const DashboardLayout = () => {
    const [sidebarOpen, setSidebarOpen] = React.useState(false);
    const { user, logout } = useAuth();
    const navigate = useNavigate();

    const handleLogout = () => {
        logout();
        navigate('/');
    };

    const navItems = [
        { path: '/dashboard/verify', icon: ShieldCheck, label: 'Verify Signature' },
        { path: '/dashboard/history', icon: History, label: 'Verification History' }
    ];

    return (
        <div className="min-h-screen bg-slate-950 flex">
            {/* Mobile Sidebar Toggle */}
            <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-slate-800 rounded-lg text-slate-300 hover:text-white"
            >
                {sidebarOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>

            {/* Sidebar */}
            <motion.aside
                initial={{ x: -280 }}
                animate={{ x: 0 }}
                className={`
          fixed lg:static inset-y-0 left-0 z-40 w-72 bg-slate-900 border-r border-slate-800
          transform transition-transform duration-300 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
            >
                <div className="flex flex-col h-full">
                    {/* Logo Section */}
                    <div className="p-6 border-b border-slate-800">
                        <div className="flex items-center gap-3">
                            <div className="bg-gradient-to-br from-emerald-500 to-emerald-600 p-2 rounded-xl">
                                <ShieldCheck className="w-6 h-6 text-white" />
                            </div>
                            <div>
                                <h1 className="text-xl font-bold text-white">AI Signature</h1>
                                <p className="text-xs text-slate-400">Verification System</p>
                            </div>
                        </div>
                    </div>

                    {/* Navigation */}
                    <nav className="flex-1 p-4 space-y-2">
                        {navItems.map((item) => (
                            <NavLink
                                key={item.path}
                                to={item.path}
                                onClick={() => setSidebarOpen(false)}
                                className={({ isActive }) =>
                                    `flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 ${
                                        isActive
                                            ? 'bg-emerald-500/10 text-emerald-400 font-semibold'
                                            : 'text-slate-400 hover:text-white hover:bg-slate-800'
                                    }`
                                }
                            >
                                {({ isActive }) => (
                                    <>
                                        <item.icon className={`w-5 h-5 ${isActive ? 'text-emerald-400' : ''}`} />
                                        <span>{item.label}</span>
                                    </>
                                )}
                            </NavLink>
                        ))}
                    </nav>

                    {/* User Section */}
                    <div className="p-4 border-t border-slate-800">
                        <div className="flex items-center gap-3 mb-4 p-3 bg-slate-800/50 rounded-xl">
                            <div className="w-10 h-10 bg-emerald-500 rounded-full flex items-center justify-center text-white font-bold">
                                {user?.name?.charAt(0).toUpperCase() || 'U'}
                            </div>
                            <div className="flex-1 min-w-0">
                                <p className="text-sm font-semibold text-white truncate">
                                    {user?.name || 'User'}
                                </p>
                                <p className="text-xs text-slate-400 truncate">{user?.email || ''}</p>
                            </div>
                        </div>

                        <motion.button
                            onClick={handleLogout}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            className="flex items-center gap-3 w-full px-4 py-3 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-xl transition-all"
                        >
                            <LogOut className="w-5 h-5" />
                            <span className="font-medium">Logout</span>
                        </motion.button>
                    </div>
                </div>
            </motion.aside>

            {/* Overlay for mobile */}
            {sidebarOpen && (
                <div
                    className="lg:hidden fixed inset-0 bg-black/50 z-30"
                    onClick={() => setSidebarOpen(false)}
                />
            )}

            {/* Main Content */}
            <main className="flex-1 overflow-auto">
                <Outlet />
            </main>
        </div>
    );
};

export default DashboardLayout;