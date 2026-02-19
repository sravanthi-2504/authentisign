import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Search, CheckCircle, XCircle, Eye, FileText, Calendar, TrendingUp } from 'lucide-react';

const VerificationHistory = () => {
    const [history, setHistory] = useState([]);
    const [filteredHistory, setFilteredHistory] = useState([]);
    const [searchTerm, setSearchTerm] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');

    useEffect(() => {
        // Load history from localStorage
        const stored = JSON.parse(localStorage.getItem('verificationHistory') || '[]');
        setHistory(stored);
        setFilteredHistory(stored);
    }, []);

    useEffect(() => {
        // Filter history based on search and status
        let filtered = history;

        if (searchTerm) {
            filtered = filtered.filter(item =>
                item.filename.toLowerCase().includes(searchTerm.toLowerCase())
            );
        }

        if (statusFilter !== 'all') {
            filtered = filtered.filter(item => item.status.toLowerCase() === statusFilter);
        }

        setFilteredHistory(filtered);
    }, [searchTerm, statusFilter, history]);

    const getStats = () => {
        const total = history.length;
        const genuine = history.filter(h => h.status === 'GENUINE').length;
        const forged = history.filter(h => h.status === 'FORGED').length;
        const avgConfidence = history.length > 0
            ? Math.round(history.reduce((sum, h) => sum + h.confidence, 0) / history.length)
            : 0;

        return { total, genuine, forged, avgConfidence };
    };

    const stats = getStats();

    const StatCard = ({ icon: Icon, label, value, color }) => (
        <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-slate-900/50 backdrop-blur-xl rounded-xl p-6 border border-slate-800"
        >
            <div className="flex items-center gap-4">
                <div className={`p-3 rounded-xl ${color}`}>
                    <Icon className="w-6 h-6 text-white" />
                </div>
                <div>
                    <p className="text-slate-400 text-sm">{label}</p>
                    <p className="text-2xl font-bold text-white">{value}</p>
                </div>
            </div>
        </motion.div>
    );

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6 lg:p-10">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-8"
                >
                    <h1 className="text-4xl font-bold text-white mb-2">Verification History</h1>
                    <p className="text-slate-400 text-lg">View all your past signature verification results</p>
                </motion.div>

                {/* Stats Grid */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8"
                >
                    <StatCard
                        icon={FileText}
                        label="Total Verifications"
                        value={stats.total}
                        color="bg-blue-500"
                    />
                    <StatCard
                        icon={CheckCircle}
                        label="Genuine"
                        value={stats.genuine}
                        color="bg-emerald-500"
                    />
                    <StatCard
                        icon={XCircle}
                        label="Forged"
                        value={stats.forged}
                        color="bg-red-500"
                    />
                    <StatCard
                        icon={TrendingUp}
                        label="Avg Confidence"
                        value={`${stats.avgConfidence}%`}
                        color="bg-purple-500"
                    />
                </motion.div>

                {/* Filters */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="bg-slate-900/50 backdrop-blur-xl rounded-xl p-6 border border-slate-800 mb-6"
                >
                    <div className="flex flex-col md:flex-row gap-4">
                        {/* Search */}
                        <div className="flex-1 relative">
                            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-500" />
                            <input
                                type="text"
                                placeholder="Search by filename..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="w-full pl-12 pr-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 transition-all"
                            />
                        </div>

                        {/* Filter Buttons */}
                        <div className="flex gap-2">
                            {['all', 'genuine', 'forged'].map((filter) => (
                                <button
                                    key={filter}
                                    onClick={() => setStatusFilter(filter)}
                                    className={`px-6 py-3 rounded-xl font-semibold transition-all ${
                                        statusFilter === filter
                                            ? 'bg-emerald-500 text-white'
                                            : 'bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700'
                                    }`}
                                >
                                    {filter.charAt(0).toUpperCase() + filter.slice(1)}
                                </button>
                            ))}
                        </div>
                    </div>
                </motion.div>

                {/* History Table */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="bg-slate-900/50 backdrop-blur-xl rounded-xl border border-slate-800 overflow-hidden"
                >
                    {filteredHistory.length === 0 ? (
                        <div className="p-12 text-center">
                            <FileText className="w-16 h-16 text-slate-700 mx-auto mb-4" />
                            <p className="text-slate-400 text-lg">No verification history found</p>
                            <p className="text-slate-600 text-sm mt-2">
                                {searchTerm || statusFilter !== 'all'
                                    ? 'Try adjusting your filters'
                                    : 'Start by verifying some signatures'}
                            </p>
                        </div>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                <tr className="bg-slate-800/50 border-b border-slate-700">
                                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                        <div className="flex items-center gap-2">
                                            <Calendar className="w-4 h-4" />
                                            Date & Time
                                        </div>
                                    </th>
                                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                        <div className="flex items-center gap-2">
                                            <FileText className="w-4 h-4" />
                                            Filename
                                        </div>
                                    </th>
                                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                        Status
                                    </th>
                                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                        Confidence
                                    </th>
                                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                        Actions
                                    </th>
                                </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-800">
                                {filteredHistory.map((item, index) => (
                                    <motion.tr
                                        key={index}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: index * 0.05 }}
                                        className="hover:bg-slate-800/30 transition-colors"
                                    >
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <div className="flex items-center gap-2 text-slate-300">
                                                <Calendar className="w-4 h-4 text-slate-500" />
                                                {new Date(item.timestamp).toLocaleDateString('en-US', {
                                                    year: 'numeric',
                                                    month: 'short',
                                                    day: 'numeric',
                                                    hour: '2-digit',
                                                    minute: '2-digit'
                                                })}
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-2 text-slate-300">
                                                <FileText className="w-4 h-4 text-slate-500" />
                                                {item.filename}
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-semibold ${
                                                item.status === 'GENUINE'
                                                    ? 'bg-emerald-500/20 text-emerald-400'
                                                    : 'bg-red-500/20 text-red-400'
                                            }`}>
                                                {item.status === 'GENUINE' ? (
                                                    <CheckCircle className="w-4 h-4" />
                                                ) : (
                                                    <XCircle className="w-4 h-4" />
                                                )}
                                                {item.status}
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <div className="flex items-center gap-3">
                                                <div className="flex-1">
                                                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                                        <div
                                                            className={`h-full ${
                                                                item.status === 'GENUINE' ? 'bg-emerald-500' : 'bg-red-500'
                                                            }`}
                                                            style={{ width: `${item.confidence}%` }}
                                                        />
                                                    </div>
                                                </div>
                                                <span className="text-slate-300 font-semibold min-w-[3rem] text-right">
                            {item.confidence}%
                          </span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <button
                                                className="p-2 hover:bg-slate-700 rounded-lg transition-colors group"
                                                title="View Details"
                                            >
                                                <Eye className="w-5 h-5 text-slate-500 group-hover:text-emerald-400 transition-colors" />
                                            </button>
                                        </td>
                                    </motion.tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    )}

                    {/* Footer */}
                    {filteredHistory.length > 0 && (
                        <div className="px-6 py-4 bg-slate-800/30 border-t border-slate-800">
                            <p className="text-sm text-slate-400">
                                Showing <span className="text-white font-semibold">{filteredHistory.length}</span> of{' '}
                                <span className="text-white font-semibold">{history.length}</span> results
                            </p>
                        </div>
                    )}
                </motion.div>
            </div>
        </div>
    );
};

export default VerificationHistory;