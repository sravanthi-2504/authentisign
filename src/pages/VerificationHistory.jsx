import React, { useState, useEffect } from 'react';
import { Search, Filter, Eye, Trash2, CheckCircle2, XCircle } from 'lucide-react';

const VerificationHistory = () => {
    const [history, setHistory] = useState([]);
    const [filteredHistory, setFilteredHistory] = useState([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [filterStatus, setFilterStatus] = useState('All');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        fetchHistory();
    }, []);

    useEffect(() => {
        filterHistoryData();
    }, [history, searchQuery, filterStatus]);

    const fetchHistory = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:5000/api/history', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                const data = await response.json();
                setHistory(data.history);
                setFilteredHistory(data.history);
            } else {
                setError('Failed to load history');
            }
        } catch (error) {
            setError('Network error. Please check if the backend is running.');
        } finally {
            setLoading(false);
        }
    };

    const filterHistoryData = () => {
        let filtered = [...history];

        // Filter by status
        if (filterStatus !== 'All') {
            filtered = filtered.filter(item => item.status === filterStatus);
        }

        // Filter by search query
        if (searchQuery) {
            filtered = filtered.filter(item =>
                item.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
                item.date.toLowerCase().includes(searchQuery.toLowerCase())
            );
        }

        setFilteredHistory(filtered);
    };

    const handleDelete = async (id) => {
        if (!window.confirm('Are you sure you want to delete this entry?')) {
            return;
        }

        try {
            const token = localStorage.getItem('token');
            const response = await fetch(`http://localhost:5000/api/history/${id}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (response.ok) {
                setHistory(history.filter(item => item.id !== id));
            }
        } catch (error) {
            console.error('Delete failed:', error);
        }
    };

    if (loading) {
        return (
            <div className="history-page">
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <p>Loading history...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="history-page">
            <div className="page-header">
                <h1>Verification History</h1>
                <p>View all your past signature verification results</p>
            </div>

            <div className="history-controls">
                <div className="search-bar">
                    <Search size={20} />
                    <input
                        type="text"
                        placeholder="Search by filename..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                    />
                </div>

                <div className="filter-buttons">
                    <button
                        className={`filter-btn ${filterStatus === 'All' ? 'active' : ''}`}
                        onClick={() => setFilterStatus('All')}
                    >
                        All
                    </button>
                    <button
                        className={`filter-btn genuine ${filterStatus === 'GENUINE' ? 'active' : ''}`}
                        onClick={() => setFilterStatus('GENUINE')}
                    >
                        Genuine
                    </button>
                    <button
                        className={`filter-btn forged ${filterStatus === 'FORGED' ? 'active' : ''}`}
                        onClick={() => setFilterStatus('FORGED')}
                    >
                        Forged
                    </button>
                </div>
            </div>

            {error && (
                <div className="error-message">
                    {error}
                </div>
            )}

            {filteredHistory.length === 0 ? (
                <div className="empty-state">
                    <Filter size={64} />
                    <h3>No verification history found</h3>
                    <p>
                        {searchQuery || filterStatus !== 'All'
                            ? 'Try adjusting your filters'
                            : 'Start verifying signatures to see your history here'}
                    </p>
                </div>
            ) : (
                <div className="history-table-container">
                    <table className="history-table">
                        <thead>
                        <tr>
                            <th>DATE & TIME</th>
                            <th>FILENAME</th>
                            <th>STATUS</th>
                            <th>CONFIDENCE</th>
                            <th>ACTIONS</th>
                        </tr>
                        </thead>
                        <tbody>
                        {filteredHistory.map((item) => (
                            <tr key={item.id}>
                                <td>
                                    <div className="date-cell">
                                        <span className="date-icon">📅</span>
                                        <span>{item.date}</span>
                                    </div>
                                </td>
                                <td>
                                    <div className="filename-cell">
                                        <span className="file-icon">📄</span>
                                        <span>{item.filename}</span>
                                    </div>
                                </td>
                                <td>
                                    <div className={`status-badge ${item.status.toLowerCase()}`}>
                                        {item.status === 'GENUINE' ? (
                                            <CheckCircle2 size={16} />
                                        ) : (
                                            <XCircle size={16} />
                                        )}
                                        <span>{item.status}</span>
                                    </div>
                                </td>
                                <td>
                                    <div className="confidence-cell">
                                        <div className="confidence-bar-small">
                                            <div
                                                className={`confidence-fill-small ${item.status.toLowerCase()}`}
                                                style={{ width: `${item.confidence}%` }}
                                            ></div>
                                        </div>
                                        <span className="confidence-text">{item.confidence}%</span>
                                    </div>
                                </td>
                                <td>
                                    <div className="action-buttons">
                                        <button
                                            className="action-btn view"
                                            title="View details"
                                        >
                                            <Eye size={16} />
                                        </button>
                                        <button
                                            className="action-btn delete"
                                            onClick={() => handleDelete(item.id)}
                                            title="Delete"
                                        >
                                            <Trash2 size={16} />
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        ))}
                        </tbody>
                    </table>
                </div>
            )}

            <div className="history-footer">
                <p>Showing {filteredHistory.length} of {history.length} results</p>
            </div>
        </div>
    );
};

export default VerificationHistory
