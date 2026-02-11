import React, { useState, useRef } from 'react';
import { Upload, RefreshCw, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react';

const VerifySignature = () => {
    const [originalImage, setOriginalImage] = useState(null);
    const [testImage, setTestImage] = useState(null);
    const [originalPreview, setOriginalPreview] = useState(null);
    const [testPreview, setTestPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const originalInputRef = useRef(null);
    const testInputRef = useRef(null);

    const handleFileSelect = (file, type) => {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            setError('Please select a valid image file');
            return;
        }

        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
            if (type === 'original') {
                setOriginalImage(file);
                setOriginalPreview(e.target.result);
            } else {
                setTestImage(file);
                setTestPreview(e.target.result);
            }
            setError('');
            setResult(null);
        };
        reader.readAsDataURL(file);
    };

    const handleDrop = (e, type) => {
        e.preventDefault();
        e.stopPropagation();

        const file = e.dataTransfer.files[0];
        handleFileSelect(file, type);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleAnalyze = async () => {
        if (!originalImage || !testImage) {
            setError('Please upload both signatures');
            return;
        }

        setLoading(true);
        setError('');
        setResult(null);

        const formData = new FormData();
        formData.append('original', originalImage);
        formData.append('test', testImage);

        try {
            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:5000/api/verify-signature', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                setResult(data);
            } else {
                setError(data.error || 'Verification failed');
            }
        } catch (error) {
            setError('Network error. Please check if the backend is running.');
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setOriginalImage(null);
        setTestImage(null);
        setOriginalPreview(null);
        setTestPreview(null);
        setResult(null);
        setError('');
    };

    const getConfidenceColor = (confidence) => {
        if (confidence >= 90) return '#10b981';
        if (confidence >= 75) return '#f59e0b';
        return '#ef4444';
    };

    return (
        <div className="verify-signature">
            <div className="page-header">
                <h1>Signature Verification</h1>
                <p>Upload signatures to verify authenticity using AI-powered analysis</p>
            </div>

            <div className="upload-section">
                <div className="upload-grid">
                    {/* Original Signature Upload */}
                    <div className="upload-card">
                        <h3>Original Reference Signature</h3>
                        <div
                            className={`upload-zone ${originalPreview ? 'has-image' : ''}`}
                            onDrop={(e) => handleDrop(e, 'original')}
                            onDragOver={handleDragOver}
                            onClick={() => originalInputRef.current?.click()}
                        >
                            {originalPreview ? (
                                <div className="image-preview">
                                    <img src={originalPreview} alt="Original signature" />
                                    <div className="image-overlay">
                                        <Upload size={24} />
                                        <span>Click to change</span>
                                    </div>
                                </div>
                            ) : (
                                <div className="upload-placeholder">
                                    <Upload size={48} />
                                    <p>Drop your image here</p>
                                    <span>or click to browse</span>
                                    <button className="select-file-btn">Select File</button>
                                </div>
                            )}
                        </div>
                        <input
                            ref={originalInputRef}
                            type="file"
                            accept="image/*"
                            onChange={(e) => handleFileSelect(e.target.files[0], 'original')}
                            style={{ display: 'none' }}
                        />
                    </div>

                    {/* Test Signature Upload */}
                    <div className="upload-card">
                        <h3>Test Signature for Verification</h3>
                        <div
                            className={`upload-zone ${testPreview ? 'has-image' : ''}`}
                            onDrop={(e) => handleDrop(e, 'test')}
                            onDragOver={handleDragOver}
                            onClick={() => testInputRef.current?.click()}
                        >
                            {testPreview ? (
                                <div className="image-preview">
                                    <img src={testPreview} alt="Test signature" />
                                    <div className="image-overlay">
                                        <Upload size={24} />
                                        <span>Click to change</span>
                                    </div>
                                </div>
                            ) : (
                                <div className="upload-placeholder">
                                    <Upload size={48} />
                                    <p>Drop your image here</p>
                                    <span>or click to browse</span>
                                    <button className="select-file-btn">Select File</button>
                                </div>
                            )}
                        </div>
                        <input
                            ref={testInputRef}
                            type="file"
                            accept="image/*"
                            onChange={(e) => handleFileSelect(e.target.files[0], 'test')}
                            style={{ display: 'none' }}
                        />
                    </div>
                </div>

                {error && (
                    <div className="error-message">
                        <AlertTriangle size={20} />
                        <span>{error}</span>
                    </div>
                )}

                <div className="action-buttons">
                    <button
                        className="analyze-btn"
                        onClick={handleAnalyze}
                        disabled={!originalImage || !testImage || loading}
                    >
                        {loading ? (
                            <>
                                <div className="btn-spinner"></div>
                                <span>Analyzing...</span>
                            </>
                        ) : (
                            <>
                                <RefreshCw size={20} />
                                <span>Run AI Analysis</span>
                            </>
                        )}
                    </button>

                    {(originalImage || testImage) && (
                        <button className="reset-btn" onClick={handleReset}>
                            Reset
                        </button>
                    )}
                </div>
            </div>

            {/* Results Section */}
            {result && (
                <div className="result-section">
                    <div className={`result-card ${result.status.toLowerCase()}`}>
                        <div className="result-header">
                            {result.status === 'GENUINE' ? (
                                <CheckCircle2 size={64} className="result-icon genuine" />
                            ) : (
                                <XCircle size={64} className="result-icon forged" />
                            )}
                            <h2 className="result-status">{result.status}</h2>
                        </div>

                        <div className="result-details">
                            <div className="confidence-display">
                                <div className="confidence-circle">
                                    <svg viewBox="0 0 200 200">
                                        <circle
                                            cx="100"
                                            cy="100"
                                            r="90"
                                            fill="none"
                                            stroke="#1e293b"
                                            strokeWidth="20"
                                        />
                                        <circle
                                            cx="100"
                                            cy="100"
                                            r="90"
                                            fill="none"
                                            stroke={getConfidenceColor(result.confidence)}
                                            strokeWidth="20"
                                            strokeDasharray={`${result.confidence * 5.65} 565`}
                                            strokeLinecap="round"
                                            transform="rotate(-90 100 100)"
                                        />
                                    </svg>
                                    <div className="confidence-text">
                                        <span className="confidence-value">{result.confidence}%</span>
                                        <span className="confidence-label">Confidence</span>
                                    </div>
                                </div>
                            </div>

                            <div className="probability-details">
                                <div className="probability-item">
                                    <span className="probability-label">Genuine Probability</span>
                                    <div className="probability-bar">
                                        <div
                                            className="probability-fill genuine"
                                            style={{ width: `${result.genuine_probability}%` }}
                                        ></div>
                                    </div>
                                    <span className="probability-value">{result.genuine_probability}%</span>
                                </div>

                                <div className="probability-item">
                                    <span className="probability-label">Forgery Probability</span>
                                    <div className="probability-bar">
                                        <div
                                            className="probability-fill forged"
                                            style={{ width: `${result.forged_probability}%` }}
                                        ></div>
                                    </div>
                                    <span className="probability-value">{result.forged_probability}%</span>
                                </div>
                            </div>
                        </div>

                        <div className="result-comparison">
                            <div className="comparison-image">
                                <h4>Original</h4>
                                <img src={originalPreview} alt="Original" />
                            </div>
                            <div className="comparison-image">
                                <h4>Test</h4>
                                <img src={testPreview} alt="Test" />
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default VerifySignature;