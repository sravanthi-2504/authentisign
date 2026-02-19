import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, CheckCircle, XCircle, RefreshCw, Image as ImageIcon } from 'lucide-react';

const VerifySignature = () => {
    const [originalImage, setOriginalImage] = useState(null);
    const [testImage, setTestImage] = useState(null);
    const [originalPreview, setOriginalPreview] = useState(null);
    const [testPreview, setTestPreview] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState(null);

    const originalInputRef = useRef(null);
    const testInputRef = useRef(null);

    const handleDrop = (e, type) => {
        e.preventDefault();
        e.stopPropagation();

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            processFile(file, type);
        }
    };

    const handleFileSelect = (e, type) => {
        const file = e.target.files[0];
        if (file) {
            processFile(file, type);
        }
    };

    const processFile = (file, type) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            if (type === 'original') {
                setOriginalImage(file);
                setOriginalPreview(e.target.result);
            } else {
                setTestImage(file);
                setTestPreview(e.target.result);
            }
            // Reset result when new image is uploaded
            setResult(null);
        };
        reader.readAsDataURL(file);
    };

    const runAnalysis = async () => {
        if (!originalImage || !testImage) return;

        setIsAnalyzing(true);
        setResult(null);

        // Simulate AI analysis with FormData
        const formData = new FormData();
        formData.append('original', originalImage);
        formData.append('test', testImage);

        try {
            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 3000));

            // Simulate result - In production, replace with actual API call
            // Example: const response = await fetch('/api/verify', { method: 'POST', body: formData });
            const randomConfidence = Math.random();
            const isGenuine = randomConfidence > 0.3;

            const analysisResult = {
                status: isGenuine ? 'GENUINE' : 'FORGED',
                confidence: isGenuine ? Math.round((0.9 + Math.random() * 0.09) * 100) : Math.round((0.85 + Math.random() * 0.14) * 100),
                timestamp: new Date().toISOString(),
                filename: `signature_comparison_${Date.now()}.jpg`
            };

            setResult(analysisResult);

            // Save to history in localStorage
            const history = JSON.parse(localStorage.getItem('verificationHistory') || '[]');
            history.unshift(analysisResult);
            localStorage.setItem('verificationHistory', JSON.stringify(history));

        } catch (error) {
            console.error('Analysis failed:', error);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const resetAnalysis = () => {
        setOriginalImage(null);
        setTestImage(null);
        setOriginalPreview(null);
        setTestPreview(null);
        setResult(null);
    };

    const UploadZone = ({ type, preview, inputRef, label }) => (
        <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold text-slate-200 mb-3">{label}</h3>

            {!preview ? (
                <div
                    onDrop={(e) => handleDrop(e, type)}
                    onDragOver={(e) => e.preventDefault()}
                    className="upload-zone h-80 flex flex-col items-center justify-center cursor-pointer"
                    onClick={() => inputRef.current?.click()}
                >
                    <UploadCloud className="w-16 h-16 text-slate-600 mb-4" />
                    <p className="text-slate-400 text-center mb-2">
                        Drop your image here
                    </p>
                    <p className="text-slate-600 text-sm mb-4">or click to browse</p>
                    <button className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition-colors">
                        Select File
                    </button>
                    <input
                        ref={inputRef}
                        type="file"
                        accept="image/*"
                        onChange={(e) => handleFileSelect(e, type)}
                        className="hidden"
                    />
                </div>
            ) : (
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="relative h-80 bg-slate-800 rounded-xl overflow-hidden border border-slate-700"
                >
                    <img
                        src={preview}
                        alt={`${type} signature`}
                        className="w-full h-full object-contain"
                    />
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            if (type === 'original') {
                                setOriginalImage(null);
                                setOriginalPreview(null);
                            } else {
                                setTestImage(null);
                                setTestPreview(null);
                            }
                            setResult(null);
                        }}
                        className="absolute top-3 right-3 p-2 bg-red-500/90 hover:bg-red-600 text-white rounded-lg transition-colors"
                    >
                        <XCircle className="w-5 h-5" />
                    </button>
                </motion.div>
            )}
        </div>
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
                    <h1 className="text-4xl font-bold text-white mb-2">Signature Verification</h1>
                    <p className="text-slate-400 text-lg">Upload signatures to verify authenticity using AI-powered analysis</p>
                </motion.div>

                {/* Upload Zones */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="grid lg:grid-cols-2 gap-6 mb-6"
                >
                    <UploadZone
                        type="original"
                        preview={originalPreview}
                        inputRef={originalInputRef}
                        label="Original Reference Signature"
                    />
                    <UploadZone
                        type="test"
                        preview={testPreview}
                        inputRef={testInputRef}
                        label="Test Signature for Verification"
                    />
                </motion.div>

                {/* Action Buttons */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="flex flex-wrap gap-4 justify-center mb-8"
                >
                    <motion.button
                        onClick={runAnalysis}
                        disabled={!originalImage || !testImage || isAnalyzing}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="px-8 py-4 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-semibold rounded-xl shadow-lg shadow-emerald-500/30 hover:shadow-emerald-500/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center gap-3"
                    >
                        <RefreshCw className={`w-5 h-5 ${isAnalyzing ? 'animate-spin' : ''}`} />
                        {isAnalyzing ? 'Analyzing...' : 'Run AI Analysis'}
                    </motion.button>

                    {(originalImage || testImage) && (
                        <motion.button
                            onClick={resetAnalysis}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            className="px-8 py-4 bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold rounded-xl transition-all duration-300"
                        >
                            Reset
                        </motion.button>
                    )}
                </motion.div>

                {/* Loading State */}
                <AnimatePresence>
                    {isAnalyzing && (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                            className="bg-slate-900/50 backdrop-blur-xl rounded-2xl p-8 border border-slate-800 mb-8"
                        >
                            <div className="flex flex-col items-center">
                                <div className="relative mb-6">
                                    <div className="w-20 h-20 border-4 border-slate-700 border-t-emerald-500 rounded-full animate-spin"></div>
                                    <ImageIcon className="w-8 h-8 text-emerald-500 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
                                </div>
                                <h3 className="text-xl font-semibold text-white mb-2">Analyzing Signatures</h3>
                                <p className="text-slate-400 text-center">Our AI is comparing the signatures for authenticity...</p>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Results */}
                <AnimatePresence>
                    {result && !isAnalyzing && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 20 }}
                            className="bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800 overflow-hidden"
                        >
                            {/* Result Header */}
                            <div className={`p-6 ${
                                result.status === 'GENUINE'
                                    ? 'bg-gradient-to-r from-emerald-500/20 to-emerald-600/20 border-b border-emerald-500/30'
                                    : 'bg-gradient-to-r from-red-500/20 to-red-600/20 border-b border-red-500/30'
                            }`}>
                                <div className="flex items-center gap-4">
                                    {result.status === 'GENUINE' ? (
                                        <CheckCircle className="w-12 h-12 text-emerald-400" />
                                    ) : (
                                        <XCircle className="w-12 h-12 text-red-400" />
                                    )}
                                    <div>
                                        <h2 className={`text-3xl font-bold ${
                                            result.status === 'GENUINE' ? 'text-emerald-400' : 'text-red-400'
                                        }`}>
                                            {result.status}
                                        </h2>
                                        <p className="text-slate-400 mt-1">
                                            Analysis completed at {new Date(result.timestamp).toLocaleString()}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Result Details */}
                            <div className="p-6">
                                <div className="grid md:grid-cols-2 gap-6 mb-6">
                                    {/* Confidence Score */}
                                    <div>
                                        <h3 className="text-lg font-semibold text-slate-200 mb-4">Confidence Score</h3>
                                        <div className="flex items-center gap-4">
                                            {/* Circular Progress */}
                                            <div className="relative w-32 h-32">
                                                <svg className="transform -rotate-90 w-32 h-32">
                                                    <circle
                                                        cx="64"
                                                        cy="64"
                                                        r="56"
                                                        stroke="currentColor"
                                                        strokeWidth="8"
                                                        fill="none"
                                                        className="text-slate-700"
                                                    />
                                                    <circle
                                                        cx="64"
                                                        cy="64"
                                                        r="56"
                                                        stroke="currentColor"
                                                        strokeWidth="8"
                                                        fill="none"
                                                        strokeDasharray={`${2 * Math.PI * 56}`}
                                                        strokeDashoffset={`${2 * Math.PI * 56 * (1 - result.confidence / 100)}`}
                                                        className={result.status === 'GENUINE' ? 'text-emerald-500' : 'text-red-500'}
                                                        strokeLinecap="round"
                                                    />
                                                </svg>
                                                <div className="absolute inset-0 flex items-center justify-center">
                                                    <span className="text-2xl font-bold text-white">{result.confidence}%</span>
                                                </div>
                                            </div>
                                            <div>
                                                <p className="text-slate-400 text-sm mb-2">
                                                    {result.status === 'GENUINE'
                                                        ? 'High probability the signatures match'
                                                        : `${result.confidence}% probability of forgery detected`
                                                    }
                                                </p>
                                                <div className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
                                                    result.confidence >= 95
                                                        ? 'bg-emerald-500/20 text-emerald-400'
                                                        : result.confidence >= 85
                                                            ? 'bg-yellow-500/20 text-yellow-400'
                                                            : 'bg-red-500/20 text-red-400'
                                                }`}>
                                                    {result.confidence >= 95 ? 'Very High Confidence' :
                                                        result.confidence >= 85 ? 'High Confidence' : 'Moderate Confidence'}
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Comparison View */}
                                    <div>
                                        <h3 className="text-lg font-semibold text-slate-200 mb-4">Side-by-Side Comparison</h3>
                                        <div className="grid grid-cols-2 gap-2">
                                            <div className="aspect-square bg-slate-800 rounded-lg overflow-hidden border border-slate-700">
                                                <img src={originalPreview} alt="Original" className="w-full h-full object-contain" />
                                            </div>
                                            <div className="aspect-square bg-slate-800 rounded-lg overflow-hidden border border-slate-700">
                                                <img src={testPreview} alt="Test" className="w-full h-full object-contain" />
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Additional Info */}
                                <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                                    <h4 className="text-sm font-semibold text-slate-300 mb-2">Analysis Details</h4>
                                    <ul className="text-sm text-slate-400 space-y-1">
                                        <li>• Filename: {result.filename}</li>
                                        <li>• Algorithm: Deep Learning CNN Model</li>
                                        <li>• Processing Time: ~3 seconds</li>
                                        <li>• Features Analyzed: Stroke patterns, pressure points, signature flow</li>
                                    </ul>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};

export default VerifySignature;