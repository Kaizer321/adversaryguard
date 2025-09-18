'use client';

import { useState } from 'react';
import axios from 'axios';

interface AttackResult {
  clean_prediction: {
    class: string;
    confidence: number;
    class_index: number;
  };
  adversarial_prediction: {
    class: string;
    confidence: number;
    class_index: number;
  };
  attack_parameters: {
    epsilon: number;
    attack_method: string;
  };
  attack_success: boolean;
  images: {
    original_base64: string;
    adversarial_base64: string;
  };
  metadata: {
    image_size: string;
    model_type: string;
    device: string;
  };
}

export default function FGSMDemo() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [epsilon, setEpsilon] = useState<number>(0.1);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AttackResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setError(null);
    }
  };

  const handleAttack = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('epsilon', epsilon.toString());

      // Use Next.js API rewrites to proxy backend calls
      const backendUrl = '/api';
      
      const response = await axios.post<AttackResult>(`${backendUrl}/attack`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout
      });

      setResult(response.data);
    } catch (err: any) {
      console.error('Attack failed:', err);
      if (err.response?.data?.detail) {
        setError(`Attack failed: ${err.response.data.detail}`);
      } else if (err.code === 'ECONNABORTED') {
        setError('Request timed out. Please try again.');
      } else if (err.message.includes('Network Error')) {
        setError('Unable to connect to backend. Make sure the API server is running.');
      } else {
        setError(`Attack failed: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            FGSM Adversarial Attack Demo
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Upload an image and see how the Fast Gradient Sign Method (FGSM) creates 
            adversarial examples that fool machine learning models. Adjust the epsilon 
            parameter to control the perturbation strength.
          </p>
        </div>

        {/* Main Controls */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload Image (PNG/JPEG)
              </label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-colors">
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer flex flex-col items-center"
                >
                  <svg className="w-12 h-12 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 48 48">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" />
                  </svg>
                  <span className="text-sm text-gray-600">
                    {selectedFile ? selectedFile.name : 'Click to upload image'}
                  </span>
                </label>
              </div>
            </div>

            {/* Epsilon Control */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Epsilon (Perturbation Strength): {epsilon.toFixed(3)}
              </label>
              <div className="space-y-4">
                <input
                  type="range"
                  min="0"
                  max="0.3"
                  step="0.01"
                  value={epsilon}
                  onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>0.0 (No perturbation)</span>
                  <span>0.3 (Max perturbation)</span>
                </div>
                <input
                  type="number"
                  min="0"
                  max="0.3"
                  step="0.001"
                  value={epsilon}
                  onChange={(e) => setEpsilon(parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter epsilon value"
                />
              </div>
            </div>
          </div>

          {/* Attack Button */}
          <div className="mt-6 text-center">
            <button
              onClick={handleAttack}
              disabled={!selectedFile || loading}
              className={`px-8 py-3 rounded-lg font-medium text-white transition-colors ${
                !selectedFile || loading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-red-600 hover:bg-red-700 focus:ring-2 focus:ring-red-500'
              }`}
            >
              {loading ? (
                <div className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Generating Attack...
                </div>
              ) : (
                'Launch FGSM Attack'
              )}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex">
              <svg className="w-5 h-5 text-red-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <span className="text-red-700">{error}</span>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Attack Results</h2>
            
            {/* Attack Success Status */}
            <div className={`p-4 rounded-lg mb-6 ${
              result.attack_success 
                ? 'bg-red-50 border border-red-200' 
                : 'bg-green-50 border border-green-200'
            }`}>
              <div className="flex items-center">
                {result.attack_success ? (
                  <svg className="w-6 h-6 text-red-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <svg className="w-6 h-6 text-green-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                )}
                <span className={`font-medium ${
                  result.attack_success ? 'text-red-700' : 'text-green-700'
                }`}>
                  {result.attack_success ? 'Attack Successful!' : 'Attack Failed'} 
                  {result.attack_success 
                    ? ' - Model prediction was changed' 
                    : ' - Model prediction remained the same'
                  }
                </span>
              </div>
            </div>

            {/* Images and Predictions */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              
              {/* Original Image */}
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Original Image</h3>
                <div className="bg-gray-50 rounded-lg p-4 mb-4">
                  <img
                    src={`data:image/png;base64,${result.images.original_base64}`}
                    alt="Original"
                    className="mx-auto max-w-full h-auto"
                    style={{ maxHeight: '200px' }}
                  />
                </div>
                <div className="bg-blue-50 rounded-lg p-4">
                  <p className="text-lg font-medium text-blue-900">
                    Prediction: {result.clean_prediction.class}
                  </p>
                  <p className="text-sm text-blue-700">
                    Confidence: {(result.clean_prediction.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Adversarial Image */}
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Adversarial Image</h3>
                <div className="bg-gray-50 rounded-lg p-4 mb-4">
                  <img
                    src={`data:image/png;base64,${result.images.adversarial_base64}`}
                    alt="Adversarial"
                    className="mx-auto max-w-full h-auto"
                    style={{ maxHeight: '200px' }}
                  />
                </div>
                <div className="bg-red-50 rounded-lg p-4">
                  <p className="text-lg font-medium text-red-900">
                    Prediction: {result.adversarial_prediction.class}
                  </p>
                  <p className="text-sm text-red-700">
                    Confidence: {(result.adversarial_prediction.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>

            {/* Attack Parameters */}
            <div className="mt-6 bg-gray-50 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-2">Attack Parameters</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600">
                <div>
                  <strong>Method:</strong> {result.attack_parameters.attack_method}
                </div>
                <div>
                  <strong>Epsilon:</strong> {result.attack_parameters.epsilon.toFixed(3)}
                </div>
                <div>
                  <strong>Model:</strong> {result.metadata.model_type}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Info Section */}
        <div className="mt-8 bg-blue-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">About FGSM</h3>
          <p className="text-blue-800 text-sm leading-relaxed">
            The Fast Gradient Sign Method (FGSM) is a simple but effective adversarial attack that creates 
            imperceptible perturbations to fool neural networks. It works by taking the sign of the gradient 
            of the loss function with respect to the input image and adding a small amount (epsilon) in that 
            direction. Even tiny changes invisible to humans can completely change the model's prediction.
          </p>
        </div>
      </div>
    </div>
  );
}