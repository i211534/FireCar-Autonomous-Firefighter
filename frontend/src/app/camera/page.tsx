'use client'; 
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Navigation from '../components/Navigation';
import styles from './Camera.module.css';

interface Threshold {
  name: string;
  value: number;
}

const CameraFeedPage = () => {
  const [alertStatus, setAlertStatus] = useState<'all-clear' | 'alert'>('all-clear');
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [thresholds, setThresholds] = useState<Threshold[]>([]);
  const [currentThreshold, setCurrentThreshold] = useState<string>('normal');
  const [customThreshold, setCustomThreshold] = useState<string>('0.75');
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Function to fetch available thresholds
  const fetchThresholds = async () => {
    try {
      const response = await axios.get('http://localhost:5001/thresholds');
      // Fix: Properly extract thresholds from the response structure
      if (response.data && response.data.object_detection) {
        setThresholds(response.data.object_detection.available || []);
        setCurrentThreshold(response.data.object_detection.current || 'normal');
      } else {
        console.error('Unexpected response format:', response.data);
        setThresholds([]);
      }
    } catch (error) {
      console.error('Error fetching thresholds:', error);
      setThresholds([]); // Ensure we always set an array
    }
  };

  // Function to set threshold by name
  const setThreshold = async (name: string) => {
    try {
      setLoading(true);
      await axios.post(`http://localhost:5001/thresholds/set/${name}`);
      setCurrentThreshold(name);
      // Refresh image with new threshold
      fetchImage();
    } catch (error) {
      console.error('Error setting threshold:', error);
      setError(`Failed to set threshold to ${name}`);
    } finally {
      setLoading(false);
    }
  };

  // Function to set custom threshold value
  const setCustomThresholdValue = async () => {
    try {
      setLoading(true);
      const thresholdValue = parseFloat(customThreshold);
      if (isNaN(thresholdValue) || thresholdValue < 0 || thresholdValue > 1) {
        setError('Threshold must be a number between 0 and 1');
        return;
      }
      
      await axios.post('http://localhost:5001/thresholds/set_value', {
        threshold: thresholdValue
      });
      setCurrentThreshold('custom');
      // Refresh image with new threshold
      fetchImage();
    } catch (error) {
      console.error('Error setting custom threshold:', error);
      setError('Failed to set custom threshold');
    } finally {
      setLoading(false);
    }
  };

  // Function to fetch the latest image
  const fetchImage = async () => {
    try {
      setLoading(true);
      // Add timestamp to URL to prevent caching
      const timestamp = new Date().getTime();
      const response = await axios.get(`http://localhost:5001/detect_objects?t=${timestamp}`, {
        responseType: 'blob'
      });
      
      // Release the previous object URL to prevent memory leaks
      if (imageSrc) {
        URL.revokeObjectURL(imageSrc);
      }
      
      const imageURL = URL.createObjectURL(response.data);
      setImageSrc(imageURL);
      setError(null);
    } catch (error) {
      console.error('Error fetching image:', error);
      setError('Failed to load camera feed. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Initial fetch of thresholds and image
    fetchThresholds();
    fetchImage();
    
    // Set up polling interval (refresh every 2 seconds)
    intervalRef.current = setInterval(fetchImage, 2000);
    
    // Clean up function to clear interval and revoke object URL
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (imageSrc) {
        URL.revokeObjectURL(imageSrc);
      }
    };
  }, []); // Empty dependency array ensures this runs only once on mount

  const toggleAlert = () => {
    setAlertStatus(prevStatus => (prevStatus === 'all-clear' ? 'alert' : 'all-clear'));
  };

  const handleRefreshClick = () => {
    fetchImage();
  };

  return (
    <div>
      <Navigation />
      <div className={styles.container}>
        {/* Alert Banner */}
        <div
          className={`${styles.alertBanner} ${
            alertStatus === 'alert' ? styles.alert : styles.allClear
          }`}
        >
          {alertStatus === 'alert'
            ? 'âš ï¸ Alert: Something requires your attention!'
            : 'âœ“ All Clear: Everything is fine.'}
        </div>
        
        {/* Threshold Controls */}
        <div className={styles.thresholdControls}>
          <h3>Environment Settings</h3>
          <div className={styles.thresholdButtons}>
            {Array.isArray(thresholds) && thresholds.length > 0 ? (
              thresholds.map((threshold) => (
                <button
                  key={threshold.name}
                  onClick={() => setThreshold(threshold.name)}
                  className={`${styles.thresholdButton} ${
                    currentThreshold === threshold.name ? styles.activeThreshold : ''
                  }`}
                >
                  {threshold.name} ({threshold.value})
                </button>
              ))
            ) : (
              <p>No threshold settings available</p>
            )}
          </div>
          
          <div className={styles.customThresholdControl}>
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={customThreshold}
              onChange={(e) => setCustomThreshold(e.target.value)}
              className={styles.customThresholdInput}
            />
            <button
              onClick={setCustomThresholdValue}
              className={styles.customThresholdButton}
            >
              Set Custom Threshold
            </button>
          </div>
        </div>
        
        {/* Camera Feed */}
        <div className={styles.cameraFeed}>
          <div className={styles.feedHeader}>
            <p className={styles.feedTitle}>Live Camera Feed with Object Detection</p>
            <div className={styles.currentThreshold}>
              Current environment: <strong>{currentThreshold}</strong>
            </div>
            <button 
              onClick={handleRefreshClick} 
              className={styles.refreshButton}
              disabled={loading}
            >
              {loading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
          
          {loading && !imageSrc ? (
            <div className={styles.feedPlaceholder}>
              <p>Loading camera feed...</p>
            </div>
          ) : error ? (
            <div className={styles.errorContainer}>
              <p className={styles.errorMessage}>{error}</p>
              <button onClick={fetchImage} className={styles.retryButton}>
                Retry
              </button>
            </div>
          ) : (
            <img
              src={imageSrc || ''}
              alt="Object Detection"
              className={styles.cameraImage}
            />
          )}
        </div>
        
        {/* Control Panel */}
        <div className={styles.controlPanel}>
          <button className={styles.controlButton} onClick={toggleAlert}>
            Toggle Alert Status
          </button>
        </div>
      </div>
    </div>
  );
};

export default CameraFeedPage;