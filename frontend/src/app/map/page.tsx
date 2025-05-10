'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Head from 'next/head';
import Navigation from '../components/Navigation';
import styles from './Map.module.css';

// Define types for threshold settings
type ThresholdSetting = {
  name: string;
  value: number;
};

type ThresholdResponse = {
  object_detection: {
    current: string;
    current_value: number;
    available: ThresholdSetting[];
  };
  fire_detection: {
    current: string;
    current_value: number;
    available: ThresholdSetting[];
  };
};

const MapPage = () => {
  const [occupancyGrid, setOccupancyGrid] = useState<number[][]>([]);
  const [path, setPath] = useState<[number, number][]>([]);
  const [goalX, setGoalX] = useState(0);
  const [goalY, setGoalY] = useState(0);
  const [inputGoalX, setInputGoalX] = useState('0');
  const [inputGoalY, setInputGoalY] = useState('0');
  
  // Separate loading states for different operations
  const [mapLoading, setMapLoading] = useState(false);
  const [thresholdLoading, setThresholdLoading] = useState(false);
  const [fireDetectionLoading, setFireDetectionLoading] = useState(false);
  const [goalUpdateLoading, setGoalUpdateLoading] = useState(false);
  const [carControlLoading, setCarControlLoading] = useState(false);
  const [refreshLoading, setRefreshLoading] = useState(false);
  const [resetLoading, setResetLoading] = useState(false);
  const [navigationLoading, setNavigationLoading] = useState(false);
  
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('');
  const [fireDetected, setFireDetected] = useState(false);
  const [firePosition, setFirePosition] = useState<{ x: number, y: number } | null>(null);
  const [mode, setMode] = useState<'fire-detection' | 'navigation'>('fire-detection');
  const [detectionInProgress, setDetectionInProgress] = useState(false);
  const [carFollowingPath, setCarFollowingPath] = useState(false);
  const [carPosition, setCarPosition] = useState<{ x: number, y: number } | null>(null);
  const [lastUpdateTime, setLastUpdateTime] = useState(Date.now());
  
  // Add states for thresholds
  const [thresholds, setThresholds] = useState<ThresholdResponse | null>(null);
  const [selectedObjectThreshold, setSelectedObjectThreshold] = useState('normal');
  const [selectedFireThreshold, setSelectedFireThreshold] = useState('normal');

  // Initial load - fetch thresholds and start with fire detection
  useEffect(() => {
    fetchThresholds();
    detectFire(); // Start with fire detection
  }, []);

  // Fetch available thresholds from backend
  const fetchThresholds = async () => {
    setThresholdLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5001/thresholds');
      if (!response.ok) {
        throw new Error(`Failed to fetch thresholds. Status: ${response.status}`);
      }
      
      const data: ThresholdResponse = await response.json();
      setThresholds(data);
      
      // Set initial values based on backend defaults
      setSelectedObjectThreshold(data.object_detection.current);
      setSelectedFireThreshold(data.fire_detection.current);
      
    } catch (err) {
      console.error('Error fetching thresholds:', err);
      if (err instanceof Error) {
        setError('Error fetching thresholds: ' + err.message);
      }
    } finally {
      setThresholdLoading(false);
    }
  };

  // Handler for changing object detection threshold
  const handleObjectThresholdChange = async (value: string) => {
    try {
      // Only disable this specific operation
      setThresholdLoading(true);
      const response = await fetch(`http://127.0.0.1:5001/thresholds/set/${value}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to update object detection threshold. Status: ${response.status}`);
      }
      
      await response.json();
      setSelectedObjectThreshold(value);
      setStatus(`Object detection threshold updated to: ${value}`);
      
      // Refresh the thresholds to ensure UI is in sync with backend
      await fetchThresholds();
      
    } catch (err) {
      console.error('Error updating object detection threshold:', err);
      if (err instanceof Error) {
        setError('Error updating threshold: ' + err.message);
      }
    } finally {
      setThresholdLoading(false);
    }
  };

  // Handler for changing fire detection threshold
  const handleFireThresholdChange = async (value: string) => {
    try {
      // Only disable this specific operation
      setThresholdLoading(true);
      const response = await fetch(`http://127.0.0.1:5001/fire_thresholds/set/${value}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to update fire detection threshold. Status: ${response.status}`);
      }
      
      await response.json();
      setSelectedFireThreshold(value);
      setStatus(`Fire detection threshold updated to: ${value}`);
      
      // Refresh the thresholds to ensure UI is in sync with backend
      await fetchThresholds();
      
    } catch (err) {
      console.error('Error updating fire detection threshold:', err);
      if (err instanceof Error) {
        setError('Error updating threshold: ' + err.message);
      }
    } finally {
      setThresholdLoading(false);
    }
  };

  // Memoize the fetchOccupancyGrid function
  const fetchOccupancyGrid = useCallback(async (forceUpdate = false, isRefresh = false) => {
    // Choose the appropriate loading state based on the operation type
    const setLoadingState = isRefresh ? setRefreshLoading : setMapLoading;
    
    // Check if this specific operation is already in progress
    const isLoading = isRefresh ? refreshLoading : mapLoading;
    if (isLoading && !forceUpdate) return;
    
    // Only set loading for this specific operation
    setLoadingState(true);
    
    try {
      const response = await fetch('http://127.0.0.1:5001/get_occupancy_grid');
      if (!response.ok) {
        throw new Error(`Failed to fetch occupancy grid. Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Response data:", data);
      
      // Update the grid without conditional checks to ensure it always refreshes
      setOccupancyGrid(data.occupancy_grid || []);
      
      // Always update path data when received
      if (data.path && Array.isArray(data.path)) {
        console.log("Setting path with length:", data.path.length);
        setPath(data.path);
      } else {
        console.log("No valid path data in response");
        setPath([]);
      }
      
      // Only update goal if we don't have a fire position yet
      if (data.goal && !firePosition) {
        setGoalX(data.goal.x);
        setGoalY(data.goal.y);
      }
      
      // Update car position if available
      if (data.car_position) {
        setCarPosition({
          x: data.car_position.x,
          y: data.car_position.y
        });
      }
      
      // Update car following status based on the status message
      if (data.status && data.status.includes("Car started following path")) {
        setCarFollowingPath(true);
      } else if (data.status && data.status.includes("Car stopped following path")) {
        setCarFollowingPath(false);
      }
      
      if (data.status) {
        setStatus(`Navigation status: ${data.status}`);
      }
      
      // Force a refresh of the component
      setLastUpdateTime(Date.now());
      
      setError(null);
    } catch (err) {
      console.error("Error in fetchOccupancyGrid:", err);
      if (err instanceof Error) {
        setError('Error fetching occupancy grid: ' + err.message);
      } else {
        setError('Unknown error occurred');
      }
    } finally {
      // Reset only the specific loading state that was set
      setLoadingState(false);
    }
  }, [mapLoading, refreshLoading, firePosition]);

  // Modify the useEffect for periodic updates with the memoized fetch function
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (mode === 'fire-detection' && !fireDetected) {
      // Keep checking for fire every 15 seconds if in fire detection mode
      interval = setInterval(detectFire, 15000);
    } else if (mode === 'navigation') {
      // Always update the map every 10 seconds when in navigation mode
      interval = setInterval(() => fetchOccupancyGrid(true, true), 10000);
    }

    return () => clearInterval(interval);
  }, [mode, fireDetected, fetchOccupancyGrid]);

  // Update input fields when goal coordinates change
  useEffect(() => {
    setInputGoalX(goalX.toString());
    setInputGoalY(goalY.toString());
  }, [goalX, goalY]);

  // When fire is detected, transition to navigation mode
  useEffect(() => {
    if (fireDetected && firePosition) {
      setMode('navigation');
      // Set the goal to the fire position
      setGoalX(firePosition.x);
      setGoalY(firePosition.y);
      // Update the input fields as well
      setInputGoalX(firePosition.x.toString());
      setInputGoalY(firePosition.y.toString());
      // Fetch the occupancy grid with the new goal
      fetchOccupancyGrid(true);
    }
  }, [fireDetected, firePosition, fetchOccupancyGrid]);

  const detectFire = async () => {
    if (detectionInProgress) return;
    
    setDetectionInProgress(true);
    setFireDetectionLoading(true);
    setStatus('Detecting fire...');
    
    try {
      const response = await fetch('http://127.0.0.1:5001/detect_fire');
      if (!response.ok) {
        throw new Error(`Failed to detect fire. Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Fire detection result:', data);
      
      setFireDetected(data.fire_detected);
      if (data.fire_detected && data.goal) {
        setFirePosition(data.goal);
        setStatus(`Fire detected at position (${data.goal.x}, ${data.goal.y})! Planning path...`);
        
        // When fire is detected, immediately fetch the occupancy grid to get path
        await fetchOccupancyGrid(true);
      } else {
        setStatus('No fire detected. Continuing to monitor...');
      }
      
      setError(null);
    } catch (err) {
      console.error('Error detecting fire:', err);
      if (err instanceof Error) {
        setError('Error detecting fire: ' + err.message);
      } else {
        setError('Unknown error occurred during fire detection');
      }
      setStatus('Fire detection failed');
    } finally {
      setFireDetectionLoading(false);
      setDetectionInProgress(false);
    }
  };

  // Function to manually start/stop car following
  const toggleCarFollowing = async () => {
    try {
      const endpoint = carFollowingPath ? 
        'http://127.0.0.1:5001/api/car/stop_following' : 
        'http://127.0.0.1:5001/api/car/start_following';
      
      setCarControlLoading(true);
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to ${carFollowingPath ? 'stop' : 'start'} car following`);
      }
      
      const data = await response.json();
      setCarFollowingPath(!carFollowingPath);
      setStatus(data.message);
      
      // Force refresh after toggling car following
      await fetchOccupancyGrid(true);
    } catch (err) {
      console.error('Error toggling car following:', err);
      if (err instanceof Error) {
        setError(err.message);
      }
    } finally {
      setCarControlLoading(false);
    }
  };

  const updateGoal = async () => {
    try {
      setGoalUpdateLoading(true);
      // Parse input values to numbers
      const newGoalX = parseInt(inputGoalX, 10);
      const newGoalY = parseInt(inputGoalY, 10);
      
      // Validate input
      if (isNaN(newGoalX) || isNaN(newGoalY)) {
        setError('Invalid goal coordinates. Please enter numbers.');
        return;
      }
      
      const response = await fetch('http://127.0.0.1:5001/set_goal', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ goal_x: newGoalX, goal_y: newGoalY }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to update goal. Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Goal updated:', data);
      
      // Update local state with the new goal values
      setGoalX(newGoalX);
      setGoalY(newGoalY);
      
      // Immediately fetch the new grid after updating goal
      await fetchOccupancyGrid(true);
    } catch (err) {
      console.error('Error updating goal:', err);
      if (err instanceof Error) {
        setError('Error updating goal: ' + err.message);
      }
    } finally {
      setGoalUpdateLoading(false);
    }
  };

  const resetDetection = async () => {
    setResetLoading(true);
    try {
      // First stop the car if it's following a path
      if (carFollowingPath) {
        const response = await fetch('http://127.0.0.1:5001/api/car/stop_following', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          }
        });
        
        if (response.ok) {
          setCarFollowingPath(false);
        }
      }
      
      // Reset the entire detection process
      setFireDetected(false);
      setFirePosition(null);
      setMode('fire-detection');
      setPath([]);
      setOccupancyGrid([]);
      setStatus('Detection reset. Starting fire detection...');
      
      // Force a UI refresh
      setLastUpdateTime(Date.now());
      
      // Start detection again
      await detectFire();
    } catch (err) {
      console.error("Error in reset:", err);
      if (err instanceof Error) {
        setError('Reset error: ' + err.message);
      }
    } finally {
      setResetLoading(false);
    }
  };

  const navigateToFire = async () => {
    setNavigationLoading(true);
    setStatus('Starting complete navigation sequence...');
    
    try {
      const response = await fetch('http://127.0.0.1:5001/navigate_to_fire');
      if (!response.ok) {
        throw new Error(`Navigation sequence failed. Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Navigation response:", data);
      
      if (data.fire_detected) {
        setFireDetected(true);
        if (data.goal) {
          setFirePosition(data.goal);
          setGoalX(data.goal.x);
          setGoalY(data.goal.y);
          setInputGoalX(data.goal.x.toString());
          setInputGoalY(data.goal.y.toString());
        }
        
        setOccupancyGrid(data.occupancy_grid || []);
        
        // Ensure path data is properly set
        if (data.path && Array.isArray(data.path)) {
          console.log("Setting navigation path:", data.path);
          setPath(data.path);
        } else {
          console.log("No valid path in navigation response");
          setPath([]);
        }
        
        setMode('navigation');
        setStatus(`Fire detected and path planned! ${data.path ? data.path.length : 0} steps to fire.`);
        
        // Update car position if available
        if (data.car_position) {
          setCarPosition({
            x: data.car_position.x,
            y: data.car_position.y
          });
        }
        
        // Force a UI refresh
        setLastUpdateTime(Date.now());
      } else {
        setStatus('No fire detected during navigation sequence.');
      }
      
      setError(null);
    } catch (err) {
      console.error('Error in navigation sequence:', err);
      if (err instanceof Error) {
        setError('Navigation error: ' + err.message);
      }
    } finally {
      setNavigationLoading(false);
    }
  };

  // Improved renderGrid function with better path cell identification
  const renderGrid = () => {
    if (occupancyGrid.length === 0) {
      return (
        <div className={styles.emptyGridMessage}>
          <p>{mode === 'fire-detection' 
            ? 'Detecting fires. Occupancy grid will appear after fire detection.' 
            : 'No grid data available yet'}</p>
        </div>
      );
    }
    
    // Create a set of path coordinates for faster lookup
    const pathCoords = new Set();
    if (Array.isArray(path)) {
      path.forEach(point => {
        if (point && point.length === 2) {
          pathCoords.add(`${point[0]},${point[1]}`);
        }
      });
    }
    
    return (
      <div className={styles.gridContainer} key={`grid-${lastUpdateTime}`}>
        {occupancyGrid.map((row, i) => (
          <div key={`row-${i}-${lastUpdateTime}`} className={styles.gridRow}>
            {row.map((cell, j) => {
              const isOccupied = cell === 1;
              const isOnPath = pathCoords.has(`${i},${j}`);
              const isGoal = i === goalY && j === goalX;
              const isCarLocation = carPosition && i === carPosition.y && j === carPosition.x;
              const isFireLocation = firePosition && i === firePosition.y && j === firePosition.x;
              
              return (
                <div
                  key={`cell-${i}-${j}-${lastUpdateTime}`}
                  className={`${styles.gridCell} 
                    ${isOccupied ? styles.occupied : ''} 
                    ${isOnPath ? styles.robotPath : ''}
                    ${isGoal ? styles.goal : ''}
                    ${isCarLocation ? styles.car : ''}
                    ${isFireLocation ? styles.fire : ''}`}
                  title={`${i},${j}`}
                  onClick={() => {
                    // Only allow manual goal setting in navigation mode
                    if (mode === 'navigation') {
                      setInputGoalX(j.toString());
                      setInputGoalY(i.toString());
                    }
                  }}
                ></div>
              );
            })}
          </div>
        ))}
      </div>
    );
  };

  // Render threshold selectors
  const renderThresholdSelectors = () => {
    if (!thresholds) return null;
    
    return (
      <div className={styles.thresholdControls}>
        <div className={styles.thresholdGroup}>
          <label className={styles.thresholdLabel}>
            Object Detection Environment:
            <select 
              className={styles.thresholdSelect}
              value={selectedObjectThreshold}
              onChange={(e) => handleObjectThresholdChange(e.target.value)}
              disabled={thresholdLoading} // Only disable for threshold operations
            >
              {thresholds.object_detection.available.map((threshold) => (
                <option key={`obj-${threshold.name}`} value={threshold.name}>
                  {threshold.name} ({threshold.value})
                </option>
              ))}
            </select>
          </label>
        </div>
        
        <div className={styles.thresholdGroup}>
          <label className={styles.thresholdLabel}>
            Fire Detection Environment:
            <select 
              className={styles.thresholdSelect}
              value={selectedFireThreshold}
              onChange={(e) => handleFireThresholdChange(e.target.value)}
              disabled={thresholdLoading} // Only disable for threshold operations
            >
              {thresholds.fire_detection.available.map((threshold) => (
                <option key={`fire-${threshold.name}`} value={threshold.name}>
                  {threshold.name} ({threshold.value})
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>
    );
  };

  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </Head>
      <div>
        <Navigation />
        <div className={styles.container}>
          <h1 className={styles.title}>Fire Detection & Navigation System</h1>
          
          <div className={styles.statusBar}>
            <div className={`${styles.statusIndicator} ${styles[mode]}`}>
              Mode: {mode === 'fire-detection' ? 'Fire Detection' : 'Navigation'}
            </div>
            {fireDetected && (
              <div className={`${styles.statusIndicator} ${styles.fireAlert}`}>
                Fire Detected at ({firePosition?.x || 0}, {firePosition?.y || 0})
              </div>
            )}
            {carFollowingPath && (
              <div className={`${styles.statusIndicator} ${styles.carFollowing}`}>
                Car is following path
              </div>
            )}
            {carPosition && (
              <div className={`${styles.statusIndicator} ${styles.carPosition}`}>
                Car position: ({carPosition.x}, {carPosition.y})
              </div>
            )}
          </div>
          
          {/* Threshold selectors - placed near the top for visibility */}
          {renderThresholdSelectors()}
          
          <div className={styles.controls}>
            {/* Fire detection button - fixed width button with loading indicator */}
            <button 
              onClick={detectFire} 
              className={`${styles.button} ${styles.fireButton} ${styles.fixedWidthButton}`} 
              disabled={fireDetectionLoading || mode === 'navigation'} // Only disable for fire detection
            >
              {fireDetectionLoading ? (
                <span className={styles.loadingText}>Detecting...</span>
              ) : (
                <span>Detect Fire</span>
              )}
            </button>
            
            {/* Complete navigation sequence button - fixed width */}
            <button 
              onClick={navigateToFire} 
              className={`${styles.button} ${styles.navigationButton} ${styles.fixedWidthButton}`} 
              disabled={navigationLoading} // Only disable during navigation operation
            >
              {navigationLoading ? (
                <span className={styles.loadingText}>Navigating...</span>
              ) : (
                <span>Run Complete Navigation</span>
              )}
            </button>
            
            {/* Manual car control button - only show in navigation mode */}
            {mode === 'navigation' && (
              <button 
                onClick={toggleCarFollowing} 
                className={`${styles.button} ${carFollowingPath ? styles.stopButton : styles.startButton} ${styles.fixedWidthButton}`} 
                disabled={carControlLoading || !path.length} // Only disable during car control operations
              >
                {carControlLoading ? (
                  <span className={styles.loadingText}>{carFollowingPath ? 'Stopping...' : 'Starting...'}</span>
                ) : (
                  <span>{carFollowingPath ? 'Stop Car' : 'Start Car'}</span>
                )}
              </button>
            )}
            
            {/* Reset button - fixed width */}
            <button 
              onClick={resetDetection} 
              className={`${styles.button} ${styles.resetButton} ${styles.fixedWidthButton}`} 
              disabled={resetLoading} // Only disable during reset operation
            >
              {resetLoading ? (
                <span className={styles.loadingText}>Resetting...</span>
              ) : (
                <span>Reset Detection</span>
              )}
            </button>
          </div>
          
          {mode === 'navigation' && (
            <div className={styles.controls}>
              <label>
                Goal X:
                <input
                  type="number"
                  value={inputGoalX}
                  onChange={(e) => setInputGoalX(e.target.value)}
                  className={styles.input}
                  disabled={goalUpdateLoading} // Only disable during goal updates
                />
              </label>
              <label>
                Goal Y:
                <input
                  type="number"
                  value={inputGoalY}
                  onChange={(e) => setInputGoalY(e.target.value)}
                  className={styles.input}
                  disabled={goalUpdateLoading} // Only disable during goal updates
                />
              </label>
              {/* Fixed width update button - Keep consistent styling */}
              <button 
                onClick={updateGoal} 
                className={`${styles.button} ${styles.fixedWidthButton}`} 
                disabled={goalUpdateLoading} // Only disable during goal update operation
              >
                {goalUpdateLoading ? 'Updating...' : 'Update Goal'}
              </button>
              {/* Fixed width refresh button - Don't change class during refreshing */}
              <button 
                onClick={() => fetchOccupancyGrid(true, true)} 
                className={`${styles.button} ${styles.fixedWidthButton}`} 
                disabled={refreshLoading} // Only disable during refresh operation
              >
                {refreshLoading ? 'Refreshing...' : 'Refresh Map'}
              </button>
            </div>
          )}
          
          {status && <p className={styles.status}>{status}</p>}
          {error && <p className={styles.error}>{error}</p>}
          
          {/* Show loading message only when both mapLoading is true AND we have no grid */}
          {mapLoading && !occupancyGrid.length ? (
            <p className={styles.loading}>
              {mode === 'fire-detection' 
                ? 'Scanning for fires...' 
                : 'Loading occupancy grid...'}
            </p>
          ) : (
            renderGrid()
          )}
          
          {path.length > 0 && (
            <p className={styles.pathInfo}>
              Path found: {path.length} steps 
              {carPosition ? ` from (${carPosition.x},${carPosition.y})` : ''}
              {' to '}({goalX},{goalY})
            </p>
          )}
          
          {/* Debug section for path data */}
          {path.length > 0 && (
            <details className={styles.debugSection}>
              <summary>Path Data (Debug)</summary>
              <pre>{JSON.stringify(path, null, 2)}</pre>
            </details>
          )}
          
          {/* Debug section for threshold data */}
          {thresholds && (
            <details className={styles.debugSection}>
              <summary>Current Threshold Settings (Debug)</summary>
              <p>Object Detection: {selectedObjectThreshold} ({thresholds.object_detection.current_value})</p>
              <p>Fire Detection: {selectedFireThreshold} ({thresholds.fire_detection.current_value})</p>
            </details>
          )}
        </div>
      </div>
    </>
  );
};

export default MapPage;