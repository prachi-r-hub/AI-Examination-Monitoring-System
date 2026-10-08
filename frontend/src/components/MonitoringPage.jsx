import React, { useState, useEffect } from 'react';

export default function MonitoringPage() {
  const [cameraStatus, setCameraStatus] = useState({
    is_active: false,
    is_running: false,
    metrics: {
      fps: 0.0,
      gpu_inference_ms: 0.0,
      people_count: 0,
      allowed_count: 0,
      restricted_count: 0,
      confirmed_incidents: 0,
      monitoring_risk_score: 0,
      monitoring_risk_level: 'LOW',
      status_label: 'CAMERA OFFLINE',
      status_color: 'GRAY'
    },
    active_detections: [],
    timeline: []
  });

  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [streamVersion, setStreamVersion] = useState(Date.now());

  // Poll Backend Camera Status Every 1 Second
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/camera/status');
        if (res.ok) {
          const data = await res.json();
          setCameraStatus(data);
          if (data.is_active && errorMessage) {
            setErrorMessage(null);
          }
        }
      } catch (err) {
        console.error('Error fetching camera status:', err);
        setErrorMessage('Backend connection failed — Please verify FastAPI server is running on http://localhost:8000');
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, [errorMessage]);

  const handleStartCamera = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      console.log('[FRONTEND] Sending POST /api/camera/start request...');
      const res = await fetch('/api/camera/start', { method: 'POST' });
      const data = await res.json();
      console.log('[FRONTEND] Start camera response:', data);

      if (!res.ok || !data.success) {
        setErrorMessage(data.message || 'Camera unavailable — Hardware access failed');
      } else {
        // Force browser image tag to open a fresh HTTP stream connection
        setStreamVersion(Date.now());
      }
    } catch (err) {
      console.error('[FRONTEND] Error starting camera:', err);
      setErrorMessage('Backend connection failed — Cannot reach FastAPI server.');
    } finally {
      setLoading(false);
    }
  };

  const handleStopCamera = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      console.log('[FRONTEND] Sending POST /api/camera/stop request...');
      await fetch('/api/camera/stop', { method: 'POST' });
      setStreamVersion(Date.now());
    } catch (err) {
      console.error('[FRONTEND] Error stopping camera:', err);
    } finally {
      setLoading(false);
    }
  };

  const isLive = cameraStatus.is_active && cameraStatus.is_running;
  const metrics = cameraStatus.metrics || {};
  const activeDets = cameraStatus.active_detections || [];
  const timeline = cameraStatus.timeline || [];

  return (
    <div>
      {/* Header Status Bar */}
      <div className="header-bar">
        <div>
          <h1 className="page-title">🎥 Live Examination Monitoring</h1>
          <p className="page-description">
            Real-time examination control room feed powered by YOLOv8s custom object detection and spatial persistence tracking.
          </p>
        </div>
        <div>
          <span className={`status-badge ${isLive ? 'live' : 'offline'}`}>
            {isLive ? '● LIVE MONITORING ACTIVE' : '● CAMERA OFFLINE'}
          </span>
        </div>
      </div>

      {/* Explicit Error Banner if camera or backend failed */}
      {errorMessage && (
        <div style={{ backgroundColor: '#7f1d1d', border: '1px solid #dc2626', color: '#f87171', padding: '12px 16px', borderRadius: '6px', marginBottom: '20px', fontSize: '13px', fontWeight: '600' }}>
          ❌ {errorMessage}
        </div>
      )}

      {/* Main Monitoring Grid */}
      <div className="monitoring-grid">
        {/* DOMINANT LARGE CAMERA FEED */}
        <div className="camera-container">
          <div className="camera-header">
            <div className="camera-title">
              <span>📡 Main Examination Feed (Camera #0)</span>
            </div>
            <div className="camera-controls">
              {!isLive ? (
                <button
                  className="btn btn-primary"
                  onClick={handleStartCamera}
                  disabled={loading}
                >
                  {loading ? 'Initializing...' : '▶ START CAMERA'}
                </button>
              ) : (
                <button
                  className="btn btn-danger"
                  onClick={handleStopCamera}
                  disabled={loading}
                >
                  {loading ? 'Stopping...' : '⏹ STOP CAMERA'}
                </button>
              )}
            </div>
          </div>

          <div className="camera-viewport">
            <img
              key={streamVersion}
              src={`/api/camera/stream?v=${streamVersion}`}
              alt="Live Examination Feed"
              className="camera-feed-img"
            />
          </div>

          {/* Status Label Banner Below Stream */}
          <div style={{ marginTop: '12px', padding: '10px 14px', borderRadius: '6px', backgroundColor: '#0f172a', border: '1px solid #334155' }}>
            <span style={{ fontSize: '13px', fontWeight: '600', color: metrics.status_color === 'RED' ? '#f87171' : metrics.status_color === 'AMBER' ? '#fbbf24' : '#4ade80' }}>
              {metrics.status_label || 'CAMERA OFFLINE'}
            </span>
          </div>
        </div>

        {/* SIDE METRICS & TABLES */}
        <div className="side-panel">
          {/* Real Backend Metrics Gauges */}
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">⚡ Inference Speed</div>
              <div className="metric-value">{metrics.fps || 0.0} FPS</div>
              <div className="metric-subtitle">{metrics.gpu_inference_ms || 24.4}ms ({metrics.device || 'GTX 1050'})</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">📹 Camera Grab</div>
              <div className="metric-value">{metrics.camera_fps || 0.0} FPS</div>
              <div className="metric-subtitle">Dropped: {metrics.dropped_frames || 0} frames</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">🟢 Allowed Objects</div>
              <div className="metric-value" style={{ color: '#4ade80' }}>{metrics.allowed_count || 0}</div>
              <div className="metric-subtitle">Stationery Tools</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">🔴 Restricted Objects</div>
              <div className="metric-value" style={{ color: '#f87171' }}>{metrics.restricted_count || 0}</div>
              <div className="metric-subtitle">Contraband Items</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">🚨 Confirmed Incidents</div>
              <div className="metric-value" style={{ color: '#f87171' }}>{metrics.confirmed_incidents || 0}</div>
              <div className="metric-subtitle">SQLite Audit Log</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">📈 Monitoring Risk</div>
              <div className="metric-value">
                {metrics.monitoring_risk_score || 0}/100
              </div>
              <div className="metric-subtitle">[{metrics.monitoring_risk_level || 'LOW'}]</div>
            </div>
          </div>

          {/* Active Detections Table */}
          <div className="panel-section">
            <h3 className="section-title">📋 Active Frame Detections</h3>
            {activeDets.length > 0 ? (
              <table className="control-table">
                <thead>
                  <tr>
                    <th>Object</th>
                    <th>Category</th>
                    <th>Confidence</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {activeDets.map((d) => (
                    <tr key={d.id}>
                      <td style={{ fontWeight: '600' }}>{d.object_name}</td>
                      <td>
                        <span className={`badge ${d.category === 'ALLOWED' ? 'badge-green' : 'badge-red'}`}>
                          {d.category}
                        </span>
                      </td>
                      <td>{d.confidence}%</td>
                      <td style={{ fontSize: '11px', color: '#94a3b8' }}>{d.status_label}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p style={{ fontSize: '12px', color: '#64748b' }}>No objects detected in current frame.</p>
            )}
          </div>

          {/* Recent Incident Timeline Table */}
          <div className="panel-section">
            <h3 className="section-title">⏱️ Recent Incident Timeline</h3>
            {timeline.length > 0 ? (
              <table className="control-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Event</th>
                    <th>Object</th>
                    <th>Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {timeline.map((t, idx) => (
                    <tr key={idx}>
                      <td style={{ fontSize: '11px', color: '#94a3b8' }}>{t.time}</td>
                      <td style={{ fontWeight: '500' }}>{t.event_type}</td>
                      <td>{t.object_name}</td>
                      <td>
                        <span className={`badge ${t.risk_level === 'HIGH' ? 'badge-red' : t.risk_level === 'MEDIUM' ? 'badge-amber' : 'badge-green'}`}>
                          {t.risk_score}/100
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p style={{ fontSize: '12px', color: '#64748b' }}>No confirmed incidents recorded in database.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
