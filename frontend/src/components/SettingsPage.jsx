import React, { useState, useEffect } from 'react';

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    conf_threshold: 0.25,
    required_consecutive_frames: 3,
    missed_frame_tolerance: 1,
    duration_threshold: 2.0,
    head_rep_threshold: 3,
    head_window_seconds: 4.0,
    voice_alerts_enabled: true,
    voice_cooldown_seconds: 10.0,
  });

  const [loading, setLoading] = useState(false);
  const [evalResults, setEvalResults] = useState(null);

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const res = await fetch('/api/settings');
        if (res.ok) {
          const data = await res.json();
          setSettings(data);
        }
      } catch (err) {
        console.error('Error fetching settings:', err);
      }
    };
    fetchSettings();
  }, []);

  const handleSaveSettings = async () => {
    // Validate inputs
    if (settings.conf_threshold < 0.0 || settings.conf_threshold > 1.0) {
      alert('Confidence threshold must be between 0.0 and 1.0.');
      return;
    }
    if (settings.required_consecutive_frames < 1) {
      alert('Required consecutive frames must be a positive integer (>= 1).');
      return;
    }

    setLoading(true);
    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      });
      if (res.ok) {
        alert('Centralized configuration updated and saved successfully!');
      }
    } catch (err) {
      alert('Error saving settings: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunEvaluation = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/evaluator/run', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setEvalResults(data);
        alert('Research evaluation and ablation experiments completed!');
      }
    } catch (err) {
      alert('Error running evaluation: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="header-bar">
        <div>
          <h1 className="page-title">⚙️ System Configuration & Threshold Settings</h1>
          <p className="page-description">
            Configure detection parameters, tracking tolerances, behavior heuristics, voice controls, and run research paper evaluation experiments.
          </p>
        </div>
      </div>

      {/* Settings Input Grid */}
      <div className="panel-section" style={{ marginBottom: '24px' }}>
        <h3 className="section-title">1. Detection & Tracking Parameters</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '16px' }}>
          <div>
            <label className="form-label">Confidence Threshold (0.05 - 1.00)</label>
            <input
              type="number"
              step="0.05"
              min="0.05"
              max="1.00"
              value={settings.conf_threshold}
              onChange={(e) => setSettings({ ...settings, conf_threshold: parseFloat(e.target.value) })}
              className="form-input"
            />
          </div>
          <div>
            <label className="form-label">Required Consecutive Frames (&ge; 1)</label>
            <input
              type="number"
              step="1"
              min="1"
              max="30"
              value={settings.required_consecutive_frames}
              onChange={(e) => setSettings({ ...settings, required_consecutive_frames: parseInt(e.target.value) })}
              className="form-input"
            />
          </div>
          <div>
            <label className="form-label">Missed-Frame Tolerance (&ge; 0)</label>
            <input
              type="number"
              step="1"
              min="0"
              max="20"
              value={settings.missed_frame_tolerance}
              onChange={(e) => setSettings({ ...settings, missed_frame_tolerance: parseInt(e.target.value) })}
              className="form-input"
            />
          </div>
          <div>
            <label className="form-label">Handling Duration Threshold (Seconds &ge; 0.0)</label>
            <input
              type="number"
              step="0.5"
              min="0.0"
              max="60.0"
              value={settings.duration_threshold}
              onChange={(e) => setSettings({ ...settings, duration_threshold: parseFloat(e.target.value) })}
              className="form-input"
            />
          </div>
        </div>

        <h3 className="section-title">2. Voice Alerts & Cooldown</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '16px' }}>
          <div>
            <label className="form-label">Enable Voice Alerts</label>
            <input
              type="checkbox"
              checked={settings.voice_alerts_enabled}
              onChange={(e) => setSettings({ ...settings, voice_alerts_enabled: e.target.checked })}
              style={{ width: '20px', height: '20px', marginTop: '6px' }}
            />
          </div>
          <div>
            <label className="form-label">Voice Alert Cooldown (Seconds &ge; 0.0)</label>
            <input
              type="number"
              step="1.0"
              min="0.0"
              max="120.0"
              value={settings.voice_cooldown_seconds}
              onChange={(e) => setSettings({ ...settings, voice_cooldown_seconds: parseFloat(e.target.value) })}
              className="form-input"
            />
          </div>
        </div>

        <button className="btn btn-primary" onClick={handleSaveSettings} disabled={loading}>
          💾 SAVE CONFIGURATION CHANGES
        </button>
      </div>

      {/* Research Paper Evaluation Suite Runner */}
      <div className="panel-section">
        <h3 className="section-title">📊 Research Paper Evaluation & Ablation Experiments Suite</h3>
        <p style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '16px' }}>
          Execute empirical CUDA evaluation and 4 research ablation study experiments for minor project report generation.
        </p>

        <button className="btn btn-secondary" onClick={handleRunEvaluation} disabled={loading} style={{ marginBottom: '16px' }}>
          🧪 {loading ? 'Running Evaluation...' : 'RUN FULL RESEARCH EVALUATION SUITE'}
        </button>

        {evalResults && (
          <div>
            {/* Object Detection Metrics */}
            <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(6, 1fr)', marginBottom: '16px' }}>
              <div className="metric-card">
                <div className="metric-label">Precision</div>
                <div className="metric-value">{(evalResults.object_detection_eval.precision * 100).toFixed(2)}%</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Recall</div>
                <div className="metric-value">{(evalResults.object_detection_eval.recall * 100).toFixed(2)}%</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">mAP@50</div>
                <div className="metric-value">{(evalResults.object_detection_eval.mAP50 * 100).toFixed(2)}%</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">mAP@50-95</div>
                <div className="metric-value">{(evalResults.object_detection_eval.mAP50_95 * 100).toFixed(2)}%</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Latency</div>
                <div className="metric-value">{evalResults.object_detection_eval.inference_time_ms} ms</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">FPS</div>
                <div className="metric-value">{evalResults.object_detection_eval.fps} FPS</div>
              </div>
            </div>

            {/* Experiment 1 Table */}
            <h4 style={{ fontSize: '13px', color: '#60a5fa', marginBottom: '8px' }}>Experiment 1: Single vs Consecutive Frame Confirmation</h4>
            <table className="control-table" style={{ marginBottom: '16px' }}>
              <thead>
                <tr>
                  <th>N Frames</th>
                  <th>Setting</th>
                  <th>Confirmed Events</th>
                  <th>False Alerts</th>
                  <th>Latency</th>
                </tr>
              </thead>
              <tbody>
                {evalResults.experiment_1.map((r, idx) => (
                  <tr key={idx}>
                    <td>{r.persistence_n}</td>
                    <td>{r.setting_name}</td>
                    <td>{r.confirmed_events}</td>
                    <td>{r.false_alerts}</td>
                    <td>{r.avg_confirmation_time_sec}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
