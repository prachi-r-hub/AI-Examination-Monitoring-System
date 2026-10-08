import React, { useState, useEffect } from 'react';

export default function VideoAnalysisPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [sampleVideos, setSampleVideos] = useState([]);
  const [sampleRate, setSampleRate] = useState(1);
  const [loading, setLoading] = useState(false);
  const [progressMsg, setProgressMsg] = useState('');
  const [report, setReport] = useState(null);
  const [selectedEvidence, setSelectedEvidence] = useState(null);

  // Fetch built-in sample videos metadata
  useEffect(() => {
    const fetchSamples = async () => {
      try {
        const res = await fetch('/api/sample_videos');
        if (res.ok) {
          const data = await res.json();
          setSampleVideos(data.sample_videos || []);
        }
      } catch (err) {
        console.error('Error fetching sample videos:', err);
      }
    };
    fetchSamples();
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const runAnalysis = async (fileObj, sampleFilename = null) => {
    setLoading(true);
    setReport(null);
    setProgressMsg('Initializing real video analysis engine...');

    try {
      let res;
      if (sampleFilename) {
        setProgressMsg(`Loading built-in demonstration video '${sampleFilename}'...`);
        res = await fetch(`/api/video/analyze?sample_filename=${encodeURIComponent(sampleFilename)}&frame_sample_rate=${sampleRate}`, {
          method: 'POST',
        });
      } else if (fileObj) {
        setProgressMsg(`Uploading '${fileObj.name}' and extracting video frames...`);
        const formData = new FormData();
        formData.append('file', fileObj);
        res = await fetch(`/api/video/analyze?frame_sample_rate=${sampleRate}`, {
          method: 'POST',
          body: formData,
        });
      } else {
        alert('Please upload a video or select a sample video.');
        setLoading(false);
        return;
      }

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Video analysis failed');
      }

      setProgressMsg('Deduplicating temporal events and extracting evidence screenshots...');
      const data = await res.json();
      setReport(data);
    } catch (err) {
      alert('Error during video analysis: ' + err.message);
    } finally {
      setLoading(false);
      setProgressMsg('');
    }
  };

  const incidents = report?.incidents || [];
  const evidenceList = report?.evidence_screenshots || [];
  const summary = report?.event_summary || {};
  const isReviewRecommended = report?.overall_status === 'REVIEW RECOMMENDED';

  return (
    <div>
      {/* Header Bar */}
      <div className="header-bar">
        <div>
          <h1 className="page-title">🎬 Video Analysis & Behavior Inspection</h1>
          <p className="page-description">
            Decode pre-recorded examination videos, run CUDA YOLOv8s inference, filter transient noise, group temporal events, and inspect extracted evidence snapshots.
          </p>
        </div>
        <div>
          <span className="badge badge-secondary" style={{ fontSize: '12px', padding: '6px 12px' }}>
            Threshold: {report?.confidence_threshold || 0.25} Conf
          </span>
        </div>
      </div>

      {/* Safety & Human-in-the-Loop Disclaimer */}
      <div style={{ backgroundColor: '#1e293b', border: '1px solid #3b82f6', borderRadius: '6px', padding: '12px 16px', marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span style={{ fontSize: '20px' }}>⚖️</span>
        <div>
          <div style={{ fontSize: '13px', fontWeight: '700', color: '#60a5fa' }}>Human Verification Required</div>
          <div style={{ fontSize: '12px', color: '#94a3b8' }}>
            AI-generated alerts indicate observable patterns for proctor review. The system does NOT determine candidate intent or make automatic cheating decisions.
          </div>
        </div>
      </div>

      {/* Built-in Demonstration Sample Videos Section */}
      {sampleVideos.length > 0 && (
        <div className="panel-section" style={{ marginBottom: '24px' }}>
          <h3 className="section-title">🧪 Built-in Viva Demonstration Sample Videos</h3>
          <p style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '12px' }}>
            Select any pre-configured examination video scenario to run through the exact same real-time computer vision analysis engine:
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px' }}>
            {sampleVideos.map((sample) => (
              <div
                key={sample.id}
                style={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '6px',
                  padding: '12px',
                  display: 'flex',
                  flexDirection: 'column',
                  justify: 'space-between'
                }}
              >
                <div>
                  <div style={{ fontSize: '13px', fontWeight: '700', color: '#f8fafc', marginBottom: '4px' }}>
                    {sample.title}
                  </div>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '10px' }}>
                    {sample.description}
                  </div>
                </div>
                <button
                  className="btn btn-secondary"
                  style={{ width: '100%', fontSize: '12px' }}
                  onClick={() => runAnalysis(null, sample.filename)}
                  disabled={loading}
                >
                  ▶ Run {sample.title.split(':')[0]} Analysis
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Custom Video File Upload Controls */}
      <div className="panel-section" style={{ marginBottom: '24px' }}>
        <h3 className="section-title">📁 Upload Custom Examination Video File</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 220px 180px', gap: '16px', alignItems: 'end' }}>
          <div>
            <label className="form-label">Select Video File (.mp4, .avi, .mov, .mkv)</label>
            <input
              type="file"
              accept="video/mp4,video/avi,video/mov,video/mkv"
              onChange={handleFileChange}
              className="form-input"
            />
          </div>
          <div>
            <label className="form-label">Frame Sampling Strategy</label>
            <select
              value={sampleRate}
              onChange={(e) => setSampleRate(parseInt(e.target.value))}
              className="form-select"
            >
              <option value="1">1 Frame / Frame (Full Precision)</option>
              <option value="2">1 Frame / 2 Frames (2x Faster)</option>
            </select>
          </div>
          <div>
            <button
              className="btn btn-primary"
              style={{ width: '100%' }}
              onClick={() => runAnalysis(selectedFile, null)}
              disabled={loading || !selectedFile}
            >
              {loading ? 'Processing...' : '▶ START VIDEO ANALYSIS'}
            </button>
          </div>
        </div>
      </div>

      {/* Loading Progress State */}
      {loading && (
        <div className="panel-section" style={{ textAlign: 'center', padding: '30px', marginBottom: '24px' }}>
          <div style={{ fontSize: '16px', fontWeight: '700', color: '#60a5fa', marginBottom: '8px' }}>
            ⏳ Processing Video Frames...
          </div>
          <div style={{ fontSize: '13px', color: '#94a3b8' }}>{progressMsg}</div>
        </div>
      )}

      {/* Analysis Results Dashboard */}
      {report && (
        <div>
          {/* Top Video Metadata & Overall Session Status */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '20px', marginBottom: '24px' }}>
            {/* Metadata Cards */}
            <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
              <div className="metric-card">
                <div className="metric-label">📹 Video File</div>
                <div className="metric-value" style={{ fontSize: '14px' }}>{report.video_name}</div>
                <div className="metric-subtitle">{report.resolution} | {report.fps} FPS</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">⏱️ Duration & Frames</div>
                <div className="metric-value">{report.duration_formatted}</div>
                <div className="metric-subtitle">{report.frames_processed} / {report.total_frames} frames</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">⚡ Analysis Speed</div>
                <div className="metric-value">{report.processing_fps} FPS</div>
                <div className="metric-subtitle">{report.processing_time_seconds}s runtime ({report.gpu_device_name.split(' ')[0]})</div>
              </div>
            </div>

            {/* Overall Session Status Box */}
            <div
              className="panel-section"
              style={{
                borderColor: isReviewRecommended ? '#dc2626' : '#16a34a',
                backgroundColor: isReviewRecommended ? 'rgba(220, 38, 38, 0.1)' : 'rgba(22, 163, 74, 0.1)',
                display: 'flex',
                flexDirection: 'column',
                justify: 'center'
              }}
            >
              <div style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.5px', color: '#94a3b8' }}>
                Overall Session Status
              </div>
              <div style={{ fontSize: '18px', fontWeight: '800', marginTop: '4px', color: isReviewRecommended ? '#f87171' : '#4ade80' }}>
                {isReviewRecommended ? '🔴 REVIEW RECOMMENDED' : '🟢 NORMAL'}
              </div>
              <div style={{ fontSize: '11px', color: '#cbd5e1', marginTop: '6px' }}>
                {report.session_message}
              </div>
            </div>
          </div>

          {/* Event Breakdown Cards */}
          <div className="panel-section" style={{ marginBottom: '24px' }}>
            <h3 className="section-title">📊 Event Summary Breakdown</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
              <div style={{ padding: '12px', backgroundColor: '#1e293b', borderRadius: '6px', border: '1px solid #334155' }}>
                <span style={{ fontSize: '11px', color: '#94a3b8' }}>Total Grouped Events:</span>
                <span style={{ fontSize: '18px', fontWeight: '700', float: 'right', color: '#f8fafc' }}>
                  {report.total_events_detected}
                </span>
              </div>
              <div style={{ padding: '12px', backgroundColor: '#1e293b', borderRadius: '6px', border: '1px solid #334155' }}>
                <span style={{ fontSize: '11px', color: '#94a3b8' }}>Restricted Items:</span>
                <span style={{ fontSize: '18px', fontWeight: '700', float: 'right', color: '#f87171' }}>
                  {report.event_summary.restricted_object_events}
                </span>
              </div>
              <div style={{ padding: '12px', backgroundColor: '#1e293b', borderRadius: '6px', border: '1px solid #334155' }}>
                <span style={{ fontSize: '11px', color: '#94a3b8' }}>Head Movement Patterns:</span>
                <span style={{ fontSize: '18px', fontWeight: '700', float: 'right', color: '#fbbf24' }}>
                  {report.summary_breakdown.head_movement}
                </span>
              </div>
              <div style={{ padding: '12px', backgroundColor: '#1e293b', borderRadius: '6px', border: '1px solid #334155' }}>
                <span style={{ fontSize: '11px', color: '#94a3b8' }}>Object Passing Events:</span>
                <span style={{ fontSize: '18px', fontWeight: '700', float: 'right', color: '#fbbf24' }}>
                  {report.summary_breakdown.object_passing}
                </span>
              </div>
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', marginTop: '10px' }}>
              Strategy: {report.processing_strategy}
            </div>
          </div>

          {/* Deduplicated Chronological Event Timeline */}
          <div className="panel-section" style={{ marginBottom: '24px' }}>
            <h3 className="section-title">⏱️ Deduplicated Chronological Event Timeline</h3>
            {incidents.length > 0 ? (
              <table className="control-table">
                <thead>
                  <tr>
                    <th>Time Range</th>
                    <th>Duration</th>
                    <th>Event Description</th>
                    <th>Detected Item</th>
                    <th>Peak Confidence</th>
                    <th>Risk Score</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {incidents.map((ev) => (
                    <tr
                      key={ev.incident_id}
                      onClick={() => ev.evidence_path && setSelectedEvidence(ev)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td style={{ color: '#60a5fa', fontFamily: 'monospace', fontWeight: '600' }}>
                        {ev.time_range || ev.time_formatted}
                      </td>
                      <td>{ev.duration_seconds}s</td>
                      <td style={{ fontWeight: '600' }}>{ev.event_type}</td>
                      <td>{ev.object_name}</td>
                      <td>{ev.confidence_pct}%</td>
                      <td>
                        <span className={`badge ${ev.risk_level === 'HIGH' ? 'badge-red' : ev.risk_level === 'MEDIUM' ? 'badge-amber' : 'badge-green'}`}>
                          {ev.risk_score}/100 [{ev.risk_level}]
                        </span>
                      </td>
                      <td>
                        <button className="btn btn-secondary" style={{ padding: '2px 8px', fontSize: '11px' }}>
                          Inspect Frame
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div style={{ padding: '16px', color: '#4ade80', fontSize: '13px' }}>
                🟢 No configured restricted-object events detected in video timeline.
              </div>
            )}
          </div>

          {/* Extracted Evidence Screenshot Gallery */}
          <div className="panel-section">
            <h3 className="section-title">🖼️ Extracted Evidence Frame Screenshots</h3>
            {evidenceList.length > 0 ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '16px' }}>
                {evidenceList.map((item) => (
                  <div
                    key={item.incident_id}
                    style={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '6px', overflow: 'hidden', padding: '10px', cursor: 'pointer' }}
                    onClick={() => setSelectedEvidence(item)}
                  >
                    <div style={{ fontSize: '12px', fontWeight: '700', color: '#f8fafc', marginBottom: '4px' }}>
                      {item.event_type}
                    </div>
                    {item.file_exists ? (
                      <img
                        src={item.evidence_url}
                        alt="Evidence Screenshot"
                        style={{ width: '100%', height: '160px', objectFit: 'cover', borderRadius: '4px', border: '1px solid #475569' }}
                      />
                    ) : (
                      <div style={{ height: '160px', backgroundColor: '#0f172a', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', fontSize: '12px' }}>
                        ⚠️ Evidence frame unavailable
                      </div>
                    )}
                    <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '6px', display: 'flex', justifyContent: 'space-between' }}>
                      <span>Time: {item.timestamp}</span>
                      <span>Conf: {item.confidence}%</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ fontSize: '12px', color: '#64748b' }}>No evidence screenshots extracted.</p>
            )}
          </div>
        </div>
      )}

      {/* Full Evidence Image Viewer Modal */}
      {selectedEvidence && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.85)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '20px'
          }}
          onClick={() => setSelectedEvidence(null)}
        >
          <div
            style={{ maxWidth: '850px', width: '100%', backgroundColor: '#1e293b', padding: '20px', borderRadius: '8px', border: '1px solid #334155' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div>
                <h3 style={{ fontSize: '16px', color: '#f8fafc', margin: 0 }}>
                  {selectedEvidence.event_type}
                </h3>
                <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>
                  Incident ID: <code style={{ color: '#60a5fa' }}>{selectedEvidence.incident_id}</code> | Time: {selectedEvidence.timestamp || selectedEvidence.time_formatted}
                </div>
              </div>
              <button
                onClick={() => setSelectedEvidence(null)}
                style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '16px' }}
              >
                ✕ Close
              </button>
            </div>

            {selectedEvidence.evidence_url || selectedEvidence.file_path ? (
              <img
                src={selectedEvidence.evidence_url || `/evidence_files/${selectedEvidence.file_path.split(/[\\\\/]/).pop()}`}
                alt="Full High Res Evidence Frame"
                style={{ width: '100%', borderRadius: '6px', border: '1px solid #334155' }}
              />
            ) : (
              <div style={{ padding: '40px', textAlign: 'center', color: '#94a3b8' }}>
                Snapshot image frame unavailable.
              </div>
            )}

            <div style={{ marginTop: '12px', fontSize: '12px', color: '#94a3b8', display: 'flex', justifyContent: 'space-between' }}>
              <span>Item: <strong>{selectedEvidence.object_name}</strong></span>
              <span>Peak Confidence: <strong>{selectedEvidence.confidence_pct || selectedEvidence.confidence}%</strong></span>
              <span>
                Risk Score:{' '}
                <span className={`badge ${selectedEvidence.risk_level === 'HIGH' ? 'badge-red' : selectedEvidence.risk_level === 'MEDIUM' ? 'badge-amber' : 'badge-green'}`}>
                  {selectedEvidence.risk_score}/100 [{selectedEvidence.risk_level}]
                </span>
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
