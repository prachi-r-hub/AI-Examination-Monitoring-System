import React, { useState, useEffect } from 'react';

export default function IncidentsPage() {
  const [incidents, setIncidents] = useState([]);
  const [statusFilter, setStatusFilter] = useState('All Statuses');
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [noteInput, setNoteInput] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchIncidents = async () => {
    try {
      const url = statusFilter === 'All Statuses' ? '/api/incidents' : `/api/incidents?status=${statusFilter}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setIncidents(data.incidents || []);
      }
    } catch (err) {
      console.error('Error fetching incidents:', err);
    }
  };

  useEffect(() => {
    fetchIncidents();
  }, [statusFilter]);

  const handleUpdateStatus = async (incidentId, newStatus) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/incidents/${incidentId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_status: newStatus }),
      });
      if (res.ok) {
        const data = await res.json();
        if (selectedIncident && selectedIncident.incident_id === incidentId) {
          setSelectedIncident(data.incident);
        }
        fetchIncidents();
      }
    } catch (err) {
      alert('Error updating status: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveNote = async (incidentId) => {
    if (!noteInput.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/incidents/${incidentId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes: noteInput }),
      });
      if (res.ok) {
        const data = await res.json();
        if (selectedIncident && selectedIncident.incident_id === incidentId) {
          setSelectedIncident(data.incident);
        }
        setNoteInput('');
        fetchIncidents();
        alert('Note saved successfully!');
      }
    } catch (err) {
      alert('Error saving note: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="header-bar">
        <div>
          <h1 className="page-title">📋 Incident Review Dashboard</h1>
          <p className="page-description">
            Inspect, review, update proctor status, and add persistent notes to confirmed examination monitoring incidents in the SQLite database.
          </p>
        </div>
        <div>
          <a href="/api/export/csv" download="examination_incidents_export.csv" className="btn btn-secondary">
            📥 Export to CSV
          </a>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="panel-section" style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <label className="form-label" style={{ margin: 0 }}>Filter by Review Status:</label>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="form-select"
            style={{ width: '220px' }}
          >
            <option value="All Statuses">All Statuses</option>
            <option value="Unreviewed">Unreviewed</option>
            <option value="Reviewed">Reviewed</option>
            <option value="Dismissed">Dismissed</option>
          </select>
          <span style={{ fontSize: '12px', color: '#94a3b8', marginLeft: 'auto' }}>
            Total Records: {incidents.length}
          </span>
        </div>
      </div>

      {/* Main Content Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: selectedIncident ? '1fr 420px' : '1fr', gap: '24px' }}>
        {/* Incidents Table */}
        <div className="panel-section">
          <h3 className="section-title">Incidents Audit Log</h3>
          {incidents.length > 0 ? (
            <table className="control-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Timestamp</th>
                  <th>Event Type</th>
                  <th>Object</th>
                  <th>Confidence</th>
                  <th>Duration</th>
                  <th>Risk Score</th>
                  <th>Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {incidents.map((inc) => (
                  <tr
                    key={inc.incident_id}
                    onClick={() => { setSelectedIncident(inc); setNoteInput(inc.notes || ''); }}
                    style={{
                      cursor: 'pointer',
                      backgroundColor: selectedIncident?.incident_id === inc.incident_id ? 'rgba(37, 99, 235, 0.15)' : 'transparent'
                    }}
                  >
                    <td style={{ fontFamily: 'monospace', fontSize: '11px', color: '#94a3b8' }}>{inc.incident_id}</td>
                    <td>{inc.time_formatted}</td>
                    <td style={{ fontWeight: '600' }}>{inc.event_type}</td>
                    <td>{inc.object_name}</td>
                    <td>{Math.round(inc.confidence * 100)}%</td>
                    <td>{inc.duration_seconds}s</td>
                    <td>
                      <span className={`badge ${inc.risk_level === 'HIGH' ? 'badge-red' : inc.risk_level === 'MEDIUM' ? 'badge-amber' : 'badge-green'}`}>
                        {inc.risk_score}/100 [{inc.risk_level}]
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${inc.review_status === 'Reviewed' ? 'badge-green' : inc.review_status === 'Dismissed' ? 'badge-amber' : 'badge-red'}`}>
                        {inc.review_status}
                      </span>
                    </td>
                    <td>
                      <button className="btn btn-secondary" style={{ padding: '2px 8px', fontSize: '11px' }}>
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p style={{ fontSize: '12px', color: '#64748b' }}>No incidents match the selected filter.</p>
          )}
        </div>

        {/* Selected Incident Inspector Panel */}
        {selectedIncident && (
          <div className="panel-section" style={{ border: '1px solid #2563eb' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <h3 className="section-title" style={{ margin: 0 }}>Incident Inspector</h3>
              <button
                onClick={() => setSelectedIncident(null)}
                style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '14px' }}
              >
                ✕ Close
              </button>
            </div>

            <div style={{ fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px' }}>
              <div><strong>Incident ID:</strong> <code style={{ color: '#60a5fa' }}>{selectedIncident.incident_id}</code></div>
              <div><strong>Timestamp:</strong> {selectedIncident.time_formatted}</div>
              <div><strong>Event Wording:</strong> <span style={{ color: '#f87171', fontWeight: '600' }}>{selectedIncident.event_type}</span></div>
              <div><strong>Detected Item:</strong> {selectedIncident.object_name} ({Math.round(selectedIncident.confidence * 100)}% conf)</div>
              <div><strong>Continuous Duration:</strong> {selectedIncident.duration_seconds} seconds</div>
              <div>
                <strong>Risk Score:</strong>{' '}
                <span className={`badge ${selectedIncident.risk_level === 'HIGH' ? 'badge-red' : selectedIncident.risk_level === 'MEDIUM' ? 'badge-amber' : 'badge-green'}`}>
                  {selectedIncident.risk_score}/100 [{selectedIncident.risk_level}]
                </span>
              </div>
              <div><strong>Review Status:</strong> <code>{selectedIncident.review_status}</code></div>
            </div>

            {/* Evidence Image Preview */}
            {selectedIncident.evidence_path && (
              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8', marginBottom: '4px' }}>Annotated Evidence Frame:</div>
                <img
                  src={`/evidence_files/${selectedIncident.evidence_path.split(/[\\\\/]/).pop()}`}
                  alt="Evidence"
                  style={{ width: '100%', borderRadius: '6px', border: '1px solid #334155' }}
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              </div>
            )}

            {/* Proctor Review Actions */}
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
              <button
                className="btn btn-primary"
                style={{ flex: 1, fontSize: '12px' }}
                onClick={() => handleUpdateStatus(selectedIncident.incident_id, 'Reviewed')}
                disabled={loading}
              >
                ✅ Mark Reviewed
              </button>
              <button
                className="btn btn-danger"
                style={{ flex: 1, fontSize: '12px' }}
                onClick={() => handleUpdateStatus(selectedIncident.incident_id, 'Dismissed')}
                disabled={loading}
              >
                ❌ Dismiss
              </button>
            </div>

            {/* Proctor Note Editor */}
            <div>
              <label className="form-label">Invigilator Review Notes</label>
              <textarea
                value={noteInput}
                onChange={(e) => setNoteInput(e.target.value)}
                placeholder="Enter invigilator notes here..."
                rows="3"
                className="form-input"
                style={{ marginBottom: '8px' }}
              />
              <button
                className="btn btn-secondary"
                style={{ width: '100%', fontSize: '12px' }}
                onClick={() => handleSaveNote(selectedIncident.incident_id)}
                disabled={loading}
              >
                📝 Save Invigilator Note
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
