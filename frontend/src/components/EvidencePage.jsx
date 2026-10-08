import React, { useState, useEffect } from 'react';

export default function EvidencePage() {
  const [evidenceItems, setEvidenceItems] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);

  useEffect(() => {
    const fetchEvidence = async () => {
      try {
        const res = await fetch('/api/evidence');
        if (res.ok) {
          const data = await res.json();
          setEvidenceItems(data.evidence || []);
        }
      } catch (err) {
        console.error('Error fetching evidence gallery:', err);
      }
    };
    fetchEvidence();
  }, []);

  return (
    <div>
      <div className="header-bar">
        <div>
          <h1 className="page-title">🖼️ Captured Evidence Screenshot Gallery</h1>
          <p className="page-description">
            Inspect annotated evidence screenshot snapshots captured automatically by the evidence buffer manager.
          </p>
        </div>
        <div>
          <span className="badge badge-green" style={{ fontSize: '13px', padding: '6px 12px' }}>
            Total Files: {evidenceItems.length}
          </span>
        </div>
      </div>

      {evidenceItems.length > 0 ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
          {evidenceItems.map((item) => (
            <div
              key={item.incident_id}
              className="panel-section"
              style={{ cursor: 'pointer', transition: 'transform 0.15s ease' }}
              onClick={() => item.file_exists && setSelectedImage(item)}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontFamily: 'monospace', fontSize: '11px', color: '#60a5fa' }}>
                  {item.incident_id}
                </span>
                <span className={`badge ${item.risk_level === 'HIGH' ? 'badge-red' : item.risk_level === 'MEDIUM' ? 'badge-amber' : 'badge-green'}`}>
                  {item.risk_score}/100 [{item.risk_level}]
                </span>
              </div>

              <div style={{ fontSize: '13px', fontWeight: '600', marginBottom: '8px' }}>
                {item.event_type}
              </div>

              {item.file_exists && item.evidence_url ? (
                <img
                  src={item.evidence_url}
                  alt="Evidence"
                  style={{ width: '100%', height: '180px', objectFit: 'cover', borderRadius: '4px', border: '1px solid #334155' }}
                />
              ) : (
                <div style={{ height: '180px', backgroundColor: '#0f172a', border: '1px border-dashed #475569', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', fontSize: '12px' }}>
                  ⚠️ Evidence unavailable on disk
                </div>
              )}

              <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '8px', display: 'flex', justifyContent: 'space-between' }}>
                <span>Object: {item.object_name}</span>
                <span>Time: {item.timestamp}</span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="panel-section">
          <p style={{ fontSize: '13px', color: '#64748b' }}>No evidence screenshot files captured yet.</p>
        </div>
      )}

      {/* Full Image Viewer Modal */}
      {selectedImage && (
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
          onClick={() => setSelectedImage(null)}
        >
          <div style={{ maxWidth: '900px', width: '100%', backgroundColor: '#1e293b', padding: '20px', borderRadius: '8px', border: '1px solid #334155' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <h3 style={{ fontSize: '16px', color: 'white' }}>{selectedImage.event_type} ({selectedImage.incident_id})</h3>
              <button style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer', fontSize: '16px' }}>✕ Close</button>
            </div>
            <img src={selectedImage.evidence_url} alt="Full High Res Evidence" style={{ width: '100%', borderRadius: '6px' }} />
            <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '12px' }}>
              File Location: <code>{selectedImage.file_path}</code>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
