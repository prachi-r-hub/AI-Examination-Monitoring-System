import React, { useState } from 'react';

export default function ImageDetectionPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleRunDetection = async () => {
    if (!selectedFile) {
      alert('Please select or upload an image first.');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`/api/image/detect?conf=${confThreshold}`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error('Image detection failed');
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      alert('Error during image detection: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="header-bar">
        <div>
          <h1 className="page-title">📷 Image Detection & Class Verification</h1>
          <p className="page-description">
            Upload custom examination photos to verify target classes (`Notebook`, `Paper`, `Pen`, `Pencil`, `Phone`, `Smartwatch`, `Earbuds`).
          </p>
        </div>
      </div>

      {/* Upload & Controls Bar */}
      <div className="panel-section" style={{ marginBottom: '24px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 200px 150px', gap: '16px', alignItems: 'end' }}>
          <div>
            <label className="form-label">Select Image File</label>
            <input
              type="file"
              accept="image/jpeg,image/png,image/jpg"
              onChange={handleFileChange}
              className="form-input"
            />
          </div>
          <div>
            <label className="form-label">Confidence Threshold ({confThreshold})</label>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>
          <div>
            <button
              className="btn btn-primary"
              style={{ width: '100%' }}
              onClick={handleRunDetection}
              disabled={loading || !selectedFile}
            >
              {loading ? 'Processing...' : '🔍 RUN DETECTION'}
            </button>
          </div>
        </div>
      </div>

      {/* Results Display */}
      {result && (
        <div>
          <div style={{ display: 'flex', gap: '20px', marginBottom: '16px' }}>
            <div className="badge badge-green" style={{ fontSize: '13px', padding: '6px 12px' }}>
              🟢 Allowed Objects: {result.allowed_count}
            </div>
            <div className="badge badge-red" style={{ fontSize: '13px', padding: '6px 12px' }}>
              🔴 Restricted Objects: {result.restricted_count}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
            <div className="panel-section">
              <h3 className="section-title">Original Image</h3>
              <img
                src={result.original_image_b64}
                alt="Original"
                style={{ width: '100%', borderRadius: '6px', border: '1px solid #334155' }}
              />
            </div>
            <div className="panel-section">
              <h3 className="section-title">Annotated YOLOv8s Detection</h3>
              <img
                src={result.annotated_image_b64}
                alt="Annotated Output"
                style={{ width: '100%', borderRadius: '6px', border: '1px solid #334155' }}
              />
            </div>
          </div>

          {/* Detections Breakdown Table */}
          <div className="panel-section">
            <h3 className="section-title">📋 Object Detection Breakdown Payload</h3>
            {result.detections.length > 0 ? (
              <table className="control-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Object Name</th>
                    <th>Category</th>
                    <th>Confidence</th>
                    <th>Bounding Box Coordinates (x1, y1, x2, y2)</th>
                    <th>Center Point</th>
                  </tr>
                </thead>
                <tbody>
                  {result.detections.map((d) => (
                    <tr key={d.id}>
                      <td>{d.id}</td>
                      <td style={{ fontWeight: '600' }}>{d.object_name}</td>
                      <td>
                        <span className={`badge ${d.category === 'ALLOWED' ? 'badge-green' : 'badge-red'}`}>
                          {d.status_label}
                        </span>
                      </td>
                      <td>{d.confidence}%</td>
                      <td style={{ fontFamily: 'monospace' }}>[{d.bounding_box.join(', ')}]</td>
                      <td style={{ fontFamily: 'monospace' }}>[{d.center_point.join(', ')}]</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p style={{ fontSize: '12px', color: '#64748b' }}>No objects detected at confidence threshold {confThreshold}.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
