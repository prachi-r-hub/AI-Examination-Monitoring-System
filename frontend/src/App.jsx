import React, { useState } from 'react';
import { Camera, Image as ImageIcon, Video, ClipboardList, ShieldAlert, Settings } from 'lucide-react';
import MonitoringPage from './components/MonitoringPage';
import ImageDetectionPage from './components/ImageDetectionPage';
import VideoAnalysisPage from './components/VideoAnalysisPage';
import IncidentsPage from './components/IncidentsPage';
import EvidencePage from './components/EvidencePage';
import SettingsPage from './components/SettingsPage';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('monitoring');

  return (
    <div className="app-container">
      {/* Control Room Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-title">
            <span>🛡️ EXAM MONITOR</span>
          </div>
          <div className="sidebar-subtitle">Control Room Dashboard</div>
        </div>

        <ul className="nav-menu">
          <li className={`nav-item ${activeTab === 'monitoring' ? 'active' : ''}`}>
            <button onClick={() => setActiveTab('monitoring')}>
              <Camera size={18} />
              <span>Monitoring</span>
            </button>
          </li>
          <li className={`nav-item ${activeTab === 'image' ? 'active' : ''}`}>
            <button onClick={() => setActiveTab('image')}>
              <ImageIcon size={18} />
              <span>Image Detection</span>
            </button>
          </li>
          <li className={`nav-item ${activeTab === 'video' ? 'active' : ''}`}>
            <button onClick={() => setActiveTab('video')}>
              <Video size={18} />
              <span>Video Analysis</span>
            </button>
          </li>
          <li className={`nav-item ${activeTab === 'incidents' ? 'active' : ''}`}>
            <button onClick={() => setActiveTab('incidents')}>
              <ClipboardList size={18} />
              <span>Incidents</span>
            </button>
          </li>
          <li className={`nav-item ${activeTab === 'evidence' ? 'active' : ''}`}>
            <button onClick={() => setActiveTab('evidence')}>
              <ShieldAlert size={18} />
              <span>Evidence</span>
            </button>
          </li>
          <li className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`}>
            <button onClick={() => setActiveTab('settings')}>
              <Settings size={18} />
              <span>Settings</span>
            </button>
          </li>
        </ul>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {activeTab === 'monitoring' && <MonitoringPage />}
        {activeTab === 'image' && <ImageDetectionPage />}
        {activeTab === 'video' && <VideoAnalysisPage />}
        {activeTab === 'incidents' && <IncidentsPage />}
        {activeTab === 'evidence' && <EvidencePage />}
        {activeTab === 'settings' && <SettingsPage />}
      </main>
    </div>
  );
}
