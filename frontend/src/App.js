import React, { useEffect, useState, useRef } from 'react';
import SettingsPanel, {defaultSettings} from './SettingsPanel';
import ChatWindow from './ChatWindow';
import InputBar from './InputBar';
import './css/App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [settings, setSettings] = useState(defaultSettings);
  const [agentStatus, setAgentStatus] = useState('idle');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const websocket = useRef(null);

  useEffect(() => {
    // Connect to the WebSocket server
    setAgentStatus('connecting')
    websocket.current = new WebSocket("ws://localhost:8000/ws");

    websocket.current.onopen = () => {
      console.log("WebSocket connected");
      setAgentStatus('connected')
    };

    websocket.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // Handle incoming messages from the backend
      console.log("Received:", data);
      if (("message" in data && data.message) || ("images" in data && data.images)){
        setMessages(prev => [...prev, { type: data.type, text: data.message, ...data }]);
      }
      if ("final" in data){
        setAgentStatus('done');
      }
    };

    return () => {
      // Cleanup on component unmount
      websocket.current.close();
    };
  }, []); // Empty dependency array ensures this runs only once

  const handleSubmit = (userInput) => {
    if (websocket.current?.readyState === WebSocket.OPEN) {
      // Clear previous messages and add the new user message
      setMessages(prev => [...prev, { type: 'user', text: userInput }]);
      const payload = {
        userInput: userInput,
        settings: settings
      };
      websocket.current.send(JSON.stringify(payload));
      setAgentStatus('processing');
    }
  };

  return (
    <div className="App">
      <button
        className="sidebar-toggle"
        onClick={() => setSidebarOpen((open) => !open)}
        aria-label="Open settings"
      >
        <span className="hamburger"></span>
        <span className="hamburger"></span>
        <span className="hamburger"></span>
      </button>
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <SettingsPanel settings={settings} setSettings={setSettings} />
      </aside>
      <main className="main-content">
        <div className="chat-area">
          <ChatWindow messages={messages} />
        </div>
        <InputBar onSubmit={handleSubmit} agentStatus={agentStatus} />
      </main>
    </div>
  );
}

export default App;
