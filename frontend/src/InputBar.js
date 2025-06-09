import React, { useState, useEffect } from 'react';
import './css/InputBar.css';

function AnimatedEllipsis() {
  const [dots, setDots] = useState('');
  useEffect(() => {
    const interval = setInterval(() => {
      setDots(d => d.length < 6 ? d + ' .' : '');
    }, 200);
    return () => clearInterval(interval);
  }, []);
  return <span>{dots}</span>;
}

function InputBar({ onSubmit, agentStatus }) {
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (input.trim()) {
      onSubmit(input);
      setInput("");
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="input-bar">
        {agentStatus === 'processing' && (
          <div className="agent-status processing">
            Agent is processing<AnimatedEllipsis />
          </div>
        )}
        {agentStatus === 'done' && (
          <div className="agent-status done">
            Agent process completed
          </div>
        )}
        {agentStatus === 'connecting' && (
          <div className="agent-status connecting">
            Agent connecting<AnimatedEllipsis />
          </div>
        )}
        {agentStatus === 'connected' && (
          <div className="agent-status connected">
            Agent connected
          </div>
        )}
        {agentStatus === 'disconnected' && (
          <div className="agent-status disconnected">
            Agent disconnected
          </div>
        )}
        {agentStatus === '' && (
            <div className="agent-status"></div>
        )}
        <div className="input-group">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter your agent request..."
            />
            <button onClick={handleSend}>Send</button>
        </div>
    </div>
  );
}

export default InputBar;