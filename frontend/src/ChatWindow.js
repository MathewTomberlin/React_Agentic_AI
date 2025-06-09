import React, { useEffect, useRef } from 'react';
import Message from './Message';
import './css/ChatWindow.css';

function ChatWindow({ messages }) {
  const endRef = useRef(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="chat-window">
      {messages.map((msg, idx) => <Message key={idx} message={msg} />)}
      <div ref={endRef} />
    </div>
  );
}

export default ChatWindow;