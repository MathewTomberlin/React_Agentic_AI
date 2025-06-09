import {useState} from 'react';

function unescapeText(text) {
  // Replace common escaped characters
  return (text ?? '')
    .replaceAll('\\n', '\n')
    .replaceAll('\\t', '\t')
    .replaceAll('\\r', '\r')
    .replaceAll("\\'", "'")
    .replaceAll('\\"', '"')
    .replaceAll('\\]', ']')
    .replaceAll('\\[', '[')
    .replaceAll('\\)', ')')
    .replaceAll('\\(', '(')
    .replaceAll('\\}', '}')
    .replaceAll('\\{', '{')
    .replaceAll('\\\\', '\\');
}

function Message({ message }) {
    const [modalImg, setModalImg] = useState(null);

    if (message.type === "image_result") {
      return (
        <div className="message image-result">
          <div className="images">
            {message.images.map((img, i) => (
              <img 
                  key={i} 
                  src={img.data_uri} 
                  alt={`Generated ${i}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setModalImg(img.data_uri)} 
              />
            ))}
          </div>
          <div><strong>Prompt:</strong> {message.enhanced_prompt}</div>
          {modalImg && (
            <div className="image-modal" onClick={() => setModalImg(null)} tabIndex={-1} >
              <img src={modalImg} alt="Enlarged" className="image-modal-img" />
            </div>
          )}
        </div>
      );
    }

    return (
      <div className={`message ${message.type}`}>
        <span style={{ whiteSpace: 'pre-wrap' }} dangerouslySetInnerHTML={{ __html: unescapeText(message.text) }} />
      </div>
    );
}

export default Message;