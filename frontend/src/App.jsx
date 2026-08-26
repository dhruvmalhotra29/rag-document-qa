import { useState, useRef, useEffect } from "react";
import api from "./services/api"
import "./App.css";

function App() {

  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [pdfUploaded, setPdfUploaded] = useState(false);
  const fileInputRef = useRef(null);
  const chatEndref = useRef(null);

  const uploadPDF = async () => {

    if(!file) {
      setStatus("❌ Please select a PDF file");
      return;
    }

    if(file.type !== "application/pdf") {
      setStatus("❌ Only PDF files are allowed");
      return;
    }

    setUploading(true);

    try {
        const formData = new FormData();

        formData.append("file", file)

        await api.post("/upload-pdf", formData);

        setStatus("✅ PDF uploaded successfully");
        setPdfUploaded(true);
        setMessages([]);
        setQuestion("");
        setFile(null);
        fileInputRef.current.value = "";
        setUploading(false);

    } catch (error) {
      setStatus("❌ Upload failed");
      setPdfUploaded(false);
      setUploading(false);
      console.error(error);
    } finally {
      setUploading(false);
    }
  };

  const askQuestion = async () => {

    if(!question.trim()) return;

    const currentQuestion = question;
    
    setQuestion("");
    setLoading(true);

    setMessages((prevMessages) => [
      ...prevMessages,
      {
        user: currentQuestion,
        ai: ""
      }
    ]);
    
    try {
      const response = await api.post("/query", {
        question: currentQuestion,
        history: messages
      });

    // Update the same message with AI answer
    setMessages((prevMessages) => {
      const updatedMessages = [...prevMessages];

      updatedMessages[updatedMessages.length - 1].ai =
        response.data.answer;

      updatedMessages[updatedMessages.length - 1].sources = 
        response.data.sources;

      console.log(updatedMessages);

      return updatedMessages;
    });
    
    } catch(error) {
      console.error(error);

      // Update the same message with error
      setMessages((prevMessages) => {
      const updatedMessages = [...prevMessages];

      updatedMessages[updatedMessages.length - 1].ai =
        "❌ Unable to get answer. Please try again.";

      return updatedMessages;
    });

    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    chatEndref.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages]);
  
  return (
    <div className="app">

    <div className="header">
      <h1>🤖 AI Document Assistant</h1>
      <p>Upload a PDF and ask questions in natural language.</p>
    </div>
      
      <div className="upload-section">
        <input ref={fileInputRef} type="file" accept=".pdf" onChange={(e) => setFile(e.target.files[0])}/>
        <p className="upload-hint">
          Supported format: PDF Only
        </p>
        
        <button onClick={uploadPDF} disabled={!file || uploading}>
          {uploading ? "Uploading..." : "Upload"}
        </button>
      </div>

      <p>{status}</p>
      {pdfUploaded && file && (
        <div className="current-file">
          📄 Current document: <strong>{file.name}</strong>  
        </div>
      )}

      <div className="chat-container">
        {messages.length === 0 && !loading && (
          <div className="empty-chat">
            {pdfUploaded ? (
              <>
                <h3>💬 Ask Questions</h3>
                <p>Ask questions in natural language.</p>
              </>
            ) : (
              <>
                <h3>📄 Upload a PDF</h3>
                <p>Upload your PDF to get started.</p>
              </>
            )}
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index}>
            <div className="message-user">
              <strong>You:</strong> {message.user}
            </div>

          {message.ai && 
            (<div className="message-ai">
              <strong>AI:</strong> {message.ai}
            </div>
          )}

              {message.sources && (
              <div className="sources">
                <strong>📄 Sources:</strong>

                {message.sources.map((source, sourceIndex) => (
                  <div key={sourceIndex}>
                    {source.filename} — Page {source.page}
                  </div>
                ))}
              </div>
            )}
            
        </div>
        ))}


        {loading && (
          <div className="message-ai">
            <strong>AI:</strong> Thinking...
          </div>
        )}

        <div ref={chatEndref}></div>
      </div>

        <div className="input-section">
          <input type="text" placeholder={pdfUploaded ? "Ask a question..." : "Upload a pdf first..." }
          value={question}
          disabled={!pdfUploaded}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if(e.key == "Enter") {
              askQuestion();
            }
          }}/>

          <button onClick={askQuestion} disabled={loading || !pdfUploaded}>
            {loading ? "Thinking...": "Send"}
          </button>
        </div>

      </div>
  );
}

export default App;