import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import "./App.css";

type Message = {
  text: string;
  sender: "user" | "bot";
};

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [distMode, setDistMode] = useState("1");
  const [isLoading, setIsLoading] = useState(false);
  
  // Model management state
  const [modelsDir, setModelsDir] = useState(localStorage.getItem("modelsDir") || "");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState(localStorage.getItem("selectedModel") || "");
  const [hasAutoStarted, setHasAutoStarted] = useState(false);

  const [chatMessages, setChatMessages] = useState<Message[]>([
    { text: "Servidor Plural Inicializado com Sucesso. Como posso ajudar hoje?", sender: "bot" }
  ]);
  const [prompt, setPrompt] = useState("");
  const wsRef = useRef<WebSocket | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Load models when directory changes
  useEffect(() => {
    if (modelsDir) {
      invoke<string[]>("list_models", { path: modelsDir })
        .then((models) => {
          setAvailableModels(models);
          if (models.length > 0) {
            // Se não tem modelo selecionado ou o selecionado não está mais na pasta
            if (!selectedModel || !models.includes(selectedModel)) {
              const firstModel = models[0];
              setSelectedModel(firstModel);
              localStorage.setItem("selectedModel", firstModel);
            }
          }
        })
        .catch((err) => {
          console.error(err);
          setAvailableModels([]);
        });
    }
  }, [modelsDir]);

  // Handle selectedModel change
  const handleModelSelect = (val: string) => {
    setSelectedModel(val);
    localStorage.setItem("selectedModel", val);
  };

  // Auto-start se tiver tudo configurado no localStorage
  useEffect(() => {
    if (!hasAutoStarted && modelsDir && selectedModel && availableModels.includes(selectedModel)) {
      setHasAutoStarted(true);
      initEngine();
    }
  }, [availableModels, modelsDir, selectedModel, hasAutoStarted]);

  const selectDirectory = async () => {
    try {
      const selectedPath = await open({
        directory: true,
        multiple: false,
      });
      if (selectedPath && typeof selectedPath === 'string') {
        setModelsDir(selectedPath);
        localStorage.setItem("modelsDir", selectedPath);
      }
    } catch (err) {
      console.error("Failed to open dialog", err);
    }
  };

  const getFullModelPath = () => {
    if (!modelsDir || !selectedModel) return "";
    return modelsDir.replace(/[\\/]+$/, "") + "/" + selectedModel;
  };

  const initEngine = async (isRetry = false) => {
    const fullPath = getFullModelPath();
    if (!fullPath) {
      alert("Por favor, selecione uma pasta válida e um modelo.");
      return;
    }

    setIsLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:11438/api/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: fullPath, distillation_mode: parseInt(distMode) })
      });
      const data = await res.json();
      if (data.status === 'success') {
        if (!isConnected) {
          connectWS();
        } else {
          // Add a system message indicating the model was switched
          setChatMessages((prev) => [...prev, { text: `[Sistema] Modelo alterado para ${selectedModel}`, sender: "bot" }]);
        }
      } else {
        if (data.message && data.message.includes("BackendAlreadyInitialized") && !isRetry) {
          console.log("Reiniciando o backend para trocar de modelo...");
          await invoke("restart_server");
          setTimeout(() => {
             initEngine(true);
          }, 1500); // Espera 1.5s para o servidor subir novamente
          return; // Retorna sem desligar o isLoading, o retry cuidará disso
        } else {
          alert(data.message);
        }
      }
    } catch (e) {
      if (!isRetry) {
          console.log("Servidor não respondeu, tentando reiniciar o backend...");
          try { await invoke("restart_server"); } catch(ex) {}
          setTimeout(() => {
             initEngine(true);
          }, 1500);
          return;
      } else {
          alert("Erro ao conectar no servidor! O binário servidor_plural_apps_sem_mic.exe pode estar indisponível no momento.");
      }
    }
    
    setIsLoading(false);
  };

  const connectWS = () => {
    const ws = new WebSocket("ws://127.0.0.1:11438/stream");
    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      setChatMessages((prev) => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg && lastMsg.sender === "bot" && lastMsg !== prev[0] && !lastMsg.text.startsWith("[Sistema]")) {
           const newMessages = [...prev];
           newMessages[newMessages.length - 1] = { ...lastMsg, text: lastMsg.text + event.data };
           return newMessages;
        } else {
           return [...prev, { text: event.data, sender: "bot" }];
        }
      });
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    wsRef.current = ws;
  };

  const sendPrompt = () => {
    const text = prompt.trim();
    if (!text || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    
    setChatMessages((prev) => [...prev, { text, sender: "user" }]);
    setPrompt("");
    
    // reset textarea height
    if (textareaRef.current) {
        textareaRef.current.style.height = '60px';
    }
    
    wsRef.current.send(JSON.stringify({
        action: "generate",
        prompt: text,
        max_tokens: 2048,
        temperature: 0.7
    }));
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setPrompt(e.target.value);
    e.target.style.height = '60px';
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
  };

  return (
    <div className="app-wrapper">
      <div className="glow-orb orb-1"></div>
      <div className="glow-orb orb-2"></div>

      <header>
        <div className="header-left">
          <h1>Servidor Plural</h1>
          <div className="status-badge" id="statusBadge">
              <div className={`indicator ${isConnected ? "connected" : ""}`}></div>
              <span>{isConnected ? "Conectado" : "Aguardando"}</span>
          </div>
        </div>
        {isConnected && (
          <div className="header-right">
             <select className="header-model-select" value={selectedModel} onChange={(e) => handleModelSelect(e.target.value)}>
                {availableModels.map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
             </select>
             <button className="icon-btn" onClick={() => initEngine()} disabled={isLoading} title="Trocar Modelo">
               {isLoading ? "..." : "🔄"}
             </button>
          </div>
        )}
      </header>

      <main>
        {!isConnected ? (
          <div className="setup-container">
            <div className="setup-title">Configurar Modelo</div>
            
            <div className="form-group">
                <label>Pasta de Modelos</label>
                <div className="file-input-group">
                  <input 
                    type="text" 
                    value={modelsDir} 
                    readOnly 
                    placeholder="Selecione a pasta com os .gguf..." 
                  />
                  <button onClick={selectDirectory} className="browse-btn">📁 Procurar</button>
                </div>
            </div>

            {modelsDir && (
              <div className="form-group">
                  <label>Selecione o Modelo</label>
                  <select value={selectedModel} onChange={(e) => handleModelSelect(e.target.value)}>
                      {availableModels.length === 0 && <option value="">Nenhum modelo .gguf encontrado</option>}
                      {availableModels.map(m => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                  </select>
              </div>
            )}
            
            <div className="form-group">
                <label>Estratégia de Destilação</label>
                <select value={distMode} onChange={(e) => setDistMode(e.target.value)}>
                    <option value="1">Padrão (Foco em Precisão)</option>
                    <option value="2">Alta Velocidade (Otimizado Q4)</option>
                    <option value="3">Baixa VRAM (Economia Máxima)</option>
                </select>
            </div>
            
            <button className="primary" disabled={isLoading || !modelsDir || !selectedModel} onClick={() => initEngine()}>
              {isLoading ? "Iniciando..." : "Carregar Modelo"}
            </button>
          </div>
        ) : (
          <div className="chat-layout">
              <div className="chat-messages">
                  <div className="chat-history">
                    {chatMessages.map((msg, idx) => (
                      <div key={idx} className={`message-row ${msg.sender}`}>
                        <div className={`avatar ${msg.sender}`}>
                           {msg.sender === "bot" ? "🤖" : "👤"}
                        </div>
                        <div className="message-content">
                          {msg.sender === "bot" ? (
                             <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                          ) : (
                             msg.text
                          )}
                        </div>
                      </div>
                    ))}
                    <div ref={chatEndRef} className="scroll-anchor" />
                  </div>
              </div>
              <div className="input-area">
                  <div className="input-container">
                    <textarea 
                      ref={textareaRef}
                      value={prompt}
                      onChange={handleInput}
                      placeholder="Pergunte ao Plural (Shift+Enter para nova linha)" 
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) { 
                          e.preventDefault(); 
                          sendPrompt(); 
                        }
                      }}
                    />
                    <button className="send-btn" onClick={sendPrompt} disabled={!prompt.trim()}>
                      <svg stroke="currentColor" fill="none" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" height="20" width="20" xmlns="http://www.w3.org/2000/svg"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                    </button>
                  </div>
                  <div className="footer-text">
                    O Servidor Plural pode cometer erros. Considere verificar informações importantes.
                  </div>
              </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
