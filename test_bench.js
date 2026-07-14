const WebSocket = require('ws');
const { performance } = require('perf_hooks');

console.log("🚀 Iniciando Benchmark do DeepSeek R1 no Servidor Plural...");

const ws = new WebSocket('ws://127.0.0.1:11438/stream');

let startTime = 0;
let firstTokenTime = 0;
let tokenCount = 0;
let fullResponse = "";

ws.on('open', function open() {
  console.log("✅ Conectado ao Motor LLM na porta 11438.");
  
  const payload = {
    prompt: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nVocê é um assistente criativo. Você DEVE obrigatoriamente responder em Português do Brasil (PT-BR) e todas as suas respostas devem ser criativas e diretas.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nEscreva um mini conto de ficção científica sobre uma inteligência artificial que descobriu como comprimir memória fisicamente. Escreva APENAS a história em português.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    max_tokens: 500,
    temperature: 0.7
  };

  console.log("📥 Enviando prompt pesado para o DeepSeek R1...");
  startTime = performance.now();
  ws.send(JSON.stringify(payload));
});

ws.on('message', function message(data) {
  const text = data.toString();
  
  if (text === "[DONE]") {
    const totalTime = (performance.now() - startTime) / 1000;
    const genTime = (performance.now() - firstTokenTime) / 1000;
    const tps = tokenCount / genTime;
    
    console.log("\n\n========================================");
    console.log("🏁 BENCHMARK CONCLUÍDO!");
    console.log("========================================");
    console.log(`⏱️ Tempo Total (Prompt + Geração): ${totalTime.toFixed(2)} segundos`);
    console.log(`⚡ Velocidade de Geração Bruta: ${tps.toFixed(2)} Tokens por Segundo (TPS)`);
    console.log(`📝 Tokens Gerados: ${tokenCount}`);
    console.log("========================================");
    
    ws.close();
    return;
  }

  if (firstTokenTime === 0) {
    firstTokenTime = performance.now();
    const ttfb = (firstTokenTime - startTime) / 1000;
    console.log(`\n⏱️ Tempo para o 1º Token (TTFB): ${ttfb.toFixed(2)} segundos`);
    console.log("💬 Resposta do Modelo:\n");
  }

  // Llama.cpp usually streams token by token
  process.stdout.write(text);
  tokenCount++;
});

ws.on('error', function error(err) {
  console.error("❌ Erro no WebSocket:", err.message);
  console.log("Verifique se você rodou 'npm install ws' nesta pasta ou se o servidor está online.");
});
