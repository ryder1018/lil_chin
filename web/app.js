// 瀏覽器假裝置：用電腦的麥克風和喇叭模擬之後的實體機器人。
// 跟 ESP32 一樣走 OTA → WebSocket 協定（見 server/protocol.py），Opus 編解碼用瀏覽器內建的 WebCodecs。

const $ = (id) => document.getElementById(id);
const IN_RATE = 16000, OUT_RATE = 24000, FRAME_US = 60000;

const STATE_TEXT = {
  offline: "按「開始」跟小魚說話",
  listening: "在聽喔，說吧",
  hearing: "在聽…",
  thinking: "想一下…",
  speaking: "",
};

let ws = null, state = "offline";
let micStream, micCtx, worklet, encoder, decoder, playCtx;
let tsIn = 0, tsOut = 0, nextTime = 0, sources = new Set(), ttsStopped = false;
let botBubble = null, sttAt = 0, firstAudioAt = 0, hearingTimer = 0;

const deviceId = (() => {
  try {
    let id = localStorage.getItem("robot_device_id");
    if (!id) {
      id = "web-" + Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => b.toString(16).padStart(2, "0")).join("");
      localStorage.setItem("robot_device_id", id);
    }
    return id;
  } catch { return "web-anon"; }
})();

function setState(s) {
  state = s;
  $("face").className = "face " + s;
  if (s !== "speaking") $("state").textContent = STATE_TEXT[s];
  const online = s !== "offline";
  $("dot").classList.toggle("on", online);
  $("conn").textContent = online ? "已連線" : "未連線";
  $("btnConnect").textContent = online ? "結束" : "開始";
  $("btnConnect").classList.toggle("primary", !online);
  $("btnAbort").disabled = s !== "speaking" && s !== "thinking";
  $("textInput").disabled = $("btnSend").disabled = !online;
}

function addMsg(kind, text) {
  const el = document.createElement("div");
  el.className = "msg " + kind;
  el.textContent = text;
  $("log").appendChild(el);
  $("log").scrollTop = $("log").scrollHeight;
  return el;
}

function addMeta(el, text) {
  const m = document.createElement("span");
  m.className = "meta";
  m.textContent = text;
  el.appendChild(m);
}

// ---------- 連線 ----------
async function connect() {
  if (!("AudioEncoder" in window) || !("AudioDecoder" in window)) {
    addMsg("sys", "這個瀏覽器不支援 WebCodecs，請用 Chrome 或 Edge。");
    return;
  }
  $("btnConnect").disabled = true;
  try {
    const ota = await fetch("/xiaozhi/ota/", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Device-Id": deviceId, "Client-Id": "web" },
      body: JSON.stringify({ mac_address: deviceId, application: { name: "ai-robot-web", version: "0.1.0" } }),
    }).then((r) => r.json());
    const url = new URL(ota.websocket.url);
    // 用區網 IP 開網頁時，WebSocket 也連同一台
    if (["127.0.0.1", "localhost"].includes(url.hostname) && !["127.0.0.1", "localhost"].includes(location.hostname)) {
      url.hostname = location.hostname;
    }
    url.searchParams.set("device-id", deviceId);
    url.searchParams.set("client-id", "web");

    await startAudio();
    ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => ws.send(JSON.stringify({
      type: "hello", version: 1, transport: "websocket",
      audio_params: { format: "opus", sample_rate: IN_RATE, channels: 1, frame_duration: 60 },
    }));
    ws.onmessage = onMessage;
    ws.onclose = () => disconnect("連線結束");
  } catch (e) {
    addMsg("sys", "連線失敗：" + e.message);
    await disconnect();
  } finally {
    $("btnConnect").disabled = false;
  }
}

async function disconnect(reason) {
  if (ws) { ws.onclose = null; try { ws.close(); } catch {} ws = null; }
  stopPlayback();
  micStream?.getTracks().forEach((t) => t.stop());
  for (const c of [encoder, decoder]) { try { c?.close(); } catch {} }
  for (const c of [micCtx, playCtx]) { try { await c?.close(); } catch {} }
  micStream = micCtx = playCtx = encoder = decoder = worklet = null;
  $("level").style.width = "0";
  if (reason && state !== "offline") addMsg("sys", reason);
  setState("offline");
}

function sendJSON(obj) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

// ---------- 伺服器訊息 ----------
function onMessage(ev) {
  if (ev.data instanceof ArrayBuffer) {
    if (!firstAudioAt) {
      firstAudioAt = performance.now();
      setState("speaking");
    }
    decoder.decode(new EncodedAudioChunk({ type: "key", timestamp: tsOut, duration: FRAME_US, data: ev.data }));
    tsOut += FRAME_US;
    return;
  }
  const m = JSON.parse(ev.data);
  if (m.type === "hello") {
    addMsg("sys", "已連線，可以開始說話");
    sendJSON({ type: "listen", state: "start", mode: "auto" });
    setState("listening");
  } else if (m.type === "stt") {
    addMsg("user", m.text);
    sttAt = performance.now();
    firstAudioAt = 0;
    botBubble = null;
    setState("thinking");
  } else if (m.type === "tts") {
    if (m.state === "start") {
      ttsStopped = false;
    } else if (m.state === "sentence_start") {
      if (!botBubble) botBubble = addMsg("bot", "");
      botBubble.textContent += m.text;
      $("state").textContent = m.text;
      $("log").scrollTop = $("log").scrollHeight;
    } else if (m.state === "stop") {
      ttsStopped = true;
      maybeFinish();
    }
  }
}

function maybeFinish() {
  if (!ttsStopped || sources.size) return;
  if (botBubble && firstAudioAt && sttAt) {
    addMeta(botBubble, `辨識完 → 開口 ${Math.round(firstAudioAt - sttAt)} ms`);
  }
  ttsStopped = false;
  botBubble = null;
  if (ws) {
    sendJSON({ type: "listen", state: "start", mode: "auto" });
    setState("listening");
  }
}

// ---------- 麥克風 → Opus → 伺服器 ----------
async function startAudio() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  micCtx = new AudioContext({ sampleRate: IN_RATE });
  await micCtx.audioWorklet.addModule("/web/recorder-worklet.js");
  worklet = new AudioWorkletNode(micCtx, "recorder");
  const src = micCtx.createMediaStreamSource(micStream);
  const mute = micCtx.createGain();
  mute.gain.value = 0;
  src.connect(worklet).connect(mute).connect(micCtx.destination);

  encoder = new AudioEncoder({
    output: (chunk) => {
      if (ws?.readyState !== WebSocket.OPEN) return;
      const buf = new Uint8Array(chunk.byteLength);
      chunk.copyTo(buf);
      ws.send(buf);
    },
    error: (e) => addMsg("sys", "編碼錯誤：" + e.message),
  });
  encoder.configure({ codec: "opus", sampleRate: IN_RATE, numberOfChannels: 1, bitrate: 24000, opus: { frameDuration: FRAME_US } });

  worklet.port.onmessage = (e) => {
    const pcm = e.data;
    let sum = 0;
    for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
    const rms = Math.sqrt(sum / pcm.length);
    $("level").style.width = Math.min(100, rms * 600) + "%";
    // 半雙工：她在想或在說的時候不送麥克風（沒有回音消除）
    if (state !== "listening" && state !== "hearing") return;
    if (rms > 0.02) {
      if (state === "listening") setState("hearing");
      clearTimeout(hearingTimer);
      hearingTimer = setTimeout(() => state === "hearing" && setState("listening"), 700);
    }
    const ad = new AudioData({ format: "f32-planar", sampleRate: IN_RATE, numberOfFrames: pcm.length, numberOfChannels: 1, timestamp: tsIn, data: pcm });
    tsIn += FRAME_US;
    encoder.encode(ad);
    ad.close();
  };

  // ---------- 伺服器 → Opus → 喇叭 ----------
  playCtx = new AudioContext();
  decoder = new AudioDecoder({
    output: (ad) => {
      const pcm = new Float32Array(ad.numberOfFrames);
      ad.copyTo(pcm, { planeIndex: 0, format: "f32-planar" });
      ad.close();
      schedule(pcm);
    },
    error: (e) => addMsg("sys", "解碼錯誤：" + e.message),
  });
  decoder.configure({ codec: "opus", sampleRate: OUT_RATE, numberOfChannels: 1 });
}

function schedule(pcm) {
  const buf = playCtx.createBuffer(1, pcm.length, OUT_RATE);
  buf.copyToChannel(pcm, 0);
  const src = playCtx.createBufferSource();
  src.buffer = buf;
  src.connect(playCtx.destination);
  nextTime = Math.max(nextTime, playCtx.currentTime + 0.05);
  src.start(nextTime);
  nextTime += buf.duration;
  sources.add(src);
  src.onended = () => { sources.delete(src); maybeFinish(); };
}

function stopPlayback() {
  for (const s of sources) { s.onended = null; try { s.stop(); } catch {} }
  sources.clear();
  nextTime = 0;
}

// ---------- 按鈕 ----------
$("btnConnect").onclick = () => (state === "offline" ? connect() : disconnect("已結束"));
$("btnAbort").onclick = () => {
  sendJSON({ type: "abort" });
  stopPlayback();
  ttsStopped = true;
  maybeFinish();
};
$("textForm").onsubmit = (e) => {
  e.preventDefault();
  const text = $("textInput").value.trim();
  if (!text || !ws) return;
  sendJSON({ type: "listen", state: "detect", text });
  $("textInput").value = "";
};
setState("offline");
