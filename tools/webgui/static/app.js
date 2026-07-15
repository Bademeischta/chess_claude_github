// ─── State ────────────────────────────────────────────────────────────
let game = new Chess(), board = null, busy = false;
let lastMoveSquares = [];
let humanWhite = true;
let gameOver = false;
let selectedSq = null, legalTargets = [], clickPolicy = {};

// ─── Helpers ──────────────────────────────────────────────────────────
async function post(u, d){
  let r = await fetch(u, {method:'POST', headers:{'Content-Type':'application/json'},
                           body:JSON.stringify(d||{})});
  return await r.json();
}
function setLive(state, txt){
  let d = document.getElementById('livedot');
  d.className = 'dot ' + (state||'');
  document.getElementById('livetxt').textContent = txt;
}
function setEval(v){
  // v ∈ [-1, 1] from side-to-move's POV. Always show from White's POV.
  let white_pov = (game.turn() === 'w') ? v : -v;
  let pct = 50 + white_pov * 50;
  pct = Math.max(2, Math.min(98, pct));
  let bar = document.getElementById('evalfill');
  bar.style.width = pct + '%';
  bar.classList.toggle('bad', white_pov < -0.05);
  let label = (white_pov >= 0 ? '+' : '') + white_pov.toFixed(2);
  document.getElementById('evalnum').textContent = label;
}
function updateMaterial(){
  // Count material directly from the chess.js board — independent of the
  // (potentially broken) value head. Standard piece values.
  const v = {p:1, n:3, b:3, r:5, q:9, k:0};
  let w = 0, b = 0;
  for (let row of game.board()){
    for (let sq of row){
      if (!sq) continue;
      let val = v[sq.type] || 0;
      if (sq.color === 'w') w += val; else b += val;
    }
  }
  let diff = w - b;
  let chip = document.getElementById('matchip');
  if (diff === 0){
    chip.textContent = 'Material: even';
    chip.className = 'matchip';
  } else if (diff > 0){
    chip.textContent = 'Material: White +' + diff;
    chip.className = 'matchip up';
  } else {
    chip.textContent = 'Material: Black +' + (-diff);
    chip.className = 'matchip down';
  }
  // Also update the move counter pill.
  let mc = document.getElementById('movecount');
  if (mc) mc.textContent = 'move ' + Math.ceil((game.history().length + 1) / 2);
}
function updateConfidence(a){
  // Compute confidence from the spread of Q-values across the AI's top moves.
  // Tiny spread = the network can't distinguish moves → don't trust eval.
  let chip = document.getElementById('confchip');
  if (!a || !a.top || a.top.length < 2){
    chip.textContent = 'confidence —';
    chip.className = 'confchip';
    return;
  }
  let qs = a.top.map(m => m.q);
  let spread = Math.max(...qs) - Math.min(...qs);
  let label, cls;
  if (spread < 0.05)      { label = 'confidence low (Δ' + spread.toFixed(3) + ')';   cls = 'low'; }
  else if (spread < 0.20) { label = 'confidence medium (Δ' + spread.toFixed(2) + ')'; cls = 'med'; }
  else                    { label = 'confidence high (Δ' + spread.toFixed(2) + ')';   cls = 'high'; }
  chip.textContent = label;
  chip.className = 'confchip ' + cls;
}

// ─── Top-move arrows overlay ───────────────────────────────────────────
function squareCenter(sq){
  // chessboard.js attaches data-square to every square cell. We measure
  // its position relative to the board's bounding rect, then return as
  // a percentage of the board so the SVG viewBox (0..100) lines up.
  let boardEl = document.getElementById('board');
  let cell = boardEl.querySelector('[data-square="' + sq + '"]');
  if (!cell || !boardEl) return null;
  let br = boardEl.getBoundingClientRect();
  let cr = cell.getBoundingClientRect();
  if (br.width <= 0) return null;
  let cx = (cr.left - br.left + cr.width / 2) / br.width * 100;
  let cy = (cr.top - br.top + cr.height / 2) / br.height * 100;
  return {x: cx, y: cy};
}
function drawArrows(top){
  let svg = document.getElementById('arrows');
  if (!svg) return;
  // Clear
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!top || !top.length) return;
  // Total visits for opacity normalisation
  let totalN = top.reduce((s, m) => s + (m.n||0), 0) || 1;
  // Heatmap: shade each candidate's destination square by visit share, so
  // you can SEE where the search concentrated (drawn first, arrows on top).
  const SQ = 12.5;
  for (let m of top){
    if (!m.uci || m.uci.length < 4) continue;
    let to = squareCenter(m.uci.slice(2, 4));
    if (!to) continue;
    let share = (m.n || 0) / totalN;
    let rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', (to.x - SQ/2).toFixed(2));
    rect.setAttribute('y', (to.y - SQ/2).toFixed(2));
    rect.setAttribute('width', SQ); rect.setAttribute('height', SQ);
    rect.setAttribute('rx', '1.5');
    rect.setAttribute('fill', '#3ddc97');
    rect.setAttribute('opacity', Math.max(0.05, Math.min(0.5, share*0.6)).toFixed(3));
    svg.appendChild(rect);
  }
  // Draw at most 3 arrows, in order. Strongest = green-ish, fading.
  const colors = ['#3ddc97', '#5aa9ff', '#ffb454'];
  for (let i = 0; i < Math.min(3, top.length); i++){
    let m = top[i];
    if (!m.uci || m.uci.length < 4) continue;
    let from = squareCenter(m.uci.slice(0, 2));
    let to   = squareCenter(m.uci.slice(2, 4));
    if (!from || !to) continue;
    let share = (m.n || 0) / totalN;
    let opacity = Math.max(0.25, Math.min(0.95, share * 1.6));
    // line + arrowhead
    let dx = to.x - from.x, dy = to.y - from.y;
    let len = Math.sqrt(dx*dx + dy*dy);
    if (len < 0.5) continue;
    // Shorten the line so it doesn't poke into the piece graphic.
    let shrink = 4.5 / len;
    let endX = to.x - dx * shrink;
    let endY = to.y - dy * shrink;
    let strokeW = 1.8 - i * 0.35;
    let line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', from.x); line.setAttribute('y1', from.y);
    line.setAttribute('x2', endX);   line.setAttribute('y2', endY);
    line.setAttribute('stroke', colors[i]);
    line.setAttribute('stroke-width', strokeW);
    line.setAttribute('stroke-linecap', 'round');
    line.setAttribute('opacity', opacity);
    svg.appendChild(line);
    // arrowhead — equilateral triangle pointing in direction (dx,dy)
    let ux = dx / len, uy = dy / len;
    let head = 3.4 - i * 0.6;
    let tipX = to.x - dx * (shrink * 0.5), tipY = to.y - dy * (shrink * 0.5);
    let p1 = (tipX + ux * head) + ',' + (tipY + uy * head);
    let p2 = (tipX - uy * head * 0.7) + ',' + (tipY + ux * head * 0.7);
    let p3 = (tipX + uy * head * 0.7) + ',' + (tipY - ux * head * 0.7);
    let poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', p1 + ' ' + p2 + ' ' + p3);
    poly.setAttribute('fill', colors[i]);
    poly.setAttribute('opacity', opacity);
    svg.appendChild(poly);
  }
}

function fmtMoveSan(uci){
  // Try to render as SAN via chess.js — but rolled back, so we use a clone.
  let g = new Chess(game.fen());
  let mv = g.move({from: uci.slice(0,2), to: uci.slice(2,4),
                   promotion: uci.length > 4 ? uci[4] : 'q'});
  return mv ? mv.san : uci;
}

// ─── Click-to-move & policy heatmap ────────────────────────────────────
function clearClickSelection(){
  selectedSq=null; legalTargets=[]; clickPolicy={};
  let ly=document.getElementById('clickLayer'); if(ly) ly.innerHTML='';
}
function _sqBox(sq){
  let c=squareCenter(sq); if(!c) return null;
  return {l:(c.x-6.25).toFixed(2), t:(c.y-6.25).toFixed(2)};
}
function renderClickLayer(){
  let ly=document.getElementById('clickLayer'); if(!ly) return;
  ly.innerHTML='';
  if(!selectedSq) return;
  // Selected-piece highlight
  let sb=_sqBox(selectedSq);
  if(sb){
    let d=document.createElement('div');
    d.className='click-sq sel';
    d.style.left=sb.l+'%'; d.style.top=sb.t+'%'; ly.appendChild(d);
  }
  // Legal target squares
  legalTargets.forEach(to=>{
    let b=_sqBox(to); if(!b) return;
    let p=clickPolicy[to]; let cap=!!game.get(to);
    let d=document.createElement('div');
    d.className='click-sq leg'+(cap?' cap':'')+(p!==undefined?' pol':'');
    d.style.left=b.l+'%'; d.style.top=b.t+'%';
    if(p!==undefined){
      // Policy heatmap: pale green → strong orange as probability rises
      let alpha=Math.max(0.14,Math.min(0.72,p));
      let hue=p<0.20?120:p<0.50?75:35;
      d.style.background='hsla('+hue+',85%,52%,'+alpha.toFixed(2)+')';
      let s=document.createElement('span'); s.className='click-pct';
      s.textContent=Math.round(p*100)+'%'; d.appendChild(s);
    }
    ly.appendChild(d);
  });
}
async function fetchPolicy(){
  let fen=game.fen(), src=selectedSq;
  try{
    let r=await fetch('/api/policy?fen='+encodeURIComponent(fen));
    let d=await r.json();
    if(!d.policy||selectedSq!==src) return;   // selection changed while awaiting
    // Aggregate by to-square (handles promotions: e7e8q + e7e8r + … → one dot)
    let toMap={}, total=0;
    legalTargets.forEach(to=>{toMap[to]=0;});
    for(let [uci,p] of Object.entries(d.policy)){
      let from=uci.slice(0,2), to=uci.slice(2,4);
      if(from===src && to in toMap){ toMap[to]+=p; total+=p; }
    }
    if(total>0){
      clickPolicy={};
      for(let [to,p] of Object.entries(toMap)) clickPolicy[to]=p/total;
    }
  }catch(e){}
  if(selectedSq===src) renderClickLayer();
}
function selectSquare(sq){
  clearClickSelection(); selectedSq=sq;
  let mvs=game.moves({square:sq,verbose:true});
  legalTargets=[...new Set(mvs.map(m=>m.to))];
  renderClickLayer(); fetchPolicy();
}
function onSquareClickCB(sq,piece){
  if(busy||gameOver) return;
  let myColor=humanWhite?'w':'b';
  let myTurn=game.turn()===myColor;
  let isOwn=piece&&piece[0]===myColor;
  if(selectedSq===null){ if(isOwn&&myTurn) selectSquare(sq); return; }
  if(sq===selectedSq){ clearClickSelection(); return; }
  if(isOwn&&myTurn){ selectSquare(sq); return; }
  if(legalTargets.includes(sq)){
    // promotion always defaults to queen (same behaviour as drag-and-drop)
    let mv=game.move({from:selectedSq,to:sq,promotion:'q'});
    if(mv){
      let uci=mv.from+mv.to+(mv.promotion||'');
      busy=true; highlightLastMove(uci); clearClickSelection();
      post('/api/move',{uci:uci}).then(res=>{
        if(res&&res.error){game.undo();board.position(game.fen());
          setLive('','illegal: '+res.error);busy=false;}
      });
    } else clearClickSelection();
  } else clearClickSelection();
}

// ─── Board ─────────────────────────────────────────────────────────────
function highlightLastMove(uci){
  // Remove previous highlights
  document.querySelectorAll('.highlight-last').forEach(el => el.classList.remove('highlight-last'));
  if (!uci || uci.length < 4) return;
  let from = uci.slice(0,2), to = uci.slice(2,4);
  let s1 = document.querySelector('[data-square="'+from+'"]');
  let s2 = document.querySelector('[data-square="'+to+'"]');
  if (s1) s1.classList.add('highlight-last');
  if (s2) s2.classList.add('highlight-last');
}
function highlightHint(uci){
  document.querySelectorAll('.highlight-hint').forEach(el => el.classList.remove('highlight-hint'));
  if (!uci || uci.length < 4) return;
  let from = uci.slice(0,2), to = uci.slice(2,4);
  let s1 = document.querySelector('[data-square="'+from+'"]');
  let s2 = document.querySelector('[data-square="'+to+'"]');
  if (s1) s1.classList.add('highlight-hint');
  if (s2) s2.classList.add('highlight-hint');
  setTimeout(() => {
    document.querySelectorAll('.highlight-hint').forEach(el => el.classList.remove('highlight-hint'));
  }, 4000);
}

function onDrop(src, tgt){
  if (busy || gameOver) return 'snapback';
  clearClickSelection();
  // Whose turn from chess.js's view?
  let turn = game.turn();   // 'w' or 'b'
  let ourTurn = (humanWhite && turn === 'w') || (!humanWhite && turn === 'b');
  if (!ourTurn) return 'snapback';
  // Try move locally
  let mv = game.move({from:src, to:tgt, promotion:'q'});
  if (mv === null) return 'snapback';
  let uci = mv.from + mv.to + (mv.promotion || '');
  busy = true;
  highlightLastMove(uci);
  // Send to server — server confirms via SSE event 'user_move'
  post('/api/move', {uci: uci}).then(res => {
    if (res && res.error){
      game.undo();
      board.position(game.fen());
      setLive('', 'illegal: '+res.error);
      busy = false;
    }
  });
}

// ─── SSE ──────────────────────────────────────────────────────────────
let es;
function connectStream(){
  es = new EventSource('/api/stream');
  es.onopen  = () => setLive('live', 'connected');
  es.onerror = () => setLive('', 'disconnected (auto-retry)');
  es.addEventListener('snapshot', e => applySnapshot(JSON.parse(e.data)));
  es.addEventListener('new_game', e => onNewGame(JSON.parse(e.data)));
  es.addEventListener('user_move', e => onUserMove(JSON.parse(e.data)));
  es.addEventListener('ai_thinking_start', e => onAIStart(JSON.parse(e.data)));
  es.addEventListener('ai_thinking', e => onAIProgress(JSON.parse(e.data)));
  es.addEventListener('ai_move', e => onAIMove(JSON.parse(e.data)));
  es.addEventListener('ai_done', e => onAIDone(JSON.parse(e.data)));
  es.addEventListener('ponder_state', e => onPonderState(JSON.parse(e.data)));
  es.addEventListener('ponder_update', e => onPonderUpdate(JSON.parse(e.data)));
  es.addEventListener('takeback', e => onTakeback(JSON.parse(e.data)));
  es.addEventListener('settings', e => applySettings(JSON.parse(e.data)));
  es.addEventListener('hint', e => onHint(JSON.parse(e.data)));
  es.addEventListener('game_over', e => onGameOver(JSON.parse(e.data)));
}

// ─── SSE handlers ─────────────────────────────────────────────────────
function applySnapshot(d){
  humanWhite = d.human_white;
  gameOver = d.game_over;
  game.load(d.fen);
  board.orientation(humanWhite ? 'white' : 'black');
  board.position(d.fen, false);
  renderHistory(d.history);
  applySettings(d.settings);
  updateMaterial();
  document.getElementById('gamestatus').textContent =
    gameOver ? ('GAME OVER — ' + (d.result||'')) : 'in progress';
}
function onNewGame(d){
  humanWhite = d.human_white;
  gameOver = false;
  clearClickSelection();
  game.load(d.fen);
  board.orientation(humanWhite ? 'white' : 'black');
  board.position(d.fen, false);
  document.getElementById('commentary').innerHTML =
    '<div class="lead">Game started</div><div>Make your first move.</div>';
  document.getElementById('commentary').className = '';
  document.getElementById('commentary').style.background = '';
  document.getElementById('commentary').style.borderColor = '';
  renderHistory([]);
  applySettings(d.settings);
  setEval(0);
  updateMaterial();
  drawArrows([]);  // clear any leftover arrows from previous game
  document.getElementById('gamestatus').textContent = 'your move';
  busy = false;
  highlightLastMove(null);
}
function onUserMove(d){
  game.load(d.fen);
  board.position(d.fen, true);
  highlightLastMove(d.uci);
  drawArrows([]);  // user just played — clear AI's recommendation arrows
  renderHistory(d.history);
  renderCommentary(d.commentary);
  updateMaterial();
  if (d.game_over) {
    onGameOver({result:d.result});
  }
}
function onAIStart(d){
  busy = true;
  resetConverge();
  document.getElementById('thinkmode').textContent = d.budget || d.mode || '';
  document.getElementById('progresstxt').textContent = 'starting…';
  document.getElementById('progressfill').style.width = '2%';
  document.getElementById('gamestatus').textContent = 'AI is thinking…';
  setLive('thinking', 'AI thinking');
}
function onAIProgress(d){
  let done = d.sims_done || 0;
  let target = d.sims_target;
  let pct = 0;
  let label = done + ' sims';
  if (target){
    pct = Math.min(99, 100 * done / target);
    label = done + ' / ' + target + ' sims';
  } else if (d.time_elapsed != null && d.time_budget){
    pct = Math.min(99, 100 * d.time_elapsed / d.time_budget);
    label = d.time_elapsed.toFixed(1) + 's / ' + d.time_budget.toFixed(1) + 's (' + done + ' sims)';
  }
  document.getElementById('progressfill').style.width = pct + '%';
  document.getElementById('progresstxt').textContent = label;
  if (d.analysis){
    renderAnalysis(d.analysis);
  }
  if (d.position) renderPosition(d.position);
}
function onAIMove(d){
  clearClickSelection();
  game.load(d.fen);
  board.position(d.fen, true);
  highlightLastMove(d.uci);
  drawArrows([]);  // AI just moved — clear the "I'm planning to play" arrows
  renderHistory(d.history);
  renderAnalysis(d.analysis);   // updates eval / top moves / pv / confidence
  drawArrows([]);  // renderAnalysis re-drew them; explicitly clear post-move
  updateMaterial();
  document.getElementById('progressfill').style.width = '100%';
  document.getElementById('progresstxt').textContent = 'done';
  busy = false;
  setLive('live', 'connected');
  document.getElementById('gamestatus').textContent =
    d.game_over ? 'GAME OVER — ' + (d.result||'') : 'your move';
  if (d.game_over) onGameOver({result:d.result});
}
function onAIDone(d){
  busy = false;
  setLive('live', 'connected');
  document.getElementById('gamestatus').textContent =
    d.game_over ? 'GAME OVER — ' + (d.result||'') : 'your move';
  if (d.game_over) onGameOver({result:d.result});
}
function onPonderState(d){
  let dot = document.getElementById('ponderdot');
  let lbl = document.getElementById('ponderlbl');
  if (d.running){
    dot.className = 'dot thinking';
    lbl.textContent = 'pondering on your time…';
  } else {
    dot.className = 'dot';
    lbl.textContent = 'ponder idle';
  }
}
function onPonderUpdate(d){
  document.getElementById('ponderlbl').textContent =
    'pondering — ' + d.sims + ' sims so far';
  if (d.analysis) renderAnalysis(d.analysis);
  if (d.position) renderPosition(d.position);
}
function onTakeback(d){
  clearClickSelection();
  game.load(d.fen);
  board.position(d.fen, true);
  renderHistory(d.history);
  drawArrows([]);
  updateMaterial();
  document.getElementById('commentary').innerHTML =
    '<div class="lead">Take back</div><div>Restored to before your last move.</div>';
  document.getElementById('commentary').className = '';
  document.getElementById('commentary').style.background = '';
  document.getElementById('commentary').style.borderColor = '';
  gameOver = false;
  document.getElementById('gamestatus').textContent = 'your move';
}
function onHint(d){
  if (d.top && d.top.length){
    highlightHint(d.top[0].uci);
    let c = document.getElementById('commentary');
    c.className = 'good';
    c.innerHTML = '<div class="lead">Hint</div>' +
                  '<div>I would play <b>' + fmtMoveSan(d.top[0].uci) +
                  '</b> (Q=' + d.top[0].q.toFixed(2) + ').</div>';
  }
  renderAnalysis(d);
}
function onGameOver(d){
  gameOver = true;
  document.getElementById('gamestatus').textContent = 'GAME OVER — ' + (d.result||'');
  let c = document.getElementById('commentary');
  c.className = '';
  c.innerHTML = '<div class="lead">Game over</div><div>' + (d.result||'') + '</div>';
}

// ─── Render ───────────────────────────────────────────────────────────
// Convergence tracking: how the engine's preferred move evolves during a
// search (does it keep changing its mind = uncertain, or lock in early?).
let convergeState = {best:null, changes:0};
function resetConverge(){
  convergeState = {best:null, changes:0};
  let el = document.getElementById('converge');
  if (el) el.textContent = '';
}
function trackConverge(top){
  if (!top || !top.length) return;
  let best = top[0].uci;
  if (convergeState.best && best !== convergeState.best) convergeState.changes++;
  convergeState.best = best;
  let el = document.getElementById('converge');
  if (!el) return;
  let san = fmtMoveSan(best);
  let tag = convergeState.changes === 0
    ? '<b>locked in</b> from the start'
    : 'changed its mind <b>' + convergeState.changes + '×</b>';
  el.innerHTML = 'Current pick: <b>' + san + '</b> · ' + tag;
}
// Replay a UCI line into SAN on a clone of the current position.
function sanLine(uciList){
  let g = new Chess(game.fen());
  let sans = [];
  for (let u of (uciList||[])){
    let mv = g.move({from:u.slice(0,2), to:u.slice(2,4),
                     promotion: u.length>4 ? u[4] : 'q'});
    if (!mv) break;
    sans.push(mv.san);
  }
  return sans;
}
function toggleMoveDetail(i){
  let row = document.getElementById('mdetail-'+i);
  if (row) row.style.display = (row.style.display === 'none' ? '' : 'none');
}
function renderAnalysis(a){
  if (!a) return;
  if (typeof a.value === 'number') setEval(a.value);
  if (typeof a.nodes === 'number') {
    document.getElementById('nodes').textContent =
      a.nodes ? (a.nodes.toLocaleString() + ' nodes · d' + (a.depth||0)) : '';
  }
  updateConfidence(a);
  drawArrows(a.top || []);
  // Prefer the richer `ext` (prior P, exploration U, mini-PV) when present;
  // fall back to plain `top` for older payloads.
  let rows = (a.ext && a.ext.length) ? a.ext : (a.top || []);
  trackConverge(rows);
  // Top moves
  let tbody = document.getElementById('topmoves');
  if (rows.length){
    let total = rows.reduce((s,m) => s + (m.n||0), 0) || 1;
    tbody.innerHTML = rows.map((m, i) => {
      let pct = (m.n||0) / total * 100;
      let cls = i === 0 ? 'top1' : '';
      let priorTxt = (typeof m.p === 'number') ? (m.p*100).toFixed(1)+'%' : '—';
      let main = '<tr class="moverow '+cls+'" onclick="toggleMoveDetail('+i+')">'+
             '<td>'+(i+1)+'</td><td>'+fmtMoveSan(m.uci)+'</td>'+
             '<td class="num">'+m.n+'</td>'+
             '<td class="num prior">'+priorTxt+'</td>'+
             '<td class="num">'+(m.q>=0?'+':'')+m.q.toFixed(2)+'</td>'+
             '<td><div class="bar"><span style="width:'+pct.toFixed(0)+'%"></span></div></td></tr>';
      // Detail row: PUCT decomposition + the move's own expected line.
      let detail = '';
      if (typeof m.p === 'number' || (m.line && m.line.length)){
        let puct = '<span class="puct">Q '+(m.q>=0?'+':'')+m.q.toFixed(3)+'</span>'+
                   '<span class="puct">prior '+priorTxt+'</span>'+
                   (typeof m.u==='number' ? '<span class="puct">explore +'+m.u.toFixed(3)+'</span>' : '');
        let sans = sanLine(m.line);
        let lineHtml = sans.length
          ? '<span class="sanline">'+sans.map(s=>'<span>'+s+'</span>').join('')+'</span>'
          : '<span style="color:var(--ink-fade)">no deeper line searched</span>';
        detail = '<tr id="mdetail-'+i+'" style="display:none"><td colspan="6" class="miniline">'+
                 puct+'<br>'+lineHtml+'</td></tr>';
      }
      return main + detail;
    }).join('');
  }
  // PV
  let pv = document.getElementById('pv');
  if (a.pv && a.pv.length){
    let sans = sanLine(a.pv);
    pv.innerHTML = sans.length ? sans.map(s => '<span>'+s+'</span>').join('') :
                                 '<span style="color:var(--ink-dim);background:transparent;border:none">—</span>';
  }
}
function renderPosition(pos){
  if (!pos) return;
  // WDL bar
  let w = pos.wdl || {};
  if (typeof w.win === 'number'){
    let pw = Math.max(0, w.win)*100, pd = Math.max(0, w.draw)*100, pl = Math.max(0, w.loss)*100;
    document.getElementById('wdl-w').style.width = pw.toFixed(1)+'%';
    document.getElementById('wdl-d').style.width = pd.toFixed(1)+'%';
    document.getElementById('wdl-l').style.width = pl.toFixed(1)+'%';
    document.getElementById('wdl-wt').textContent = pw.toFixed(0)+'%';
    document.getElementById('wdl-dt').textContent = pd.toFixed(0)+'%';
    document.getElementById('wdl-lt').textContent = pl.toFixed(0)+'%';
  }
  // Material / phase / aux
  if (typeof pos.material === 'number'){
    let m = pos.material;
    document.getElementById('pi-mat').textContent =
      (m>0?'White +'+m : m<0?'Black +'+(-m) : 'even');
  }
  if (pos.phase) document.getElementById('pi-phase').textContent = pos.phase;
  if (w && typeof w.aux === 'number')
    document.getElementById('pi-aux').textContent = (w.aux>=0?'+':'')+w.aux.toFixed(2);
  // Tactics
  let t = pos.tactics || {};
  let chips = [];
  if (t.in_check) chips.push('<span class="chk">in check</span>');
  (t.captures||[]).forEach(c =>
    chips.push('<span class="cap">×'+c.victim+' '+fmtMoveSan(c.uci)+'</span>'));
  (t.hanging||[]).forEach(h =>
    chips.push('<span class="hang">hanging '+h.piece+'</span>'));
  let tac = document.getElementById('pi-tac');
  tac.innerHTML = chips.length ? chips.join('') :
    '<span style="color:var(--ink-fade);border:none;background:none">nothing obvious</span>';
}
// Eval-over-game line chart from the move history (side-to-move POV evals).
function renderEvalChart(h){
  let svg = document.getElementById('evalchart');
  if (!svg) return;
  let pts = (h||[]).filter(x => typeof x.eval === 'number');
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const W=300, H=90, mid=H/2;
  const mkline = (cls, d) => {
    let p = document.createElementNS('http://www.w3.org/2000/svg','line');
    p.setAttribute('class',cls);
    p.setAttribute('x1',d[0]);p.setAttribute('y1',d[1]);
    p.setAttribute('x2',d[2]);p.setAttribute('y2',d[3]);
    svg.appendChild(p);
  };
  mkline('zero',[0,mid,W,mid]);
  if (pts.length < 2){
    let t=document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x',W/2);t.setAttribute('y',mid-6);
    t.setAttribute('text-anchor','middle');
    t.setAttribute('fill','#7a8590');t.setAttribute('font-size','11');
    t.textContent='not enough moves yet';svg.appendChild(t);
    return;
  }
  // eval in [-1,1] → y; clamp. Top = +1 (good for side to move at that ply).
  let path='';
  pts.forEach((x,i) => {
    let xx = i/(pts.length-1)*W;
    let e = Math.max(-1, Math.min(1, x.eval));
    let yy = mid - e*(mid-4);
    path += (i===0?'M':'L')+xx.toFixed(1)+' '+yy.toFixed(1)+' ';
  });
  let pl=document.createElementNS('http://www.w3.org/2000/svg','path');
  pl.setAttribute('class','line');pl.setAttribute('d',path);
  svg.appendChild(pl);
}
function renderHistory(h){
  let el = document.getElementById('history');
  renderEvalChart(h);
  if (!h || !h.length) {
    el.innerHTML = '<div style="color:var(--ink-dim);text-align:center;padding:20px">no moves yet</div>';
    return;
  }
  let rows = [];
  let g2 = new Chess();
  for (let i = 0; i < h.length; i++){
    let mv = g2.move({from:h[i].uci.slice(0,2), to:h[i].uci.slice(2,4),
                      promotion: h[i].uci.length>4 ? h[i].uci[4] : 'q'});
    let san = mv ? mv.san : h[i].uci;
    let cls = h[i].by === 'ai' ? 'ai' : '';
    let evstr = (h[i].eval != null) ? ((h[i].eval>=0?'+':'')+h[i].eval.toFixed(2)) : '';
    rows.push('<div class="row"><div class="num">'+Math.ceil(h[i].ply/2)+(h[i].ply%2===1?'.':'…')+'</div>'+
              '<div class="'+cls+'">'+san+'</div>'+
              '<div style="color:var(--ink-dim);text-align:right">'+evstr+'</div>'+
              (h[i].verdict ? '<div class="verdict '+verdictClass(h[i].verdict)+'">'+escapeHtml(h[i].verdict)+'</div>':'')+
              '</div>');
  }
  el.innerHTML = rows.join('');
  el.scrollTop = el.scrollHeight;
}
function verdictClass(v){
  if (!v) return '';
  if (v.startsWith('Best')) return 'good';
  if (v.startsWith('Mistake')) return 'bad';
  if (v.includes('inaccuracy') || v.includes('Off my radar')) return 'warn';
  return '';
}
function escapeHtml(s){ return s.replace(/[&<>"']/g, m => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m])); }
function renderCommentary(c){
  if (!c) return;
  let el = document.getElementById('commentary');
  let cls;
  if (c.low_confidence){
    cls = 'warn-conf';
    el.className = '';   // neutral background for the "no opinion" case
    el.style.background = 'rgba(255,180,84,.07)';
    el.style.borderColor = 'rgba(255,180,84,.25)';
  } else {
    el.style.background = '';
    el.style.borderColor = '';
    el.className = verdictClass(c.verdict);
  }
  let head;
  if (c.low_confidence){
    head = 'AI on your move (low confidence)';
  } else {
    head = 'AI on your move' + (c.rank ? ' (ranked #'+c.rank+' on my list)' : '');
  }
  el.innerHTML = '<div class="lead">' + head + '</div>' +
                 '<div>' + escapeHtml(c.verdict || '—') + '</div>';
}

// ─── Settings ─────────────────────────────────────────────────────────
let settingsDebounce;
function settingChanged(){
  // Sync slider <-> number for sims/time
  document.getElementById('thinkSimsR').value = document.getElementById('thinkSims').value;
  document.getElementById('thinkTimeR').value = document.getElementById('thinkTime').value;
  // Show/hide appropriate row
  let mode = document.getElementById('thinkMode').value;
  document.getElementById('sims-row').style.display = (mode==='sims') ? 'grid' : 'none';
  document.getElementById('time-row').style.display = (mode==='time') ? 'grid' : 'none';
  document.getElementById('dyn-row').style.display  = (mode==='dynamic') ? 'grid' : 'none';
  // Debounced push to server
  clearTimeout(settingsDebounce);
  settingsDebounce = setTimeout(pushSettings, 250);
}
function pushSettings(){
  post('/api/settings', {
    think_mode:   document.getElementById('thinkMode').value,
    think_sims:   parseInt(document.getElementById('thinkSims').value),
    think_time_s: parseFloat(document.getElementById('thinkTime').value),
    dynamic_min:  parseInt(document.getElementById('dynMin').value),
    dynamic_max:  parseInt(document.getElementById('dynMax').value),
    ponder_on:    document.getElementById('ponderOn').checked,
    temperature:  parseFloat(document.getElementById('temperature').value),
    contempt:     parseFloat(document.getElementById('contempt').value),
  });
}
function applySettings(s){
  if (!s) return;
  document.getElementById('thinkMode').value = s.think_mode;
  document.getElementById('thinkSims').value = s.think_sims;
  document.getElementById('thinkSimsR').value = s.think_sims;
  document.getElementById('thinkTime').value = s.think_time_s;
  document.getElementById('thinkTimeR').value = s.think_time_s;
  document.getElementById('dynMin').value = s.dynamic_min;
  document.getElementById('dynMax').value = s.dynamic_max;
  document.getElementById('ponderOn').checked = s.ponder_on;
  document.getElementById('temperature').value = s.temperature;
  document.getElementById('tempLbl').textContent = parseFloat(s.temperature).toFixed(3);
  if (s.contempt !== undefined) {
    document.getElementById('contempt').value = s.contempt;
    document.getElementById('contemptLbl').textContent = parseFloat(s.contempt).toFixed(2);
  }
  document.getElementById('sims-row').style.display = (s.think_mode==='sims') ? 'grid' : 'none';
  document.getElementById('time-row').style.display = (s.think_mode==='time') ? 'grid' : 'none';
  document.getElementById('dyn-row').style.display  = (s.think_mode==='dynamic') ? 'grid' : 'none';
}

// ─── Actions ──────────────────────────────────────────────────────────
function newGame(asWhite){
  post('/api/new', {human_white: !!asWhite});
}
function takeback(){ post('/api/takeback', {}); }
function resign(){ post('/api/resign', {}); }
function hint(){ post('/api/hint', {}); }

// ─── Boot ─────────────────────────────────────────────────────────────
board = Chessboard('board', {
  draggable: true,
  position: 'start',
  onDrop: onDrop,
  pieceTheme: '/piece/{piece}.svg',
});

// chessboard.js 1.0.0 has NO `onSquareClick` option — the one that used to be
// passed here was silently ignored, so point-and-click never fired and only
// drag worked. Wire it ourselves with a single DELEGATED click listener on the
// stable #board container: it survives chessboard.js re-creating its cells on
// every position() call, and clicks pass through the pointer-events:none
// #clickLayer / #arrows overlays to the square beneath. We translate the
// clicked cell's data-square + chess.js piece into the (square, 'wP'-style
// code) arguments onSquareClickCB already expects, so click and drag share all
// downstream move logic. A genuine drag is ignored here because onDrop sets
// busy=true (and flips the turn) before the trailing click bubbles up.
(function(){
  const host = document.getElementById('board');
  if (!host) return;

  // Resolve a viewport point to a board square. Uses elementsFromPoint
  // (the full hit stack) rather than elementFromPoint: when you click a
  // piece, chessboard.js has already started a drag and its floating piece
  // clone — appended to <body>, not to a square — sits directly under the
  // cursor. The PLURAL variant lets us skip that clone (and any overlay)
  // and find the real square cell beneath. Restricted to cells inside
  // #board so clicks on other UI never resolve to a square.
  function squareAt(x, y){
    for (const el of document.elementsFromPoint(x, y)){
      const cell = el.closest ? el.closest('[data-square]') : null;
      if (cell && host.contains(cell)) return cell.getAttribute('data-square');
    }
    return null;
  }
  function tapSquare(sq){
    if (!sq) return;
    const pc = game.get(sq);                                     // {type,color}|null
    onSquareClickCB(sq, pc ? (pc.color + pc.type.toUpperCase()) : null); // 'wP'|null
  }

  // Touch (iPad / phone): iOS does NOT reliably fire `click` on plain <div>
  // squares, and chessboard.js drag needs a mouse — so a `click`-only handler
  // does nothing on touch. Detect a TAP (touchstart+touchend on ~the same
  // point; a larger move is a drag/scroll, left alone) and route it through the
  // same handler. `touchTap` then swallows the ghost `click` iOS synthesises
  // ~300 ms later so the tap isn't processed twice.
  let tStart = null, touchTap = false;
  document.addEventListener('touchstart', ev => {
    const t = ev.touches[0]; if (t) tStart = {x: t.clientX, y: t.clientY};
  }, {passive: true});
  document.addEventListener('touchend', ev => {
    const t = ev.changedTouches && ev.changedTouches[0];
    if (!t || !tStart) { tStart = null; return; }
    const moved = Math.abs(t.clientX - tStart.x) > 14 ||
                  Math.abs(t.clientY - tStart.y) > 14;
    tStart = null;
    if (moved) return;                        // swipe / drag, not a tap
    touchTap = true; setTimeout(() => { touchTap = false; }, 500);
    tapSquare(squareAt(t.clientX, t.clientY));
  }, {passive: true});

  // Mouse (desktop): the listener MUST live on document, not #board.
  // Clicking a piece makes chessboard.js start a drag on mousedown and
  // reparent the pointer under its floating clone on <body>; the resulting
  // click event then fires on the common ancestor — BODY — and never
  // bubbles through #board (verified via event logging: "click -> BODY").
  // squareAt() keeps this safe: coordinates that don't sit over a board
  // square resolve to null, so page-wide clicks are untouched. Skipped
  // right after a touch so iOS's synthetic click doesn't double-fire.
  document.addEventListener('click', ev => {
    if (touchTap) return;
    tapSquare(squareAt(ev.clientX, ev.clientY));
  });

  // Escape clears the current selection (spec: empty square OR Esc deselects;
  // the empty-square case is already handled inside onSquareClickCB).
  document.addEventListener('keydown', ev => {
    if (ev.key === 'Escape' && selectedSq) clearClickSelection();
  });
})();

// Fetch engine info for the pill
fetch('/api/state').then(r => r.json()).then(s => {
  document.getElementById('enginepill').textContent = 'engine: ready';
  applySnapshot(s);
});

connectStream();

window.addEventListener('resize', () => { if (board) board.resize(); });
