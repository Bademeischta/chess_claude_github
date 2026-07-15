(function(){
  'use strict';
  const $  = id => document.getElementById(id);
  const elc = (tag, cls, html) => { const e=document.createElement(tag);
        if(cls) e.className=cls; if(html!=null) e.innerHTML=html; return e; };
  const LS = window.localStorage;
  const get = (k,d) => { const v=LS.getItem(k); return v===null?d:v; };

  // ── State ────────────────────────────────────────────────────────────
  let proHist=[], prevFen=null, reviewPly=null;
  let threatsOn = get('pp_threats','0')==='1';
  let soundOn   = get('pp_sound','1')==='1';

  // ── Threat overlay (inside the board stage) ──────────────────────────
  const stage = document.querySelector('.board-stage');
  const tov = elc('div','threats'); tov.id='threats';
  if (stage) stage.appendChild(tov);

  // ── Pro toolbar (prepended to the left column) ───────────────────────
  const bar = elc('div','card');
  bar.innerHTML =
    '<h3>Pro tools</h3>'+
    '<div class="protools">'+
      '<button class="tbtn" data-a="flip" title="Flip board (F)">⟲ Flip</button>'+
      '<button class="tbtn" data-a="coords" title="Toggle coordinates">#</button>'+
      '<button class="tbtn" data-a="theme" title="Cycle theme">🎨 Theme</button>'+
      '<button class="tbtn" data-a="sound" title="Toggle sound">🔊</button>'+
      '<button class="tbtn" data-a="threats" title="Threat overlay">⚠ Threats</button>'+
      '<button class="tbtn" data-a="focus" title="Focus / zen mode">⛶ Focus</button>'+
      '<button class="tbtn" data-a="pgn" title="Download PGN">⭳ PGN</button>'+
      '<button class="tbtn" data-a="cfen" title="Copy FEN">FEN</button>'+
      '<button class="tbtn" data-a="cpgn" title="Copy PGN">⧉ PGN</button>'+
      '<button class="tbtn" data-a="loadfen" title="Play from a FEN">Load FEN</button>'+
      '<button class="tbtn" data-a="guess" title="Guess-the-AI-move game">🎯 Guess</button>'+
      '<button class="tbtn" data-a="help" title="Keyboard shortcuts (?)">?</button>'+
      '<span class="opening-chip" id="openingChip">—</span>'+
    '</div>';
  const leftCol = document.querySelector('.left');
  if (leftCol) leftCol.insertBefore(bar, leftCol.firstChild);

  // ── Training monitor card (right column) ─────────────────────────────
  const tmCard = elc('div','card');
  tmCard.innerHTML = '<h3>Training monitor <span id="tmStale" class="tm-stale"></span></h3>'+
    '<div class="tm-grid" id="tmGrid"></div>';
  // ── Accuracy / review card (right column) ────────────────────────────
  const accCard = elc('div','card');
  accCard.innerHTML = '<h3>Game accuracy &amp; review</h3>'+
    '<div class="acc-row">'+
      '<div class="acc-pill"><div class="big" id="accW">—</div><div class="lbl">White</div></div>'+
      '<div class="acc-pill"><div class="big" id="accB">—</div><div class="lbl">Black</div></div>'+
    '</div>'+
    '<div class="acc-counts" id="accCounts"></div>'+
    '<div style="font-size:11px;color:var(--ink-fade);margin-top:8px">'+
      'Click a move below, or use ← / → to step through the game.</div>';
  const rightCol = document.querySelector('.right');
  if (rightCol){ rightCol.appendChild(tmCard); rightCol.appendChild(accCard); }

  // ── Search-tree visualiser (left column, wide) ───────────────────────
  const treeCard = elc('div','card');
  treeCard.innerHTML = '<h3>🌳 Search tree — watch me think</h3>'+
    '<div id="treeWrap"><svg id="treeSvg" viewBox="0 0 400 80"></svg></div>'+
    '<div class="tree-meta"><span id="treeRootEval">root —</span>'+
    '<span>green = good for the side that just moved · thickness = visits</span></div>';
  if (leftCol) leftCol.appendChild(treeCard);

  // ── Guess-the-move scoreboard (right column) ─────────────────────────
  const guessCard = elc('div','card');
  guessCard.innerHTML = '<h3>🎯 Guess the AI’s move</h3>'+
    '<div class="gs-row">'+
      '<div class="cell"><div class="big" id="gsScore">0</div><div class="lbl">Score</div></div>'+
      '<div class="cell"><div class="big" id="gsStreak">0</div><div class="lbl">Streak</div></div>'+
      '<div class="cell"><div class="big" id="gsBest">0</div><div class="lbl">Best</div></div>'+
      '<div class="cell"><div class="big" id="gsAcc">—</div><div class="lbl">Hit %</div></div>'+
    '</div>'+
    '<div id="guessNow">Turn on <b>🎯 Guess</b>, then after your move click '+
    'two squares to predict my reply. Exact = +3, top-3 = +1.</div>';
  if (rightCol) rightCol.appendChild(guessCard);

  // ── Fixed chrome: review bar + help overlay ──────────────────────────
  const rbar = elc('div'); rbar.id='reviewbar';
  rbar.innerHTML = '<button data-r="first">⏮</button><button data-r="prev">◀</button>'+
    '<span id="reviewlbl" style="font-variant-numeric:tabular-nums">review</span>'+
    '<button data-r="next">▶</button><button data-r="last">⏭</button>'+
    '<button data-r="live" class="primary">● Live</button>';
  document.body.appendChild(rbar);

  const help = elc('div'); help.id='helpov';
  help.innerHTML = '<div class="box"><h3 style="margin-top:0">Keyboard shortcuts</h3>'+
    '<table>'+
    '<tr><td><kbd>N</kbd></td><td>New game (you as White)</td></tr>'+
    '<tr><td><kbd>F</kbd></td><td>Flip board</td></tr>'+
    '<tr><td><kbd>H</kbd></td><td>Hint</td></tr>'+
    '<tr><td><kbd>T</kbd></td><td>Takeback</td></tr>'+
    '<tr><td><kbd>R</kbd></td><td>Resign</td></tr>'+
    '<tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>Step through game (review)</td></tr>'+
    '<tr><td><kbd>Esc</kbd></td><td>Back to live / close</td></tr>'+
    '<tr><td><kbd>?</kbd></td><td>This help</td></tr>'+
    '</table><div style="text-align:right;margin-top:10px">'+
    '<button data-r="closehelp" class="primary">Got it</button></div></div>';
  document.body.appendChild(help);

  // ── Sound (WebAudio, no assets) ──────────────────────────────────────
  let actx=null;
  function beep(freq,dur,type,gain){ if(!soundOn) return;
    try{ actx=actx||new (window.AudioContext||window.webkitAudioContext)();
      const o=actx.createOscillator(), g=actx.createGain();
      o.type=type||'sine'; o.frequency.value=freq; g.gain.value=gain||0.04;
      o.connect(g); g.connect(actx.destination); o.start();
      g.gain.exponentialRampToValueAtTime(0.0001, actx.currentTime+(dur||0.12));
      o.stop(actx.currentTime+(dur||0.13)); }catch(e){}
  }
  const sMove=()=>beep(440,0.07,'triangle',0.05);
  const sCap =()=>beep(190,0.12,'sawtooth',0.06);
  const sChk =()=>beep(720,0.16,'square',0.045);
  const sEnd =()=>{ beep(330,0.28,'sine',0.06); setTimeout(()=>beep(247,0.4,'sine',0.06),150); };

  // ── Opening detection (longest-prefix match) ─────────────────────────
  const OPEN = [
    [['e2e4'],'King’s Pawn'],[['d2d4'],'Queen’s Pawn'],
    [['c2c4'],'English Opening'],[['g1f3'],'Réti Opening'],
    [['e2e4','e7e5'],'Open Game'],[['e2e4','c7c5'],'Sicilian Defense'],
    [['e2e4','e7e6'],'French Defense'],[['e2e4','c7c6'],'Caro-Kann'],
    [['e2e4','d7d5'],'Scandinavian'],[['e2e4','g8f6'],'Alekhine Defense'],
    [['e2e4','d7d6'],'Pirc Defense'],[['e2e4','b8c6'],'Nimzowitsch Def.'],
    [['e2e4','g7g6'],'Modern Defense'],
    [['e2e4','e7e5','g1f3','b8c6','f1b5'],'Ruy López'],
    [['e2e4','e7e5','g1f3','b8c6','f1c4'],'Italian Game'],
    [['e2e4','e7e5','g1f3','b8c6','d2d4'],'Scotch Game'],
    [['e2e4','e7e5','g1f3','g8f6'],'Petrov Defense'],
    [['e2e4','c7c5','g1f3','d7d6'],'Sicilian: Najdorf-ish'],
    [['d2d4','d7d5'],'Closed Game'],[['d2d4','g8f6'],'Indian Defense'],
    [['d2d4','d7d5','c2c4'],'Queen’s Gambit'],
    [['d2d4','d7d5','c2c4','e7e6'],'QGD'],
    [['d2d4','d7d5','c2c4','c7c6'],'Slav Defense'],
    [['d2d4','g8f6','c2c4','g7g6'],'King’s Indian'],
    [['d2d4','g8f6','c2c4','e7e6'],'Nimzo/QID'],
    [['d2d4','f7f5'],'Dutch Defense'],
  ];
  function openingName(ucis){ let best='—', bl=0;
    for(const [seq,name] of OPEN){
      if(seq.length<=ucis.length && seq.length>bl && seq.every((u,i)=>u===ucis[i])){
        best=name; bl=seq.length; } }
    return best;
  }

  // ── Accuracy (from per-move eval vs best_eval) ───────────────────────
  function classify(delta){
    if(delta==null) return ['',''];
    if(delta<0.02) return ['!!','bril'];
    if(delta<0.05) return ['!','good'];
    if(delta<0.15) return ['','' ];
    if(delta<0.40) return ['?!','inacc'];
    if(delta<0.90) return ['?','mis'];
    return ['??','blu'];
  }
  function accFromDelta(delta){ // crude human-move accuracy curve
    if(delta==null) return null;
    return Math.max(0, Math.min(100, 100*Math.exp(-2.6*Math.max(0,delta))));
  }
  function renderAccuracy(){
    let accs={w:[],b:[]}, cnt={bril:0,blu:0,mis:0,inacc:0};
    for(const h of proHist){
      if(h.by!=='human') continue;
      if(h.eval==null||h.best_eval==null||h.low_confidence) continue;
      const delta=Math.max(0,(h.best_eval||0)-(h.eval||0));
      const side=(h.ply%2===1)?'w':'b';
      const a=accFromDelta(delta); if(a!=null) accs[side].push(a);
      const [g,c]=classify(delta);
      if(c==='bril') cnt.bril++; else if(c==='blu') cnt.blu++;
      else if(c==='mis') cnt.mis++; else if(c==='inacc') cnt.inacc++;
    }
    const avg=a=>a.length?Math.round(a.reduce((s,x)=>s+x,0)/a.length):null;
    const aw=avg(accs.w), ab=avg(accs.b);
    $('accW').textContent = aw==null?'—':aw+'%';
    $('accB').textContent = ab==null?'—':ab+'%';
    $('accCounts').innerHTML =
      '<span class="bril">'+cnt.bril+' brilliant</span>'+
      '<span class="inacc">'+cnt.inacc+' inacc.</span>'+
      '<span class="mis">'+cnt.mis+' mistakes</span>'+
      '<span class="blu">'+cnt.blu+' blunders</span>';
    decorateHistory();
  }
  // Make the existing history rows hoverable + clickable for review.
  function decorateHistory(){
    const hist=$('history'); if(!hist) return;
    hist.querySelectorAll('.row').forEach(row=>row.classList.add('moverow'));
  }

  // ── Threat overlay ───────────────────────────────────────────────────
  function squareBox(sq){ const c=(typeof squareCenter==='function')?squareCenter(sq):null;
    if(!c) return null; return {l:c.x-6.25, t:c.y-6.25}; }
  async function refreshThreats(){
    if(!threatsOn || reviewPly!=null){ tov.innerHTML=''; return; }
    try{ const r=await fetch('/api/threats'); const t=await r.json(); drawThreats(t); }
    catch(e){ /* offline / between games */ }
  }
  function drawThreats(t){
    tov.innerHTML=''; if(!t||!t.hanging) return;
    for(const h of t.hanging){
      const b=squareBox(h.sq); if(!b) continue;
      const d=elc('div','tsq hang'); d.style.left=b.l+'%'; d.style.top=b.t+'%';
      d.title=(h.white?'White ':'Black ')+h.sym+' is loose on '+h.sq;
      tov.appendChild(d);
    }
  }

  // ── Themes ───────────────────────────────────────────────────────────
  const THEMES = {
    midnight:{}, // default :root
    light:{'--bg-0':'#eef1f5','--bg-1':'#ffffff','--bg-2':'#f4f6f9','--bg-3':'#e7ebf1',
           '--ink':'#10151b','--ink-dim':'#475263','--ink-fade':'#8b97a6','--border':'#d3dae3',
           '--shadow':'0 6px 24px rgba(40,60,90,.12)'},
    emerald:{'--bg-0':'#06120e','--bg-1':'#0c1d16','--bg-2':'#102a20','--bg-3':'#16362a',
             '--accent':'#37e0a0','--good':'#7ef0c8','--border':'#1c4234'},
    rose:{'--bg-0':'#160a10','--bg-1':'#221019','--bg-2':'#2c1521','--bg-3':'#3a1c2c',
          '--accent':'#ff7eb6','--good':'#ffa9d0','--border':'#43202f'},
  };
  const THORDER=['midnight','light','emerald','rose'];
  let themeStyle=elc('style'); themeStyle.id='protheme'; document.head.appendChild(themeStyle);
  function applyTheme(name){
    const vars=THEMES[name]||{}; let css=':root{';
    for(const k in vars) css+=k+':'+vars[k]+';'; css+='}';
    themeStyle.textContent=css; LS.setItem('pp_theme',name);
    document.body.classList.toggle('lighttheme', name==='light');
  }
  applyTheme(get('pp_theme','midnight'));

  // ── PGN / FEN ────────────────────────────────────────────────────────
  function curPGN(){ try{ return game.pgn({max_width:80,newline_char:'\n'})||''; }catch(e){ return ''; } }
  function dl(name,text){ const b=new Blob([text],{type:'text/plain'});
    const u=URL.createObjectURL(b); const a=elc('a'); a.href=u; a.download=name;
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(u); }
  async function copy(text){ try{ await navigator.clipboard.writeText(text); flash('Copied'); }
    catch(e){ const ta=elc('textarea'); ta.value=text; document.body.appendChild(ta);
      ta.select(); try{document.execCommand('copy');flash('Copied');}catch(_){} ta.remove(); } }
  let flashT=null;
  function flash(msg){ let f=$('ppflash'); if(!f){ f=elc('div'); f.id='ppflash';
      f.style.cssText='position:fixed;bottom:18px;right:18px;z-index:70;padding:8px 14px;'+
        'background:var(--accent);color:#0b0d10;border-radius:8px;font-weight:600;box-shadow:var(--shadow)';
      document.body.appendChild(f); }
    f.textContent=msg; f.style.opacity='1'; clearTimeout(flashT);
    flashT=setTimeout(()=>{ f.style.transition='opacity .4s'; f.style.opacity='0'; },1100);
  }
  function pgnHeaders(){ const d=new Date().toISOString().slice(0,10);
    return '[Event "Chess AI web"]\n[Site "local"]\n[Date "'+d+'"]\n'+
      '[White "'+(humanWhite?'You':'ChessAI')+'"]\n[Black "'+(humanWhite?'ChessAI':'You')+'"]\n\n'; }

  // ── Board controls ───────────────────────────────────────────────────
  function flip(){ if(board) board.flip(); clearClickSelection(); setTimeout(refreshThreats,60); }
  function toggleCoords(){ document.body.classList.toggle('no-coords');
    LS.setItem('pp_coords', document.body.classList.contains('no-coords')?'0':'1'); }
  function toggleFocus(){ document.body.classList.toggle('focusmode');
    setTimeout(()=>{ if(board) board.resize(); refreshThreats(); },60); }
  function cycleTheme(){ const cur=get('pp_theme','midnight');
    applyTheme(THORDER[(THORDER.indexOf(cur)+1)%THORDER.length]); }
  function toggleSound(){ soundOn=!soundOn; LS.setItem('pp_sound',soundOn?'1':'0'); syncBtns();
    if(soundOn) sMove(); }
  function toggleThreats(){ threatsOn=!threatsOn; LS.setItem('pp_threats',threatsOn?'1':'0');
    syncBtns(); refreshThreats(); }
  function loadFen(){ const f=prompt('Paste a FEN to play from:'); if(f&&f.trim())
    post('/api/setfen',{fen:f.trim()}).then(r=>{ if(r&&r.error) alert(r.error); }); }

  // ── Review mode ──────────────────────────────────────────────────────
  function maxPly(){ return proHist.length; }
  function fenAtPly(p){ // reconstruct from startpos by replaying recorded ucis
    try{ const tmp=new Chess();
      for(let i=0;i<p && i<proHist.length;i++){ const u=proHist[i].uci||'';
        if(u.length<4) return null;
        tmp.move({from:u.slice(0,2),to:u.slice(2,4),promotion:(u[4]||'q')}); }
      return tmp.fen();
    }catch(e){ return null; }
  }
  function enterReview(p){ p=Math.max(0,Math.min(maxPly(),p));
    const fen=fenAtPly(p); if(fen==null){ flash('Review N/A (custom start)'); return; }
    reviewPly=p; if(board) board.position(fen,false); tov.innerHTML='';
    rbar.classList.add('show');
    $('reviewlbl').textContent='ply '+p+' / '+maxPly();
  }
  function reviewLive(){ reviewPly=null; rbar.classList.remove('show');
    if(board && typeof game!=='undefined') board.position(game.fen()); refreshThreats(); }
  function reviewStep(d){ if(reviewPly==null) enterReview(maxPly()+d); else enterReview(reviewPly+d); }

  // ── Training monitor poll ────────────────────────────────────────────
  function fmt(v,dp){ return (v==null)?'—':(dp!=null?Number(v).toFixed(dp):v); }
  let lastStep=null, lastStepTs=0;
  async function refreshTraining(){
    let d; try{ d=await (await fetch('/api/training')).json(); }catch(e){ return; }
    const cells=[
      ['Step', d.step!=null?d.step.toLocaleString():'—',''],
      ['ELO', fmt(d.elo,1), 'good'],
      ['Loss total', fmt(d.loss_total,3),''],
      ['Policy loss', fmt(d.p,3),''],
      ['Value loss', fmt(d.v,3),''],
      ['Teacher KL', fmt(d.kl,3),''],
      ['Entropy', fmt(d.entropy,2),''],
      ['LR', d.lr!=null?Number(d.lr).toExponential(2):'—',''],
      ['Throughput', d.pos_s!=null?(Number(d.pos_s).toFixed(0)+' pos/s'):'—',''],
      ['Games/hr', d.games_hr!=null?Math.round(d.games_hr).toLocaleString():'—',''],
      ['Replay buf', d.replay!=null?Math.round(d.replay).toLocaleString():'—',''],
      ['VRAM', d.vram!=null?(Math.round(d.vram)+' MB'):'—',''],
    ];
    $('tmGrid').innerHTML = cells.map(([k,v,c])=>
      '<div class="tm-cell"><span class="k">'+k+'</span><span class="v '+(c||'')+'">'+v+'</span></div>'
    ).join('');
    // "alive" detector: did the step advance recently?
    const now=Date.now()/1000;
    if(d.step!=null && d.step!==lastStep){ lastStep=d.step; lastStepTs=now; }
    let st='';
    if(d.step==null) st='(no run data yet)';
    else if(now-lastStepTs>120) st='● idle (no new steps)';
    else st='● live';
    $('tmStale').textContent=st;
    $('tmStale').style.color = st==='● live' ? 'var(--good)' : 'var(--warn)';
  }

  // ── Wire toolbar + review + help buttons ─────────────────────────────
  function syncBtns(){
    document.querySelectorAll('.tbtn').forEach(b=>{
      const a=b.dataset.a;
      if(a==='sound'){ b.classList.toggle('on',soundOn); b.textContent=soundOn?'🔊':'🔇'; }
      if(a==='threats') b.classList.toggle('on',threatsOn);
      if(a==='guess') b.classList.toggle('on',guessMode);
      if(a==='focus') b.classList.toggle('on',document.body.classList.contains('focusmode'));
    });
  }
  bar.addEventListener('click',e=>{ const b=e.target.closest('.tbtn'); if(!b) return;
    const a=b.dataset.a;
    if(a==='flip') flip();
    else if(a==='coords') toggleCoords();
    else if(a==='theme') cycleTheme();
    else if(a==='sound') toggleSound();
    else if(a==='threats') toggleThreats();
    else if(a==='focus') toggleFocus();
    else if(a==='pgn') dl('chessai_game.pgn', pgnHeaders()+curPGN());
    else if(a==='cfen'){ try{ copy(game.fen()); }catch(_){ } }
    else if(a==='cpgn') copy(pgnHeaders()+curPGN());
    else if(a==='loadfen') loadFen();
    else if(a==='guess') toggleGuess();
    else if(a==='help') help.classList.add('show');
  });
  rbar.addEventListener('click',e=>{ const b=e.target.closest('button'); if(!b) return;
    const r=b.dataset.r;
    if(r==='first') enterReview(0); else if(r==='prev') reviewStep(-1);
    else if(r==='next') reviewStep(1); else if(r==='last') enterReview(maxPly());
    else if(r==='live') reviewLive(); });
  help.addEventListener('click',e=>{ if(e.target===help||e.target.dataset.r==='closehelp')
    help.classList.remove('show'); });
  // Click a move in the existing history list → review that ply.
  const histEl=$('history');
  if(histEl) histEl.addEventListener('click',e=>{ const row=e.target.closest('.row');
    if(!row) return; const rows=[...histEl.querySelectorAll('.row')];
    const idx=rows.indexOf(row); if(idx<0) return;
    // each row holds a move PAIR (white+black) → jump to end of that pair
    enterReview(Math.min(maxPly(),(idx+1)*2)); });

  // ── Keyboard shortcuts ───────────────────────────────────────────────
  document.addEventListener('keydown',e=>{
    if(e.target.matches('input,textarea,select')) return;
    const k=e.key;
    if(k==='ArrowLeft'){ reviewStep(-1); e.preventDefault(); }
    else if(k==='ArrowRight'){ reviewStep(1); e.preventDefault(); }
    else if(k==='Escape'){ clearClickSelection();
      if(help.classList.contains('show')) help.classList.remove('show');
      else if(reviewPly!=null) reviewLive(); }
    else if(k==='?'||(k==='/'&&e.shiftKey)) help.classList.toggle('show');
    else if(k==='f'||k==='F') flip();
    else if(k==='n'||k==='N'){ if(typeof newGame==='function') newGame(true); }
    else if(k==='h'||k==='H'){ if(typeof hint==='function') hint(); }
    else if(k==='t'||k==='T'){ if(typeof takeback==='function') takeback(); }
    else if(k==='r'||k==='R'){ if(typeof resign==='function') resign(); }
  });

  // ── Search-tree visualiser ───────────────────────────────────────────
  const esc = (window.escapeHtml) ? window.escapeHtml : (s=>String(s));
  function renderTree(t){
    const svg=$('treeSvg'); if(!svg) return;
    const meta=$('treeRootEval');
    if(!t || !t.kids || !t.kids.length){
      svg.setAttribute('viewBox','0 0 400 60');
      svg.innerHTML='<text x="12" y="30" fill="#7a8590" font-size="12">'+
        'The tree grows here while the AI is thinking…</text>';
      if(meta) meta.textContent='root —';
      return;
    }
    let leaf=0; const nodes=[], edges=[];
    const rootN=t.n||1;
    const root={san:'start',uci:'',n:t.n,q:t.q,p:1,kids:t.kids,_root:true};
    (function layout(node,depth,parent){
      node._d=depth;
      if(!node.kids||!node.kids.length){ node._y=leaf++; }
      else { node.kids.forEach(k=>layout(k,depth+1,node));
             node._y=(node.kids[0]._y + node.kids[node.kids.length-1]._y)/2; }
      nodes.push(node); if(parent) edges.push([parent,node]);
    })(root,0,null);
    const maxD=Math.max(1,...nodes.map(n=>n._d)), leaves=Math.max(1,leaf);
    const colW=128, rowH=30, padX=12, padY=14, NW=78;
    const W=padX*2+maxD*colW+NW, H=padY*2+leaves*rowH;
    svg.setAttribute('viewBox','0 0 '+W+' '+H);
    svg.style.height=Math.min(440,H+8)+'px';
    const X=n=>padX+n._d*colW, Y=n=>padY+n._y*rowH+rowH/2;
    let s='';
    for(const [a,b] of edges){
      const w=Math.max(0.6,Math.min(7,(b.n||0)/rootN*9));
      const x1=X(a)+NW, y1=Y(a), x2=X(b), y2=Y(b);
      s+='<path d="M'+x1+','+y1+' C'+(x1+colW*0.4)+','+y1+' '+(x2-colW*0.4)+','+y2+' '+x2+','+y2+
         '" fill="none" stroke="rgba(120,140,160,.4)" stroke-width="'+w.toFixed(1)+'"/>';
    }
    for(const n of nodes){
      const q=n.q||0;
      const col=n._root?'#5aa9ff':(q>0.15?'#3ddc97':(q<-0.15?'#ff6f6f':'#9bd1ff'));
      const x=X(n), y=Y(n);
      const share=Math.round((n.n||0)/rootN*100);
      s+='<g class="tnode">';
      s+='<rect x="'+x+'" y="'+(y-10)+'" rx="6" width="'+NW+'" height="20" '+
         'fill="rgba(18,23,28,.95)" stroke="'+col+'" stroke-width="1.3"/>';
      s+='<text x="'+(x+6)+'" y="'+(y+4)+'" font-size="11" fill="#e6edf3" '+
         'font-family="Consolas,monospace">'+esc(n.san)+'</text>';
      s+='<text x="'+(x+NW-4)+'" y="'+(y+4)+'" font-size="9" fill="#7a8590" '+
         'text-anchor="end">'+(n._root?'':share+'%')+'</text>';
      s+='<title>'+esc(n.san)+'   N='+n.n+'   Q='+(q>=0?'+':'')+q.toFixed(2)+
         '   prior='+Math.round((n.p||0)*100)+'%</title></g>';
    }
    svg.innerHTML=s;
    if(meta){ const rq=t.q||0; meta.textContent='root eval '+(rq>=0?'+':'')+rq.toFixed(2)+
              ' · '+(t.n||0).toLocaleString()+' sims'; }
  }

  // ── Guess-the-move game ──────────────────────────────────────────────
  let guessMode = get('pp_guess','0')==='1';
  let awaitingGuess=false, guessFrom=null, guessUci=null;
  let gScore=+get('pp_gscore','0'), gStreak=0, gBest=+get('pp_gbest','0'),
      gTries=+get('pp_gtries','0'), gHits=+get('pp_ghits','0');
  function clearGuessSq(){ document.querySelectorAll('.gsq').forEach(e=>e.classList.remove('gsq')); }
  function markSq(sq){ const c=document.querySelector('#board [data-square="'+sq+'"]');
    if(c) c.classList.add('gsq'); }
  function renderGuessCard(){
    $('gsScore').textContent=gScore; $('gsStreak').textContent=gStreak;
    $('gsBest').textContent=gBest;
    $('gsAcc').textContent = gTries? Math.round(100*gHits/gTries)+'%' : '—';
  }
  function setGuessNow(msg){ const e=$('guessNow'); if(e) e.innerHTML=msg; }
  function openGuessWindow(){ if(!guessMode) return;
    awaitingGuess=true; guessFrom=null; guessUci=null; clearGuessSq();
    setGuessNow('🎯 <b>Predict my move</b> — click the FROM then the TO square.'); }
  function evalGuess(actualUci, top){
    if(!guessMode){ return; }
    if(!guessUci){ awaitingGuess=false; return; }
    gTries++; const top3=(top||[]).slice(0,3).map(m=>m.uci);
    let pts=0,msg='';
    if(guessUci===actualUci){ pts=3; gStreak++; gHits++;
      msg='🎯 <b>Spot on!</b> +3 — streak '+gStreak; burstConfetti(20); sChk(); }
    else if(top3.includes(guessUci)){ pts=1; gHits++; gStreak=0;
      msg='Close — that was in my top 3! +1 (I played <b>'+actualUci+'</b>)'; }
    else { gStreak=0; msg='Missed — I played <b>'+actualUci+'</b> (you said '+guessUci+')'; }
    gScore+=pts; gBest=Math.max(gBest,gStreak);
    LS.setItem('pp_gscore',gScore); LS.setItem('pp_gbest',gBest);
    LS.setItem('pp_gtries',gTries); LS.setItem('pp_ghits',gHits);
    setGuessNow(msg); renderGuessCard(); guessUci=null; awaitingGuess=false;
    setTimeout(clearGuessSq,1200);
  }
  function toggleGuess(){ guessMode=!guessMode; LS.setItem('pp_guess',guessMode?'1':'0');
    syncBtns(); if(guessMode) setGuessNow('Guess mode ON — make your move, then predict mine.');
    else { awaitingGuess=false; clearGuessSq(); setGuessNow('Guess mode off.'); } }
  // Capture two board clicks as a from→to prediction.
  const _boardEl=$('board');
  if(_boardEl) _boardEl.addEventListener('click',e=>{
    if(!guessMode||!awaitingGuess) return;
    const cell=e.target.closest('[data-square]'); if(!cell) return;
    const sq=cell.getAttribute('data-square');
    if(!guessFrom){ guessFrom=sq; clearGuessSq(); markSq(sq);
      setGuessNow('From <b>'+sq+'</b> → now click the target square.'); }
    else { guessUci=guessFrom+sq; markSq(sq);
      setGuessNow('Locked in: <b>'+guessUci+'</b> — waiting for my move…'); guessFrom=null; }
  });

  // ── Confetti (Web Animations, no assets) ─────────────────────────────
  function burstConfetti(n){ n=n||40;
    const cols=['#5aa9ff','#3ddc97','#ffb454','#ff6f6f','#9bd1ff','#7ee0d2'];
    for(let i=0;i<n;i++){ const d=elc('div');
      d.style.cssText='position:fixed;z-index:80;width:9px;height:9px;border-radius:2px;'+
        'pointer-events:none;top:42%;left:'+(50+(Math.random()*36-18))+'%;background:'+cols[i%cols.length];
      document.body.appendChild(d);
      const dx=(Math.random()*2-1)*260, rot=Math.random()*900;
      d.animate([{transform:'translate(0,0) rotate(0)',opacity:1},
                 {transform:'translate('+dx+'px,'+(360+Math.random()*220)+'px) rotate('+rot+'deg)',opacity:0}],
                {duration:1200+Math.random()*800,easing:'cubic-bezier(.2,.7,.3,1)'});
      setTimeout(()=>d.remove(),2100);
    }
  }

  // ── Own SSE stream (additive; server supports many clients) ──────────
  function pieceCount(fen){ return (fen.split(' ')[0].match(/[a-zA-Z]/g)||[]).length; }
  function reactMove(newFen){
    let cap=false, chk=false;
    if(prevFen && newFen) cap = pieceCount(newFen) < pieceCount(prevFen);
    try{ chk = new Chess(newFen).in_check(); }catch(_){}
    if(chk) sChk(); else if(cap) sCap(); else sMove();
    prevFen=newFen;
  }
  function ingest(data, isMove){
    if(data.history) proHist=data.history;
    if(data.fen!=null) prevFen=prevFen||data.fen;
    const ucis=proHist.map(h=>h.uci).filter(Boolean);
    const oc=$('openingChip'); if(oc) oc.textContent=openingName(ucis);
    renderAccuracy();
    if(reviewPly==null) refreshThreats();
    if(isMove && data.fen) reactMove(data.fen);
    if(data.game_over) sEnd();
  }
  try{
    const es=new EventSource('/api/stream');
    const on=(name,fn)=>es.addEventListener(name,e=>{ try{ fn(JSON.parse(e.data)); }catch(_){} });
    on('snapshot', d=>{ proHist=d.history||[]; prevFen=d.fen||null; reviewPly=null;
                        rbar.classList.remove('show'); ingest(d,false); });
    on('new_game', d=>{ proHist=[]; prevFen=d.fen||null; reviewPly=null;
                        rbar.classList.remove('show');
                        const oc=$('openingChip'); if(oc) oc.textContent='—';
                        renderAccuracy(); refreshThreats(); });
    on('user_move', d=>ingest(d,true));
    on('ai_move',   d=>ingest(d,true));
    on('takeback',  d=>{ if(d.history) proHist=d.history; reviewLive(); ingest(d,false); });
    on('game_over', d=>{ sEnd(); ingest(d,false); });
    // ── Big-feature wiring: live tree, guess game, win confetti ──────────
    on('tree',      d=>renderTree(d));
    on('user_move', d=>openGuessWindow());
    on('ai_move',   d=>evalGuess(d.uci, (d.analysis && d.analysis.top) || []));
    on('user_move', d=>{ if(d.game_over && d.result && !/draw|stalemate/i.test(d.result)) burstConfetti(70); });
  }catch(e){ /* EventSource unsupported */ }

  // ── Boot ─────────────────────────────────────────────────────────────
  if(get('pp_coords','1')==='0') document.body.classList.add('no-coords');
  syncBtns();
  renderGuessCard();
  renderTree(null);
  refreshTraining(); setInterval(refreshTraining, 4000);
  setTimeout(refreshThreats, 800);
  console.log('[ChessAI] Pro feature pack + big expansions loaded.');
})();
