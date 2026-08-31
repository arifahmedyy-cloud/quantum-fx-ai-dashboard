import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const Icon = ({ children, size=18 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{children}</svg>;
const icons = {
  grid:<><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
  chart:<><path d="M4 19V5"/><path d="M4 19h16"/><path d="m7 15 3-4 3 2 5-6"/></>,
  bot:<><rect x="5" y="7" width="14" height="12" rx="3"/><path d="M9 7V4h6v3"/><circle cx="9" cy="13" r="1"/><circle cx="15" cy="13" r="1"/><path d="M9 16h6"/></>,
  backtest:<><path d="M4 8h16"/><path d="M4 16h16"/><path d="M8 4 6 8l2 4"/><path d="M16 12l2 4-2 4"/></>,
  shield:<><path d="M12 3 20 6v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3Z"/><path d="m9 12 2 2 4-4"/></>,
  settings:<><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-1.8 1.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5V20h-2.6v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.9.3l-.1.1-1.8-1.8.1-.1A1.7 1.7 0 0 0 8 15a1.7 1.7 0 0 0-1.5-1H6v-2.6h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1L9 6.6l.1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.5V5h2.6v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1 1.8 1.8-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.5 1h.1V14h-.1a1.7 1.7 0 0 0-1.5 1Z"/></>,
  bell:<><path d="M18 9a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9"/><path d="M10 21h4"/></>,
  search:<><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
  play:<><path d="m8 5 11 7-11 7V5Z"/></>,
  pause:<><path d="M8 5v14"/><path d="M16 5v14"/></>,
  bolt:<><path d="m13 2-8 12h7l-1 8 8-12h-7l1-8Z"/></>,
  menu:<><path d="M4 7h16M4 12h16M4 17h16"/></>,
  wallet:<><path d="M4 7a3 3 0 0 1 3-3h11v16H7a3 3 0 0 1-3-3V7Z"/><path d="M4 7h14"/><path d="M16 12h4"/></>,
  cpu:<><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/></>,
  database:<><ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5"/><path d="M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7"/></>,
  chevron:<path d="m9 18 6-6-6-6"/>
};

const API_BASE = import.meta.env.VITE_QFX_API_URL || 'http://127.0.0.1:8808';

async function apiGet(path){
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return response.json();
}

function MiniChart({rows=[]}){
  const source = rows.slice(-54).map(c=>({open:c.open, high:c.high, low:c.low, close:c.close}));
  const W=760,H=300,pad=22;
  if (!source.length) return <div className="empty-chart">No broker/paper candles available.</div>;
  const all=source.flatMap(c=>[c.high,c.low]); const min=Math.min(...all), max=Math.max(...all);
  const span=Math.max(0.00001,max-min);
  const y=v=>pad+(max-v)/span*(H-pad*2);
  const x=i=>pad+i*((W-pad*2)/Math.max(1,source.length-1));
  return <svg className="main-chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
    {[0,.25,.5,.75,1].map((t)=><line key={t} x1={pad} x2={W-pad} y1={pad+t*(H-pad*2)} y2={pad+t*(H-pad*2)} stroke="rgba(140,170,200,.09)" strokeDasharray="4 6"/>)}
    {source.map((c,i)=>{const up=c.close>=c.open,xx=x(i),yo=y(c.open),yc=y(c.close),yh=y(c.high),yl=y(c.low); return <g key={i}><line x1={xx} x2={xx} y1={yh} y2={yl} stroke={up?'#36d38d':'#ff647c'} strokeWidth="1.3"/><rect x={xx-2.5} y={Math.min(yo,yc)} width="5" height={Math.max(2,Math.abs(yc-yo))} rx="1" fill={up?'#36d38d':'#ff647c'}/></g>})}
  </svg>
}

const nav=[['Overview','grid'],['Live Trading','chart'],['AI Analysis','bot'],['Backtesting','backtest'],['Risk Manager','shield']];
const navSectionIds={'Overview':'sec-overview','Live Trading':'sec-live-trading','AI Analysis':'sec-ai-analysis','Backtesting':'sec-backtesting','Risk Manager':'sec-risk-manager','Settings':'sec-overview'};

function App(){
  const [active,setActive]=useState('Overview');
  const [compact,setCompact]=useState(false);
  const [symbol,setSymbol]=useState('XAUUSDm');
  const [tf,setTf]=useState('M15');
  const [tab,setTab]=useState('Positions');
  const [snapshot,setSnapshot]=useState(null);
  const [account,setAccount]=useState(null);
  const [positions,setPositions]=useState([]);
  const [history,setHistory]=useState([]);
  const [risk,setRisk]=useState(null);
  const [market,setMarket]=useState(null);
  const [error,setError]=useState('');
  const [botRunning,setBotRunning]=useState(false);
  const [busy,setBusy]=useState('');
  const [result,setResult]=useState(null);
  const [mode,setMode]=useState('mt5');
  const [riskSettings,setRiskSettings]=useState(null);
  const [riskDraft,setRiskDraft]=useState(null);
  const [riskSaving,setRiskSaving]=useState(false);
  const [riskSaveMsg,setRiskSaveMsg]=useState('');

  const loadRiskSettings=async()=>{
    try{ const r=await apiGet('/api/settings/risk'); setRiskSettings(r); setRiskDraft(prev=>prev||r); }
    catch(e){ /* non-fatal; risk panel will show a load error */ }
  };
  useEffect(()=>{ loadRiskSettings(); },[]);

  const saveRiskSettings=async()=>{
    if(!riskDraft) return;
    setRiskSaving(true); setRiskSaveMsg('');
    try{
      const changed={};
      for(const k in riskDraft){ if(riskSettings && riskDraft[k]!==riskSettings[k]) changed[k]=riskDraft[k]; }
      if(Object.keys(changed).length===0){ setRiskSaveMsg('No changes to save.'); return; }
      const r=await fetch(`${API_BASE}/api/settings/risk`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(changed)});
      const data=await r.json();
      if(!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setRiskSettings(data.applied?{...riskSettings,...data.applied}:riskSettings);
      setRiskSaveMsg('Saved. ' + (data.note||''));
      await loadRiskSettings();
    }catch(e){ setRiskSaveMsg('Failed to save: ' + (e.message||'unknown error')); }
    finally{ setRiskSaving(false); }
  };

  const refresh=async()=>{
    try{
      const prefix=mode==='paper'?'/api/paper':'/api/mt5';
      const [health, acct, pos, mkt, hist, riskData] = await Promise.all([
        apiGet(`${prefix}/health`),
        apiGet(`${prefix}/account`),
        apiGet(`${prefix}/positions`),
        apiGet(`${prefix}/market?symbol=${encodeURIComponent(mode==='paper'?'XAUUSD':symbol)}&timeframe=${tf}&bars=80`),
        apiGet(`/api/history?limit=50`),
        apiGet(`${prefix}/risk`)
      ]);
      setSnapshot(health); setBotRunning(Boolean(health.bot_running)); setAccount(acct.account);
      setPositions(pos.positions); setHistory(hist.trades || []); setRisk(riskData.summary || null); setMarket(mkt); setError('');
    }catch(err){ setError(err.message || 'Dashboard API unavailable'); }
  };

  useEffect(()=>{ refresh(); const id=setInterval(refresh,5000); return ()=>clearInterval(id); },[symbol,tf,mode]);

  const abortRef=useRef(null);
  const action=async(path, params='')=>{
    setBusy(path); setResult(null); setError('');
    const controller=new AbortController();
    abortRef.current=controller;
    try{
      const r=await fetch(`${API_BASE}${path}${params}`, {method:'POST', signal:controller.signal});
      const data=await r.json();
      if(!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setResult(data); await refresh();
    }catch(e){
      if(e.name==='AbortError') setError('Cancelled by user.');
      else setError(e.message || 'Action failed');
    }
    finally{ setBusy(''); abortRef.current=null; }
  };
  const cancelAction=()=>{ if(abortRef.current) abortRef.current.abort(); };

  const runAnalysis=()=>action(mode==='paper'?'/api/analysis/paper':'/api/analysis/mt5',`?symbol=${encodeURIComponent(mode==='paper'?'XAUUSD':symbol)}&timeframe=${tf}`);
  const [btMonths,setBtMonths]=useState(1);
  const runPaperBacktest=()=>action('/api/backtest/paper',`?symbol=XAUUSD&timeframe=${tf}&months=${btMonths}`);
  const runMT5Backtest=()=>action('/api/backtest/mt5',`?symbol=${encodeURIComponent(symbol)}&timeframe=${tf}&months=${btMonths}`);
  const startMode=(m)=>{ setMode(m); action(m==='paper'?'/api/bot/start/paper':'/api/bot/start/mt5'); };
  const goToSection=(label)=>{
    setActive(label);
    const id=navSectionIds[label];
    const el=id&&document.getElementById(id);
    if(el) el.scrollIntoView({behavior:'smooth', block:'start'});
  };


  const live = Boolean(snapshot?.connected);
  const price = market?.bid || market?.candles?.at(-1)?.close || 0;
  const balance = account?.balance ?? 0;
  const equity = account?.equity ?? 0;
  const floating = positions.reduce((sum,p)=>sum+(p.profit||0),0);

  return <div className="app">
    <aside className={compact?'sidebar compact':'sidebar'}>
      <div className="brand"><div className="brand-mark">Q</div>{!compact&&<div><div className="brand-name">QUANTUM FX</div><div className="brand-sub">AI TRADING CORE</div></div>}</div>
      <div className="nav-label">CONTROL CENTER</div>
      {nav.map(([label,key])=><button key={label} className={`nav-item ${active===label?'active':''}`} onClick={()=>goToSection(label)}><Icon>{icons[key]}</Icon>{!compact&&<span>{label}</span>}</button>)}
      <div className="nav-label spacer">SYSTEM</div>
      <button className={`nav-item ${active==='Settings'?'active':''}`} onClick={()=>goToSection('Settings')}><Icon>{icons.settings}</Icon>{!compact&&<span>Settings</span>}</button>
      <div className="side-bottom"><div className="status-row"><span className={`status-dot ${live?'':'offline'}`}></span>{!compact&&<span>{live?'Broker Connected':'Broker Offline'}</span>}</div><button className="icon-btn" onClick={()=>setCompact(!compact)}><Icon>{icons.chevron}</Icon></button></div>
    </aside>

    <main className="main">
      <header className="topbar">
        <div className="title-wrap"><button className="mobile-menu"><Icon>{icons.menu}</Icon></button><div><h1>{active}</h1><p>Quantum FX AI · Production control console</p></div></div>
        <div className="top-actions"><div className="connection"><span className={`status-dot ${live?'':'offline'}`}></span><span>{snapshot?.broker?.toUpperCase() || 'BROKER'}</span><b>{live?'CONNECTED':'OFFLINE'}</b></div><button className="icon-btn" onClick={refresh}><Icon>{icons.bolt}</Icon></button><div className="user-chip"><div className="avatar">QA</div><div><strong>Dashboard</strong><span>{snapshot?.resolved_symbol || symbol}</span></div></div></div>
      </header>

      <section className="content">
        {error&&<div className="panel" style={{marginBottom:16,padding:12}}><span className="negative">Connection: {error}</span></div>}
        <div className="hero-row" id="sec-overview">
          <div><div className="eyebrow">{mode==='paper'?'PAPER MARKET':'MT5 LIVE MARKET'}</div><h2>{mode==='paper'?'XAUUSD':symbol} <span>· {tf}</span></h2><div className="price-row"><strong>{price ? `$${price.toFixed(2)}` : '—'}</strong><span className={floating>=0?'up':'negative'}>{positions.length ? `${floating>=0?'+':''}$${floating.toFixed(2)} floating` : 'No open positions'}</span><span className="muted">Source: {market?.source || '—'}</span></div></div>
          <div className="hero-actions"><span className="ghost-btn">Read-only API <span className={`toggle ${live?'on':''}`}></span></span><button className="primary-btn" onClick={refresh}><Icon>{icons.bolt}</Icon>Refresh</button></div>
        </div>

        <section className="panel" style={{marginBottom:16}} id="sec-ai-analysis">
          <div className="panel-head"><div><h3>Bot Control</h3><p>Real actions through the adapter · existing trading engine unchanged</p></div><span className={botRunning?'signal-badge':'tiny-badge'}>{botRunning?'RUNNING':'STOPPED'}</span></div>
          <div className="mode-switch">
            <button className={mode==='paper'?'selected':''} onClick={()=>setMode('paper')}>PAPER</button>
            <button className={mode==='mt5'?'selected':''} onClick={()=>setMode('mt5')}>MT5 LIVE</button>
          </div>
          <div style={{display:'flex',gap:10,flexWrap:'wrap',marginTop:10}}>
            {!botRunning ? <><button className="primary-btn" disabled={!!busy} onClick={()=>startMode('paper')}><Icon>{icons.play}</Icon>Start Paper</button><button className="primary-btn" disabled={!!busy} onClick={()=>startMode('mt5')}><Icon>{icons.play}</Icon>Start MT5</button></> : <button className="flat-btn" disabled={busy==='/api/bot/stop'} onClick={()=>action('/api/bot/stop')}><Icon>{icons.pause}</Icon>{busy==='/api/bot/stop'?'Stopping…':'Stop Bot'}</button>}
            <button className="flat-btn" disabled={!!busy} onClick={runAnalysis}><Icon>{icons.bot}</Icon>{busy==='/api/analysis'?'Analyzing…':'Run Analysis'}</button>
            <select value={btMonths} onChange={e=>setBtMonths(Number(e.target.value))} title="Backtest duration (longer ranges + fine timeframes like M5/M15 take longer to compute — use H1/H4 for multi-year runs)">
              <option value={1}>1 month</option>
              <option value={3}>3 months</option>
              <option value={6}>6 months</option>
              <option value={12}>1 year</option>
              <option value={24}>2 years</option>
            </select>
            <button className="flat-btn" disabled={!!busy} onClick={runPaperBacktest}><Icon>{icons.backtest}</Icon>{busy==='/api/backtest/paper'?'Running…':'Paper Backtest'}</button>
            <button className="flat-btn" disabled={!!busy} onClick={runMT5Backtest}><Icon>{icons.backtest}</Icon>{busy==='/api/backtest/mt5'?'Running…':'MT5 Backtest'}</button>
            {(busy==='/api/backtest/paper'||busy==='/api/backtest/mt5')&&<button className="flat-btn" style={{borderColor:'#e5484d',color:'#e5484d'}} onClick={cancelAction}>Cancel</button>}
            <button className="flat-btn" disabled={!!busy} onClick={refresh}><Icon>{icons.bolt}</Icon>Refresh All</button>
          </div>
          {(busy==='/api/backtest/paper'||busy==='/api/backtest/mt5')&&<p style={{marginTop:8,fontSize:12,opacity:0.75}}>Backtest is running candle-by-candle (real SMC/AI analysis per bar) — longer durations on M5/M15 can take a few minutes. Use the Cancel button if you don't want to wait, or pick a shorter duration / H1-H4 timeframe next time.</p>}
          {result&&<pre style={{marginTop:12,maxHeight:180,overflow:'auto',fontSize:12}}>{JSON.stringify(result,null,2)}</pre>}
        </section>

        <div className="kpi-grid">
          {[['Account Balance',(balance!=null)?`$${balance.toLocaleString(undefined,{minimumFractionDigits:2})}`:'—','wallet',account?`${account.currency || 'USD'}`:'Waiting','positive'],['Equity',(equity!=null)?`$${equity.toLocaleString(undefined,{minimumFractionDigits:2})}`:'—','chart',account?`Free $${(account.free_margin||0).toFixed(2)}`:'Waiting','positive'],['Floating P&L',positions.length?`${floating>=0?'+':''}$${floating.toFixed(2)}`:'$0.00','bolt',`${positions.length} open position(s)`,'positive'],['Margin Level',(account?.margin_level!=null)?`${account.margin_level.toFixed(1)}%`:'—','shield',account?'Broker native':'Waiting','neutral'],['Positions',String(positions.length),'bot',live?'Live broker data':'Offline','positive']].map(([label,value,ic,sub,cls])=><div className="kpi" key={label}><div className="kpi-icon"><Icon>{icons[ic]}</Icon></div><div><span>{label}</span><strong>{value}</strong><small className={cls}>{sub}</small></div></div>)}
        </div>

        <div className="grid-two">
          <section className="panel chart-panel" id="sec-live-trading">
            <div className="panel-head"><div><h3>Price Action</h3><p>{market?`Broker feed · ${market.candles.length} candles`:'Waiting for broker data'}</p></div><div className="control-group"><select value={mode==='paper'?'XAUUSD':symbol} onChange={e=>setSymbol(e.target.value)} disabled={mode==='paper'}><option>{mode==='paper'?'XAUUSD':'XAUUSDm'}</option><option>XAUUSD</option><option>EURUSD</option><option>GBPUSD</option><option>USDJPY</option><option>AUDUSD</option></select><select value={tf} onChange={e=>setTf(e.target.value)}><option>M5</option><option>M15</option><option>H1</option><option>H4</option></select></div></div>
            <div className="chart-legend"><span><i className="dot cyan"></i>Broker candles</span><span><i className="line green"></i>Live feed</span></div>
            <MiniChart rows={market?.candles || []}/>
            <div className="chart-footer"><span>Broker-native</span><span>{market?.updated_at ? new Date(market.updated_at).toLocaleTimeString() : '—'}</span><span>Auto refresh 5s</span></div>
          </section>
          <section className="panel signal-panel">
            <div className="panel-head"><div><h3>Connection Monitor</h3><p>Real backend status · no mock signal data</p></div><span className={live?'signal-badge':'tiny-badge'}>{live?'LIVE':'OFFLINE'}</span></div>
            <div className="signal-main"><div className="confidence"><div className="confidence-ring"><span>{live?'OK':'—'}</span></div><div><b>{live?'Broker Connected':'Not Connected'}</b><small>{snapshot?.broker || 'No broker configured'}</small></div></div><div className="signal-levels"><div><span>Resolved Symbol</span><b>{snapshot?.resolved_symbol || '—'}</b></div><div><span>Terminal</span><b className={snapshot?.terminal_ok?'profit':'loss'}>{snapshot?.terminal_ok?'OK':'OFF'}</b></div><div><span>Account</span><b className={snapshot?.account_ok?'profit':'loss'}>{snapshot?.account_ok?'OK':'OFF'}</b></div></div></div>
            <div className="ai-grid">{[['Market Data',market?'LIVE':'WAITING'],['Account',account?'LIVE':'WAITING'],['Positions',positions.length?String(positions.length):'0'],['API','READ-ONLY']].map(([name,val])=><div className="ai-card" key={name}><div><span>{name}</span><small>backend state</small></div><strong className={val==='WAITING'?'':'bull'}>{val}</strong></div>)}</div>
            <div className="decision-row"><span><i className={`status-dot ${live?'':'offline'}`}></i>Status: <b>{live?'Healthy':'Unavailable'}</b></span><span>Source: <b>{snapshot?.broker || '—'}</b></span><button className="flat-btn" onClick={refresh}>Refresh status</button></div>
          </section>
        </div>

        <div className="grid-three">
          <section className="panel positions-panel span-two"><div className="panel-head"><div><h3>Trading Activity</h3><p>Read-only broker positions · {positions.length} open</p></div><div className="segmented">{['Positions','History'].map(x=><button className={tab===x?'selected':''} onClick={()=>setTab(x)} key={x}>{x}</button>)}</div></div>
            <div className="table-wrap"><table><thead>{tab==='Positions'?<tr><th>Ticket</th><th>Symbol</th><th>Side</th><th>Size</th><th>Entry</th><th>P&L</th><th>Status</th></tr>:<tr><th>Time</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Result</th></tr>}</thead><tbody>{tab==='Positions' ? (positions.length?positions.map(p=><tr key={p.ticket}><td>{p.ticket}</td><td>{p.symbol}</td><td className={p.side==='BUY'?'positive':'negative'}>{p.side}</td><td>{p.volume}</td><td>{p.entry}</td><td className={p.profit>=0?'positive':'negative'}>{p.profit>=0?'+':''}${p.profit.toFixed(2)}</td><td><span className="tiny-badge">LIVE</span></td></tr>):<tr><td colSpan="7" style={{textAlign:'center',padding:'24px'}}>No broker positions returned.</td></tr>) : (history.length?history.map((p,i)=><tr key={p.ticket||p.id||i}><td>{p.open_time||p.entry_time||'—'}</td><td>{p.symbol||'—'}</td><td>{p.side||p.direction||'—'}</td><td>{p.entry||p.entry_price||'—'}</td><td>{p.exit||p.exit_price||'—'}</td><td className={(p.profit||p.profit_loss||0)>=0?'positive':'negative'}>{Number(p.profit||p.profit_loss||0)>=0?'+':''}${Number(p.profit||p.profit_loss||0).toFixed(2)}</td><td>{p.result||p.status||'CLOSED'}</td></tr>):<tr><td colSpan="7" style={{textAlign:'center',padding:'24px'}}>No trade history returned.</td></tr>)}</tbody></table></div>
          </section>
          <section className="panel health-panel"><div className="panel-head"><div><h3>System Health</h3><p>Runtime diagnostics</p></div><span className={live?'healthy':'tiny-badge'}>{live?'HEALTHY':'OFFLINE'}</span></div>
            {[['Broker Connection',live?'Stable':'Offline',snapshot?.broker||'—'],['Market Data',market?'Live':'Waiting',market?'Fresh':'—'],['Account',account?'Live':'Waiting',account?account.currency:'—'],['Positions',`${positions.length}`, 'broker-native'],['Risk Guard',risk?.guard_active?'BLOCKED':'CLEAR',risk?.guard_reason||'daily risk state'],['Dashboard API',snapshot?'Online':'Offline','127.0.0.1:8808']].map(([a,b,c])=><div className="health-row" key={a}><span className={`status-dot ${b==='Offline'?'offline':''}`}></span><div><b>{a}</b><small>{b}</small></div><em>{c}</em></div>)}
          </section>
        </div>

        <div className="grid-three">
          <section className="panel" id="sec-risk-manager" style={{gridColumn:'span 3'}}>
            <div className="panel-head"><div><h3>Risk Manager</h3><p>Live-editable risk parameters — saved to .env, applied immediately to this dashboard session</p></div><span className={live?'healthy':'tiny-badge'}>{riskSettings?'LOADED':'LOADING'}</span></div>
            {!riskDraft ? <p style={{padding:'8px 0',opacity:0.7}}>Loading risk settings…</p> : <>
              <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(200px,1fr))',gap:14,marginTop:8}}>
                <label style={{display:'flex',flexDirection:'column',gap:4}}><span style={{fontSize:12,opacity:0.75}}>Risk per trade (%)</span><input type="number" step="0.1" min="0.1" max="100" value={riskDraft.base_risk_pct} onChange={e=>setRiskDraft({...riskDraft,base_risk_pct:Number(e.target.value)})} /></label>
                <label style={{display:'flex',flexDirection:'column',gap:4}}><span style={{fontSize:12,opacity:0.75}}>Max risk per trade (%)</span><input type="number" step="0.1" min="0.1" max="100" value={riskDraft.max_risk_pct} onChange={e=>setRiskDraft({...riskDraft,max_risk_pct:Number(e.target.value)})} /></label>
                <label style={{display:'flex',flexDirection:'column',gap:4}}><span style={{fontSize:12,opacity:0.75}}>Daily loss limit (%)</span><input type="number" step="0.1" min="0.1" max="100" value={riskDraft.daily_loss_limit_pct} onChange={e=>setRiskDraft({...riskDraft,daily_loss_limit_pct:Number(e.target.value)})} /></label>
                <label style={{display:'flex',flexDirection:'column',gap:4}}><span style={{fontSize:12,opacity:0.75}}>Max open trades</span><input type="number" step="1" min="1" max="50" value={riskDraft.max_open_trades} onChange={e=>setRiskDraft({...riskDraft,max_open_trades:Number(e.target.value)})} /></label>
              </div>
              <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(200px,1fr))',gap:10,marginTop:16}}>
                {[['use_auto_drawdown_risk','Auto drawdown risk'],['use_trailing_stop','Trailing stop'],['use_kelly_sizing','Kelly sizing'],['enable_ml_filter','ML filter'],['strict_smc_confluence','Strict SMC confluence']].map(([key,label])=>
                  <label key={key} style={{display:'flex',alignItems:'center',justifyContent:'space-between',gap:8,padding:'8px 10px',border:'1px solid rgba(255,255,255,0.08)',borderRadius:8}}>
                    <span style={{fontSize:13}}>{label}</span>
                    <span className={`toggle ${riskDraft[key]?'on':''}`} onClick={()=>setRiskDraft({...riskDraft,[key]:!riskDraft[key]})} style={{cursor:'pointer'}}></span>
                  </label>)}
              </div>
              <h4 style={{marginTop:20,marginBottom:6,fontSize:13,opacity:0.75}}>Circuit Breaker (consecutive-loss cooldown)</h4>
              <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(220px,1fr))',gap:14}}>
                <label style={{display:'flex',flexDirection:'column',gap:4}}><span style={{fontSize:12,opacity:0.75}}>Consecutive losses before pause</span><input type="number" step="1" min="1" max="20" value={riskDraft.max_consecutive_losses} onChange={e=>setRiskDraft({...riskDraft,max_consecutive_losses:Number(e.target.value)})} /></label>
                <label style={{display:'flex',flexDirection:'column',gap:4}}><span style={{fontSize:12,opacity:0.75}}>Cooldown hours (blank = pause until next day)</span><input type="number" step="0.5" min="0" max="168" value={riskDraft.consecutive_loss_cooldown_hours ?? ''} placeholder="next day" onChange={e=>setRiskDraft({...riskDraft,consecutive_loss_cooldown_hours:e.target.value===''?null:Number(e.target.value)})} /></label>
              </div>
              <h4 style={{marginTop:20,marginBottom:6,fontSize:13,opacity:0.75}}>Break-Even &amp; Trailing Stop (in R-multiples of initial risk)</h4>
              <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(220px,1fr))',gap:14}}>
                <label style={{display:'flex',flexDirection:'column',gap:4}}><span style={{fontSize:12,opacity:0.75}}>Break-even trigger (R)</span><input type="number" step="0.1" min="0.1" max="10" value={riskDraft.breakeven_trigger_r} onChange={e=>setRiskDraft({...riskDraft,breakeven_trigger_r:Number(e.target.value)})} /></label>
                <label style={{display:'flex',flexDirection:'column',gap:4}}><span style={{fontSize:12,opacity:0.75}}>Break-even buffer (fraction of risk)</span><input type="number" step="0.01" min="0" max="1" value={riskDraft.breakeven_buffer_pct} onChange={e=>setRiskDraft({...riskDraft,breakeven_buffer_pct:Number(e.target.value)})} /></label>
                <label style={{display:'flex',flexDirection:'column',gap:4}}><span style={{fontSize:12,opacity:0.75}}>Trailing trigger (R)</span><input type="number" step="0.1" min="0.1" max="10" value={riskDraft.trailing_trigger_r} onChange={e=>setRiskDraft({...riskDraft,trailing_trigger_r:Number(e.target.value)})} /></label>
                <label style={{display:'flex',flexDirection:'column',gap:4}}><span style={{fontSize:12,opacity:0.75}}>Trailing distance (R)</span><input type="number" step="0.1" min="0.1" max="10" value={riskDraft.trailing_distance_r} onChange={e=>setRiskDraft({...riskDraft,trailing_distance_r:Number(e.target.value)})} /></label>
              </div>
              <div style={{display:'flex',alignItems:'center',gap:12,marginTop:16}}>
                <button className="primary-btn" disabled={riskSaving} onClick={saveRiskSettings}><Icon>{icons.shield}</Icon>{riskSaving?'Saving…':'Save Risk Settings'}</button>
                <button className="flat-btn" disabled={riskSaving} onClick={()=>setRiskDraft(riskSettings)}>Reset</button>
                {riskSaveMsg && <small style={{opacity:0.8}}>{riskSaveMsg}</small>}
              </div>
            </>}
          </section>
        </div>

        <div className="grid-three">
          <section className="panel metric-panel"><div className="panel-head"><div><h3>Broker Snapshot</h3><p>Native account values</p></div></div><div className="metric-list">{[['Balance',(balance!=null)?`$${balance.toFixed(2)}`:'—'],['Equity',(equity!=null)?`$${equity.toFixed(2)}`:'—'],['Free Margin',account?`$${account.free_margin.toFixed(2)}`:'—'],['Margin',account?`$${account.margin.toFixed(2)}`:'—'],['Leverage',account?`1:${account.leverage}`:'—']].map(x=><div key={x[0]}><span>{x[0]}</span><b>{x[1]}</b></div>)}</div></section>
          <section className="panel backtest-panel" id="sec-backtesting"><div className="panel-head"><div><h3>Backtest</h3><p>Existing Python engine remains unchanged</p></div><span className="tiny-badge">SEPARATE</span></div><div className="backtest-stats"><div><span>Engine</span><b>Python</b></div><div><span>UI</span><b>React</b></div><div><span>Core</span><b>Untouched</b></div><div><span>Mode</span><b>Read-only</b></div></div></section>
          <section className="panel logs-panel"><div className="panel-head"><div><h3>System Events</h3><p>Dashboard bridge</p></div><span className="tiny-badge">LIVE</span></div><div className="logs"><div><time>{new Date().toLocaleTimeString()}</time><span className="log-ok">API</span><p>{live?'Broker status healthy':'Waiting for broker connection'}</p></div><div><time>{market?.updated_at?new Date(market.updated_at).toLocaleTimeString():'—'}</time><span className="log-ai">DATA</span><p>{market?`${market.symbol} ${market.timeframe} data received`:'No market data yet'}</p></div><div><time>—</time><span className="log-risk">SAFE</span><p>Read-only dashboard bridge · no order endpoint</p></div></div></section>
        </div>

        <footer className="footer"><span>Quantum FX AI · React Dashboard</span><span><i className={`status-dot ${live?'':'offline'}`}></i> {live?'Broker data active':'Broker data unavailable'} · Auto refresh 5s</span></footer>
      </section>
    </main>
  </div>
}

createRoot(document.getElementById('root')).render(<App />)
