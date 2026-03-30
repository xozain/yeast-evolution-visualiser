"""
S. cerevisiae ALE — Live Population Visualiser
================================================
Flask + SSE + Canvas particle renderer.
Each visible yeast cell is drawn as a budding ellipse, coloured by Topt.
Run:  python yeast_visualiser.py
Open: http://localhost:5001
"""

import json, math, time, threading, queue, os
from flask import Flask, Response, render_template_string, request, jsonify
import numpy as np

app = Flask(__name__)

# ══════════════════════════════════════════════════════════
#  SIMULATION ENGINE  (same physics as v2, stream-optimised)
# ══════════════════════════════════════════════════════════
GENE_MODULES = {
    "ERG3_LOF":   (1.6, 2.2,-0.08,3.5,0.4),
    "ETC_LOF":    (1.0, 1.8,-0.12,3.0,0.3),
    "CDC25_reg":  (0.9, 1.1,-0.03,2.5,0.9),
    "HSF1_pt":    (0.5, 0.9,-0.01,0.6,2.2),
    "SKN7_pt":    (0.4, 0.7,-0.01,0.5,1.8),
    "HSP104":     (0.45,0.8,-0.015,1.2,2.2),
    "TPS1_TPS2":  (0.35,0.6,-0.01,0.7,1.8),
    "HOG1_reg":   (0.35,0.6,-0.01,0.9,1.6),
    "RAS2_IRA2":  (0.5, 0.7,-0.02,1.5,1.0),
    "Cell_wall":  (0.2, 0.3,-0.005,0.7,1.3),
}
MODULE_NAMES = list(GENE_MODULES.keys())
N_MODULES    = len(MODULE_NAMES)
N_CHROM      = 16
WT_TOPT=30.8; WT_TMAX=40.0; WT_MUOPT=0.40
CHROM_BENEFIT={3:0.04,6:0.03,9:0.035,13:0.03}
CHROM_NGENES=np.array([119,428,170,773,289,262,578,298,439,373,349,579,476,405,290,513],dtype=float)
MAX_LINEAGES=300; DILUTION_FREQ=8; N_BOTTLE=100

def build_temp_schedule(ramp_speed):
    base=[(0,30),(100,33),(200,36),(300,38),(450,40),(600,42)]
    if ramp_speed==1.0: return base
    return [(int(g/ramp_speed),T) for g,T in base]

def current_temp_fn(gen, schedule):
    T=schedule[0][1]
    for g0,t0 in schedule:
        if gen>=g0: T=t0
    return float(T)

def growth_rate_vec(T,To,Tx,mu,Tmin=4.0):
    r=np.zeros(len(To))
    m=(T>Tmin)&(T<Tx)
    if not np.any(m): return r
    to=To[m];tx=Tx[m];u=mu[m]
    n=(T-tx)*(T-Tmin)**2
    d=(to-Tmin)*((to-Tmin)*(T-to)-(to-tx)*(to+Tmin-2*T))
    r[m]=np.where(np.abs(d)>1e-14,np.maximum(0,u*n/d),0)
    return r

def growth_rate(T,To,Tx,mu,Tmin=4.0):
    if T<=Tmin or T>=Tx: return 0.
    n=(T-Tx)*(T-Tmin)**2
    d=(To-Tmin)*((To-Tmin)*(T-To)-(To-Tx)*(To+Tmin-2*T))
    return float(max(0,mu*n/d)) if abs(d)>1e-14 else 0.

class SimPool:
    def __init__(self,Ne,mu_mult,aneu_on,rng):
        self.Ne=Ne; self.mu_mult=mu_mult; self.aneu_on=aneu_on; self.rng=rng
        self.Topt=np.array([WT_TOPT]); self.Tmax=np.array([WT_TMAX])
        self.muopt=np.array([WT_MUOPT]); self.counts=np.array([Ne],dtype=np.int64)
        self.modules=[frozenset()]; self.aneu=[frozenset()]; self.n=1
        self._sgv()

    def _sgv(self):
        for _ in range(60):
            f=self.rng.beta(0.4,10); nc=max(1,int(f*self.Ne))
            dT=self.rng.normal(0.08,0.18)
            self.Topt=np.append(self.Topt,WT_TOPT+dT)
            self.Tmax=np.append(self.Tmax,WT_TMAX+dT*1.3)
            self.muopt=np.append(self.muopt,max(WT_MUOPT*(1+self.rng.normal(0,0.03)),0.04))
            self.counts=np.append(self.counts,nc)
            self.modules.append(frozenset()); self.aneu.append(frozenset()); self.n+=1
        self.counts=(self.counts/self.counts.sum()*self.Ne).astype(np.int64)
        self.counts[0]=max(1,self.Ne-self.counts[1:].sum())

    def freqs(self):
        s=self.counts.sum()
        return self.counts/s if s>0 else np.ones(self.n)/self.n

    def step(self,T,gen):
        if gen>0 and gen%DILUTION_FREQ==0:
            p=self.freqs(); surv=self.rng.multinomial(N_BOTTLE,p).astype(np.int64)
            p2=surv/N_BOTTLE; self.counts=self.rng.multinomial(self.Ne,p2).astype(np.int64)
        w=growth_rate_vec(T,self.Topt,self.Tmax,self.muopt)
        p=self.freqs(); wbar=float(np.dot(w,p))
        if wbar>1e-12:
            ps=np.where(np.isfinite(p*w/wbar),p*w/wbar,0.0)
            ps=np.clip(ps,0,None); s=ps.sum()
            ps=ps/s if s>1e-12 else np.ones(self.n)/self.n
        else:
            ps=np.ones(self.n)/self.n
        self.counts=self.rng.multinomial(self.Ne,ps).astype(np.int64)
        U=2e-10*1.2e7*(1+(2.6*max(0,(T-30)/12)))*self.mu_mult
        phase=min(1.,gen/300)
        nTo=[]; nTx=[]; nmu=[]; nc=[]; nmo=[]; nan_=[]
        for i in range(self.n):
            ni=int(self.counts[i])
            if ni==0: continue
            for _ in range(min(self.rng.poisson(max(ni*U*0.10,0)),5)):
                sr=(self.rng.gamma(0.4,0.008) if self.rng.random()<0.65 else self.rng.gamma(1.5,0.03))
                wts=np.array([max(((1-phase)*we+phase*wl)*(0.05 if m in self.modules[i] else 1),0.001)
                              for m,(dTo,dTx,dmu,we,wl) in GENE_MODULES.items()])
                wts/=wts.sum(); mi=self.rng.choice(N_MODULES,p=wts); mod=MODULE_NAMES[mi]
                dTo,dTx,dmu,_,_=GENE_MODULES[mod]
                Wbg=growth_rate(T,self.Topt[i],self.Tmax[i],self.muopt[i])
                sc=(sr/(1+3.*Wbg)+self.rng.normal(0,0.004))/0.05
                nTo.append(min(self.Topt[i]+dTo*sc*self.rng.uniform(0.5,1.5),WT_TOPT+8))
                nTx.append(min(self.Tmax[i]+dTx*sc*self.rng.uniform(0.5,1.5),WT_TMAX+10))
                nmu.append(max(self.muopt[i]*(1+dmu*abs(sc)),0.04))
                nmo.append(self.modules[i]|{mod}); nan_.append(self.aneu[i])
                nc.append(max(1,self.rng.poisson(1)))
            for _ in range(min(self.rng.poisson(max(ni*U*0.35,0)),3)):
                sd=-self.rng.gamma(0.5,0.015)
                nTo.append(self.Topt[i]+self.rng.normal(0,0.05))
                nTx.append(self.Tmax[i]+self.rng.normal(0,0.08))
                nmu.append(max(self.muopt[i]*(1+sd*0.3),0.04))
                nmo.append(self.modules[i]); nan_.append(self.aneu[i])
                nc.append(max(1,self.rng.poisson(1)))
            if self.aneu_on:
                for _ in range(min(self.rng.poisson(max(ni*5e-6*N_CHROM,0)),2)):
                    ch=self.rng.integers(0,N_CHROM)
                    if ch in self.aneu[i]: continue
                    b=CHROM_BENEFIT.get(ch,0.005)*max(0,(T-32)/10)
                    cc=0.002*CHROM_NGENES[ch]/500
                    if b-cc>-0.015:
                        nTo.append(self.Topt[i]+max(0,self.rng.normal(0.4,0.2))*(T>34))
                        nTx.append(self.Tmax[i]+max(0,self.rng.normal(0.6,0.3))*(T>34))
                        nmu.append(max(self.muopt[i]*(1-cc),0.04))
                        nmo.append(self.modules[i]); nan_.append(self.aneu[i]|{ch})
                        nc.append(max(1,self.rng.poisson(1)))
        if nTo:
            self.Topt=np.append(self.Topt,nTo); self.Tmax=np.append(self.Tmax,nTx)
            self.muopt=np.append(self.muopt,nmu); self.counts=np.append(self.counts,nc)
            self.modules+=nmo; self.aneu+=nan_; self.n=len(self.modules)
        alive=self.counts>0
        if not np.all(alive):
            self.Topt=self.Topt[alive]; self.Tmax=self.Tmax[alive]
            self.muopt=self.muopt[alive]; self.counts=self.counts[alive]
            idx=np.where(alive)[0]
            self.modules=[self.modules[k] for k in idx]
            self.aneu=[self.aneu[k] for k in idx]; self.n=len(self.modules)
        if self.n>MAX_LINEAGES:
            order=np.argsort(self.counts)[::-1][:MAX_LINEAGES]
            lost=self.counts[~np.isin(np.arange(self.n),order)].sum()
            self.Topt=self.Topt[order]; self.Tmax=self.Tmax[order]
            self.muopt=self.muopt[order]; self.counts=self.counts[order]; self.counts[0]+=lost
            self.modules=[self.modules[k] for k in order]
            self.aneu=[self.aneu[k] for k in order]; self.n=len(self.modules)
        fr=self.freqs()
        w2=growth_rate_vec(T,self.Topt,self.Tmax,self.muopt)
        wbar2=float(np.dot(w2,fr))
        # Build cell data for visualiser: top lineages only
        top_n=min(80,self.n)
        order=np.argsort(fr)[::-1][:top_n]
        cells=[]
        for k in order:
            f=float(fr[k])
            if f<0.001: continue
            cells.append({
                "topt": round(float(self.Topt[k]),2),
                "tmax": round(float(self.Tmax[k]),2),
                "freq": round(f,4),
                "muts": len(self.modules[k]),
                "aneu": len(self.aneu[k])>0,
                "modules": list(self.modules[k])[:3],
            })
        return {
            "wbar":  round(wbar2+float(np.random.normal(0,0.006)),5),
            "topt":  round(float(np.dot(self.Topt,fr)),3),
            "tmax":  round(float(np.dot(self.Tmax,fr)),3),
            "aneu":  round(float(sum(f for a,f in zip(self.aneu,fr) if a)),4),
            "n_lin": int(np.sum(fr>0.01)),
            "temp":  T,
            "cells": cells,
            "n_lineages": self.n,
        }

# ── Global state ──
sim_state={"running":False,"gen":0,"config":{},"pools":[],"lock":threading.Lock(),"clients":[]}

def reset_sim(cfg):
    with sim_state["lock"]:
        sim_state["running"]=False; sim_state["gen"]=0; sim_state["config"]=cfg
        nr=cfg.get("n_reps",2); Ne=cfg.get("ne",200000)
        seeds=[int(np.random.randint(0,2**31)) for _ in range(nr)]
        sim_state["pools"]=[SimPool(Ne,cfg.get("mu_mult",1.0),cfg.get("aneu_on",True),
                                    np.random.default_rng(seeds[r])) for r in range(nr)]

def run_sim_thread():
    cfg=sim_state["config"]; n_reps=cfg.get("n_reps",2); n_gens=cfg.get("n_gens",700)
    schedule=build_temp_schedule(cfg.get("ramp_speed",1.0))
    for gen in range(n_gens+1):
        with sim_state["lock"]:
            if not sim_state["running"]: break
        T=current_temp_fn(gen,schedule)
        rep_data=[]
        for r in range(n_reps):
            m=sim_state["pools"][r].step(T,gen)
            m["rep"]=r; rep_data.append(m)
        with sim_state["lock"]:
            sim_state["gen"]=gen
        if gen%2==0:
            payload=json.dumps({"gen":gen,"n_gens":n_gens,"reps":rep_data,
                                 "schedule":schedule},default=float)
            dead=[]
            for q in sim_state["clients"]:
                try: q.put_nowait(payload)
                except: dead.append(q)
            for q in dead:
                try: sim_state["clients"].remove(q)
                except: pass
        time.sleep(0.008)
    with sim_state["lock"]:
        sim_state["running"]=False
    for q in sim_state["clients"]:
        try: q.put_nowait(json.dumps({"done":True,"gen":sim_state["gen"]}))
        except: pass

@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/api/start",methods=["POST"])
def api_start():
    cfg=request.get_json()
    config={"n_reps":int(cfg.get("n_reps",2)),"n_gens":int(cfg.get("n_gens",700)),
            "ne":int(cfg.get("ne",200000)),"mu_mult":float(cfg.get("mu_mult",1.0)),
            "aneu_on":bool(cfg.get("aneu_on",True)),"ramp_speed":float(cfg.get("ramp_speed",1.0))}
    reset_sim(config)
    with sim_state["lock"]: sim_state["running"]=True
    threading.Thread(target=run_sim_thread,daemon=True).start()
    return jsonify({"status":"started"})

@app.route("/api/stop",methods=["POST"])
def api_stop():
    with sim_state["lock"]: sim_state["running"]=False
    return jsonify({"status":"stopped"})

@app.route("/stream")
def stream():
    q=queue.Queue(maxsize=60); sim_state["clients"].append(q)
    def generate():
        try:
            while True:
                try:
                    d=q.get(timeout=30); yield f"data: {d}\n\n"
                    if json.loads(d).get("done"): break
                except queue.Empty:
                    yield "data: {\"ping\":1}\n\n"
        finally:
            try: sim_state["clients"].remove(q)
            except: pass
    return Response(generate(),mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ══════════════════════════════════════════════════════════
#  HTML / JS — live particle renderer
# ══════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Yeast ALE — Live Population View</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  /* Topt colour scale: cool blue → warm white → hot red */
  --c-cold:   #1a6fff;   /* Topt ~28°C  */
  --c-wt:     #40e0c0;   /* Topt ~30.8°C (WT) */
  --c-warm:   #ffe566;   /* Topt ~33°C  */
  --c-hot:    #ff7b1a;   /* Topt ~36°C  */
  --c-xhot:   #ff2244;   /* Topt ~38°C+ */

  --bg:       #03060b;
  --surface:  #080f18;
  --panel:    #0b1520;
  --border:   #142030;
  --text:     #b8d4e8;
  --muted:    #3a5570;
  --accent:   #40e0c0;
}

html, body { height: 100%; background: var(--bg); color: var(--text);
  font-family: 'Space Mono', monospace; overflow: hidden; }

/* ── scanlines ── */
body::after {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:9000;
  background: repeating-linear-gradient(0deg,transparent,transparent 3px,
    rgba(0,0,0,.06) 3px,rgba(0,0,0,.06) 4px);
}

/* ── layout ── */
.shell { display: grid; grid-template-columns: 260px 1fr 260px;
         grid-template-rows: 56px 1fr; height: 100vh; }

/* ── top bar ── */
.topbar {
  grid-column: 1/-1; display: flex; align-items: center;
  justify-content: space-between; padding: 0 24px;
  background: linear-gradient(90deg,rgba(64,224,192,.07),rgba(255,123,26,.04));
  border-bottom: 1px solid var(--border); z-index: 10;
}
.brand { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:800;
         letter-spacing:.04em;
         background:linear-gradient(90deg,var(--c-wt),var(--c-hot));
         -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.brand-sub { font-size:.58rem; color:var(--muted); letter-spacing:.18em;
             font-family:'Space Mono',monospace; text-transform:uppercase; margin-top:1px; }
.topbar-stats { display:flex; gap:20px; align-items:center; }
.ts { font-size:.7rem; display:flex; gap:6px; align-items:center; }
.ts-label { color:var(--muted); }
.ts-val { font-weight:700; }
.pill { padding:3px 12px; border-radius:20px; border:1px solid var(--border);
        font-size:.7rem; display:flex; align-items:center; gap:6px;
        background:var(--surface); }
.dot { width:8px; height:8px; border-radius:50%; background:var(--muted); }
.dot.run { background:var(--c-wt); box-shadow:0 0 8px var(--c-wt); animation:blink 1.2s infinite; }
.dot.done { background:var(--c-hot); }
@keyframes blink { 0%,100%{opacity:1}50%{opacity:.25} }

/* ── left panel ── */
aside {
  background: var(--surface); border-right:1px solid var(--border);
  padding:18px 14px; display:flex; flex-direction:column; gap:14px;
  overflow-y:auto;
}
.sec { font-size:.58rem; letter-spacing:.2em; text-transform:uppercase;
       color:var(--muted); padding-bottom:7px; border-bottom:1px solid var(--border); }
.ctrl { display:flex; flex-direction:column; gap:5px; }
.ctrl label { font-size:.68rem; color:var(--muted); display:flex;
              justify-content:space-between; }
.ctrl label span { color:var(--accent); font-weight:700; }
input[type=range] {
  -webkit-appearance:none; width:100%; height:3px; outline:none; cursor:pointer;
  background:linear-gradient(90deg,var(--c-wt),var(--c-hot)); border-radius:2px;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance:none; width:13px; height:13px; border-radius:50%;
  background:#fff; border:2px solid var(--c-wt); box-shadow:0 0 6px var(--c-wt);
}
.tog-row { display:flex; justify-content:space-between; align-items:center; }
.tog-lbl { font-size:.74rem; }
.tog { position:relative; width:42px; height:21px; cursor:pointer; }
.tog input { opacity:0; width:0; height:0; }
.tog-track { position:absolute; inset:0; border-radius:21px;
             background:var(--border); transition:.3s; }
.tog-track::before { content:''; position:absolute; width:15px; height:15px;
  left:3px; bottom:3px; border-radius:50%; background:var(--muted); transition:.3s; }
.tog input:checked + .tog-track { background:var(--c-wt); }
.tog input:checked + .tog-track::before { transform:translateX(21px); background:#fff; }
.btn { width:100%; padding:10px; border:none; border-radius:5px; cursor:pointer;
       font-family:'Space Mono',monospace; font-size:.8rem; font-weight:700;
       letter-spacing:.06em; transition:.2s; }
.btn-run { background:linear-gradient(135deg,var(--c-wt),#1adfaf); color:#000; }
.btn-run:hover { filter:brightness(1.15); box-shadow:0 0 24px rgba(64,224,192,.35); }
.btn-stop { background:transparent; color:#ff4455; border:1px solid #ff4455; }
.btn-stop:hover { background:rgba(255,68,85,.1); }
.btn:disabled { opacity:.35; cursor:not-allowed; }

/* progress */
.prog-wrap { height:2px; background:var(--border); border-radius:1px; overflow:hidden; }
.prog-fill { height:100%; border-radius:1px; width:0%;
             background:linear-gradient(90deg,var(--c-wt),var(--c-hot)); transition:width .3s; }

/* ── centre canvas area ── */
.canvas-area {
  position:relative; background:var(--bg); overflow:hidden;
  display:flex; flex-direction:column;
}

/* rep selector tabs */
.rep-tabs { display:flex; gap:1px; background:var(--border); flex-shrink:0; }
.rep-tab { flex:1; padding:6px; text-align:center; font-size:.65rem;
           cursor:pointer; background:var(--surface); color:var(--muted);
           letter-spacing:.08em; transition:.2s; border:none; font-family:inherit; }
.rep-tab.active { background:var(--panel); color:var(--accent);
                  border-bottom:2px solid var(--accent); }

/* flask vessel */
.vessel-wrap { flex:1; position:relative; display:flex;
               align-items:center; justify-content:center; padding:16px; }
canvas#petri { border-radius:50%; cursor:crosshair; display:block; }

/* tooltip */
.cell-tip {
  position:absolute; background:rgba(8,15,24,.95); border:1px solid var(--border);
  border-radius:6px; padding:8px 11px; font-size:.65rem; pointer-events:none;
  opacity:0; transition:opacity .15s; z-index:100; min-width:130px;
  backdrop-filter:blur(6px);
}
.cell-tip.show { opacity:1; }
.tip-row { display:flex; justify-content:space-between; gap:12px; padding:1px 0; }
.tip-key { color:var(--muted); }
.tip-val { color:var(--accent); font-weight:700; }

/* ── right panel ── */
.right-panel {
  background:var(--surface); border-left:1px solid var(--border);
  padding:18px 14px; display:flex; flex-direction:column; gap:14px;
  overflow-y:auto;
}

/* colour legend */
.legend { display:flex; flex-direction:column; gap:5px; }
.leg-grad {
  height:14px; border-radius:7px; width:100%;
  background:linear-gradient(90deg,var(--c-cold),var(--c-wt),var(--c-warm),var(--c-hot),var(--c-xhot));
}
.leg-labels { display:flex; justify-content:space-between; font-size:.6rem; color:var(--muted); }

/* live stats */
.stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:7px; }
.stat { background:var(--panel); border:1px solid var(--border); border-radius:5px;
        padding:7px 9px; }
.stat-n { font-size:.57rem; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; }
.stat-v { font-size:.95rem; font-weight:700; margin-top:2px; }

/* mutation badge list */
.mut-list { display:flex; flex-direction:column; gap:4px; }
.mut-row { display:flex; align-items:center; gap:7px; font-size:.62rem; }
.mut-swatch { width:8px; height:8px; border-radius:2px; flex-shrink:0; }
.mut-name { color:var(--muted); flex:1; white-space:nowrap; overflow:hidden;
            text-overflow:ellipsis; }
.mut-bar-bg { width:60px; height:4px; background:var(--border); border-radius:2px; overflow:hidden; }
.mut-bar-fill { height:100%; border-radius:2px; transition:width .4s; }
.mut-pct { color:var(--accent); width:26px; text-align:right; }

/* temp ramp mini-timeline */
.ramp-track { position:relative; height:24px; background:var(--panel);
              border-radius:4px; overflow:hidden; border:1px solid var(--border); }
.ramp-fill { position:absolute; top:0; left:0; height:100%; transition:width .4s;
             background:linear-gradient(90deg,var(--c-cold),var(--c-wt),var(--c-warm),var(--c-hot),var(--c-xhot)); }
.ramp-label { position:absolute; inset:0; display:flex; align-items:center;
              justify-content:center; font-size:.65rem; font-weight:700;
              mix-blend-mode:overlay; color:#fff; }

/* environment glow */
.env-glow {
  position:absolute; inset:0; pointer-events:none; border-radius:50%;
  transition:box-shadow 1.5s;
}

/* aneuploidy flash indicator */
.aneu-indicator {
  position:absolute; top:10px; right:10px; font-size:.6rem; padding:3px 8px;
  border-radius:10px; background:rgba(189,147,249,.15); border:1px solid #bd93f9;
  color:#bd93f9; opacity:0; transition:opacity .5s;
}
.aneu-indicator.show { opacity:1; }

::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
</style>
</head>
<body>
<div class="shell">

<!-- ── TOP BAR ── -->
<header class="topbar">
  <div>
    <div class="brand">S. cerevisiae ALE</div>
    <div class="brand-sub">Live Population Visualiser</div>
  </div>
  <div class="topbar-stats">
    <div class="ts"><span class="ts-label">GEN</span>
      <span class="ts-val" id="hGen" style="color:var(--c-wt)">0</span></div>
    <div class="ts"><span class="ts-label">TEMP</span>
      <span class="ts-val" id="hTemp" style="color:var(--c-hot)">30°C</span></div>
    <div class="ts"><span class="ts-label">FITNESS</span>
      <span class="ts-val" id="hFit" style="color:var(--c-warm)">—</span></div>
    <div class="ts"><span class="ts-label">Topt</span>
      <span class="ts-val" id="hTopt" style="color:var(--accent)">—</span></div>
    <div class="pill">
      <div class="dot" id="statusDot"></div>
      <span id="statusTxt">Ready</span>
    </div>
  </div>
</header>

<!-- ── LEFT SIDEBAR ── -->
<aside>
  <div class="sec">Experiment Setup</div>

  <div class="ctrl">
    <label>Generations <span id="vGens">700</span></label>
    <input type="range" id="sGens" min="200" max="1000" step="50" value="700"
           oninput="V('vGens',this.value)">
  </div>
  <div class="ctrl">
    <label>Population Ne <span id="vNe">200k</span></label>
    <input type="range" id="sNe" min="1" max="5" step="1" value="2"
           oninput="neLabel(this.value)">
  </div>
  <div class="ctrl">
    <label>Replicates <span id="vReps">2</span></label>
    <input type="range" id="sReps" min="1" max="4" step="1" value="2"
           oninput="V('vReps',this.value); buildTabs(+this.value)">
  </div>
  <div class="ctrl">
    <label>Temp Ramp Speed <span id="vRamp">1.0×</span></label>
    <input type="range" id="sRamp" min="0.5" max="3.0" step="0.25" value="1.0"
           oninput="V('vRamp',parseFloat(this.value).toFixed(2)+'×')">
  </div>
  <div class="ctrl">
    <label>Mutation Rate <span id="vMu">1.0×</span></label>
    <input type="range" id="sMu" min="0.1" max="5.0" step="0.1" value="1.0"
           oninput="V('vMu',parseFloat(this.value).toFixed(1)+'×')">
  </div>
  <div class="tog-row">
    <span class="tog-lbl">Aneuploidy</span>
    <label class="tog"><input type="checkbox" id="cbAneu" checked>
      <span class="tog-track"></span></label>
  </div>
  <div class="ctrl">
    <label>Cell density <span id="vDens">medium</span></label>
    <input type="range" id="sDens" min="100" max="800" step="50" value="350"
           oninput="densLabel(this.value)">
  </div>

  <div class="prog-wrap"><div class="prog-fill" id="prog"></div></div>

  <button class="btn btn-run" id="btnRun" onclick="startSim()">▶ START CULTURE</button>
  <button class="btn btn-stop" id="btnStop" onclick="stopSim()" disabled>■ STOP</button>

  <!-- temperature timeline -->
  <div>
    <div class="sec" style="margin-bottom:8px">Temperature Ramp</div>
    <div class="ramp-track">
      <div class="ramp-fill" id="rampFill" style="width:0%"></div>
      <div class="ramp-label" id="rampLabel">30°C</div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:.57rem;
                color:var(--muted);margin-top:4px">
      <span>30°C</span><span>33°C</span><span>36°C</span>
      <span>38°C</span><span>40°C</span><span>42°C</span>
    </div>
  </div>
</aside>

<!-- ── CENTRE: PETRI DISH ── -->
<div class="canvas-area">
  <div class="rep-tabs" id="repTabs"></div>
  <div class="vessel-wrap" id="vesselWrap">
    <div class="env-glow" id="envGlow"></div>
    <canvas id="petri"></canvas>
    <div class="cell-tip" id="cellTip"></div>
    <div class="aneu-indicator" id="aneuInd">⬡ ANEUPLOIDY EVENT</div>
  </div>
</div>

<!-- ── RIGHT PANEL ── -->
<div class="right-panel">
  <div class="sec">Topt Colour Map</div>
  <div class="legend">
    <div class="leg-grad"></div>
    <div class="leg-labels">
      <span>&lt;29°C</span><span>30.8°C</span><span>33°C</span>
      <span>36°C</span><span>&gt;38°C</span>
    </div>
    <div style="font-size:.6rem;color:var(--muted);margin-top:4px;line-height:1.6">
      Cell colour = thermal optimum.<br>
      Size = relative frequency.<br>
      Glow ring = aneuploidy carrier.<br>
      Opacity = fitness at current T.
    </div>
  </div>

  <div class="sec">Live Stats</div>
  <div class="stat-grid">
    <div class="stat">
      <div class="stat-n">Fitness</div>
      <div class="stat-v" id="sv-w" style="color:var(--c-warm)">—</div>
    </div>
    <div class="stat">
      <div class="stat-n">Mean Topt</div>
      <div class="stat-v" id="sv-to" style="color:var(--accent)">—</div>
    </div>
    <div class="stat">
      <div class="stat-n">Mean Tmax</div>
      <div class="stat-v" id="sv-tx" style="color:var(--c-hot)">—</div>
    </div>
    <div class="stat">
      <div class="stat-n">Lineages</div>
      <div class="stat-v" id="sv-nl" style="color:var(--c-wt)">—</div>
    </div>
    <div class="stat">
      <div class="stat-n">Aneuploidy</div>
      <div class="stat-v" id="sv-an" style="color:#bd93f9">—</div>
    </div>
    <div class="stat">
      <div class="stat-n">Total Lineages</div>
      <div class="stat-v" id="sv-nt" style="color:var(--muted)">—</div>
    </div>
  </div>

  <div class="sec">Adaptive Mutations</div>
  <div class="mut-list" id="mutList"></div>

  <div class="sec">Population Composition</div>
  <canvas id="miniBar" width="230" height="60"
          style="width:100%;border-radius:4px;background:var(--panel);border:1px solid var(--border)"></canvas>
</div>

</div><!-- .shell -->

<script>
// ════════════════════════════════════════════════════════
//  CONSTANTS & STATE
// ════════════════════════════════════════════════════════
const WT_TOPT = 30.8;
const MODULE_NAMES = ["ERG3_LOF","ETC_LOF","CDC25_reg","HSF1_pt","SKN7_pt",
                      "HSP104","TPS1_TPS2","HOG1_reg","RAS2_IRA2","Cell_wall"];
const MOD_COLS = ["#ff6b35","#ff2244","#40e0c0","#ffe566","#bd93f9",
                  "#1adfaf","#f1fa8c","#79c0ff","#ffb86c","#50fa7b"];
const NE_MAP  = {1:50000,2:200000,3:500000,4:1000000,5:3000000};
const NE_LABS = {1:"50k",2:"200k",3:"500k",4:"1M",5:"3M"};

let activeRep = 0;
let nReps = 2;
let allRepData = {};
let evtSource = null;
let simRunning = false;
let maxNGens = 700;
let particles = [];          // current frame's particle objects
let animId = null;
let lastFrameData = null;
let aneuFlashTimer = 0;

// ── canvas setup ──
const canvas = document.getElementById('petri');
const ctx    = canvas.getContext('2d');
const miniBar= document.getElementById('miniBar');
const mctx   = miniBar.getContext('2d');

function resizeCanvas() {
  const wrap = document.getElementById('vesselWrap');
  const dim  = Math.min(wrap.clientWidth - 32, wrap.clientHeight - 32, 680);
  canvas.width  = dim;
  canvas.height = dim;
  if (lastFrameData) renderCells(lastFrameData, document.getElementById('hTemp').textContent);
}
window.addEventListener('resize', resizeCanvas);

// ════════════════════════════════════════════════════════
//  COLOUR: Topt → colour (thermal gradient)
// ════════════════════════════════════════════════════════
function toptToRGB(topt) {
  // Maps Topt ~26-42°C onto cool→hot colour scale
  const stops = [
    { t:26,  r:26,  g:111, b:255 },   // deep blue
    { t:29,  r:30,  g:160, b:230 },   // cyan-blue
    { t:30.8,r:64,  g:224, b:192 },   // WT teal
    { t:32,  r:120, g:240, b:140 },   // green
    { t:33.5,r:220, g:240, b:80  },   // yellow-green
    { t:35,  r:255, g:229, b:60  },   // yellow
    { t:36.5,r:255, g:160, b:40  },   // amber
    { t:38,  r:255, g:100, b:26  },   // orange
    { t:39.5,r:255, g:40,  b:60  },   // red-orange
    { t:42,  r:200, g:10,  b:80  },   // deep red
  ];
  const clamp = Math.max(stops[0].t, Math.min(stops[stops.length-1].t, topt));
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i], b = stops[i+1];
    if (clamp >= a.t && clamp <= b.t) {
      const f = (clamp - a.t) / (b.t - a.t);
      return [
        Math.round(a.r + f*(b.r-a.r)),
        Math.round(a.g + f*(b.g-a.g)),
        Math.round(a.b + f*(b.b-a.b)),
      ];
    }
  }
  return [200,10,80];
}

function rgbToStr(rgb, alpha=1) {
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
}

// ════════════════════════════════════════════════════════
//  PARTICLE LAYOUT ENGINE
//  Converts lineage data → stable, physics-influenced positions
// ════════════════════════════════════════════════════════
let particleStore = {};   // id → {x, y, vx, vy, bud angle, phase}

function buildParticles(cells, W, H, maxCells) {
  const R     = W * 0.46;   // dish radius
  const cx    = W / 2, cy = H / 2;
  const result = [];
  const neMap  = {};

  // Sort by freq descending, take top maxCells
  const sorted = [...cells].sort((a,b) => b.freq - a.freq).slice(0, maxCells);
  const totalFreq = sorted.reduce((s,c) => s+c.freq, 0) || 1;

  sorted.forEach((cell, idx) => {
    const key = `${Math.round(cell.topt*10)}_${cell.muts}_${cell.aneu?1:0}`;
    let p = particleStore[key];
    if (!p) {
      // Place by angle sector based on Topt
      const sector = (cell.topt - 26) / 16 * Math.PI * 2;
      const rad    = R * (0.15 + Math.random() * 0.75);
      p = {
        x: cx + rad * Math.cos(sector + Math.random()*0.8-0.4),
        y: cy + rad * Math.sin(sector + Math.random()*0.8-0.4),
        vx: (Math.random()-0.5)*0.3,
        vy: (Math.random()-0.5)*0.3,
        budAngle: Math.random() * Math.PI * 2,
        phase: Math.random() * Math.PI * 2,
      };
      particleStore[key] = p;
    }

    // Brownian drift
    p.vx += (Math.random()-0.5)*0.12;
    p.vy += (Math.random()-0.5)*0.12;
    p.vx *= 0.88; p.vy *= 0.88;
    p.x  += p.vx;  p.y  += p.vy;

    // Keep inside dish with soft boundary
    const dx = p.x - cx, dy = p.y - cy;
    const dist = Math.sqrt(dx*dx+dy*dy);
    if (dist > R*0.92) {
      p.x -= dx * 0.04; p.y -= dy * 0.04;
      p.vx *= -0.3; p.vy *= -0.3;
    }

    // Cell body size: proportional to sqrt(freq), with floor
    const baseR = Math.max(3.5, Math.sqrt(cell.freq / totalFreq) * W * 0.55);
    const rgb   = toptToRGB(cell.topt);
    const fitness_opacity = Math.max(0.3, Math.min(1.0, cell.freq * 12 + 0.4));

    result.push({
      ...p, key, cell,
      r: baseR, rgb, alpha: fitness_opacity,
      budPhase: p.phase,
    });
    p.phase += 0.025;  // cell-cycle phase drift
  });
  return result;
}

// ════════════════════════════════════════════════════════
//  RENDERER: draw yeast cells (ellipsoid + bud)
// ════════════════════════════════════════════════════════
function drawYeastCell(x, y, r, rgb, alpha, budAngle, budPhase, isAneu, muts) {
  ctx.save();

  // ── Aneuploidy glow ring ──
  if (isAneu) {
    const grd = ctx.createRadialGradient(x,y,r*0.8,x,y,r*2.4);
    grd.addColorStop(0, 'rgba(189,147,249,0.22)');
    grd.addColorStop(1, 'rgba(189,147,249,0)');
    ctx.beginPath(); ctx.arc(x,y,r*2.4,0,Math.PI*2);
    ctx.fillStyle = grd; ctx.fill();
  }

  // ── Mutation complexity → cell shape irregularity ──
  const irregularity = Math.min(muts * 0.06, 0.28);

  // ── Mother cell body ──
  const grad = ctx.createRadialGradient(
    x - r*0.28, y - r*0.28, r*0.05,
    x, y, r*1.1
  );
  grad.addColorStop(0, rgbToStr(rgb.map(c=>Math.min(255,c+60)), alpha));
  grad.addColorStop(0.45, rgbToStr(rgb, alpha));
  grad.addColorStop(1,    rgbToStr(rgb.map(c=>Math.max(0,c-40)), alpha*0.7));

  ctx.beginPath();
  // Slightly irregular ellipse for evolved cells
  const scaleX = 1.0 + irregularity * Math.sin(budPhase*0.7);
  const scaleY = 1.0 - irregularity * Math.sin(budPhase*0.7) * 0.5;
  ctx.ellipse(x, y, r*scaleX, r*scaleY, budAngle*0.3, 0, Math.PI*2);
  ctx.fillStyle = grad;
  ctx.fill();

  // Cell wall outline
  ctx.strokeStyle = rgbToStr(rgb.map(c=>Math.min(255,c+80)), alpha*0.6);
  ctx.lineWidth = 0.7;
  ctx.stroke();

  // ── Nucleus (small dark circle) ──
  ctx.beginPath();
  ctx.arc(x + r*0.1, y + r*0.05, r*0.28, 0, Math.PI*2);
  ctx.fillStyle = rgbToStr([0,0,0], alpha*0.35);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x + r*0.1, y + r*0.05, r*0.28, 0, Math.PI*2);
  ctx.strokeStyle = rgbToStr(rgb, alpha*0.4);
  ctx.lineWidth = 0.5;
  ctx.stroke();

  // ── Bud (daughter cell, size pulsates with cell-cycle phase) ──
  const budSize = r * (0.35 + 0.28 * Math.abs(Math.sin(budPhase)));
  if (budSize > 1.5) {
    const bx = x + (r + budSize*0.65) * Math.cos(budAngle);
    const by = y + (r + budSize*0.65) * Math.sin(budAngle);

    // Bud neck
    ctx.beginPath();
    ctx.moveTo(x + r*0.6*Math.cos(budAngle), y + r*0.6*Math.sin(budAngle));
    ctx.lineTo(bx - budSize*0.5*Math.cos(budAngle), by - budSize*0.5*Math.sin(budAngle));
    ctx.strokeStyle = rgbToStr(rgb, alpha*0.5);
    ctx.lineWidth = budSize * 0.55;
    ctx.lineCap = 'round'; ctx.stroke();

    // Bud body
    const budGrad = ctx.createRadialGradient(
      bx-budSize*0.2, by-budSize*0.2, budSize*0.05, bx, by, budSize
    );
    budGrad.addColorStop(0, rgbToStr(rgb.map(c=>Math.min(255,c+50)), alpha*0.9));
    budGrad.addColorStop(1, rgbToStr(rgb.map(c=>Math.max(0,c-30)),   alpha*0.6));
    ctx.beginPath(); ctx.arc(bx, by, budSize, 0, Math.PI*2);
    ctx.fillStyle = budGrad; ctx.fill();
    ctx.strokeStyle = rgbToStr(rgb.map(c=>Math.min(255,c+70)), alpha*0.5);
    ctx.lineWidth = 0.5; ctx.stroke();
  }

  ctx.restore();
}

// ════════════════════════════════════════════════════════
//  PETRI DISH BACKGROUND
// ════════════════════════════════════════════════════════
function drawDish(W, H, T) {
  const cx=W/2, cy=H/2, R=W*0.46;

  // Dish glass fill
  const diskGrad = ctx.createRadialGradient(cx,cy,0,cx,cy,R);
  diskGrad.addColorStop(0,   'rgba(15,30,50,0.95)');
  diskGrad.addColorStop(0.75,'rgba(8,18,32,0.98)');
  diskGrad.addColorStop(1,   'rgba(4,10,20,1)');
  ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2);
  ctx.fillStyle = diskGrad; ctx.fill();

  // Temperature-tinted environment glow
  const tempFrac = Math.max(0,Math.min(1,(T-30)/12));
  const glowR = Math.round(26  + tempFrac*229);
  const glowG = Math.round(224 - tempFrac*214);
  const glowB = Math.round(192 - tempFrac*158);
  const envGrad = ctx.createRadialGradient(cx,cy,R*0.5,cx,cy,R);
  envGrad.addColorStop(0,'rgba(0,0,0,0)');
  envGrad.addColorStop(0.7,'rgba(0,0,0,0)');
  envGrad.addColorStop(1,`rgba(${glowR},${glowG},${glowB},${0.12+tempFrac*0.18})`);
  ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2);
  ctx.fillStyle=envGrad; ctx.fill();

  // Dish rim
  const rimGrad = ctx.createRadialGradient(cx,cy,R*0.93,cx,cy,R*1.01);
  rimGrad.addColorStop(0,'rgba(60,120,180,0.0)');
  rimGrad.addColorStop(0.4,'rgba(60,120,180,0.25)');
  rimGrad.addColorStop(1,'rgba(20,60,100,0.15)');
  ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2);
  ctx.strokeStyle=rimGrad; ctx.lineWidth=6; ctx.stroke();

  // Dish reflection arc
  ctx.save();
  ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2); ctx.clip();
  ctx.beginPath();
  ctx.arc(cx-R*0.3, cy-R*0.35, R*0.55, -0.3, Math.PI*0.4);
  ctx.strokeStyle='rgba(255,255,255,0.04)'; ctx.lineWidth=R*0.18; ctx.stroke();
  ctx.restore();
}

// ════════════════════════════════════════════════════════
//  MAIN RENDER CALL
// ════════════════════════════════════════════════════════
function renderCells(repData, tempStr) {
  if (!repData) return;
  const W=canvas.width, H=canvas.height;
  const T=repData.temp||30;
  const maxCells=parseInt(document.getElementById('sDens').value)||350;

  ctx.clearRect(0,0,W,H);
  drawDish(W,H,T);

  particles = buildParticles(repData.cells||[], W, H, maxCells);

  // Draw back-to-front (smallest first)
  const sorted = [...particles].sort((a,b)=>a.r-b.r);
  for (const p of sorted) {
    drawYeastCell(p.x, p.y, p.r, p.rgb, p.alpha,
                  p.budAngle, p.budPhase, p.cell.aneu, p.cell.muts);
  }
}

// ════════════════════════════════════════════════════════
//  ANIMATION LOOP (keeps cells wiggling between SSE frames)
// ════════════════════════════════════════════════════════
function animLoop() {
  animId = requestAnimationFrame(animLoop);
  if (!lastFrameData) return;
  const tempStr = document.getElementById('hTemp').textContent;
  renderCells(lastFrameData, tempStr);
}
animLoop();

// ════════════════════════════════════════════════════════
//  MINI BAR CHART (right panel composition)
// ════════════════════════════════════════════════════════
function renderMiniBar(cells) {
  const W=miniBar.width, H=miniBar.height;
  mctx.clearRect(0,0,W,H);
  if (!cells||cells.length===0) return;
  const total=cells.reduce((s,c)=>s+c.freq,0)||1;
  let x=0;
  const sorted=[...cells].sort((a,b)=>b.freq-a.freq).slice(0,30);
  for (const c of sorted) {
    const w=c.freq/total*W;
    const rgb=toptToRGB(c.topt);
    mctx.fillStyle=rgbToStr(rgb,0.9);
    mctx.fillRect(x,0,Math.max(w,1),H);
    x+=w;
  }
  // Topt axis labels
  mctx.fillStyle='rgba(0,0,0,0.5)';
  mctx.fillRect(0,H-14,W,14);
  mctx.font='8px Space Mono';
  mctx.fillStyle='rgba(255,255,255,0.5)';
  mctx.fillText('← cold lineages          hot lineages →',4,H-3);
}

// ════════════════════════════════════════════════════════
//  MODULE BARS (right panel)
// ════════════════════════════════════════════════════════
function renderModBars(cells) {
  const modFreq={};
  MODULE_NAMES.forEach(m=>modFreq[m]=0);
  if (!cells) return;
  const total=cells.reduce((s,c)=>s+c.freq,0)||1;
  cells.forEach(c=>{
    (c.modules||[]).forEach(m=>{ if(modFreq[m]!==undefined) modFreq[m]+=c.freq/total; });
  });
  const el=document.getElementById('mutList');
  el.innerHTML='';
  MODULE_NAMES.forEach((m,i)=>{
    const v=Math.min(1,modFreq[m]||0);
    const pct=Math.round(v*100);
    const row=document.createElement('div'); row.className='mut-row';
    row.innerHTML=`<div class="mut-swatch" style="background:${MOD_COLS[i]}"></div>
      <span class="mut-name" title="${m}">${m.replace(/_/g,' ')}</span>
      <div class="mut-bar-bg"><div class="mut-bar-fill"
        style="width:${pct}%;background:${MOD_COLS[i]}"></div></div>
      <span class="mut-pct">${pct}%</span>`;
    el.appendChild(row);
  });
}

// ════════════════════════════════════════════════════════
//  SSE UPDATE HANDLER
// ════════════════════════════════════════════════════════
let prevAneu=0;

function onSimData(payload) {
  const { gen, n_gens, reps } = payload;
  allRepData = {};
  (reps||[]).forEach(r=>{ allRepData[r.rep]=r; });

  // Header counters
  document.getElementById('hGen').textContent = gen;
  document.getElementById('prog').style.width = (gen/n_gens*100)+'%';

  const rep = allRepData[activeRep] || reps[0];
  if (!rep) return;

  lastFrameData = rep;

  const T = rep.temp||30;
  document.getElementById('hTemp').textContent = T+'°C';
  document.getElementById('hFit').textContent  = (rep.wbar||0).toFixed(4);
  document.getElementById('hTopt').textContent = (rep.topt||0).toFixed(2)+'°C';
  document.getElementById('sv-w').textContent  = (rep.wbar||0).toFixed(4);
  document.getElementById('sv-to').textContent = (rep.topt||0).toFixed(2)+'°C';
  document.getElementById('sv-tx').textContent = (rep.tmax||0).toFixed(2)+'°C';
  document.getElementById('sv-nl').textContent = rep.n_lin||0;
  document.getElementById('sv-an').textContent = ((rep.aneu||0)*100).toFixed(1)+'%';
  document.getElementById('sv-nt').textContent = rep.n_lineages||0;

  // Ramp progress
  const tempFrac=Math.max(0,Math.min(1,(T-30)/12));
  document.getElementById('rampFill').style.width=(tempFrac*100)+'%';
  document.getElementById('rampLabel').textContent=T+'°C';

  // Aneuploidy flash
  if ((rep.aneu||0)>prevAneu+0.01) {
    const ind=document.getElementById('aneuInd');
    ind.classList.add('show'); aneuFlashTimer=80;
  }
  prevAneu=rep.aneu||0;
  if (aneuFlashTimer>0) {
    aneuFlashTimer--;
    if (aneuFlashTimer===0) document.getElementById('aneuInd').classList.remove('show');
  }

  renderMiniBar(rep.cells);
  renderModBars(rep.cells);
}

// ════════════════════════════════════════════════════════
//  TOOLTIP on hover
// ════════════════════════════════════════════════════════
canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX-rect.left) * (canvas.width/rect.width);
  const my = (e.clientY-rect.top)  * (canvas.height/rect.height);
  const tip = document.getElementById('cellTip');
  let hit = null;
  for (const p of [...particles].reverse()) {
    const dx=mx-p.x, dy=my-p.y;
    if (dx*dx+dy*dy < p.r*p.r*1.8) { hit=p; break; }
  }
  if (hit) {
    const c=hit.cell;
    tip.innerHTML=`
      <div class="tip-row"><span class="tip-key">Topt</span>
        <span class="tip-val" style="color:${rgbToStr(hit.rgb)}">${c.topt.toFixed(2)}°C</span></div>
      <div class="tip-row"><span class="tip-key">Tmax</span>
        <span class="tip-val">${c.tmax.toFixed(2)}°C</span></div>
      <div class="tip-row"><span class="tip-key">Frequency</span>
        <span class="tip-val">${(c.freq*100).toFixed(2)}%</span></div>
      <div class="tip-row"><span class="tip-key">Mutations</span>
        <span class="tip-val">${c.muts}</span></div>
      <div class="tip-row"><span class="tip-key">Aneuploidy</span>
        <span class="tip-val">${c.aneu?'yes':'no'}</span></div>
      ${c.modules.length?`<div class="tip-row"><span class="tip-key">Modules</span>
        <span class="tip-val" style="font-size:.6rem">${c.modules.join(', ')}</span></div>`:''}`;
    tip.style.left=(e.clientX-rect.left+14)+'px';
    tip.style.top =(e.clientY-rect.top -10)+'px';
    tip.classList.add('show');
  } else {
    tip.classList.remove('show');
  }
});
canvas.addEventListener('mouseleave',()=>document.getElementById('cellTip').classList.remove('show'));

// ════════════════════════════════════════════════════════
//  REP TABS
// ════════════════════════════════════════════════════════
function buildTabs(n) {
  nReps=n;
  const el=document.getElementById('repTabs');
  el.innerHTML='';
  for(let r=0;r<n;r++){
    const b=document.createElement('button');
    b.className='rep-tab'+(r===activeRep?' active':'');
    b.textContent=`Flask ${r+1}`;
    b.onclick=(()=>{const ri=r;return()=>{
      activeRep=ri;
      document.querySelectorAll('.rep-tab').forEach((t,i)=>t.classList.toggle('active',i===ri));
      lastFrameData=allRepData[ri]||null;
    };})();
    el.appendChild(b);
  }
}
buildTabs(2);

// ════════════════════════════════════════════════════════
//  SIM CONTROLS
// ════════════════════════════════════════════════════════
function startSim() {
  if(evtSource){evtSource.close();evtSource=null;}
  particleStore={};
  lastFrameData=null;
  allRepData={};
  prevAneu=0;
  const nr=parseInt(document.getElementById('sReps').value);
  maxNGens=parseInt(document.getElementById('sGens').value);
  activeRep=0; buildTabs(nr);
  const cfg={
    n_reps:nr, n_gens:maxNGens,
    ne:NE_MAP[parseInt(document.getElementById('sNe').value)]||200000,
    mu_mult:parseFloat(document.getElementById('sMu').value),
    aneu_on:document.getElementById('cbAneu').checked,
    ramp_speed:parseFloat(document.getElementById('sRamp').value),
  };
  fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(cfg)}).then(r=>r.json()).then(()=>{
    simRunning=true;
    document.getElementById('btnRun').disabled=true;
    document.getElementById('btnStop').disabled=false;
    document.getElementById('statusDot').className='dot run';
    document.getElementById('statusTxt').textContent='Evolving…';
    evtSource=new EventSource('/stream');
    evtSource.onmessage=e=>{
      const d=JSON.parse(e.data);
      if(d.ping) return;
      if(d.done){
        simRunning=false;
        document.getElementById('btnRun').disabled=false;
        document.getElementById('btnStop').disabled=true;
        document.getElementById('statusDot').className='dot done';
        document.getElementById('statusTxt').textContent='Complete';
        evtSource.close(); return;
      }
      onSimData(d);
    };
  });
}

function stopSim(){
  fetch('/api/stop',{method:'POST'});
  simRunning=false;
  document.getElementById('btnRun').disabled=false;
  document.getElementById('btnStop').disabled=true;
  document.getElementById('statusDot').className='dot';
  document.getElementById('statusTxt').textContent='Stopped';
  if(evtSource){evtSource.close();evtSource=null;}
}

// ── Slider label helpers ──
function V(id,v){document.getElementById(id).textContent=v;}
function neLabel(v){V('vNe',NE_LABS[v]||v+'k');}
function densLabel(v){V('vDens',v<200?'sparse':v<400?'medium':v<600?'dense':'packed');}

// ── Init ──
resizeCanvas();
renderModBars([]);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n{'='*55}")
    print(f"  S. cerevisiae ALE — Live Population Visualiser")
    print(f"  Open your browser at:  http://localhost:{port}")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
