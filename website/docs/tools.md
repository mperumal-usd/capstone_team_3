---
id: tools
title: Interactive Tools
sidebar_label: Interactive Tools
---

# Interactive MIDI Tools

Two browser-based tools for exploring and comparing MIDI files. Both run entirely in the browser — no server needed.

---

## MIDI Player & Analyzer

Visualize and play any MIDI file. Renders a real-time piano roll, shows note statistics, and plays back the melody using the Web Audio API (Tone.js).

<div style={{background:'linear-gradient(135deg,#667eea,#764ba2)',borderRadius:'14px',padding:'2.5rem',marginBottom:'2rem',display:'flex',alignItems:'center',justifyContent:'space-between',gap:'2rem',flexWrap:'wrap'}}>
  <div>
    <div style={{fontSize:'2.5rem',marginBottom:'0.5rem'}}>🎹</div>
    <h3 style={{color:'#fff',margin:'0 0 0.5rem',fontSize:'1.3rem'}}>MIDI Player & Analyzer</h3>
    <ul style={{color:'rgba(255,255,255,0.85)',margin:0,paddingLeft:'1.2rem',fontSize:'13px',lineHeight:'1.8'}}>
      <li>Drag-and-drop MIDI upload</li>
      <li>Real-time piano roll visualization</li>
      <li>Note density and pitch range stats</li>
      <li>Playback controls (play / pause / stop)</li>
    </ul>
  </div>
  <a href="/capstone_team_3/apps/MidiAnalyzer.html" target="_blank" rel="noopener noreferrer"
     style={{background:'#fff',color:'#6d28d9',padding:'0.75rem 2rem',borderRadius:'8px',fontWeight:600,fontSize:'15px',textDecoration:'none',whiteSpace:'nowrap',flexShrink:0,boxShadow:'0 4px 16px rgba(0,0,0,0.2)'}}>
    Launch Tool ↗
  </a>
</div>

---

## MIDI Comparator

Upload two MIDI files side-by-side and compute their similarity. Overlays both piano rolls, computes a similarity score, and shows a detailed note-level comparison chart (Chart.js).

<div style={{background:'linear-gradient(135deg,#f093fb,#f5576c)',borderRadius:'14px',padding:'2.5rem',marginBottom:'2rem',display:'flex',alignItems:'center',justifyContent:'space-between',gap:'2rem',flexWrap:'wrap'}}>
  <div>
    <div style={{fontSize:'2.5rem',marginBottom:'0.5rem'}}>🎼</div>
    <h3 style={{color:'#fff',margin:'0 0 0.5rem',fontSize:'1.3rem'}}>MIDI Comparator</h3>
    <ul style={{color:'rgba(255,255,255,0.85)',margin:0,paddingLeft:'1.2rem',fontSize:'13px',lineHeight:'1.8'}}>
      <li>Side-by-side MIDI upload</li>
      <li>Overlapping piano roll visualization</li>
      <li>Similarity score computation</li>
      <li>Note overlap and distribution charts</li>
    </ul>
  </div>
  <a href="/capstone_team_3/apps/MidiComparator.html" target="_blank" rel="noopener noreferrer"
     style={{background:'#fff',color:'#c0392b',padding:'0.75rem 2rem',borderRadius:'8px',fontWeight:600,fontSize:'15px',textDecoration:'none',whiteSpace:'nowrap',flexShrink:0,boxShadow:'0 4px 16px rgba(0,0,0,0.2)'}}>
    Launch Tool ↗
  </a>
</div>

---

## Technical Notes

Both tools are built with vanilla HTML/CSS/JavaScript using:

| Library | Purpose |
|---------|---------|
| [Tone.js](https://tonejs.github.io/) | Web Audio synthesis and MIDI playback |
| [@tonejs/midi](https://github.com/Tonejs/Midi) | MIDI file parsing |
| [Chart.js](https://www.chartjs.org/) | Similarity and distribution charts (Comparator) |

The tools run **entirely in the browser** — no data is uploaded to any server. MIDI files are processed locally using the Web File API.

Source files: [`FrontEnd/MidiAnalyzer.html`](https://github.com/mperumal-usd/capstone_team_3/blob/main/FrontEnd/MidiAnalyzer.html) · [`FrontEnd/MidiComparator.html`](https://github.com/mperumal-usd/capstone_team_3/blob/main/FrontEnd/MidiComparator.html)
