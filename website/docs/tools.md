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

**Features:**
- Drag-and-drop MIDI upload
- Real-time piano roll visualization
- Note density and pitch range stats
- Playback controls (play / pause / stop)

<div style={{border: '1px solid var(--ifm-color-emphasis-300)', borderRadius: '12px', overflow: 'hidden', marginBottom: '2rem'}}>
  <div style={{background: 'var(--ifm-color-emphasis-100)', padding: '8px 16px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: 'var(--ifm-color-emphasis-600)'}}>
    <span style={{width: 12, height: 12, borderRadius: '50%', background: '#ef4444', display: 'inline-block'}}></span>
    <span style={{width: 12, height: 12, borderRadius: '50%', background: '#f59e0b', display: 'inline-block'}}></span>
    <span style={{width: 12, height: 12, borderRadius: '50%', background: '#22c55e', display: 'inline-block'}}></span>
    <span style={{marginLeft: 8}}>MIDI Player & Analyzer</span>
    <a href="/capstone_team_3/tools/MidiAnalyzer.html" target="_blank" rel="noopener noreferrer" style={{marginLeft: 'auto', fontSize: '12px'}}>Open in new tab ↗</a>
  </div>
  <iframe
    src="/capstone_team_3/tools/MidiAnalyzer.html"
    style={{width: '100%', height: '700px', border: 'none', display: 'block'}}
    title="MIDI Player & Analyzer"
  />
</div>

---

## MIDI Comparator

Upload two MIDI files side-by-side and compute their similarity. Overlays both piano rolls, computes a similarity score, and shows a detailed note-level comparison chart (Chart.js).

**Features:**
- Side-by-side MIDI upload
- Overlapping piano roll visualization
- Similarity score computation
- Note overlap and distribution charts

<div style={{border: '1px solid var(--ifm-color-emphasis-300)', borderRadius: '12px', overflow: 'hidden', marginBottom: '2rem'}}>
  <div style={{background: 'var(--ifm-color-emphasis-100)', padding: '8px 16px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: 'var(--ifm-color-emphasis-600)'}}>
    <span style={{width: 12, height: 12, borderRadius: '50%', background: '#ef4444', display: 'inline-block'}}></span>
    <span style={{width: 12, height: 12, borderRadius: '50%', background: '#f59e0b', display: 'inline-block'}}></span>
    <span style={{width: 12, height: 12, borderRadius: '50%', background: '#22c55e', display: 'inline-block'}}></span>
    <span style={{marginLeft: 8}}>MIDI Comparator</span>
    <a href="/capstone_team_3/tools/MidiComparator.html" target="_blank" rel="noopener noreferrer" style={{marginLeft: 'auto', fontSize: '12px'}}>Open in new tab ↗</a>
  </div>
  <iframe
    src="/capstone_team_3/tools/MidiComparator.html"
    style={{width: '100%', height: '800px', border: 'none', display: 'block'}}
    title="MIDI Comparator"
  />
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
