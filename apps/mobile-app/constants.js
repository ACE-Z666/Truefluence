// ── TrueFluence Mobile App — Constants ────────────────────────────────────────
// Change this IP to your machine's local IP if it changes
export const API_BASE = 'http://192.168.220.2:5000';

export const COLORS = {
  bg:        '#0a0a0f',
  surface:   '#111118',
  card:      '#16161f',
  border:    'rgba(255,255,255,0.07)',
  accent:    '#6c63ff',
  accent2:   '#a78bfa',
  blue:      '#38bdf8',
  green:     '#4ade80',
  orange:    '#fb923c',
  red:       '#f87171',
  yellow:    '#facc15',
  text:      '#f1f5f9',
  muted:     '#64748b',
  white:     '#ffffff',
};

export const VERDICT_COLORS = {
  'REAL':         { bg: '#4ade8020', text: '#4ade80', border: '#4ade8040' },
  'LIKELY SCAM':  { bg: '#fb923c20', text: '#fb923c', border: '#fb923c40' },
  'UNCERTAIN':    { bg: '#facc1520', text: '#facc15', border: '#facc1540' },
  'SCAM':         { bg: '#f8717120', text: '#f87171', border: '#f8717140' },
  'DEEPFAKE':     { bg: '#f8717133', text: '#f87171', border: '#f87171' },
  'Pending':      { bg: '#6c63ff20', text: '#a78bfa', border: '#6c63ff40' },
};
