import React, { useState, useRef } from 'react';
import {
    View, Text, StyleSheet, TouchableOpacity, Image,
    Dimensions, ActivityIndicator, ScrollView,
} from 'react-native';
import { Video, ResizeMode } from 'expo-av';
import { COLORS, VERDICT_COLORS, API_BASE } from '../constants';

const { width } = Dimensions.get('window');

// ── Verdict Badge ─────────────────────────────────────────────────────────────
export function VerdictBadge({ verdict }) {
    const v = verdict || 'Pending';
    const c = VERDICT_COLORS[v] || VERDICT_COLORS['Pending'];
    const emoji =
        v === 'REAL' ? '🟢' :
            v === 'SCAM' ? '🔴' :
                v === 'LIKELY SCAM' ? '🟠' :
                    v === 'UNCERTAIN' ? '🟡' :
                        v === 'DEEPFAKE' ? '⛔' : '⏳';

    return (
        <View style={[styles.badge, { backgroundColor: c.bg, borderColor: c.border }]}>
            <Text style={[styles.badgeText, { color: c.text }]}>{emoji} {v}</Text>
        </View>
    );
}

// ── Score Bar ─────────────────────────────────────────────────────────────────
export function ScoreBar({ score }) {
    const pct = Math.round((score || 0) * 100);
    const color =
        pct <= 30 ? COLORS.red :
            pct <= 50 ? COLORS.orange :
                pct <= 70 ? COLORS.yellow : COLORS.green;

    return (
        <View>
            <View style={styles.barTrack}>
                <View style={[styles.barFill, { width: `${pct}%`, backgroundColor: color }]} />
            </View>
            <View style={styles.barLabels}>
                <Text style={styles.barLabel}>SCAM</Text>
                <Text style={styles.barLabel}>0.3</Text>
                <Text style={styles.barLabel}>0.5</Text>
                <Text style={styles.barLabel}>0.7</Text>
                <Text style={styles.barLabel}>REAL</Text>
            </View>
        </View>
    );
}

// ── Reel Card ─────────────────────────────────────────────────────────────────
export function ReelCard({ reel, onAnalyze }) {
    const [analyzing, setAnalyzing] = useState(false);
    const [status, setStatus] = useState({});
    const videoRef = useRef(null);
    const a = reel.analysis;

    const handleAnalyze = async () => {
        if (!reel.video_url) return;
        setAnalyzing(true);
        try { await onAnalyze(reel); }
        finally { setAnalyzing(false); }
    };

    const fmtNum = (n) =>
        n >= 1_000_000 ? (n / 1_000_000).toFixed(1) + 'M' :
            n >= 1_000 ? (n / 1_000).toFixed(1) + 'K' : String(n || 0);

    const initials = (reel.username || '?')[0]?.toUpperCase() || '?';

    return (
        <View style={styles.card}>
            {/* ── Header ── */}
            <View style={styles.cardHeader}>
                <View style={styles.avatar}>
                    <Text style={styles.avatarText}>{initials}</Text>
                </View>
                <View style={{ flex: 1 }}>
                    <Text style={styles.username}>{reel.username}</Text>
                    <Text style={styles.time}>{reel.timestamp}</Text>
                </View>
                <VerdictBadge verdict={a?.verdict} />
            </View>

            {/* ── Video ── */}
            {reel.video_url ? (
                <View>
                    <Video
                        ref={videoRef}
                        source={{ uri: `${API_BASE}${reel.video_url}` }}
                        style={styles.video}
                        useNativeControls
                        resizeMode={ResizeMode.COVER}
                        onPlaybackStatusUpdate={s => setStatus(s)}
                    />
                    {a?.is_deepfake && (
                        <View style={styles.deepfakeBanner}>
                            <Text style={styles.deepfakeBannerText}>⛔ DEEPFAKE DETECTED — Pipeline Aborted</Text>
                        </View>
                    )}
                </View>
            ) : (
                <View style={styles.noVideo}>
                    <Text style={{ fontSize: 40 }}>🎬</Text>
                    <Text style={[styles.time, { marginTop: 8 }]}>Demo reel — no video file</Text>
                </View>
            )}

            {/* ── Caption ── */}
            <Text style={styles.caption}>
                <Text style={styles.username}>{reel.username} </Text>
                {reel.caption}
            </Text>

            {/* ── Engagement ── */}
            <View style={styles.engRow}>
                <EngItem icon="👥" value={fmtNum(reel.followers)} label="followers" />
                <EngItem icon="❤️" value={fmtNum(reel.likes)} label="likes" />
                <EngItem icon="💬" value={fmtNum(reel.num_comments)} label="comments" />
            </View>

            {/* ── Comments ── */}
            {reel.comments?.slice(0, 3).map((c, i) => (
                <View key={i} style={styles.commentRow}>
                    <View style={styles.commentAvatar}>
                        <Text style={styles.commentAvatarText}>{c[0]?.toUpperCase()}</Text>
                    </View>
                    <Text style={styles.commentText} numberOfLines={2}>
                        <Text style={{ color: COLORS.accent2, fontWeight: '700' }}>user_{Math.floor(Math.random() * 9999)} </Text>
                        {c}
                    </Text>
                </View>
            ))}

            {/* ── Score Panel ── */}
            {a && !a.is_deepfake && (
                <View style={styles.scorePanel}>
                    <ScoreRow label="🎥 Video + Audio" value={(a.video_audio_score || 0).toFixed(4)} />
                    <ScoreRow label="💬 Comments + Engagement" value={(a.comments_eng_score || 0).toFixed(4)} />
                    <ScoreBar score={a.final_score} />
                    <View style={styles.finalRow}>
                        <View>
                            <Text style={styles.finalLabel}>Final Credibility</Text>
                            <Text style={styles.finalScore}>{Math.round((a.final_score || 0) * 100)}%</Text>
                        </View>
                        <VerdictBadge verdict={a.verdict} />
                    </View>
                    {a.deepfake_prob > 0 && (
                        <Text style={styles.mesoNote}>
                            🔍 Deepfake prob: {Math.round(a.deepfake_prob * 100)}% — Gate passed ✅
                        </Text>
                    )}
                </View>
            )}

            {a?.is_deepfake && (
                <View style={styles.scorePanel}>
                    <Text style={styles.deepfakeTitle}>⛔ DEEPFAKE BLOCKED</Text>
                    <Text style={styles.deepfakeSub}>
                        MesoNet detected {Math.round((a.deepfake_prob || 0) * 100)}% deepfake probability
                    </Text>
                </View>
            )}

            {/* ── Analyze Button ── */}
            {!a && (
                <TouchableOpacity
                    style={[styles.analyzeBtn, (!reel.video_url || analyzing) && styles.analyzeBtnDisabled]}
                    onPress={handleAnalyze}
                    disabled={!reel.video_url || analyzing}
                    activeOpacity={0.8}
                >
                    {analyzing
                        ? <><ActivityIndicator color="#fff" size="small" />
                            <Text style={styles.analyzeBtnText}>  Analyzing…</Text></>
                        : <Text style={styles.analyzeBtnText}>🛡 Analyze Credibility</Text>
                    }
                </TouchableOpacity>
            )}
        </View>
    );
}

// ── helpers ──
function EngItem({ icon, value, label }) {
    return (
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 4, marginRight: 16 }}>
            <Text>{icon}</Text>
            <Text style={styles.engValue}>{value}</Text>
            <Text style={styles.engLabel}> {label}</Text>
        </View>
    );
}

function ScoreRow({ label, value }) {
    return (
        <View style={styles.scoreRow}>
            <Text style={styles.scoreLabel}>{label}</Text>
            <Text style={styles.scoreVal}>{value}</Text>
        </View>
    );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
    card: {
        backgroundColor: COLORS.card,
        borderRadius: 18,
        overflow: 'hidden',
        marginHorizontal: 12,
        marginBottom: 20,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    cardHeader: {
        flexDirection: 'row', alignItems: 'center',
        padding: 14, gap: 12,
    },
    avatar: {
        width: 40, height: 40, borderRadius: 20,
        backgroundColor: COLORS.accent,
        alignItems: 'center', justifyContent: 'center',
    },
    avatarText: { color: '#fff', fontWeight: '800', fontSize: 16 },
    username: { color: COLORS.text, fontWeight: '700', fontSize: 14 },
    time: { color: COLORS.muted, fontSize: 12, marginTop: 2 },
    badge: {
        paddingHorizontal: 10, paddingVertical: 5,
        borderRadius: 999, borderWidth: 1,
    },
    badgeText: { fontSize: 11, fontWeight: '700' },
    video: { width: '100%', height: 300 },
    noVideo: {
        height: 180, backgroundColor: '#16213e',
        alignItems: 'center', justifyContent: 'center',
    },
    deepfakeBanner: {
        position: 'absolute', top: 0, left: 0, right: 0,
        backgroundColor: 'rgba(248,113,113,0.3)',
        padding: 10,
    },
    deepfakeBannerText: { color: COLORS.red, fontWeight: '700', fontSize: 13 },
    caption: { color: COLORS.text, fontSize: 13, lineHeight: 20, padding: 14, paddingTop: 10 },
    engRow: { flexDirection: 'row', paddingHorizontal: 14, paddingBottom: 10 },
    engValue: { color: COLORS.text, fontWeight: '700', fontSize: 13 },
    engLabel: { color: COLORS.muted, fontSize: 12 },
    commentRow: { flexDirection: 'row', alignItems: 'flex-start', paddingHorizontal: 14, marginBottom: 8, gap: 8 },
    commentAvatar: {
        width: 24, height: 24, borderRadius: 12,
        backgroundColor: COLORS.surface, borderWidth: 1, borderColor: COLORS.border,
        alignItems: 'center', justifyContent: 'center',
    },
    commentAvatarText: { color: COLORS.muted, fontSize: 10, fontWeight: '700' },
    commentText: { flex: 1, color: '#cbd5e1', fontSize: 12, lineHeight: 18 },
    scorePanel: {
        margin: 12, backgroundColor: COLORS.surface,
        borderRadius: 14, padding: 14,
        borderWidth: 1, borderColor: COLORS.border,
    },
    scoreRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
    scoreLabel: { color: COLORS.muted, fontSize: 12 },
    scoreVal: { color: COLORS.text, fontWeight: '700', fontSize: 12 },
    barTrack: {
        height: 6, backgroundColor: 'rgba(255,255,255,0.06)',
        borderRadius: 99, overflow: 'hidden', marginBottom: 6,
    },
    barFill: { height: '100%', borderRadius: 99 },
    barLabels: { flexDirection: 'row', justifyContent: 'space-between' },
    barLabel: { color: COLORS.muted, fontSize: 10 },
    finalRow: {
        flexDirection: 'row', alignItems: 'center',
        justifyContent: 'space-between',
        marginTop: 12, paddingTop: 12,
        borderTopWidth: 1, borderTopColor: COLORS.border,
    },
    finalLabel: { color: COLORS.muted, fontSize: 11, marginBottom: 2 },
    finalScore: { color: COLORS.accent2, fontSize: 28, fontWeight: '900' },
    mesoNote: { color: COLORS.muted, fontSize: 11, marginTop: 8, textAlign: 'center' },
    deepfakeTitle: { color: COLORS.red, fontWeight: '900', fontSize: 18, textAlign: 'center', marginBottom: 6 },
    deepfakeSub: { color: COLORS.muted, fontSize: 12, textAlign: 'center' },
    analyzeBtn: {
        margin: 14, padding: 14, borderRadius: 12,
        backgroundColor: COLORS.accent,
        alignItems: 'center', justifyContent: 'center',
        flexDirection: 'row',
    },
    analyzeBtnDisabled: { opacity: 0.5 },
    analyzeBtnText: { color: '#fff', fontWeight: '700', fontSize: 15 },
});
