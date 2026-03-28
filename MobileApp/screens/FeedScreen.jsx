import React, { useState, useEffect, useCallback } from 'react';
import {
    View, Text, FlatList, StyleSheet, TouchableOpacity,
    RefreshControl, StatusBar, SafeAreaView, ActivityIndicator,
} from 'react-native';
import { COLORS, API_BASE } from '../constants';
import { ReelCard } from '../components/ReelCard';

// ── Seed demo reels (shown even without backend) ───────────────────────────────
const SEED_REELS = [
    {
        id: 'demo01',
        username: '@investwithraj',
        caption: '🚀 Secret crypto strategy that made me $10K in a week! DM NOW 💰',
        video_url: null,
        followers: 980, likes: 8420, num_comments: 312,
        comments: [
            'This changed my life!! Already made $500',
            'Fake!! Reported this account',
            'Bro this is 100% scam don\'t trust',
        ],
        timestamp: '2026-03-24 00:01',
        analysis: null,
    },
    {
        id: 'demo02',
        username: '@techreviews_official',
        caption: 'Honest 30-day review of the new smartphone. Full breakdown 👇',
        video_url: null,
        followers: 142000, likes: 5300, num_comments: 410,
        comments: [
            'Great review as always!',
            'Very helpful, subscribed!',
            'Accurate and unbiased 👏',
        ],
        timestamp: '2026-03-24 00:00',
        analysis: null,
    },
];

export default function FeedScreen({ navigation }) {
    const [reels, setReels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);

    const fetchReels = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/reels`, { signal: AbortSignal.timeout(5000) });
            const data = res.ok ? await res.json() : [];
            setReels([...data, ...SEED_REELS]);
        } catch {
            setReels([...SEED_REELS]);
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    }, []);

    useEffect(() => {
        fetchReels();
        // refresh when coming back from AddReel screen
        const unsub = navigation.addListener('focus', fetchReels);
        return unsub;
    }, [navigation]);

    const onRefresh = () => { setRefreshing(true); fetchReels(); };

    const handleAnalyze = async (reel) => {
        // Reels uploaded via AddReel already have analysis — this is for demo reels
        if (!reel.video_url) return;
    };

    if (loading) {
        return (
            <SafeAreaView style={styles.center}>
                <ActivityIndicator color={COLORS.accent} size="large" />
                <Text style={styles.loadingText}>Loading feed…</Text>
            </SafeAreaView>
        );
    }

    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

            {/* ── Top Bar ── */}
            <View style={styles.topbar}>
                <View style={styles.logo}>
                    <Text style={styles.logoIcon}>🛡</Text>
                    <Text style={styles.logoText}>TrueFluence</Text>
                </View>
                <TouchableOpacity
                    style={styles.addBtn}
                    onPress={() => navigation.navigate('AddReel')}
                    activeOpacity={0.8}
                >
                    <Text style={styles.addBtnText}>+ Add Reel</Text>
                </TouchableOpacity>
            </View>

            {/* ── Stories bar header ── */}
            <View style={styles.statsBar}>
                <StatChip label="Feed" value={`${reels.length} reels`} />
                <StatChip label="Analyzed" value={`${reels.filter(r => r.analysis).length}`} color={COLORS.green} />
                <StatChip label="Deepfakes" value={`${reels.filter(r => r.analysis?.is_deepfake).length}`} color={COLORS.red} />
            </View>

            {/* ── Reel List ── */}
            <FlatList
                data={reels}
                keyExtractor={r => r.id}
                renderItem={({ item }) => (
                    <ReelCard reel={item} onAnalyze={handleAnalyze} />
                )}
                contentContainerStyle={{ paddingTop: 12, paddingBottom: 40 }}
                refreshControl={
                    <RefreshControl
                        refreshing={refreshing}
                        onRefresh={onRefresh}
                        tintColor={COLORS.accent}
                    />
                }
                ListEmptyComponent={
                    <View style={styles.center}>
                        <Text style={{ fontSize: 40 }}>📭</Text>
                        <Text style={styles.loadingText}>No reels yet</Text>
                    </View>
                }
            />
        </SafeAreaView>
    );
}

function StatChip({ label, value, color }) {
    return (
        <View style={styles.statChip}>
            <Text style={[styles.statValue, color && { color }]}>{value}</Text>
            <Text style={styles.statLabel}>{label}</Text>
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: COLORS.bg },
    center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: COLORS.bg },
    loadingText: { color: COLORS.muted, marginTop: 14, fontSize: 14 },
    topbar: {
        flexDirection: 'row', alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16, paddingVertical: 12,
        borderBottomWidth: 1, borderBottomColor: COLORS.border,
    },
    logo: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    logoIcon: { fontSize: 22 },
    logoText: { color: COLORS.accent2, fontWeight: '900', fontSize: 20, letterSpacing: -0.5 },
    addBtn: {
        backgroundColor: COLORS.accent, borderRadius: 999,
        paddingHorizontal: 16, paddingVertical: 8,
    },
    addBtnText: { color: '#fff', fontWeight: '700', fontSize: 13 },
    statsBar: {
        flexDirection: 'row', paddingHorizontal: 16,
        paddingVertical: 10, gap: 12,
        borderBottomWidth: 1, borderBottomColor: COLORS.border,
    },
    statChip: {
        backgroundColor: COLORS.surface, borderRadius: 10,
        paddingHorizontal: 12, paddingVertical: 8, alignItems: 'center',
        borderWidth: 1, borderColor: COLORS.border,
    },
    statValue: { color: COLORS.text, fontWeight: '800', fontSize: 16 },
    statLabel: { color: COLORS.muted, fontSize: 10, marginTop: 2 },
});
