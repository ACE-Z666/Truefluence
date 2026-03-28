import React, { useState } from 'react';
import {
    View, Text, TextInput, StyleSheet, ScrollView,
    TouchableOpacity, Alert, ActivityIndicator,
    SafeAreaView, StatusBar, KeyboardAvoidingView,
    Platform, Image,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { COLORS, API_BASE } from '../constants';

const COMMENT_COUNT = 10;

export default function AddReelScreen({ navigation }) {
    const [video, setVideo] = useState(null);       // { uri, filename }
    const [username, setUsername] = useState('');
    const [caption, setCaption] = useState('');
    const [followers, setFollowers] = useState('');
    const [likes, setLikes] = useState('');
    const [numComments, setNumComments] = useState('');
    const [comments, setComments] = useState(Array(COMMENT_COUNT).fill(''));
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState('');

    // ── Pick video from gallery ────────────────────────────────────────────────
    const pickVideo = async () => {
        const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (!perm.granted) {
            Alert.alert('Permission required', 'Allow access to your media library.');
            return;
        }
        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Videos,
            allowsEditing: false,
            quality: 1,
        });
        if (!result.canceled && result.assets?.[0]) {
            const asset = result.assets[0];
            setVideo({ uri: asset.uri, filename: asset.fileName || 'video.mp4' });
        }
    };

    // ── Record video with camera ───────────────────────────────────────────────
    const recordVideo = async () => {
        const perm = await ImagePicker.requestCameraPermissionsAsync();
        if (!perm.granted) {
            Alert.alert('Permission required', 'Allow camera access.');
            return;
        }
        const result = await ImagePicker.launchCameraAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Videos,
            videoMaxDuration: 120,
            quality: 0.8,
        });
        if (!result.canceled && result.assets?.[0]) {
            const asset = result.assets[0];
            setVideo({ uri: asset.uri, filename: asset.fileName || 'recorded.mp4' });
        }
    };

    const setComment = (i, val) => {
        const next = [...comments];
        next[i] = val;
        setComments(next);
    };

    // ── Submit ─────────────────────────────────────────────────────────────────
    const handleSubmit = async () => {
        if (!video) { Alert.alert('No video', 'Please select or record a video first.'); return; }
        if (!username.trim()) { Alert.alert('Missing field', 'Please enter a username.'); return; }

        setUploading(true);
        setProgress('Preparing upload…');

        try {
            const activeComments = comments.filter(c => c.trim());

            const formData = new FormData();
            formData.append('video', {
                uri: video.uri,
                name: video.filename,
                type: 'video/mp4',
            });
            formData.append('username', username.trim() || 'anonymous');
            formData.append('caption', caption.trim());
            formData.append('followers', followers || '0');
            formData.append('likes', likes || '0');
            formData.append('num_comments', numComments || '0');
            formData.append('comments', JSON.stringify(activeComments));

            setProgress('Uploading to TrueFluence…');

            const res = await fetch(`${API_BASE}/api/analyze`, {
                method: 'POST',
                body: formData,
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            if (!res.ok) {
                const err = await res.text();
                throw new Error(err || 'Server error');
            }

            const reel = await res.json();
            setProgress('Done!');

            Alert.alert(
                `${reel.analysis.verdict_emoji} Analysis Complete`,
                `Verdict: ${reel.analysis.verdict}\nFinal Score: ${Math.round(reel.analysis.final_score * 100)}%`,
                [{ text: 'View Feed', onPress: () => navigation.navigate('Feed') }]
            );

            // Reset form
            setVideo(null);
            setUsername('');
            setCaption('');
            setFollowers('');
            setLikes('');
            setNumComments('');
            setComments(Array(COMMENT_COUNT).fill(''));

        } catch (e) {
            Alert.alert('Error', e.message || 'Failed to upload. Make sure the backend is running.');
        } finally {
            setUploading(false);
            setProgress('');
        }
    };

    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

            {/* ── Top bar ── */}
            <View style={styles.topbar}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
                    <Text style={styles.backBtnText}>‹ Back</Text>
                </TouchableOpacity>
                <Text style={styles.topTitle}>Add New Reel</Text>
                <View style={{ width: 60 }} />
            </View>

            <KeyboardAvoidingView
                style={{ flex: 1 }}
                behavior={Platform.OS === 'ios' ? 'padding' : undefined}
            >
                <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">

                    {/* ── Video picker ── */}
                    <SectionLabel>🎬 Video</SectionLabel>
                    <View style={styles.videoPickerRow}>
                        <PickerBtn icon="📁" label="Gallery" onPress={pickVideo} />
                        <PickerBtn icon="📷" label="Camera" onPress={recordVideo} />
                    </View>

                    {video && (
                        <View style={styles.videoPreview}>
                            <Text style={styles.videoName}>✅ {video.filename}</Text>
                            <TouchableOpacity onPress={() => setVideo(null)}>
                                <Text style={{ color: COLORS.red, fontSize: 13 }}>Remove</Text>
                            </TouchableOpacity>
                        </View>
                    )}

                    {/* ── Creator info ── */}
                    <SectionLabel style={styles.sectionGap}>👤 Creator Info</SectionLabel>
                    <Field
                        placeholder="@username"
                        value={username}
                        onChangeText={setUsername}
                        autoCapitalize="none"
                    />
                    <Field
                        placeholder="Caption…"
                        value={caption}
                        onChangeText={setCaption}
                        multiline
                        style={{ minHeight: 70 }}
                    />

                    {/* ── Engagement data ── */}
                    <SectionLabel style={styles.sectionGap}>📊 Engagement Data</SectionLabel>
                    <View style={styles.row3}>
                        <Field
                            placeholder="Followers"
                            value={followers}
                            onChangeText={setFollowers}
                            keyboardType="numeric"
                            style={{ flex: 1 }}
                            label="Followers"
                        />
                        <Field
                            placeholder="Likes"
                            value={likes}
                            onChangeText={setLikes}
                            keyboardType="numeric"
                            style={{ flex: 1 }}
                            label="Video Likes"
                        />
                        <Field
                            placeholder="# Comments"
                            value={numComments}
                            onChangeText={setNumComments}
                            keyboardType="numeric"
                            style={{ flex: 1 }}
                            label="Total Comments"
                        />
                    </View>

                    {/* ── Comments ── */}
                    <SectionLabel style={styles.sectionGap}>💬 Comments (up to 10)</SectionLabel>
                    {comments.map((c, i) => (
                        <Field
                            key={i}
                            placeholder={`Comment ${i + 1}…`}
                            value={c}
                            onChangeText={v => setComment(i, v)}
                        />
                    ))}

                    {/* ── Submit ── */}
                    <TouchableOpacity
                        style={[styles.submitBtn, uploading && styles.submitBtnDisabled]}
                        onPress={handleSubmit}
                        disabled={uploading}
                        activeOpacity={0.85}
                    >
                        {uploading
                            ? <><ActivityIndicator color="#fff" size="small" />
                                <Text style={styles.submitBtnText}> {progress}</Text></>
                            : <Text style={styles.submitBtnText}>🛡 Upload & Analyze</Text>
                        }
                    </TouchableOpacity>

                    <Text style={styles.hint}>
                        💡 The video will be saved on your system and analyzed by the TrueFluence pipeline.
                    </Text>

                </ScrollView>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
}

// ── Small helpers ──
function SectionLabel({ children, style }) {
    return <Text style={[styles.sectionLabel, style]}>{children}</Text>;
}

function Field({ label, style, ...props }) {
    return (
        <View style={[styles.fieldWrap, style]}>
            {label && <Text style={styles.fieldLabel}>{label}</Text>}
            <TextInput
                style={styles.input}
                placeholderTextColor={COLORS.muted}
                {...props}
            />
        </View>
    );
}

function PickerBtn({ icon, label, onPress }) {
    return (
        <TouchableOpacity style={styles.pickerBtn} onPress={onPress} activeOpacity={0.8}>
            <Text style={{ fontSize: 28 }}>{icon}</Text>
            <Text style={styles.pickerLabel}>{label}</Text>
        </TouchableOpacity>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: COLORS.bg },
    topbar: {
        flexDirection: 'row', alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16, paddingVertical: 14,
        borderBottomWidth: 1, borderBottomColor: COLORS.border,
    },
    backBtn: { width: 60 },
    backBtnText: { color: COLORS.accent2, fontSize: 16, fontWeight: '600' },
    topTitle: { color: COLORS.text, fontWeight: '800', fontSize: 17 },
    content: { padding: 16, paddingBottom: 60 },
    sectionLabel: {
        color: COLORS.muted, fontWeight: '700', fontSize: 12,
        textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10
    },
    sectionGap: { marginTop: 24 },
    videoPickerRow: { flexDirection: 'row', gap: 12 },
    pickerBtn: {
        flex: 1, backgroundColor: COLORS.surface,
        borderWidth: 2, borderStyle: 'dashed', borderColor: COLORS.border,
        borderRadius: 14, alignItems: 'center', justifyContent: 'center',
        paddingVertical: 22, gap: 8,
    },
    pickerLabel: { color: COLORS.muted, fontWeight: '600', fontSize: 14 },
    videoPreview: {
        flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
        backgroundColor: COLORS.surface, borderRadius: 10,
        padding: 12, marginTop: 10,
        borderWidth: 1, borderColor: COLORS.border,
    },
    videoName: { color: COLORS.accent2, fontWeight: '600', fontSize: 13, flex: 1 },
    row3: { flexDirection: 'row', gap: 8 },
    fieldWrap: { marginBottom: 10 },
    fieldLabel: {
        color: COLORS.muted, fontSize: 10, fontWeight: '600',
        textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4
    },
    input: {
        backgroundColor: COLORS.surface, borderWidth: 1, borderColor: COLORS.border,
        borderRadius: 10, padding: 12, color: COLORS.text,
        fontSize: 14, fontFamily: Platform.OS === 'ios' ? 'System' : undefined,
    },
    submitBtn: {
        backgroundColor: COLORS.accent, borderRadius: 14,
        padding: 16, alignItems: 'center', justifyContent: 'center',
        flexDirection: 'row', marginTop: 28,
    },
    submitBtnDisabled: { opacity: 0.6 },
    submitBtnText: { color: '#fff', fontWeight: '800', fontSize: 16 },
    hint: { color: COLORS.muted, fontSize: 12, textAlign: 'center', marginTop: 14, lineHeight: 18 },
});
