import streamlit as st
import numpy as np
from streamlit_mic_recorder import mic_recorder
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cdist
import os
from io import BytesIO
import tempfile

# Configure page
st.set_page_config(page_title="Speaker Recognition System", layout="wide")
st.title("🎤 Speaker Recognition & Enrollment System")

# Initialize pyannote model
@st.cache_resource
def load_model():
    model = Model.from_pretrained("pyannote/embedding", 
                                )
    inference = Inference(model, window="whole")
    return inference

inference = load_model()

# Initialize session state for storing embeddings
if 'speaker_embeddings' not in st.session_state:
    st.session_state.speaker_embeddings = {}

def save_audio_and_get_embedding(audio_data, filename):
    """Save audio to temp file and extract embedding"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_data)
        tmp_file_path = tmp_file.name
    
    try:
        # Get embedding from temporary file
        embedding = inference(tmp_file_path)
        return embedding
    finally:
        # Clean up temp file
        os.unlink(tmp_file_path)

def compute_similarity(query_embedding, speaker_embeddings):
    """Compute cosine similarity between query and enrolled speakers"""
    if not speaker_embeddings:
        return None, None
    
    similarities = []
    speaker_names = []
    
    for speaker_name, speaker_embedding in speaker_embeddings.items():
        # Reshape embeddings for distance calculation
        query_2d = query_embedding.reshape(1, -1)
        speaker_2d = speaker_embedding.reshape(1, -1)
        
        # Calculate cosine distance (0 = identical, 1 = completely different)
        distance = cdist(query_2d, speaker_2d, metric="cosine")[0,0]
        similarity = 1 - distance  # Convert to similarity score
        
        similarities.append(similarity)
        speaker_names.append(speaker_name)
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_speaker = speaker_names[best_idx]
    best_similarity = similarities[best_idx]
    
    return best_speaker, best_similarity, dict(zip(speaker_names, similarities))

# Create two columns for enrollment and recognition
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Speaker Enrollment")
    st.write("Record audio samples for up to 3 speakers:")
    
    # Speaker enrollment section
    for i in range(1, 4):
        st.subheader(f"Speaker {i}")
        
        # Record audio for speaker
        audio = mic_recorder(
            start_prompt=f"🔴 Record Speaker {i}",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            use_container_width=True,
            format="wav",
            key=f"recorder_speaker_{i}"
        )
        
        if audio:
            # Play back the recorded audio
            st.audio(audio['bytes'])
            
            try:
                # Extract embedding and store
                embedding = save_audio_and_get_embedding(audio['bytes'], f"speaker_{i}.wav")
                st.session_state.speaker_embeddings[f"Speaker {i}"] = embedding
                st.success(f"✅ Speaker {i} enrolled successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing audio for Speaker {i}: {str(e)}")
        
        # Show enrollment status
        if f"Speaker {i}" in st.session_state.speaker_embeddings:
            st.info(f"✓ Speaker {i} is enrolled")
        
        st.divider()

with col2:
    st.header("🔍 Speaker Recognition")
    st.write("Record a query audio to identify the speaker:")
    
    # Show enrolled speakers count
    enrolled_count = len(st.session_state.speaker_embeddings)
    st.metric("Enrolled Speakers", enrolled_count)
    
    if enrolled_count == 0:
        st.warning("⚠️ Please enroll at least one speaker first!")
    else:
        # Query audio recording
        query_audio = mic_recorder(
            start_prompt="🎙️ Record Query Audio",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            use_container_width=True,
            format="wav",
            key="query_recorder"
        )
        
        if query_audio:
            st.audio(query_audio['bytes'])
            
            try:
                # Extract embedding from query audio
                query_embedding = save_audio_and_get_embedding(query_audio['bytes'], "query.wav")
                
                # Find best matching speaker
                best_speaker, best_similarity, all_similarities = compute_similarity(
                    query_embedding, st.session_state.speaker_embeddings
                )
                
                if best_speaker:
                    st.subheader("🎯 Recognition Results")
                    
                    # Display best match prominently
                    similarity_percentage = best_similarity * 100
                    st.metric(
                        "Best Match", 
                        best_speaker, 
                        f"{similarity_percentage:.1f}% similarity"
                    )
                    
                    # Show confidence level
                    if similarity_percentage > 80:
                        st.success("🟢 High Confidence Match")
                    elif similarity_percentage > 60:
                        st.warning("🟡 Medium Confidence Match")
                    else:
                        st.error("🔴 Low Confidence Match")
                    
                    # Show all similarity scores
                    st.subheader("📊 All Similarity Scores")
                    for speaker, similarity in all_similarities.items():
                        similarity_pct = similarity * 100
                        st.progress(similarity, text=f"{speaker}: {similarity_pct:.1f}%")
                
            except Exception as e:
                st.error(f"❌ Error processing query audio: {str(e)}")

# Sidebar with additional controls
with st.sidebar:
    st.header("🛠️ Controls")
    
    # Clear all enrollments
    if st.button("🗑️ Clear All Enrollments", type="secondary"):
        st.session_state.speaker_embeddings = {}
        st.rerun()
    
    # Show enrollment status
    st.subheader("📋 Enrollment Status")
    if st.session_state.speaker_embeddings:
        for speaker in st.session_state.speaker_embeddings.keys():
            st.write(f"✅ {speaker}")
    else:
        st.write("No speakers enrolled yet")
    
    # Instructions
    st.subheader("📖 Instructions")
    st.write("""
    1. **Enroll Speakers**: Record audio samples for up to 3 speakers
    2. **Record Query**: Record new audio to identify the speaker
    3. **View Results**: See similarity scores and best match
    
    **Tips:**
    - Record clear audio for better accuracy
    - Each enrollment should be 3-5 seconds long
    - Avoid background noise when possible
    """)
