import streamlit as st
import numpy as np
from streamlit_mic_recorder import mic_recorder
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cdist
import os
from io import BytesIO
import tempfile
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Configure page
st.set_page_config(page_title="Speaker Recognition System", layout="wide")
st.title("🎤 Speaker Recognition & Enrollment System")

# Initialize pyannote model
@st.cache_resource
def load_model():
    model = Model.from_pretrained("pyannote/embedding")
    inference = Inference(model, window="whole")
    return inference

inference = load_model()

# Initialize session state for storing embeddings
if 'speaker_embeddings' not in st.session_state:
    st.session_state.speaker_embeddings = {}
if 'query_embedding' not in st.session_state:
    st.session_state.query_embedding = None

def save_audio_and_get_embedding(audio_data, filename):
    """Save audio to temp file and extract embedding"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_data)
        tmp_file_path = tmp_file.name
    
    try:
        embedding = inference(tmp_file_path)
        return embedding
    finally:
        os.unlink(tmp_file_path)

def compute_similarity(query_embedding, speaker_embeddings):
    """Compute cosine similarity between query and enrolled speakers"""
    if not speaker_embeddings:
        return None, None
    
    similarities = []
    speaker_names = []
    
    for speaker_name, speaker_embedding in speaker_embeddings.items():
        query_2d = query_embedding.reshape(1, -1)
        speaker_2d = speaker_embedding.reshape(1, -1)
        distance = cdist(query_2d, speaker_2d, metric="cosine")[0,0]
        similarity = 1 - distance
        similarities.append(similarity)
        speaker_names.append(speaker_name)
    
    best_idx = np.argmax(similarities)
    best_speaker = speaker_names[best_idx]
    best_similarity = similarities[best_idx]
    
    return best_speaker, best_similarity, dict(zip(speaker_names, similarities))

def visualize_embeddings_with_query(embeddings_dict, query_embedding=None, method='t-SNE'):
    """Create 2D visualization of speaker embeddings with optional query point"""
    if len(embeddings_dict) < 1:
        st.warning("At least 1 speaker needed for visualization")
        return None
    
    # Prepare enrolled speaker data
    embeddings_list = []
    speaker_names = []
    for speaker, embedding in embeddings_dict.items():
        embeddings_list.append(embedding.flatten())
        speaker_names.append(speaker)
    
    embeddings_array = np.array(embeddings_list)
    
    # Add query embedding if provided
    if query_embedding is not None:
        query_vec = query_embedding.flatten()
        embeddings_array = np.vstack([embeddings_array, query_vec])
        all_names = speaker_names + ['Query Audio']
    else:
        all_names = speaker_names
    
    # Create numeric labels for coloring
    unique_speakers = list(embeddings_dict.keys())
    labels = [unique_speakers.index(name) for name in speaker_names]
    
    # Special label for query point
    if query_embedding is not None:
        labels.append(-1)  # Special label for query
    
    # Apply dimensionality reduction
    if method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_array)-1))
        embeddings_2d = reducer.fit_transform(embeddings_array)
    elif method == 'LDA':
        if len(unique_speakers) < 2:
            st.error("LDA requires at least 2 different speakers")
            return None
        # For LDA, we need to handle the query point separately
        if query_embedding is not None:
            # Fit LDA on enrolled speakers only
            reducer = LDA(n_components=min(2, len(unique_speakers)-1))
            enrolled_2d = reducer.fit_transform(embeddings_array[:-1], labels[:-1])
            # Transform the query point using the fitted LDA
            query_2d = reducer.transform(query_vec.reshape(1, -1))
            embeddings_2d = np.vstack([enrolled_2d, query_2d])
        else:
            reducer = LDA(n_components=min(2, len(unique_speakers)-1))
            embeddings_2d = reducer.fit_transform(embeddings_array, labels)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot enrolled speakers
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_speakers)))
    for i, speaker in enumerate(unique_speakers):
        speaker_idx = speaker_names.index(speaker)
        ax.scatter(embeddings_2d[speaker_idx, 0], embeddings_2d[speaker_idx, 1], 
                  c=[colors[i]], label=speaker, s=120, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add speaker name annotation
        ax.annotate(speaker, (embeddings_2d[speaker_idx, 0], embeddings_2d[speaker_idx, 1]), 
                   xytext=(8, 8), textcoords='offset points', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Plot query point if available
    if query_embedding is not None:
        query_idx = len(speaker_names)  # Query is the last point
        ax.scatter(embeddings_2d[query_idx, 0], embeddings_2d[query_idx, 1], 
                  c='red', label='Query Audio', s=200, marker='X', alpha=0.9, 
                  edgecolor='darkred', linewidth=2)
        
        # Add query annotation
        ax.annotate('Query Audio', (embeddings_2d[query_idx, 0], embeddings_2d[query_idx, 1]), 
                   xytext=(10, 10), textcoords='offset points', fontsize=12, 
                   color='red', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Draw lines from query to all enrolled speakers
        for i in range(len(speaker_names)):
            ax.plot([embeddings_2d[query_idx, 0], embeddings_2d[i, 0]], 
                   [embeddings_2d[query_idx, 1], embeddings_2d[i, 1]], 
                   'r--', alpha=0.3, linewidth=1)
    
    ax.set_title(f'{method} Visualization of Speaker Embeddings' + 
                (' with Query Audio' if query_embedding is not None else ''), fontsize=14)
    ax.set_xlabel(f'{method} Dimension 1')
    ax.set_ylabel(f'{method} Dimension 2')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    return fig

# Create three columns for enrollment, recognition, and visualization
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.header("📝 Speaker Enrollment")
    st.write("Record audio samples for up to 3 speakers:")
    
    # Speaker enrollment section
    for i in range(1, 4):
        st.subheader(f"Speaker {i}")
        
        audio = mic_recorder(
            start_prompt=f"🔴 Record Speaker {i}",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            use_container_width=True,
            format="wav",
            key=f"recorder_speaker_{i}"
        )
        
        if audio:
            st.audio(audio['bytes'])
            
            try:
                embedding = save_audio_and_get_embedding(audio['bytes'], f"speaker_{i}.wav")
                st.session_state.speaker_embeddings[f"Speaker {i}"] = embedding
                st.success(f"✅ Speaker {i} enrolled successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing audio for Speaker {i}: {str(e)}")
        
        if f"Speaker {i}" in st.session_state.speaker_embeddings:
            st.info(f"✓ Speaker {i} is enrolled")
        
        st.divider()

with col2:
    st.header("🔍 Speaker Recognition")
    st.write("Record a query audio to identify the speaker:")
    
    enrolled_count = len(st.session_state.speaker_embeddings)
    st.metric("Enrolled Speakers", enrolled_count)
    
    if enrolled_count == 0:
        st.warning("⚠️ Please enroll at least one speaker first!")
    else:
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
                query_embedding = save_audio_and_get_embedding(query_audio['bytes'], "query.wav")
                
                # Store query embedding in session state for visualization
                st.session_state.query_embedding = query_embedding
                
                best_speaker, best_similarity, all_similarities = compute_similarity(
                    query_embedding, st.session_state.speaker_embeddings
                )
                
                if best_speaker:
                    st.subheader("🎯 Recognition Results")
                    
                    similarity_percentage = best_similarity * 100
                    st.metric(
                        "Best Match", 
                        best_speaker, 
                        f"{similarity_percentage:.1f}% similarity"
                    )
                    
                    if similarity_percentage > 80:
                        st.success("🟢 High Confidence Match")
                    elif similarity_percentage > 60:
                        st.warning("🟡 Medium Confidence Match")
                    else:
                        st.error("🔴 Low Confidence Match")
                    
                    st.subheader("📊 All Similarity Scores")
                    for speaker, similarity in all_similarities.items():
                        similarity_pct = similarity * 100
                        st.progress(similarity, text=f"{speaker}: {similarity_pct:.1f}%")
                
            except Exception as e:
                st.error(f"❌ Error processing query audio: {str(e)}")

# Visualization Column
with col3:
    st.header("📈 Embedding Visualization")
    st.write("Visualize speaker embeddings in 2D:")
    
    enrolled_count = len(st.session_state.speaker_embeddings)
    
    if enrolled_count < 1:
        st.info("🔔 Enroll at least 1 speaker to see visualization")
    else:
        # Method selection
        viz_method = st.selectbox(
            "Choose visualization method:",
            ["t-SNE", "LDA"],
            help="t-SNE: Better for exploring clusters. LDA: Better for class separation."
        )
        
        # Show query status
        if st.session_state.query_embedding is not None:
            st.success("🎙️ Query audio available for visualization")
        else:
            st.info("🔔 Record a query audio to see it in the plot")
        
        # Visualization button
        if st.button("🎨 Generate Visualization", use_container_width=True):
            with st.spinner(f"Generating {viz_method} visualization..."):
                # Pass query embedding if available
                fig = visualize_embeddings_with_query(
                    st.session_state.speaker_embeddings, 
                    st.session_state.query_embedding, 
                    viz_method
                )
                if fig:
                    st.pyplot(fig)
                    
                    # Add interpretation
                    st.subheader("💡 Interpretation")
                    base_text = ""
                    if viz_method == "t-SNE":
                        base_text = """
                        **t-SNE Plot**: Points that are close together represent speakers with similar voice characteristics.
                        - Tight clusters suggest consistent voice features
                        - Distant points indicate distinct speakers
                        """
                    else:
                        base_text = """
                        **LDA Plot**: Shows the most discriminative dimensions for separating speakers.
                        - Maximizes separation between different speakers
                        - Better separation indicates more distinguishable voices
                        """
                    
                    if st.session_state.query_embedding is not None:
                        base_text += """
                        
                        **Query Point (Red X)**: Shows where your query audio falls in the embedding space.
                        - Closer to a speaker = higher similarity
                        - Dotted lines show distances to each enrolled speaker
                        """
                    
                    st.write(base_text)

# Sidebar with additional controls
with st.sidebar:
    st.header("🛠️ Controls")
    
    if st.button("🗑️ Clear All Enrollments", type="secondary"):
        st.session_state.speaker_embeddings = {}
        st.session_state.query_embedding = None
        st.rerun()
    
    if st.button("🔄 Clear Query Audio", type="secondary"):
        st.session_state.query_embedding = None
        st.success("Query audio cleared!")
    
    st.subheader("📋 Enrollment Status")
    if st.session_state.speaker_embeddings:
        for speaker in st.session_state.speaker_embeddings.keys():
            st.write(f"✅ {speaker}")
    else:
        st.write("No speakers enrolled yet")
    
    # Show query status
    st.subheader("🎙️ Query Status")
    if st.session_state.query_embedding is not None:
        st.write("✅ Query audio recorded")
    else:
        st.write("No query audio recorded")
    
    st.subheader("📖 Instructions")
    st.write("""
    1. **Enroll Speakers**: Record audio samples for up to 3 speakers
    2. **Record Query**: Record new audio to identify the speaker
    3. **View Results**: See similarity scores and best match
    4. **Visualize**: Generate 2D plots of speaker embeddings
    
    **Tips:**
    - Record clear audio for better accuracy
    - Each enrollment should be 3-5 seconds long
    - Avoid background noise when possible
    - Use visualization to understand speaker separation
    """)
    
    # Additional visualization info
    st.subheader("🎯 Visualization Guide")
    st.write("""
    **t-SNE**: Good for exploring natural clusters in speaker embeddings
    
    **LDA**: Shows optimal separation between enrolled speakers
    
    **Query Point**: Red X shows where your query audio falls in the space
    
    Both methods help understand how well your speakers can be distinguished.
    """)
