import torch
import numpy as np
from scipy.spatial.distance import cosine
from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

class SpeakerIdentifier:
    def __init__(self,):
        """
        Initialize the speaker identifier with pyannote models.
        """
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the embedding model
        print("Loading embedding model...")
        self.embedding_model = Model.from_pretrained(
            "pyannote/embedding", 

        )
        self.inference = Inference(
            self.embedding_model, 
            window="whole",
            device=self.device
        )
        
        # Load the diarization pipeline
        print("Loading diarization pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "collinbarnwell/pyannote-speaker-diarization-31",
        
        )
        self.diarization_pipeline.to(self.device)
        
        self.reference_embeddings = {}
        self.speaker_names = []
        
    def extract_reference_embeddings(self, speaker_files: Dict[str, str]):
        """
        Extract embeddings from reference speaker audio files.
        
        Args:
            speaker_files: Dictionary mapping speaker names to audio file paths
        """
        print("Extracting reference embeddings...")
        
        for speaker_name, audio_path in speaker_files.items():
            print(f"Processing {speaker_name}: {audio_path}")
            
            try:
                # Extract embedding for the entire reference audio
                embedding = self.inference(audio_path)
                
                # Debug: Print embedding shape
                print(f"  Raw embedding shape for {speaker_name}: {embedding.shape}")
                
                # Handle different embedding formats
                if isinstance(embedding, np.ndarray):
                    if embedding.ndim == 2:
                        # If 2D, take the mean across time dimension
                        embedding = np.mean(embedding, axis=0)
                    elif embedding.ndim == 1:
                        # Already 1D, use as is
                        pass
                    else:
                        raise ValueError(f"Unexpected embedding dimensions: {embedding.shape}")
                else:
                    # Convert to numpy if it's a tensor
                    embedding = np.array(embedding)
                    if embedding.ndim == 2:
                        embedding = np.mean(embedding, axis=0)
                
                print(f"  Final embedding shape for {speaker_name}: {embedding.shape}")
                
                self.reference_embeddings[speaker_name] = embedding
                self.speaker_names.append(speaker_name)
                
            except Exception as e:
                print(f"Error processing {speaker_name}: {str(e)}")
                continue
                
        print(f"Successfully loaded {len(self.reference_embeddings)} reference speakers")
    
    def identify_speakers_in_audio(self, combined_audio_path: str, min_segment_duration: float = 1.0):
        """
        Perform diarization on combined audio and match segments to reference speakers.
        
        Args:
            combined_audio_path: Path to the combined audio file
            min_segment_duration: Minimum duration (seconds) for a segment to be considered
            
        Returns:
            List of dictionaries containing segment information
        """
        print(f"Processing combined audio: {combined_audio_path}")
        
        # Perform diarization
        print("Running speaker diarization...")
        try:
            # Try to get embeddings with diarization
            diarization_result = self.diarization_pipeline(combined_audio_path)
            
            # Extract embeddings for each segment manually
            results = []
            
            for segment, _, speaker_label in diarization_result.itertracks(yield_label=True):
                # Skip very short segments
                if segment.duration < min_segment_duration:
                    continue
                    
                start_time = segment.start
                end_time = segment.end
                duration = segment.duration
                
                # Extract embedding for this specific segment
                try:
                    segment_embedding = self.inference({'audio': combined_audio_path, 'segment': segment})
                    
                    # Debug: Print segment embedding shape
                    print(f"Segment {start_time:.2f}-{end_time:.2f}s embedding shape: {segment_embedding.shape}")
                    
                    # Handle segment embedding format
                    if isinstance(segment_embedding, np.ndarray):
                        if segment_embedding.ndim == 2:
                            segment_embedding = np.mean(segment_embedding, axis=0)
                    else:
                        segment_embedding = np.array(segment_embedding)
                        if segment_embedding.ndim == 2:
                            segment_embedding = np.mean(segment_embedding, axis=0)
                    
                    # Find the best matching reference speaker
                    best_match = self._find_best_speaker_match(segment_embedding)
                    
                    results.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'diarization_label': speaker_label,
                        'identified_speaker': best_match['speaker'],
                        'similarity_score': best_match['similarity'],
                        'confidence': best_match['confidence']
                    })
                    
                except Exception as e:
                    print(f"Error processing segment {start_time:.2f}-{end_time:.2f}s: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"Error in diarization: {str(e)}")
            return []
    
    def _find_best_speaker_match(self, segment_embedding: np.ndarray) -> Dict:
        """
        Find the best matching reference speaker for a given segment embedding.
        
        Args:
            segment_embedding: Embedding vector for the audio segment
            
        Returns:
            Dictionary with best match information
        """
        similarities = {}
        
        # Ensure segment embedding is 1D
        if segment_embedding.ndim > 1:
            segment_embedding = segment_embedding.flatten()
        
        # Calculate cosine similarity with each reference speaker
        for speaker_name, ref_embedding in self.reference_embeddings.items():
            try:
                # Ensure reference embedding is 1D
                if ref_embedding.ndim > 1:
                    ref_embedding = ref_embedding.flatten()
                
                # Check if embeddings have compatible dimensions
                if segment_embedding.shape[0] != ref_embedding.shape[0]:
                    print(f"Warning: Dimension mismatch for {speaker_name}: "
                          f"segment={segment_embedding.shape}, ref={ref_embedding.shape}")
                    
                    # Try to handle dimension mismatch
                    min_dim = min(segment_embedding.shape[0], ref_embedding.shape[0])
                    segment_emb_truncated = segment_embedding[:min_dim]
                    ref_emb_truncated = ref_embedding[:min_dim]
                    
                    similarity = 1 - cosine(segment_emb_truncated, ref_emb_truncated)
                else:
                    # Calculate cosine similarity (1 - cosine distance)
                    similarity = 1 - cosine(segment_embedding, ref_embedding)
                
                similarities[speaker_name] = similarity
                
            except Exception as e:
                print(f"Error calculating similarity with {speaker_name}: {str(e)}")
                similarities[speaker_name] = 0.0
        
        if not similarities:
            return {
                'speaker': 'Unknown',
                'similarity': 0.0,
                'confidence': 'Low',
                'all_similarities': {}
            }
        
        # Find the speaker with highest similarity
        best_speaker = max(similarities, key=similarities.get)
        best_similarity = similarities[best_speaker]
        
        # Calculate confidence (normalized similarity)
        confidence = "High" if best_similarity > 0.8 else "Medium" if best_similarity > 0.6 else "Low"
        
        return {
            'speaker': best_speaker,
            'similarity': best_similarity,
            'confidence': confidence,
            'all_similarities': similarities
        }
    
    def generate_report(self, results: List[Dict]) -> pd.DataFrame:
        """
        Generate a detailed report of the speaker identification results.
        
        Args:
            results: List of identification results
            
        Returns:
            Pandas DataFrame with formatted results
        """
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        
        # Format time columns
        df['start_time_formatted'] = df['start_time'].apply(lambda x: f"{int(x//60):02d}:{x%60:06.3f}")
        df['end_time_formatted'] = df['end_time'].apply(lambda x: f"{int(x//60):02d}:{x%60:06.3f}")
        df['duration_formatted'] = df['duration'].apply(lambda x: f"{x:.3f}s")
        
        # Round similarity scores
        df['similarity_score'] = df['similarity_score'].round(4)
        
        # Reorder columns
        df = df[['start_time_formatted', 'end_time_formatted', 'duration_formatted', 
                'identified_speaker', 'similarity_score', 'confidence', 'diarization_label']]
        
        df.columns = ['Start Time', 'End Time', 'Duration', 'Speaker', 'Similarity', 'Confidence', 'Diarization Label']
        
        return df
    
    def print_summary(self, results: List[Dict]):
        """Print a summary of the speaker identification results."""
        if not results:
            print("No segments identified.")
            return
        
        df = pd.DataFrame(results)
        
        print(f"\n{'='*60}")
        print("SPEAKER IDENTIFICATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"Total segments analyzed: {len(results)}")
        print(f"Total audio duration: {df['duration'].sum():.2f} seconds")
        
        print(f"\nSpeaker distribution:")
        speaker_counts = df['identified_speaker'].value_counts()
        for speaker, count in speaker_counts.items():
            total_duration = df[df['identified_speaker'] == speaker]['duration'].sum()
            print(f"  {speaker}: {count} segments ({total_duration:.2f}s total)")
        
        print(f"\nConfidence distribution:")
        confidence_counts = df['confidence'].value_counts()
        for conf, count in confidence_counts.items():
            print(f"  {conf}: {count} segments")

# Usage example
def main():
    # Initialize with your HuggingFace token
   
    identifier = SpeakerIdentifier()
    
    # Define your reference speaker files
    reference_speakers = {
        "Speaker_A": "/Users/cmi_10128/Desktop/documents/projects/Amongst/audio_files/speakerA.wav",
        "Speaker_B": "/Users/cmi_10128/Desktop/documents/projects/Amongst/audio_files/speakerB.wav", 
        "Speaker_C": "/Users/cmi_10128/Desktop/documents/projects/Amongst/audio_files/speakerC.wav"
    }
    
    # Extract reference embeddings
    identifier.extract_reference_embeddings(reference_speakers)
    
    # Process the combined audio file
    combined_audio = "/Users/cmi_10128/Desktop/documents/projects/Amongst/audio_files/mix.wav"
    results = identifier.identify_speakers_in_audio(combined_audio)
    
    if results:
        # Generate and display report
        report_df = identifier.generate_report(results)
        print(report_df.to_string(index=False))
        
        # Print summary
        identifier.print_summary(results)
        
        # Save results to CSV
        report_df.to_csv("speaker_identification_results.csv", index=False)
        print(f"\nResults saved to 'speaker_identification_results.csv'")
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()
