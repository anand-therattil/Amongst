

# Let me create speech segment extraction functions using libraries commonly available with pyannote
import os
import shutil
import librosa
import soundfile as sf
import numpy as np

def extract_speech_segments(audio_path, diarization_output, output_dir="speech_segments", sr=16000):
    """
    Extract speech segments from pyannote diarization output
    
    Parameters:
    -----------
    audio_path: str
        Path to the input audio file (your WAV file)
    diarization_output: Annotation
        Pyannote diarization output from pipeline
    output_dir: str
        Directory to save the extracted segments
    sr: int
        Sample rate for output files (default: 16000)
    
    Returns:
    --------
    List of dictionaries with segment information
    """
    # Clear and create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original audio file
    print(f"Loading audio file: {audio_path}")
    audio, original_sr = librosa.load(audio_path, sr=sr)
    print(f"Audio loaded: {len(audio)} samples at {sr} Hz")
    
    segments_info = []
    
    # Extract each segment
    for i, (segment, _, speaker) in enumerate(diarization_output.itertracks(yield_label=True)):
        # Convert time to sample indices
        start_sample = int(segment.start * sr)
        end_sample = int(segment.end * sr)
        
        # Extract the audio segment
        segment_audio = audio[start_sample:end_sample]
        
        # Create filename for this segment
        filename = os.path.basename(audio_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_segment_{i+1:04d}_{segment.start:.3f}s-{segment.end:.3f}s_speaker_{speaker}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the segment
        sf.write(output_path, segment_audio, sr)
        
        # Store segment information
        segment_info = {
            'segment_id': i+1,
            'speaker': speaker,
            'start_time': segment.start,
            'end_time': segment.end,
            'duration': segment.end - segment.start,
            'file_path': output_path,
            'filename': output_filename
        }
        segments_info.append(segment_info)
        
        print(f"Segment {i+1:3d}: {speaker} | {segment.start:7.3f}s - {segment.end:7.3f}s | Duration: {segment_info['duration']:6.3f}s | Saved: {output_filename}")
    
    return segments_info

def get_diarization_summary(diarization_output):
    """
    Print a summary of the diarization results
    
    Parameters:
    -----------
    diarization_output: Annotation
        Pyannote diarization output
    """
    segments = []
    speakers = set()
    
    for segment, _, speaker in diarization_output.itertracks(yield_label=True):
        segments.append({
            'speaker': speaker,
            'start': segment.start,
            'end': segment.end,
            'duration': segment.end - segment.start
        })
        speakers.add(speaker)
    
    # Calculate statistics
    total_duration = sum(seg['duration'] for seg in segments)
    speaker_stats = {}
    
    for speaker in speakers:
        speaker_segments = [seg for seg in segments if seg['speaker'] == speaker]
        speaker_duration = sum(seg['duration'] for seg in speaker_segments)
        speaker_stats[speaker] = {
            'segment_count': len(speaker_segments),
            'total_duration': speaker_duration,
            'percentage': (speaker_duration / total_duration) * 100 if total_duration > 0 else 0
        }
    
    print("=== DIARIZATION SUMMARY ===")
    print(f"Total segments: {len(segments)}")
    print(f"Total duration: {total_duration:.3f} seconds")
    print(f"Number of speakers: {len(speakers)}")
    print("\nSpeaker breakdown:")
    for speaker, stats in sorted(speaker_stats.items()):
        print(f"  {speaker}: {stats['segment_count']:2d} segments | {stats['total_duration']:7.3f}s | {stats['percentage']:5.1f}%")
    
    return segments, speaker_stats

def extract_by_speaker(audio_path, diarization_output, target_speaker, output_dir="single_speaker"):
    """
    Extract all segments for a specific speaker
    
    Parameters:
    -----------
    audio_path: str
        Path to input audio file
    diarization_output: Annotation
        Pyannote diarization output
    target_speaker: str
        Speaker to extract (e.g., 'SPEAKER_00')
    output_dir: str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Find all segments for target speaker
    target_segments = []
    for segment, _, speaker in diarization_output.itertracks(yield_label=True):
        if speaker == target_speaker:
            target_segments.append(segment)
    
    print(f"Found {len(target_segments)} segments for {target_speaker}")
    
    # Extract and save each segment
    for i, segment in enumerate(target_segments):
        start_sample = int(segment.start * sr)
        end_sample = int(segment.end * sr)
        segment_audio = audio[start_sample:end_sample]
        
        filename = os.path.basename(audio_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_{target_speaker}_part_{i+1:03d}.wav")
        
        sf.write(output_path, segment_audio, sr)
        print(f"  Part {i+1:3d}: {segment.start:7.3f}s - {segment.end:7.3f}s -> {output_path}")
    
    return target_segments

def create_combined_speaker_audio(audio_path, diarization_output, target_speaker, output_path):
    """
    Combine all segments from one speaker into a single audio file
    
    Parameters:
    -----------
    audio_path: str
        Path to input audio file
    diarization_output: Annotation
        Pyannote diarization output
    target_speaker: str
        Speaker to combine
    output_path: str
        Path for combined audio file
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Collect all segments for the speaker
    speaker_audio_segments = []
    for segment, _, speaker in diarization_output.itertracks(yield_label=True):
        if speaker == target_speaker:
            start_sample = int(segment.start * sr)
            end_sample = int(segment.end * sr)
            speaker_audio_segments.append(audio[start_sample:end_sample])
    
    if speaker_audio_segments:
        # Concatenate all segments
        combined_audio = np.concatenate(speaker_audio_segments)
        sf.write(output_path, combined_audio, sr)
        print(f"Combined {len(speaker_audio_segments)} segments for {target_speaker}")
        print(f"Total duration: {len(combined_audio) / sr:.3f} seconds")
        print(f"Saved to: {output_path}")
        return len(combined_audio) / sr
    else:
        print(f"No segments found for {target_speaker}")
        return 0

print("Speech segment extraction functions created!")
print("\nMain functions:")
print("1. extract_speech_segments(audio_path, diarization_output) - Extract all segments")  
print("2. get_diarization_summary(diarization_output) - Show summary stats")
print("3. extract_by_speaker(audio_path, diarization_output, speaker) - Extract one speaker")
print("4. create_combined_speaker_audio(audio_path, diarization_output, speaker, output_path) - Combine speaker segments")


# instantiate the pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
  "collinbarnwell/pyannote-speaker-diarization-31",
  )

# run the pipeline on an audio file
diarization = pipeline("/Users/cmi_10128/Downloads/test.wav")
print(diarization)
# dump the diarization output to disk using RTTM format
# with open("audio.rttm", "w") as rttm:
#     diarization.write_rttm(rttm)
# extract speech segments
segments = extract_speech_segments(
    audio_path="/Users/cmi_10128/Desktop/documents/projects/audio_processing/data/wav/LJ001-0012.wav",
    diarization_output=diarization,
    output_dir="speech_segments",
    sr=16000
)