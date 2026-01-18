# Streamlit Review Dashboard
import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Some visualizations will be disabled.")


def load_events(json_path: str) -> List[Dict]:
    """Load events from JSON file"""
    if not os.path.exists(json_path):
        return []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data.get('events', [])


def load_checkpoints(checkpoint_dir: str) -> List[str]:
    """List available checkpoint files"""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('checkpoint_') and f.endswith('.json')
    ])
    return checkpoints


def render_event_summary(events: List[Dict]):
    """Render event summary statistics"""
    if not events:
        st.info("No events loaded")
        return
    
    # Count events by type
    event_counts = {}
    for event in events:
        event_type = event.get('type', 'unknown')
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Events", len(events))
    with col2:
        st.metric("Passes", event_counts.get('pass', 0))
    with col3:
        st.metric("Dribbles", event_counts.get('dribble', 0))
    with col4:
        st.metric("Shots", event_counts.get('shot', 0))
    with col5:
        st.metric("Recoveries", event_counts.get('recovery', 0))


def render_event_timeline(events: List[Dict]):
    """Render event timeline visualization"""
    if not events:
        return
    
    if not PLOTLY_AVAILABLE:
        st.info("Plotly not available for timeline visualization")
        return
    
    # Prepare data
    timeline_data = []
    for event in events:
        timeline_data.append({
            'frame': event.get('start_frame', 0),
            'type': event.get('type', 'unknown'),
            'confidence': event.get('confidence', 0.0)
        })
    
    df = pd.DataFrame(timeline_data)
    
    # Create timeline plot
    fig = px.scatter(
        df,
        x='frame',
        y='type',
        color='confidence',
        color_continuous_scale='Viridis',
        title='Event Timeline',
        labels={'frame': 'Frame Number', 'type': 'Event Type'}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_event_table(events: List[Dict]):
    """Render events in a table"""
    if not events:
        return
    
    # Convert to DataFrame
    table_data = []
    for event in events:
        table_data.append({
            'ID': event.get('id', ''),
            'Type': event.get('type', ''),
            'Start Frame': event.get('start_frame', 0),
            'End Frame': event.get('end_frame', 0),
            'Confidence': f"{event.get('confidence', 0.0):.2f}",
            'Players': len(event.get('involved_players', []))
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)


def render_pitch_map(events: List[Dict]):
    """Render events on pitch map"""
    if not events:
        return
    
    if not PLOTLY_AVAILABLE:
        st.info("Plotly not available for pitch map visualization")
        return
    
    # Extract locations
    pass_locs = []
    shot_locs = []
    dribble_locs = []
    
    for event in events:
        event_type = event.get('type', '')
        start_loc = event.get('start_location', {})
        end_loc = event.get('end_location', {})
        
        if event_type == 'pass':
            pass_locs.append({
                'x': [start_loc.get('x', 0), end_loc.get('x', 0)],
                'y': [start_loc.get('y', 0), end_loc.get('y', 0)]
            })
        elif event_type == 'shot':
            shot_locs.append({
                'x': end_loc.get('x', 0),
                'y': end_loc.get('y', 0)
            })
        elif event_type == 'dribble':
            dribble_locs.append({
                'x': end_loc.get('x', 0),
                'y': end_loc.get('y', 0)
            })
    
    # Create pitch visualization
    fig = go.Figure()
    
    # Draw pitch outline (105m x 68m)
    pitch_x = [-52.5, 52.5, 52.5, -52.5, -52.5]
    pitch_y = [-34, -34, 34, 34, -34]
    fig.add_trace(go.Scatter(
        x=pitch_x,
        y=pitch_y,
        mode='lines',
        name='Pitch',
        line=dict(color='green', width=2)
    ))
    
    # Draw passes
    for pass_loc in pass_locs:
        fig.add_trace(go.Scatter(
            x=pass_loc['x'],
            y=pass_loc['y'],
            mode='lines+markers',
            name='Pass',
            line=dict(color='blue', width=1),
            marker=dict(size=5)
        ))
    
    # Draw shots
    if shot_locs:
        shot_x = [loc['x'] for loc in shot_locs]
        shot_y = [loc['y'] for loc in shot_locs]
        fig.add_trace(go.Scatter(
            x=shot_x,
            y=shot_y,
            mode='markers',
            name='Shot',
            marker=dict(size=10, color='red', symbol='star')
        ))
    
    # Draw dribbles
    if dribble_locs:
        dribble_x = [loc['x'] for loc in dribble_locs]
        dribble_y = [loc['y'] for loc in dribble_locs]
        fig.add_trace(go.Scatter(
            x=dribble_x,
            y=dribble_y,
            mode='markers',
            name='Dribble',
            marker=dict(size=8, color='orange', symbol='circle')
        ))
    
    fig.update_layout(
        title='Event Map on Pitch',
        xaxis_title='X (meters)',
        yaxis_title='Y (meters)',
        showlegend=True,
        width=800,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit app"""
    st.set_page_config(page_title="Soccer Analysis Dashboard", layout="wide")
    st.title("Soccer Analysis Pipeline - Review Dashboard")
    
    # Sidebar for file selection
    st.sidebar.header("Data Selection")
    
    output_dir = st.sidebar.text_input("Output Directory", value="data/output")
    
    # Load events
    json_path = os.path.join(output_dir, "events.json")
    events = load_events(json_path)
    
    # Checkpoint selection
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    checkpoints = load_checkpoints(checkpoint_dir)
    
    if checkpoints:
        st.sidebar.subheader("Checkpoints")
        selected_checkpoint = st.sidebar.selectbox(
            "Select Checkpoint",
            options=checkpoints,
            index=len(checkpoints) - 1 if checkpoints else 0
        )
        
        if selected_checkpoint:
            checkpoint_path = os.path.join(checkpoint_dir, selected_checkpoint)
            checkpoint_events = load_events(checkpoint_path)
            if checkpoint_events:
                st.sidebar.info(f"Loaded {len(checkpoint_events)} events from checkpoint")
    
    # Main content
    if events:
        st.header("Event Summary")
        render_event_summary(events)
        
        st.header("Event Timeline")
        render_event_timeline(events)
        
        st.header("Pitch Map")
        render_pitch_map(events)
        
        st.header("Event Details")
        render_event_table(events)
    else:
        st.warning(f"No events found at {json_path}")
        st.info("Run the main pipeline to generate events")


if __name__ == "__main__":
    main()
