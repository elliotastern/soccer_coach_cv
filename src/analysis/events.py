# Event Detection Logic
from typing import List, Dict, Optional
import numpy as np
from src.types import Event, EventType, Player, Ball, Location, FrameData
from src.analysis.mapping import PitchMapper


class EventDetector:
    """Heuristic-based event detection"""
    
    def __init__(self, pitch_mapper: PitchMapper,
                 pass_velocity_threshold: float = 5.0,
                 dribble_distance_threshold: float = 2.0,
                 shot_velocity_threshold: float = 15.0,
                 recovery_proximity: float = 1.0):
        """
        Initialize event detector
        
        Args:
            pitch_mapper: Pitch coordinate mapper
            pass_velocity_threshold: Minimum velocity for pass (m/s)
            dribble_distance_threshold: Maximum distance for dribble (m)
            shot_velocity_threshold: Minimum velocity for shot (m/s)
            recovery_proximity: Proximity for recovery event (m)
        """
        self.pitch_mapper = pitch_mapper
        self.pass_velocity_threshold = pass_velocity_threshold
        self.dribble_distance_threshold = dribble_distance_threshold
        self.shot_velocity_threshold = shot_velocity_threshold
        self.recovery_proximity = recovery_proximity
        
        # State tracking
        self.player_history: Dict[int, List[Player]] = {}
        self.ball_history: List[Ball] = []
    
    def _calculate_velocity(self, loc1: Location, loc2: Location, time_diff: float) -> float:
        """Calculate velocity in m/s"""
        if time_diff <= 0:
            return 0.0
        distance = np.sqrt((loc2.x - loc1.x)**2 + (loc2.y - loc1.y)**2)
        return distance / time_diff
    
    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate distance in meters"""
        return np.sqrt((loc2.x - loc1.x)**2 + (loc2.y - loc1.y)**2)
    
    def detect_pass(self, frame_data: FrameData, prev_frame_data: Optional[FrameData]) -> Optional[Event]:
        """Detect pass event"""
        if prev_frame_data is None or not frame_data.ball:
            return None
        
        if not prev_frame_data.ball:
            return None
        
        # Calculate ball velocity
        ball_vel = self._calculate_velocity(
            Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch),
            Location(frame_data.ball.x_pitch, frame_data.ball.y_pitch),
            frame_data.timestamp - prev_frame_data.timestamp
        )
        
        if ball_vel < self.pass_velocity_threshold:
            return None
        
        # Find closest player to ball start position
        start_loc = Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch)
        end_loc = Location(frame_data.ball.x_pitch, frame_data.ball.y_pitch)
        
        closest_player_id = None
        min_dist = float('inf')
        
        for player in prev_frame_data.players:
            player_loc = Location(player.x_pitch, player.y_pitch)
            dist = self._calculate_distance(start_loc, player_loc)
            if dist < min_dist and dist < 3.0:  # Within 3 meters
                min_dist = dist
                closest_player_id = player.object_id
        
        if closest_player_id is None:
            return None
        
        return Event(
            id=f"pass_{frame_data.frame_id}",
            type=EventType.PASS,
            start_frame=prev_frame_data.frame_id,
            end_frame=frame_data.frame_id,
            start_location=start_loc,
            end_location=end_loc,
            involved_players=[closest_player_id],
            confidence=min(1.0, ball_vel / 20.0),
            timestamp_start=prev_frame_data.timestamp,
            timestamp_end=frame_data.timestamp
        )
    
    def detect_dribble(self, frame_data: FrameData, prev_frame_data: Optional[FrameData]) -> Optional[Event]:
        """Detect dribble event"""
        if prev_frame_data is None or not frame_data.ball:
            return None
        
        # Find player with ball
        ball_loc = Location(frame_data.ball.x_pitch, frame_data.ball.y_pitch)
        prev_ball_loc = Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch)
        
        for player in frame_data.players:
            player_loc = Location(player.x_pitch, player.y_pitch)
            dist = self._calculate_distance(ball_loc, player_loc)
            
            if dist < self.dribble_distance_threshold:
                # Check if same player had ball in previous frame
                if prev_frame_data:
                    prev_dist = self._calculate_distance(
                        prev_ball_loc,
                        Location(player.x_pitch, player.y_pitch)
                    )
                    if prev_dist < self.dribble_distance_threshold:
                        return Event(
                            id=f"dribble_{frame_data.frame_id}",
                            type=EventType.DRIBBLE,
                            start_frame=prev_frame_data.frame_id,
                            end_frame=frame_data.frame_id,
                            start_location=prev_ball_loc,
                            end_location=ball_loc,
                            involved_players=[player.object_id],
                            confidence=0.7,
                            timestamp_start=prev_frame_data.timestamp,
                            timestamp_end=frame_data.timestamp
                        )
        
        return None
    
    def detect_shot(self, frame_data: FrameData, prev_frame_data: Optional[FrameData]) -> Optional[Event]:
        """Detect shot event"""
        if prev_frame_data is None or not frame_data.ball:
            return None
        
        ball_vel = self._calculate_velocity(
            Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch),
            Location(frame_data.ball.x_pitch, frame_data.ball.y_pitch),
            frame_data.timestamp - prev_frame_data.timestamp
        )
        
        if ball_vel < self.shot_velocity_threshold:
            return None
        
        # Check if ball is in goal area
        ball_loc = Location(frame_data.ball.x_pitch, frame_data.ball.y_pitch)
        goal_x = 52.5  # End of pitch
        if abs(ball_loc.x) > goal_x - 5.0:  # Within 5m of goal
            # Find closest player
            closest_player_id = None
            min_dist = float('inf')
            for player in prev_frame_data.players:
                player_loc = Location(player.x_pitch, player.y_pitch)
                dist = self._calculate_distance(
                    Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch),
                    player_loc
                )
                if dist < min_dist:
                    min_dist = dist
                    closest_player_id = player.object_id
            
            if closest_player_id:
                return Event(
                    id=f"shot_{frame_data.frame_id}",
                    type=EventType.SHOT,
                    start_frame=prev_frame_data.frame_id,
                    end_frame=frame_data.frame_id,
                    start_location=Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch),
                    end_location=ball_loc,
                    involved_players=[closest_player_id],
                    confidence=min(1.0, ball_vel / 25.0),
                    timestamp_start=prev_frame_data.timestamp,
                    timestamp_end=frame_data.timestamp
                )
        
        return None
    
    def detect_recovery(self, frame_data: FrameData, prev_frame_data: Optional[FrameData]) -> Optional[Event]:
        """Detect ball recovery event"""
        if prev_frame_data is None or not frame_data.ball:
            return None
        
        ball_loc = Location(frame_data.ball.x_pitch, frame_data.ball.y_pitch)
        
        # Find player who recovered ball
        for player in frame_data.players:
            player_loc = Location(player.x_pitch, player.y_pitch)
            dist = self._calculate_distance(ball_loc, player_loc)
            
            if dist < self.recovery_proximity:
                # Check if ball was not with this player in previous frame
                if prev_frame_data:
                    prev_dist = self._calculate_distance(
                        Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch),
                        player_loc
                    )
                    if prev_dist > self.recovery_proximity:
                        return Event(
                            id=f"recovery_{frame_data.frame_id}",
                            type=EventType.RECOVERY,
                            start_frame=prev_frame_data.frame_id,
                            end_frame=frame_data.frame_id,
                            start_location=Location(prev_frame_data.ball.x_pitch, prev_frame_data.ball.y_pitch),
                            end_location=ball_loc,
                            involved_players=[player.object_id],
                            confidence=0.8,
                            timestamp_start=prev_frame_data.timestamp,
                            timestamp_end=frame_data.timestamp
                        )
        
        return None
    
    def detect_events(self, frame_data: FrameData, prev_frame_data: Optional[FrameData]) -> List[Event]:
        """Detect all events for current frame"""
        events = []
        
        # Try to detect each event type
        pass_event = self.detect_pass(frame_data, prev_frame_data)
        if pass_event:
            events.append(pass_event)
        
        dribble_event = self.detect_dribble(frame_data, prev_frame_data)
        if dribble_event:
            events.append(dribble_event)
        
        shot_event = self.detect_shot(frame_data, prev_frame_data)
        if shot_event:
            events.append(shot_event)
        
        recovery_event = self.detect_recovery(frame_data, prev_frame_data)
        if recovery_event:
            events.append(recovery_event)
        
        return events
