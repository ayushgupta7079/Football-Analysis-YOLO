from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_distance_estimator import SpeedDistanceEstimator
import numpy as np
import cv2

def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #initialize tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True, 
                                       stub_path='stubs/tracks_stubs.pkl')
    
    #get object position
    tracker.add_position_to_track(tracks)

    #camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement.pkl')
    
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)


    #view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)



    #interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])


    #speed and distance estimator
    speed_distance_estimator = SpeedDistanceEstimator()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)


    #assign teams to players 
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])      # assign players to teams in first frame, don't want to give ball and what not

    for frame_num, player_track in enumerate(tracks['players']):    # assign team for each frame
        for player_id, track in player_track.items():               # assign for each player in frame
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
            tracks['players'][frame_num][player_id]['team'] = team      # add new value in dict as team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]      # add new value in dict as team color


    #assign ball to a player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])     # which team has ball
        else:
            team_ball_control.append(team_ball_control[-1])     # if ball not assigned then give the ball control to last player who touched ball

    team_ball_control = np.array(team_ball_control)


    #save cropped image of a player
    for track_id, player in tracks['players'][0].items():   #0th frame
        bbox = player['bbox']
        frame = video_frames[0]

        #crop bbox from frame
        cropped_image =frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)

        break


    #draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    #draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    #draw speed and distance
    output_video_frames = speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()