# Football Analysis with YOLO

This project performs detailed football analysis using YOLOv5 for object detection and tracking, ByteTrack for object tracking, and various other techniques to enhance player and ball analysis.

## Features

- **Player and Ball Detection**: Utilizes YOLOv5 trained on a Roboflow dataset to annotate players, balls, and referees.
- **Tracking**: Employs ByteTrack to track player and ball movements across frames.
- **Custom Annotations**: 
  - Draws ellipses at the foot of players.
  - Marks balls with triangles at the top.
- **Team Identification**: Applies K-means clustering to identify player teams based on jersey colors.
- **Interpolation**: Enhances ball movement accuracy between frames.
- **Ball Acquisition**: Highlights the player with the ball using a triangle marker.
- **Ball Control Analysis**: Visualizes each team's ball control.
- **Camera Movement Compensation**: Adjusts player positions to account for camera movement.
- **Perspective Transformation**: Calculates the distance covered and speed of each player.

## Installation

1. **Clone the Repository:**

   ```sh
   git clone <repository_url>
   cd football_analysis
