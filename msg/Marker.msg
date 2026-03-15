# Detected marker (ArUco or chessboard) with pose in camera frame.
# id: ArUco marker ID as string (e.g. "42"), or "chessboard" for chessboard targets.
# pose.position  : translation vector (tvec) in metres from the camera origin.
# pose.orientation : rotation as quaternion (converted from OpenCV Rodrigues rvec).

string id
geometry_msgs/Pose pose
