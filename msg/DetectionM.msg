# Message that encodes a defect on the car.

uint8 SEVERITY_CLASS_A = 1  # enumeration value for a type-A defect.
uint8 SEVERITY_CLASS_B = 2  # enumeration value for a type-B defect.
uint8 SEVERITY_CLASS_C = 3  # enumeration value for a type-C defect.

std_msgs/Header header
uint8 severity                      # severity class of the defect.
geometry_msgs/Pose location  # location of the defect. The reference frame of the data is embedded in the PoseStamped type.

