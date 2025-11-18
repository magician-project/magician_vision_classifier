import sys
import csv
import numpy as np
import cv2

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# ----------------------------------------------------------------------
# Hyperparameters / paths (EDIT THESE)
# ----------------------------------------------------------------------
MODEL_PATH = "bugatti.obj"            # Wavefront OBJ (triangles)
CAMERA_CSV_PATH = "camera_poses.csv"
#HEATMAP_PATH = "heatmap.png"
HEATMAP_PATH_TEMPLATE = "heatmap_%d.png"

CAMERA_INDEX = 0                  # which row from CSV to use

# Camera intrinsics (assumed pinhole); will be adjusted to heatmap size
FX = 800.0
FY = 800.0

# ----------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------
vertices = None   # Nx3
faces = None      # Mx3 (indices)
colors = None     # Nx3
angle = 0.0       # for simple spinning view

# ----------------------------------------------------------------------
# Simple OBJ loader (vertices & triangular faces only)
# ----------------------------------------------------------------------
def load_obj(path):
    v = []
    f = []
    with open(path, "r") as fobj:
        for line in fobj:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()[:4]
                v.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                # faces like "f 1 2 3" or "f 1/1/1 2/2/2 3/3/3"
                idx = []
                for p in parts:
                    i = p.split("/")[0]
                    idx.append(int(i) - 1)  # OBJ is 1-based
                if len(idx) == 3:
                    f.append(idx)
                else:
                    # fan triangulation if needed
                    for k in range(1, len(idx) - 1):
                        f.append([idx[0], idx[k], idx[k + 1]])
    return np.array(v, dtype=np.float32), np.array(f, dtype=np.int32)

# ----------------------------------------------------------------------
# Camera pose loader: CSV rows: x,y,z, rx,ry,rz (angles in degrees)
# ----------------------------------------------------------------------
def load_camera_poses(path):
    poses = []
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            vals = list(map(float, row[:6]))
            poses.append(vals)
    return np.array(poses, dtype=np.float32)  # shape (N, 6)

# ----------------------------------------------------------------------
# Euler (rx,ry,rz in degrees) -> rotation matrix (R)
# Convention: R = Rz * Ry * Rx
# ----------------------------------------------------------------------
def euler_to_R(rx, ry, rz):
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]], dtype=np.float32)
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0,          1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]], dtype=np.float32)
    return Rz @ Ry @ Rx

# ----------------------------------------------------------------------
# Project vertices into image given intrinsics & extrinsics
# X_cam = R * X + t   (R: 3x3, t: 3x1)
# u,v from K * X_cam
# ----------------------------------------------------------------------
def project_vertices(vertices, K, R, t, img_shape):
    N = vertices.shape[0]
    X = vertices.T  # 3xN
    X_cam = R @ X + t.reshape(3, 1)  # 3xN

    z = X_cam[2, :]
    valid = z > 1e-6
    uvs = np.zeros((N, 2), dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    if not np.any(valid):
        return uvs, valid_mask

    X_cam_valid = X_cam[:, valid]
    proj = (K @ X_cam_valid).T  # Nx3
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]

    h, w = img_shape[:2]
    u_i = np.round(u).astype(int)
    v_i = np.round(v).astype(int)

    in_bounds = (u_i >= 0) & (u_i < w) & (v_i >= 0) & (v_i < h)

    valid_idx = np.where(valid)[0]
    kept_idx = valid_idx[in_bounds]

    uvs[kept_idx, 0] = u_i[in_bounds]
    uvs[kept_idx, 1] = v_i[in_bounds]
    valid_mask[kept_idx] = True

    return uvs, valid_mask

# ----------------------------------------------------------------------
# Compute colors per vertex from heatmap
# ----------------------------------------------------------------------
def compute_vertex_colors(vertices, pose, heatmap):
    # pose: [x,y,z, rx,ry,rz]
    tx, ty, tz, rx, ry, rz = pose
    R = euler_to_R(rx, ry, rz)
    t = np.array([tx, ty, tz], dtype=np.float32)

    h, w = heatmap.shape[:2]

    fx = FX
    fy = FY
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)

    uvs, valid_mask = project_vertices(vertices, K, R, t, heatmap.shape)

    colors = np.zeros((vertices.shape[0], 3), dtype=np.float32)

    # Base color for non-visible vertices
    colors[:, :] = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    hm = heatmap.astype(np.float32)
    if hm.ndim == 3:
        # If heatmap is color, convert to grayscale
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)
    hm /= 255.0

    for i in np.where(valid_mask)[0]:
        u, v = int(uvs[i, 0]), int(uvs[i, 1])
        val = hm[v, u]  # [0,1]
        # Simple blue-red ramp
        r = val
        g = 0.0
        b = 1.0 - val
        colors[i] = np.array([r, g, b], dtype=np.float32)

    return colors

# ----------------------------------------------------------------------
# OpenGL callbacks
# ----------------------------------------------------------------------
def display():
    global angle, vertices, faces, colors

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Simple fixed view (not same as camera used for coloring)
    gluLookAt(0.0, 0.0, 5.0,
              0.0, 0.0, 0.0,
              0.0, 1.0, 0.0)

    glRotatef(angle, 0.0, 1.0, 0.0)

    glBegin(GL_TRIANGLES)
    for f in faces:
        for idx in f:
            glColor3fv(colors[idx])
            glVertex3fv(vertices[idx])
    glEnd()

    glutSwapBuffers()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = w / float(h if h != 0 else 1)
    gluPerspective(45.0, aspect, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def idle():
    global angle
    angle += 0.2
    if angle > 360.0:
        angle -= 360.0
    glutPostRedisplay()


def keyboard(key, x, y):
    if key == b'\x1b':  # ESC
        import sys
        sys.exit(0)

# ----------------------------------------------------------------------
# Main setup
# ----------------------------------------------------------------------
def main():
    global vertices, faces, colors

    # Load data
    print("Loading model...")
    vertices, faces = load_obj(MODEL_PATH)

    # ------------- Normalize / Center / Scale -------------
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    vertices = vertices - center
    size = (vmax - vmin).max()
    vertices = vertices / size * 2.0     # approx [-1,1]

    # ------------- Fix orientation (Z-up → Y-up) -----------
    # Swap Y and Z
    vertices = vertices[:, [0, 2, 1]]

    # Flip Z axis (put car's front toward +Z)
    vertices[:, 2] *= -1

    # Optional: lift slightly above origin
    vertices[:, 1] += 0.2

    print("Vertices ready:", vertices.min(axis=0), vertices.max(axis=0))

    print("Model normalized:")
    print("  min:", vertices.min(axis=0))
    print("  max:", vertices.max(axis=0))

    print("Loading camera poses...")
    poses = load_camera_poses(CAMERA_CSV_PATH)

    for CAMERA_INDEX in range(len(poses)):
      pose = poses[CAMERA_INDEX]
      print("Loading heatmap...")
      hm_path = HEATMAP_PATH_TEMPLATE % CAMERA_INDEX
      heatmap = cv2.imread(hm_path)
      if heatmap is None:
        print("Could not load heatmap image:", HEATMAP_PATH)
        sys.exit(1)

      print("Computing vertex colors from heatmap projection %u ..."% CAMERA_INDEX)
      colors = compute_vertex_colors(vertices, pose, heatmap)

    print("Handover to OpenGL...")
    # Initialize OpenGL / GLUT
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"Heatmap-colored car (PyOpenGL + OpenCV + NumPy)")

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    
    glutKeyboardFunc(keyboard)
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutIdleFunc(idle)

    glutMainLoop()

if __name__ == "__main__":
    main()

