import numpy as np
import os
import airsim

from utils import get_projection_mask, rgb_plane2rgb_world, compute_camera_params, tell_direction

blockWidths = [-1, -3, -6, -9, 0, 1, 3, 6, 9]
relDepthThresh = 0.05


def compute_local_planes(X, Y, Z, H, W):
    """
    Computes local surface normal information. Note that in this file, the Y
    coordinate points up, consistent with the image coordinate frame.
    :param X: Nx1 column vector of 3D point cloud X-coordinates
    :param Y: Nx1 column vector of 3D point cloud Y-coordinates
    :param Z: Nx1 column vector of 3D point cloud Z-coordinates
    :param params:
    :return: imgPlanes - an 'image' of the plane parameters for each pixel
             imgNormals - HxWx3 matrix of surface normals at each pixel.
             imgConfs - HxW image of confidences
    """
#    _, sz = get_projection_mask()
#    H, W = sz
    N = H * W  # количество точек
    pts = np.stack([X, Y, Z, np.ones(N)], axis=1)
    u, v = np.meshgrid(range(W), range(H))
    nu, nv = np.meshgrid(blockWidths, blockWidths)
    nx = np.zeros((H, W), dtype=np.float32)
    ny = np.zeros((H, W), dtype=np.float32)
    nz = np.zeros((H, W), dtype=np.float32)
    nd = np.zeros((H, W), dtype=np.float32)
    imgConfs = np.zeros((H, W), dtype=np.float32)

    ind_all = np.nonzero(Z)[0]
    for k in ind_all:
        i = k % H
        j = k // H
        u2 = u[i][j] + nu
        v2 = v[i][j] + nv

        # Check that u2 and v2 are in image.
        valid = (u2 >= 0) & (v2 >= 0) & (u2 < W) & (v2 < H)
        u2 = u2[valid]
        v2 = v2[valid]
        ind2 = v2 * W + u2

        # Check that depth difference is not too large.
        valid = abs(Z[ind2] - Z[i * W + j]) < Z[i * W + j] * relDepthThresh
        u2 = u2[valid]
        v2 = v2[valid]
        ind2 = v2 * W + u2

        if len(u2) < 3:
            continue

        A = pts[ind2, :]
        w, eigv = np.linalg.eig(np.matmul(A.T, A))
        idx = w.argmin()
        vector = eigv[:, idx]
        # if min(vector) < 0 and abs(min(vector)) > abs(max(vector)) or \
        #         max(vector) < 0:
        #     vector *= -1
        nx[i][j] = vector[0]
        ny[i][j] = vector[1]
        nz[i][j] = vector[2]
        nd[i][j] = vector[3]
        imgConfs[i][j] = 1 - np.sqrt(w[-1] / w[-2])

    # Normalize so that first three coordinates form a unit normal vector and
    # the largest normal component is positive
    imgPlanes = np.stack([nx, ny, nz, nd], axis = 2)
    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + np.spacing(1.0)
    imgPlanes /= np.stack([length] * 4, axis=2)
    imgNormals = imgPlanes[:, :, 0: 3]
    return imgPlanes, imgNormals, imgConfs





def main():
    resolutions = ["640x480", "1200x600", "1920x1080"]
    root = "/Users/yuliya/Desktop"
    for resolution in resolutions:
        dir = os.path.join(root, resolution)
        files = os.listdir(dir)
        depths = filter(lambda x: x.endswith('depth_in_meters.pfm'), files)
        normal_dir = os.path.join(root, "normals_v2", resolution)
        if not os.path.exists(normal_dir):
            os.makedirs(normal_dir)
        for i, depth in enumerate(depths):
            depth_path = os.path.join(dir, depth)
            imgDepthOrig, _ = airsim.read_pfm(depth_path)
            imgDepthOrig = np.flipud(imgDepthOrig)
            H, W = imgDepthOrig.shape
            projection_mask, projection_size = get_projection_mask(H, W)
            camera_params = compute_camera_params(H, W, 90)

            # Use rgb_plane2rgb_world since the depth has already been projected
            # onto the RGB image plane.
            points3d = rgb_plane2rgb_world(imgDepthOrig, camera_params)
            points3d = points3d[projection_mask]

            # Note that we need to swap X and Y here because compute_local_planes
            # keeps XZ to be the ground plane, just like in the image-plane.
            X = points3d[:, 0]
            Y = points3d[:, 2]
            Z = points3d[:, 1]

            imgPlanes, imgNormals, normalConf = compute_local_planes(X, Z, Y, H, W)

            for k in range(H):
                for j in range(W):
                    if not tell_direction(imgNormals[k][j], k, j, imgDepthOrig[k][j], camera_params):
                        imgNormals[k][j] *= -1

            normal_path = os.path.join(normal_dir, f"{i}.pfm")
            airsim.write_pfm(normal_path, imgNormals)

            normal_for_visualization = ((imgNormals + 1) * 100).astype('uint8')
            normal_visualization_path = os.path.join(normal_dir, f"{i}.png")
            airsim.write_png(normal_visualization_path, normal_for_visualization)


if __name__ == "__main__":
    main()
    

