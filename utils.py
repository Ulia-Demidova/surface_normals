import numpy as np
import cv2 as cv


def get_projection_mask(H, W):
    """
    Gets a mask for the projected images that is most conservative with
    respect to the regions that maintain the kinect depth signal following
    projection.
    :return:
     mask - HxW binary image where the projection falls.
     sz - the size of the valid region.
    """
    mask = np.ones((H, W), dtype=bool)
    sz = (H, W)
#    mask = np.zeros((480, 640), dtype=bool)  # size of the nyuv2 depth image
#    mask[44:471, 40:601] = True
#    sz = (427, 561)
    return mask, sz


def rgb_plane2rgb_world(imgDepth, camera_params):
    """
    Projects the depth points from the image plane to the 3D world
    coordinates.
    :param imgDepth: depth map which has already been projected onto the RGB
                     image plane, an HxW matrix where H and W are the height and
                     width of the matrix, respectively.
    :return: points3d - the point cloud in the world coordinate frame, an Nx3
             matrix.
    """
    fx_rgb = camera_params['fx_rgb']
    fy_rgb = camera_params['fy_rgb']
    cx_rgb = camera_params['cx_rgb']
    cy_rgb = camera_params['cy_rgb']

    H, W = imgDepth.shape
    xx, yy = np.meshgrid(range(W), range(H))
    x3 = (xx - cx_rgb) * imgDepth / fx_rgb  # здесь есть небольшие различия в значениях с матлабом
    y3 = (yy - cy_rgb) * imgDepth / fy_rgb
    z3 = imgDepth
    points3d = np.stack([x3, -y3, z3], axis=2)
    return points3d


def compute_camera_params(depth_high, depth_width, fov):
    fx = fy = depth_width / (2 * np.tan(fov * np.pi / 360))
    cx = depth_width / 2
    cy = depth_high / 2
    return {'fx_rgb': fx,
            'fy_rgb': fy,
            'cx_rgb': cx,
            'cy_rgb': cy}


def tell_direction(norm, i, j, d, camera_params):
    fx_rgb = camera_params['fx_rgb']
    fy_rgb = camera_params['fy_rgb']
    cx_rgb = camera_params['cx_rgb']
    cy_rgb = camera_params['cy_rgb']
    x = (j - cx_rgb) * d / fx_rgb
    y = (i - cy_rgb) * d / fy_rgb
    z = d
    cor = np.array([x, -y, z])
    corner = cor.dot(norm)
    if corner >= 0:
        return 1
    else:
        return 0


def merge_similar_planes(planes, plane_idx, pts, distthresh):
    number_p = planes.shape[0]
    err = []
    for i in range(number_p):
        err.append(abs(np.dot(pts, planes[i].T)))

    newassign = np.arange(number_p)
    for p1 in newassign:
        for p2 in newassign[p1 + 1:]:
            if np.mean(err[p1][plane_idx[p2]] < distthresh[plane_idx[p2]] * 2) > 0.5 or \
                    np.mean(err[p2][plane_idx[p1]] < distthresh[plane_idx[p1]] * 2) > 0.5:
                newassign[p2] = p1
    uid = np.unique(newassign)
    planes2 = []
    plane_idx2 = []

    for p in range(len(uid)):
        ind = np.nonzero(newassign == uid[p])[0]
        planes2.append(planes[ind[0]])
        plane_idx2.append(np.concatenate([plane_idx[i] for i in ind]))
    return np.array(planes2), np.array(plane_idx)


def xyz2planes_ransac(X, Y, Z, normals, isvalid):
    """
    Finds the major scene surfaces(planes) using ransac.

    Args:
       X - HxW matrix of X coordinates.
       Y - HxW matrix of Y coordinates.
       Z - HxW matrix of Z coordinates.
       normals - Nx3 matrix of surface normals where N=H*W
       isvalid - HxW matrix indicating whether each point is valid.

    Returns:
       planes - Mx4 matrix of plane parameters where M is the number of major
                surfaces found.
       plane_idx - 1xM cell array of indices of each plane.
    """
    H, W = X.shape
    distthresh = 0.0075 * Z.ravel(order='F')
    normthresh = 0.1
    min_pts_per_plane = 2500
    offset = 30
    maxOverlap = 0.5

    X = X.ravel(order='F')
    Y = Y.ravel(order='F')
    Z = Z.ravel(order='F')

    npts = len(X)
    pts = np.stack([X, Y, Z, np.ones(npts)], axis=1)

    ui = np.arange(offset // 2, W - offset // 2, offset // 2)
    vi = np.arange(offset // 2, H - offset // 2, offset // 2)
    ui, vi = np.meshgrid(ui, vi)
    ui = ui.ravel(order='F')
    vi = vi.ravel(order='F')
    i1 = vi + (ui - 1) * H
    i2 = i1 + (offset - (2 * offset) * (ui > W / 2)) * H
    i3 = i1 + (offset - (2 * offset) * (vi > H / 2))
    validi = isvalid.ravel(order='F')[i1 - 1] * isvalid.ravel(order='F')[i2 - 1] * isvalid.ravel(order='F')[i3 - 1]
    i1 = i1[validi]
    i2 = i2[validi]
    i3 = i3[validi]

    niter = len(i1)
    planes = np.zeros((4, niter))
    inliers = [[] for _ in range(niter)]
    count = np.zeros(niter, dtype=int)

    for t in range(niter):
        A = pts[[i1[t] - 1, i2[t] - 1, i3[t] - 1]]
        w, eigv = np.linalg.eig(np.matmul(A.T, A))
        idx = w.argmin()
        planes[:, t] = eigv[:, idx]
        planes[:, t] = planes[:, t] / np.sqrt(sum(planes[:3, t] ** 2))
        dist = abs(np.dot(pts, planes[:, t]))
        distN = 1 - abs(np.dot(normals, planes[:3, t]))
        inliers[t] = np.nonzero(isvalid.ravel(order='F') * (dist < distthresh) * (distN < normthresh))[0]
        count[t] = len(inliers[t])

    si = np.argsort(count)[::-1]
    isused = np.zeros(isvalid.shape[0] * isvalid.shape[1], dtype=int)

    plane_idx = []
    c = np.zeros(niter, dtype=int)

    for t in range(niter):
        c[t] = sum(np.logical_not(isused[inliers[si[t]]]))
        if c[t] < maxOverlap * count[si[t]]:
            c[t] = 0
        elif c[t] > min_pts_per_plane:
            err = abs(np.dot(pts, planes[:, si[t]]))
            tmpidx = isvalid.ravel(order='F') * (err < distthresh / 2)
            A = pts[tmpidx]
            w, eigv = np.linalg.eig(np.matmul(A.T, A))
            idx = w.argmin()
            planes[:, si[t]] = eigv[:, idx]
            planes[:, si[t]] = planes[:, si[t]] / np.sqrt(sum(planes[:3, si[t]] ** 2))
            # dist = abs(np.dot(pts, planes[:, si[t]]))
            # distN = 1 - abs(np.dot(normals, planes[:3, si[t]]))

            plane_idx.append(inliers[si[t]])
            isused[inliers[si[t]]] = 1

    ind = np.nonzero(c > min_pts_per_plane)[0]
    planes = planes[:, si[ind]].T
    # count = c[ind]

    return merge_similar_planes(planes, plane_idx, pts, distthresh)







