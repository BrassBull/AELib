import numpy as np



def NewtonK109(dt, x0, xt, yt):
    """

    Args:
        dt: Array of time differences [dt1, dt2, dt3].
        x0: Initial guess for [x, y, v, t].
        xt: Array of x-coordinates of data points [x1, x2, x3, x4].
        yt: Array of y-coordinates of data points [y1, y2, y3, y4].

    Returns:
        Xk: The refined estimate of [x, y, v, t].  Returns None if the method fails to converge.
    """

    x1, x2, x3, x4 = xt
    y1, y2, y3, y4 = yt
    dt1, dt2, dt3 = dt
    x, y, v, t = x0

    Xkm = np.array([x, y, v, t])  # Vector of unknowns

    for i in range(10):  # Maximum 10 iterations
        F = np.array([
            np.sqrt((x1 - Xkm[0])**2 + (y1 - Xkm[1])**2) - Xkm[2] * Xkm[3],
            np.sqrt((x2 - Xkm[0])**2 + (y2 - Xkm[1])**2) - Xkm[2] * (Xkm[3] + dt1),
            np.sqrt((x3 - Xkm[0])**2 + (y3 - Xkm[1])**2) - Xkm[2] * (Xkm[3] + dt2),
            np.sqrt((x4 - Xkm[0])**2 + (y4 - Xkm[1])**2) - Xkm[2] * (Xkm[3] + dt3)
        ])

        Ja = np.array([
            [-(x1 - Xkm[0]) / np.sqrt((x1 - Xkm[0])**2 + (y1 - Xkm[1])**2),
             -(y1 - Xkm[1]) / np.sqrt((x1 - Xkm[0])**2 + (y1 - Xkm[1])**2),
             -Xkm[3], -Xkm[2]],
            [-(x2 - Xkm[0]) / np.sqrt((x2 - Xkm[0])**2 + (y2 - Xkm[1])**2),
             -(y2 - Xkm[1]) / np.sqrt((x2 - Xkm[0])**2 + (y2 - Xkm[1])**2),
             -(Xkm[3] + dt1), -Xkm[2]],
            [-(x3 - Xkm[0]) / np.sqrt((x3 - Xkm[0])**2 + (y3 - Xkm[1])**2),
             -(y3 - Xkm[1]) / np.sqrt((x3 - Xkm[0])**2 + (y3 - Xkm[1])**2),
             -(Xkm[3] + dt2), -Xkm[2]],
            [-(x4 - Xkm[0]) / np.sqrt((x4 - Xkm[0])**2 + (y4 - Xkm[1])**2),
             -(y4 - Xkm[1]) / np.sqrt((x4 - Xkm[0])**2 + (y4 - Xkm[1])**2),
             -(Xkm[3] + dt3), -Xkm[2]]
        ])
        try:
            Jac = np.linalg.pinv(Ja)  # Inverse of the Jacobian matrix
            dXk = -Jac @ F  # Corrections to Xkm
            Xk = Xkm + dXk  # New approximation
            Xkm = Xk
            if np.max(np.abs(dXk)) < 1e-1:
                break  # Convergence criterion

        except:
            return(np.zeros(4))


    return Xk

