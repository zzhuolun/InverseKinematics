import numpy as np
import matplotlib.pyplot as plt
cos = np.cos
sin = np.sin
# Draw RR robot
def drawRRrobot(q, xd, pause=True):
    q1 = q[0]
    q2 = q[1]
    p1 = np.array([np.cos(q1), np.sin(q1)])
    p2 = p1 + np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
    plt.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]])
    plt.plot(p1[0], p1[1], 'ro')
    plt.plot(p1[0], p1[1], 'ro')
    plt.plot(xd[0], xd[1], 'bx')
    plt.axis([-2, 2, -2, 2])
    # if pause:
    #     plt.pause(0.005)
    # else:
    #     plt.show()


# This function solves the corresponding q1,q2 by exhaustive test
# e.g. xd=[1.2,1.2]
def bruteforce_search(xd):
    threshold = 0.003
    for q1 in np.arange(0, 2 * np.pi, 0.005):
        for q2 in np.arange(0, 2 * np.pi, 0.005):
            f1 = np.cos(q1) + np.cos(q1 + q2)
            f2 = np.sin(q1) + np.sin(q1 + q2)
            if abs(f1 - xd[0]) < threshold and abs(f2 - xd[1]) < threshold:
                print('brute force solution:')
                print('theta angle:', '[', q1, ',', q2, ']')
                print('corresponding end-effector position:', '[', f1, ',', f2, ']')

def numeric_IK(xd, q, isNewton, epsilon, max_iter):
    fig = plt.figure(figsize=plt.figaspect(1))
    plt.title('Newton\'s Method' if isNewton else 'Gradient descent')
    mini = [0.22779704, 1.11520942]  # one of the minimizer
    x = np.linspace(-np.pi, np.pi, 50)
    y = np.linspace(-np.pi, np.pi, 50)
    X, Y = np.meshgrid(x, y)
    Z1 = xd[0] - (np.cos(X + Y) + np.cos(X))
    Z2 = xd[1] - (np.sin(X + Y) + np.sin(X))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_wireframe(X, Y, Z1, color=np.array([[0.5, 0.5, 0.5, 0.5]]))
    ax.scatter3D([mini[0]], [mini[1]], [xd[0] - (np.cos(mini[0] + mini[1]) + np.cos(mini[0]))], c=np.array([[0, 0, 0]]),
                 marker='o')
    ax.set_title('first element of function g=xd-f(q)')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot_wireframe(X, Y, Z2, color=np.array([[0.5, 0.5, 0.5, 0.5]]))
    ax2.scatter3D([mini[0]], [mini[1]], [xd[0] - (np.sin(mini[0] + mini[1]) + np.sin(mini[0]))],
                  c=np.array([[0, 0, 0]]),
                  marker='o')
    ax2.set_title('second element of function g=xd-f(q)')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.plot_wireframe(X, Y, 0.5 * (Z1 ** 2 + Z2 ** 2), color=np.array([[0.5, 0.5, 0.5, 0.5]]))
    ax3.set_title('square loss')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('RR Robot')
    for i in range(max_iter):
        q1 = q[0] % (2 * np.pi)
        if q1 > np.pi:
            q1 -= 2 * np.pi
        q2 = q[1] % (2 * np.pi)
        if q2 > np.pi:
            q2 -= 2 * np.pi
        drawRRrobot(q, xd)
        fx = xd[0] - (np.cos(q1 + q2) + np.cos(q1))
        fy = xd[0] - (np.sin(q1 + q2) + np.sin(q1))
        ax.scatter3D([q1], [q2], fx, c=np.array([[1, 0, 0]]), marker='o')
        ax2.scatter3D([q1], [q2], fy, c=np.array([[1, 0, 0]]), marker='o')
        ax3.scatter3D([q1], [q2], 0.5 * (fx ** 2 + fy ** 2), c=np.array([[1, 0, 0]]), marker='o')

        J = np.array(
            [[-sin(q1) - sin(q2), -sin(q1 + q2)],
             [cos(q1) + cos(q1 + q2), cos(q1 + q2)]]
        )
        f = np.array([cos(q1) + cos(q1 + q2), sin(q1) + sin(q1 + q2)])
        e = xd - f
        if isNewton:
            # Newton' Method
            plt.pause(1)
            q = q + np.dot(np.linalg.inv(J), e)
        else:
            # Gradient decent
            plt.pause(0.01)
            q = q + 0.1 * np.dot(np.transpose(J), e)
        if np.linalg.norm(e) < epsilon:
            plt.pause(3)
            return q,i
    plt.pause(3)
    return q,max_iter