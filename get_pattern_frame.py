import pickle
from utils.communicate import Robot

arm = Robot(connection=True)
arm.tohome()


def run():
    input('P1')
    P1 = arm.getpose()
    print(P1)
    input('P2')
    P2 = arm.getpose()
    print(P2)
    input('P3')
    P3 = arm.getpose()
    print(P3)
    with open('cam_matrix/axis_pose2.dat', 'wb') as f:
        pickle.dump([P1, P2, P3], f)
    arm.stop()
    return P1, P2, P3


if __name__ == '__main__':
    run()
