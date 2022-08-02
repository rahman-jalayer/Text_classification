from models import Train
import sys

if __name__ == '__main__':
    t = Train()
    t.train(retrain=sys.argv)
