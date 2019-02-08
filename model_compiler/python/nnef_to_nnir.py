import os, sys
import _nnef

def main ():
    if len(sys.argv) < 3:
        print('Usage: python nnef2nnir.py <nnefInputFolder> <outputFolder>')
        sys.exit(1)
    inputFolder = sys.argv[1]
    outputFolder = sys.argv[2]
    print('reading NNEF model from ' + inputFolder + '...')

if __name__ == '__main__':
    main()

