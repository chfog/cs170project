import argparse, os
import multiprocessing as mp
import utils, parse, solver

parser = argparse.ArgumentParser()
parser.add_argument("infile", nargs="?", type=str, default="inputs")
parser.add_argument("outfile", nargs='?', type=str, default="outputs")
parser.add_argument("--low", nargs='?', type=int, default=-1)
parser.add_argument("--high", nargs='?', type=int, default=-1)
args = parser.parse_args()

lock = None

def make_lock(l):
    global lock
    lock = l


if __name__ == "__main__":
    if os.path.isfile(args.infile):
        assert not os.path.exists(args.outfile) or os.path.isfile(args.outfile), "Outfile must be a file if infile is a file."
        solver.solve_file((args.infile, args.outfile))
    elif os.path.isdir(args.infile):
        assert os.path.isdir(args.outfile), "Outfile must be a directory if infile is a directory."
        files = sorted([f[:-3] for f in os.listdir(args.infile) if f.endswith(".in")])#, reverse=True)
        low = 0 if args.low < 0 or args.low > len(files) else args.low
        high = len(files) if args.high < low or args.high > len(files) else args.high
        files = files[low:high]
        with mp.Lock() as l:
            with mp.Pool(processes=5, initializer=make_lock, initargs=(l,)) as pool:
                mapped = pool.map_async(solver.solve_file, 
                        [(os.path.join(args.infile, f + ".in"), os.path.join(args.outfile, f + ".out")) for f in files]
                        )
                mapped.get()
#        print(list(map(solver.solve_file, [(os.path.join(args.infile, f + ".in"), os.path.join(args.outfile, f + ".out")) for f in files])))
