from subprocess import check_call, check_output
import argparse

def run_kalp(fn, start, end):
    out = check_output(['deploy/kalp', fn, '--start_vertex', start,
                        '--target_vertex', end])
    print(out)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="Filename to run the algorithm")
    parser.add_argument("-s", type=int, help="Start vertex")
    parser.add_argument("-t", type=int, help="End vertex")

    opts = parser.parse_args()

    run_kalp(opts.f, opts.s, opts.t)