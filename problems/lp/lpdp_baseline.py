from subprocess import check_call, check_output
import argparse
import os
from subprocess import check_call
from urllib.parse import urlparse

def run_kalp(fn, start, end):
    out = check_output(['KaLPv2.0/deploy/kalp', fn, '--start_vertex', start,
                        '--target_vertex', end])
    print(out)



def extract_lpdp(fn="KaLPv2.0.tar.gz"):

    cwd = os.path.abspath(os.path.join("problems", "lp", "lpdp"))
    os.makedirs(cwd, exist_ok=True)

    fn = os.path.join(cwd, fn)

    if not os.path.isfile(fn):
        check_call(["wget", "http://algo2.iti.kit.edu/schulz/software_releases/KaLPv2.0.tar.gz"], cwd=cwd)

    assert os.path.isfile(fn), "Download failed, {} does not exist".format(fn)
    check_call(["tar", "xvfz", fn], cwd=cwd)


def install_argtable2(cwd = "lpdp",
                      argtable2_url = "http://prdownloads.sourceforge.net/argtable/argtable2-13.tar.gz"):
    cwd = os.path.abspath(os.path.join(cwd))
    os.makedirs(cwd, exist_ok=True)

    argtable2_download = os.path.join(cwd, 'argtable2-download')
    argtable2 = os.path.join(cwd, 'argtable2')
    if not os.path.isdir(argtable2):
        print('Installing argtable2 locally...')
        try:
            # downloading...
            argtable2_fn = os.path.join(cwd, os.path.split(urlparse(argtable2_url).path)[-1])
            if not os.path.isfile(argtable2_fn):
                check_call([f"wget {argtable2_url}"], cwd=cwd, shell=True)
                assert os.path.isfile(argtable2_fn), "Download failed, {} does not exist".format(argtable2_fn)
            os.makedirs(argtable2_download, exist_ok=True)
            check_call([f"tar xvfz {argtable2_fn} -C {argtable2_download} --strip-components 1"], shell=True)
            os.makedirs(argtable2, exist_ok=True)

            # installing...
            check_call(f"cd {argtable2_download} && ./configure --prefix={argtable2}", shell=True)
            check_call(f"cd {argtable2_download} && make && make install && make clean", shell=True)

            # cleaning...
            check_call(f"rm -rf {argtable2_download} {argtable2_fn}", shell=True)
        except Exception as e:
            print("Installation failed. Cleaning directories...")
            check_call(f'rm -rf {cwd}/argtable2*', shell=True)
            raise e

    assert os.path.isdir(argtable2), "Argtable2 didn't install properly"

def install_tbb(cwd = "lpdp",
                tbb_url="https://github.com/01org/tbb/archive/2019_U5.tar.gz"):
    cwd = os.path.abspath(os.path.join(cwd))
    os.makedirs(cwd, exist_ok=True)

    tbb = os.path.join(cwd, 'tbb')
    if not os.path.isdir(tbb):
        print('Installing tbb locally...')
        try:
            # downloading...
            tbb_fn = os.path.join(cwd, os.path.split(urlparse(tbb_url).path)[-1])
            if not os.path.isfile(tbb_fn):
                check_call(f"wget {tbb_url}", cwd=cwd, shell=True)
                assert os.path.isfile(tbb_fn), "Download failed, {} does not exist".format(tbb_fn)
            os.makedirs(tbb, exist_ok=True)
            check_call(f"tar xvfz {tbb_fn} -C {tbb} --strip-components 1", shell=True)

            # installing
            check_call(f"cd {tbb} && make", shell=True)

            # cleaning
            check_call(f"rm -rf {tbb_fn}", shell=True)
        except Exception as e:
            print("Installation failed. Cleaning directories...")
            check_call(f'rm -rf {cwd}/tbb*', shell=True)
            raise e

    assert os.path.exists(tbb), "TBB didn't install properly"

def install_scons(cwd = "lpdp",
                  scons_url="http://prdownloads.sourceforge.net/scons/scons-3.0.5.tar.gz"):
    cwd = os.path.abspath(os.path.join(cwd))
    os.makedirs(cwd, exist_ok=True)
    
    scons_download = os.path.join(cwd, 'scons-download')
    scons = os.path.join(cwd, 'scons')
    if not os.path.isdir(scons):
        print('Installing scons locally...')
        try:
            # downloading...
            scons_fn = os.path.join(cwd, os.path.split(urlparse(scons_url).path)[-1])
            if not os.path.isfile(scons_fn):
                check_call(f"wget {scons_url}", cwd=cwd, shell=True)
                assert os.path.isfile(scons_fn), "Download failed, {} does not exist".format(scons_fn)
            os.makedirs(scons_download, exist_ok=True)
            check_call(f"tar xvfz {scons_fn} -C {scons_download} --strip-components 1", shell=True)

            # installing
            check_call(f"cd {scons_download} && python setup.py install --prefix={scons}", shell=True)

            # cleaning
            check_call(f"rm -rf {scons_download} {scons_fn}", shell=True)
        except Exception as e:
            print("Installation failed. Cleaning directories...")
            check_call(f'rm -rf {cwd}/scons*', shell=True)
            raise e

    assert os.path.exists(scons), "Scons didn't install properly"

def install_dependencies():

    install_argtable2()
    install_tbb()
    install_scons()




def update_environ():
    from glob import glob
    
    cwd = os.path.abspath(os.path.join("lpdp"))

    os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + \
        f":{cwd}/argtable2/lib:{cwd}/argtable2/lib/libargtable2.so.0"
    os.environ['LD_LIBRARY_PATH'] += ':' + glob(f"{cwd}/tbb/build/*_release")[0]
    os.environ['PATH'] = os.environ.get('PATH', '') + f"{cwd}/scons/bin/"



if __name__ == '__main__':
    install_dependencies()
    
    update_environ()
    print(os.environ['LD_LIBRARY_PATH'])
    print(os.environ['PATH'])

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-f", help="Filename to run the algorithm")
#     parser.add_argument("-s", type=str, help="Start vertex")
#     parser.add_argument("-t", type=str, help="End vertex")

#     opts = parser.parse_args()

#     run_kalp(opts.f, opts.s, opts.t)
    
