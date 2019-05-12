import subprocess
from subprocess import check_call, check_output
import argparse
import os
from subprocess import check_call
from urllib.parse import urlparse
from glob import glob
import time

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

def install_dependencies(cwd = "lpdp",
                        argtable2_url = "http://prdownloads.sourceforge.net/argtable/argtable2-13.tar.gz",
                        tbb_url="https://github.com/01org/tbb/archive/2019_U5.tar.gz",
                        scons_url="http://prdownloads.sourceforge.net/scons/scons-3.0.5.tar.gz"):

    install_argtable2(cwd, argtable2_url)
    install_tbb(cwd, tbb_url)
    install_scons(cwd, scons_url)

def download_lpdp_datasets(cwd = "lpdp",
                  lpdp_ds_url = "https://algo2.iti.kit.edu/schulz/lp_benchmark.tar.gz"):
    cwd = os.path.abspath(os.path.join(cwd))
    os.makedirs(cwd, exist_ok=True)
    
    lpdpds = os.path.join(cwd, 'lpdp_datasets')
    if not os.path.isdir(lpdpds):
        print('Downloading lpdp datasets locally...')
        try:
            # downloading...
            lpdpds_fn = os.path.join(cwd, os.path.split(urlparse(lpdp_ds_url).path)[-1])
            if not os.path.isfile(lpdpds_fn):
                check_call(f"wget {lpdp_ds_url}", cwd=cwd, shell=True)
                assert os.path.isfile(lpdpds_fn), "Download failed, {} does not exist".format(lpdpds_fn)
            os.makedirs(lpdpds, exist_ok=True)
            check_call(f"tar xvfz {lpdpds_fn} -C {lpdpds} --strip-components 1", shell=True)
        except Exception as e:
            print("Download dplp datasets failed.")
            raise e
    assert os.path.exists(lpdpds), "lpdp datasets didn't download properly"

def download_kalp(cwd = "lpdp",
                  kalp_url = "http://algo2.iti.kit.edu/schulz/software_releases/KaLPv2.0.tar.gz"):
    cwd = os.path.abspath(os.path.join(cwd))
    os.makedirs(cwd, exist_ok=True)
    
    kalp = os.path.join(cwd, 'kalp')
    if not os.path.isdir(kalp):
        kalp_fn = os.path.join(cwd, os.path.split(urlparse(kalp_url).path)[-1])
        if not os.path.isfile(kalp_fn):
            check_call([f"wget {kalp_url}"], cwd=cwd, shell=True)
            assert os.path.isfile(kalp_fn), "Download failed, {} does not exist".format(kalp_fn)
        os.makedirs(kalp, exist_ok=True)
        check_call([f"tar xvfz {kalp_fn} -C {kalp} --strip-components 1"], shell=True)
        os.makedirs(kalp, exist_ok=True)
        
    assert os.path.isdir(kalp), "Kalp didn't download properly"
    
def compile_kalp(cwd = "lpdp"):
    cwd = os.path.abspath(os.path.join(cwd))
    os.makedirs(cwd, exist_ok=True)
    
    kalp = os.path.join(cwd, 'kalp')
    deploy = os.path.join(kalp, 'deploy')
    
    if not os.path.isdir(deploy):
        check_call(f"2to3 -w SConstruct extern/KaHIP/SConstruct", cwd=kalp, shell=True)
        check_call(f"cd {kalp} && ./compile.sh", shell=True)
        
    assert os.path.isdir(deploy), "Kalp didn't compile properly"

def install_kalp(cwd = "lpdp",
                  kalp_url = "http://algo2.iti.kit.edu/schulz/software_releases/KaLPv2.0.tar.gz"):
    update_environ(cwd)
    download_kalp(cwd, kalp_url)
    compile_kalp(cwd)
    
def update_environ(cwd = "lpdp"):
    cwd = os.path.abspath(os.path.join(cwd))
    
    argtable2 = os.path.join(cwd, 'argtable2')
    tbb = os.path.join(cwd, 'tbb')
    scons = os.path.join(cwd, 'scons')
    
    assert os.path.isdir(argtable2) and os.path.isdir(tbb) and \
            os.path.isdir(scons), "You should first install dependencies"

    os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + \
        f":{cwd}/argtable2/lib:{cwd}/argtable2/lib/libargtable2.so.0"
    os.environ['LD_LIBRARY_PATH'] += ':' + glob(f"{cwd}/tbb/build/*_release")[0]
    os.environ['PATH'] = os.environ.get('PATH', '') + f":{cwd}/scons/bin/"


def run_kalp(graph_fn, start_vertex, target_vertex,
             output_filename=None,
             partition_configuration='eco',
             cwd='lpdp',
             results_filename=None,
             routes_filename=None):
    
    # installing dependencies if don't exist
    # cwd = os.path.abspath(os.path.join(cwd))
    argtable2 = os.path.join(cwd, 'argtable2')
    tbb = os.path.join(cwd, 'tbb')
    scons = os.path.join(cwd, 'scons')
    if not (os.path.isdir(argtable2) and os.path.isdir(tbb) and os.path.isdir(scons)):
        print('Missing dependencies. Installing locally...')
        time.sleep(3)
        install_dependencies()
        
    # prepare kalp to run 
    update_environ()
    # cwd = os.path.abspath(os.path.join(cwd))
    os.makedirs(cwd, exist_ok=True)
    
    kalp = os.path.join(cwd, 'kalp')
    deploy = os.path.join(kalp, 'deploy')
    
    if not os.path.isdir(kalp):
        download_kalp()
    if not os.path.isdir(deploy):
        compile_kalp()
    
    # run kalp
    cmd = f'''{deploy}/kalp {graph_fn} --start_vertex={start_vertex} --target_vertex={target_vertex} --partition_configuration={partition_configuration}'''
    if output_filename:
        cmd += f" --output_filename={output_filename}"
    
    graph = os.path.split(graph_fn)[-1]
    print(f"{graph} {start_vertex} {target_vertex}")

    try:
        start = time.time()
        out = check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        end = time.time()
        print("Finished kalp in {:.2f}".format(end-start))

        if routes_filename is not None:
            save_path(output_filename, routes_filename)

        # write results to a file
        if results_filename:
            with open(results_filename, 'a+') as f:
                length = -1
                if os.path.isfile(output_filename):
                    length = int(check_output(f"wc -l {output_filename} | cut -d' ' -f1", shell=True))
                    check_call(f"rm {output_filename}", shell=True)

                f.write(f"{graph} {start_vertex} {target_vertex} {end - start} {length}\n")


    except subprocess.CalledProcessError:
        print("Failed running command:", cmd)

    except Exception as e:
        print("Somethign went wrong", e)


def save_path(path_fn, output_fn):
    with open(path_fn) as f:
        path = list(map(lambda x: x.strip(), f.readlines()))
    with open(output_fn, "a+") as f:
        f.write(",".join(path) + "\n")

            

if __name__ == '__main__':

    # lpdp datasets are heavy
    # download_lpdp_datasets()


    install_dependencies()
    install_kalp()

    fns = ["1ba.dimacs", "1bipartite.dimacs", "1bp_seed1234.dimacs",
           "1er.dimacs",
           "1path.dimacs", "1regular.dimacs"]

    cwd = "lpdp"
    kalp = os.path.join(cwd, 'kalp')

    i = 2
    fn = fns[i]
    graph_fn = f"{kalp}/examples/{fn}"
    for start in range(19):
        for target in range(start+1, 20):
            run_kalp(graph_fn, start, target, output_filename='test.txt',
                     results_filename=f"{fn.split('.')[0]}.results",
                     routes_filename=f"{fn.split('.')[0]}.routes")

    
    # cwd = os.path.abspath(os.path.join("lpdp"))
    # kalp = os.path.join(cwd, 'kalp')
    # graph_fn = f"{kalp}/examples/Grid8x8.graph"
    # start_vertex = 0
    # import glob
    # N = 20
    # fns = glob.glob("f{kalp}/examples/regulars/regular_n{N}*")
    # for fn in fns:
    #     for target_vertex in range(1, 20):
    #         run_kalp(graph_fn, start_vertex, target_vertex, output_filename='test.txt',
    #                 results_filename='results.txt')
    
    

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-f", help="Filename to run the algorithm")
#     parser.add_argument("-s", type=str, help="Start vertex")
#     parser.add_argument("-t", type=str, help="End vertex")

#     opts = parser.parse_args()

#     run_kalp(opts.f, opts.s, opts.t)
    
