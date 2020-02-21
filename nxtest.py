import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz

path = './pycallgraph.dot'

G = nx.Graph(nx.drawing.nx_pydot.read_dot(path))

# print(list(G.nodes))

# print(G.edges)

rme = [
    'kron',
    'trace',
    'inv',
    'eig',
    'real',
    'numpy.lingalg.linalg._unary_dispacher',
    'numpy.linalg.linalg.inv',
    'numpy.linalg.linalg._assertNdSquareness',
    'numpy.linalg.linalg._commonType',
    'numpy.linalg.linalg.isComplexType',
    'numpy.linalg.linalg.get_linalg_error_extobj',
    '_bootlocale.getpreferredencoding',
    'codecs.IncrementalEncoder.__init__',
    'codecs.IncrementalEncoder.setstate',
    '_bootlocale.getpreferredencoding',
    'codecs.IncrementalDecoder.__init__',
    'codecs.IncrementalDecoder.decode',
    'numpy.core.numeric.identity','numpy.lib.twodim_base.eye',
    'numpy.lib.type_check._real_dispatcher',
    'numpy.lib.type_check.real',
    'scipy.linalg._matfuncs_sqrtm.sqrtm',
    'scipy._lib._util._asarray_validated',
    'isrealobj',
    'scipy.linalg.decomp_schur.schur',
    'array_equal',
    'triu',
    'scipy.linalg._matfuncs_sqrtm._sqrtm_triu',
    'numpy.core.numeric.binary_repr',
    'warnings._showwarnmsg_impl',
    'warnings.WarningMessage.__init__',
    'numpy.core.numeric.warn_if_insufficient',
    'warnings._showwarnmsg',
    '<listcomp>',
    'genericpath.exists',
    '_handle_fromlist',
    '_find_and_load',
    '_ModuleLock.acquire',
    '_call_with_frames_removed',
    '_ModuleLock.release',
    '_ModuleLockManager.__enter__',
    '_find_and_load_unlocked',
    '_find_spec',
    'SourceFileLoader.exec_module',
    '_installed_safely.__exit__',
    '_call_with_frames_removed',
    '_ImportLockContext.__enter__',
    '_get_module_lock',
    '_load_unlocked',
    '_installed_safely.__init__',
    '_installed_safely.__enter__',
    '_ImportLockContext.__exit__',
    'module_from_spec',
    '_ModuleLockManager.__exit__',
    'cb',
    'numpy.lib.shape_base._kron_dispatcher',
    'numpy.linalg.linalg._makearray',
    'numpy.linalg.linalg.eig',
    'numpy.linalg.linalg._assertRankAtLeast2',
    'numpy.core.fromnumeric._trace_dispatcher',
    '_ModuleLock.__init__',
    'numpy.lib.shape_base.kron',
    'numpy.linalg.linalg._unary_dispatcher',
    '_ModuleLockManager.__init__',
    'numpy.core.fromnumeric.trace',
    'find_spec'
]

G.remove_nodes_from(rme)

print(G.nodes)

G = nx.nx_agraph.to_agraph(G)

G = pygraphviz.AGraph(directed=True)
G.graph_attr['rankdir'] = "TB"
G.graph_attr['splines'] = "ortho"
G.graph_attr['ordering'] = "out"
G.layout(prog='dot')

G.draw('graph.png', prog='circo')