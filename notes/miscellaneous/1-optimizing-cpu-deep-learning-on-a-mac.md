# Optimizing CPU deep learning on a Mac

Making deep learning go faster in the hardware you have is not trivial at all.
If you can, get a GPU (although that in turn has a lot of configuration challenges).

However, if you are like me and you don't have access to a GPU, not all is lost; there
are still plenty of things you can do to optimize your deep learning experience on you CPU.

## Use Intel's builds of Python packages
Intel (the processor manufacturer) has an [Anaconda Channel] where they publish intel optimized builds
of many core python libraries like `numpy`, `tensorflow` or even a custom build of `python`.

Using this in you mac can seriously boost the performance of your deep learning models, since the 
libraries have been optimized to take full advantage of your intel processor.

If you see a warning like `The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.`,
this means that you have a 'generic' tensorflow build.  Using intel's one will remove the warning and make you code go
seriously faster.

### Pitfalls: 

#### `libiomp5.dylib already initialized`
You may see the following error when running your code: `Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized`

This is apparently caused [by some parallelization redundancy macOS Accelerate Framework and MKL][mkl-error-answer].

The way I solved is was to by replacing the MKL variants of the least amount of dependencies I could get away with.
In other words, I preserved as many Intel builds as I could.


## Explicitly configure parallelization in your code
   
[Intel recommends fine tuning some environment and tensorflow variables to optimize your performance.][intel-tensorflow-optimization]

Getting the syntax right can be tricky because of changes of syntax between TensorFlow 1 and 2. Many
of the resources posted online still use old TensorFlow 1 syntax. As of
May 2020, this run configuration worked for TensorFlow 2:

```python
NUM_PARALLEL_EXEC_UNITS = 4
import os
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(NUM_PARALLEL_EXEC_UNITS)  # Number of physical cores.
print('Tensorflow run configuration:')
print(f'inter_op_parallelism_threads = {tf.config.threading.get_inter_op_parallelism_threads()}')
print(f'intra_op_parallelism_threads (number of physical cores) = {tf.config.threading.get_intra_op_parallelism_threads()}')
print(' ')
os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "0"  # Intel recommends 0 for CNNs
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
```

Alternatively, you can still use TF1 syntax on your TensorFlow 2 project by using the `compat`ibility layer.

```python
from keras import backend as K
import os
import tensorflow as tf
NUM_PARALLEL_EXEC_UNITS = 4 # Number of cores
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
    inter_op_parallelism_threads=2,
    allow_soft_placement=True,
    device_count={'CPU': NUM_PARALLEL_EXEC_UNITS}
)
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "0" # Intel recommends 0 for CNNs
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
```

## Related Reading
https://towardsdatascience.com/optimize-your-cpu-for-deep-learning-424a199d7a87

[intel-anaconda-channel]: https://software.intel.com/en-us/articles/using-intel-distribution-for-python-with-anaconda
[mkl-error-answer]: https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial/58869103#58869103
[intel-tensorflow-optimization]: https://software.intel.com/en-us/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference