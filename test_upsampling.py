from fib_upsampling import fib_upsampling
obj = fib_upsampling('data/fibers_connecting.trk')
obj.prepareTrafoClustering()
obj.upsample()
