module Titan

export @kernel, @gpu

using CUDArt
using MacroTools

# package code goes here
function _kernel(fn)
    ir = codegen(fn)
    ptx = compile_ir(ir)
    mod = create_module(ptx)
    @capture fn function name_(args__)
        body_
    end
    MODULES[name] = mod
    fn′ = wrap_kernel(fn)
    fn′
end

macro kernel(fn)
    _kernel(fn)
end

function codegen(fn)
    ir = readall("/Users/malmaud/Dropbox/Titan/src/kernel.ll")
    ir
end

function compile_ir(ir)
    d = mktempdir()
    kernel_path = joinpath(d, "kernel.ll")
    ptx_path = joinpath(d, "kernel.ptx")
    open(kernel_path, "w") do file
        write(file, ir)
    end
    cmd = `$(LLVM_PATH)/llc $kernel_path -o $ptx_path`
    run(cmd)
    local ptx
    open(ptx_path) do file
        ptx = readall(file)
    end
    ptx
end

function create_module(ptx)
    d = mktempdir()
    path = joinpath(d, "kernel.ptx")
    open(path, "w") do file
        write(file, ptx)
    end

    mod = CuModule(path, false)
end

function wrap_kernel(fn)
    @capture fn function name_(args__)
        body_
    end
    q_name = QuoteNode(name)
    quote
        function $(esc(name))($(args...); griddim=1, blockdim=1)
            mod = MODULES[$q_name]
            fn = CuFunction(mod, string($q_name))
            CUDArt.launch(fn, griddim, blockdim, $(Expr(:tuple, args...)))
        end
    end
end

macro gpu(expr)
    @capture expr name_<griddim_,blockdim_>(args__)
    quote
        $name(($args...); griddim=$griddim, blockdim=$blockdim)
    end
end

function __init__()
    const global LLVM_PATH = get(ENV, "LLVM_PATH", "/Users/malmaud/llvm/install/bin")
    const global MODULES = Dict()
    device(0)
    device_reset(0)
    CUDArt.free(CUDArt.malloc(UInt8, 1))
end

end # module
