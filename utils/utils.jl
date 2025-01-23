import Pkg
import UUIDs
using FileIO
using Base.Threads
using UTCGP
using DataFlowTasks
import PNGFiles
using Images
using Flux: sigmoid
using StatsBase: sample, Weights
using UnicodePlots
using ThreadPools

pyexec(
    """
    import numpy as np
    from skimage import data
    from skimage.color import rgb2hed, hed2rgb, rgb2gray
    import skimage 
    """,
    Main)

const PYLOCK = ReentrantLock()

function gc_()
    PythonCall.GIL.lock(PythonCall.GC.gc)
end

function pycall_lock(f::Function)
    lock(PYLOCK)
    try
        local h
        local e
        local d
        t = @tspawnat 1 begin
            h, e, d = PythonCall.GIL.lock(f)
        end
        fetch(t)
        return h, e, d
    finally
        # if rand() > 0.99
        # PythonCall.GIL.lock(PythonCall.GC.gc)
        # end
        unlock(PYLOCK)
    end
end


# SAVE FROM HDF5 TO IMG  ---------
# images = h5open("datasets/pcam/camelyonpatch_level_2_split_train_x.h5")
# labels = h5open("datasets/pcam/camelyonpatch_level_2_split_train_y.h5")
# ys = labels["y"]
# xs = images["x"]
# x = [xs[:, :, :, i] for i in 1:size(xs)[end]]
# y = [ys[:, :, :, i][1] for i in 1:size(ys)[end]]
# n = 1
# for (img, class) in zip(x, y)
#     global n
#     colimg = colorview(RGB, reinterpret.(N0f8, img))
#     class_as_int = Int(class)
#     save("datasets/pcam/train/train_$(n)_$class_as_int.png", colimg)
#     n += 1
# end

# PCAMENDPOINT ---------
struct PCAMEndpoint <: UTCGP.BatchEndpoint
    fitness_results::Vector{Float64}
    function PCAMEndpoint(
        pop_preds::Vector{<:Vector{<:Number}}, # pop[ ind1[ out1, out2 ], ind2... ]. 
        truth::Vector{<:Int}, # one obs
    )
        global ACC_WEIGHT
        l = unique(length.(pop_preds))[1]
        pop_res = Float64[]
        for ind_predictions in pop_preds
            ind_predictions = replace(ind_predictions, NaN => 0.0)
            pred = argmax(ind_predictions)
            gt = truth[1]
            one_hot_gt = collect(1:l) .== gt
            nll_loss = Flux.logitcrossentropy(
                ind_predictions,
                one_hot_gt,
                dims=1)
            nll_loss = clamp(nll_loss, 0.0, 1.0)
            acc = pred == gt ? 0.0 : 1.0
            push!(pop_res, ACC_WEIGHT * acc + (1 - ACC_WEIGHT) * nll_loss)
        end
        return new(pop_res)
    end
end

struct PCAMBinaryEndpoint <: UTCGP.BatchEndpoint
    fitness_results::Vector{Float64}
    function PCAMBinaryEndpoint(
        pop_preds::Vector{<:Vector{<:Number}}, # pop[ ind1[ out1, out2 ], ind2... ]. 
        truth::Vector{<:Int}, # one obs
    )
        global ACC_WEIGHT
        l = unique(length.(pop_preds))[1]
        pop_res = Float64[]
        for ind_predictions in pop_preds
            ind_predictions = replace(ind_predictions, NaN => 0.0)
            nll_loss, acc, err = binary_metrics(ind_predictions, truth)
            loss = ACC_WEIGHT * acc + (1 - ACC_WEIGHT) * nll_loss
            push!(pop_res, loss)
        end
        return new(pop_res)
    end
end

function binary_metrics(outputs::Vector{Float64}, gt::Vector{Int64})
    @assert length(outputs) == 1
    @assert length(gt) == 1
    @assert gt[1] == 1 || gt[1] == 2

    binary_metrics(outputs[1], gt)
end
function binary_metrics(output::Float64, gt::Vector{Int64})
    nll_loss = _binary_ce(output[1], gt[1])
    gt_int = Int(gt[1])
    acc = _acc(output[1], gt_int)
    err = 1 - acc
    nll_loss, acc, err
end

function _binary_ce(pred::Float64, gt::Int)
    nll_loss = 0.0
    if gt == 1 # class 1
        nll_loss += log((1.0 - sigmoid(pred)) + eps(Float64))
    else # class 2
        nll_loss += log(sigmoid(pred) + eps(Float64))
    end
    -nll_loss
end

function _acc(pred::Float64, gt::Int)
    acc = 0.0
    p = Int(pred >= 0.5) + 1 # <0.5 => 1 || => 0.5 => 2
    Float64(p == gt)
end

struct PCAMBinaryDropout <: UTCGP.BatchEndpoint
    fitness_results::Vector{Float64}
    function PCAMBinaryDropout(
        pop_preds::Vector{<:Vector{<:Number}}, # pop[ ind1[ out1, out2 ], ind2... ]. 
        truth::Vector{<:Int}, # one obs
    )
        global ACC_WEIGHT
        l = unique(length.(pop_preds))[1]
        pop_res = Float64[]
        for ind_predictions in pop_preds
            ind_predictions = replace(ind_predictions, NaN => 0.0) # ALL predictions are support for class 1
            sampled_preds = dropout(ind_predictions) # gives 75 % of preds
            pred = sum(sampled_preds)
            pred = pred / 0.75 # scaled pred
            nll_loss, acc, err = binary_metrics(pred, truth)
            loss = ACC_WEIGHT * acc + (1 - ACC_WEIGHT) * nll_loss
            push!(pop_res, loss)
        end
        return new(pop_res)
    end
end

function dropout(preds)
    dropout_prob = 0.25
    n = length(preds)
    how_many = round(Int, (1 - dropout_prob) * n)
    sampled_preds = sample(preds, how_many, replace=false)
    sampled_preds
end

# DATALOADERS ---------
abstract type AbstractDataLoader end
struct DataLoader <: AbstractDataLoader
    path::String
    n::Int
    files
    batch_size::Int # For multi threading
    function DataLoader(path, n, bs)
        files = readdir(path)
        new(path, n, files, bs)
    end
end
struct RamDataLoader <: AbstractDataLoader
    xs
    ys
    indices
    batch_size::Int # How many to send to a single T
    n::Int # How many to sample
end
struct WRamDataLoader <: AbstractDataLoader
    xs
    ys
    indices
    batch_size::Int # How many to send to a single T
    n::Int # How many to sample
    ws::Vector{Float64}
    function WRamDataLoader(xs, ys, indices, batch_size, n)
        ws = ones(Float64, length(xs))
        new(xs, ys, indices, batch_size, n, ws)
    end
end
Base.length(rd::WRamDataLoader) = rd.n # the length of the whole sample
Base.length(d::AbstractDataLoader) = d.n
function Base.getindex(d::DataLoader, i::Int, random::Bool)
    if random
        subset = rand(d.files, 1)
    else
        subset = [d.files[i]]
    end
    vw_x = Vector(undef, 1)
    vw_y = Vector{Vector{Int}}(undef, 1)
    read_files(subset, d.path, vw_x, vw_y)
    return (vw_x[1], vw_y[1])
end

function Base.getindex(d::DataLoader, I::UnitRange{Int})
    tid = Threads.threadid()
    @info "Dataloader IO ($I) at $tid start $(now())"
    tuples = []
    midp = 0
    for i in I
        if i < midp
            push!(tuples, Base.getindex(d, i, false))
        else
            push!(tuples, Base.getindex(d, i, true))
        end
    end
    @info "Dataloader IO ($I) at $tid end $(now())"
    tuples
end
function Base.getindex(d::RamDataLoader, I::UnitRange{Int})
    how_many = length(I)
    which_samples = sample(d.indices, how_many, replace=false)
    xs = d.xs[which_samples]
    ys = d.ys[which_samples]
    collect(zip(xs, ys))
end
function Base.getindex(d::WRamDataLoader, I::UnitRange{Int})
    how_many = length(I)
    which_samples = sample(d.indices, Weights(d.ws), how_many, replace=false)
    try
        println("Dataloader Histogram")
        histogram(d.ws) |> println
    catch
        println("Could not print histogram")
    end
    xs = d.xs[which_samples]
    ys = d.ys[which_samples]
    which_samples, collect(zip(xs, ys))
end

# LOAD DATASET ---------
function load_dataset(dir::String, n::Union{Nothing,Int})
    images_dir = readdir(dir)
    nt = nthreads()
    isnothing(n) ? N = length(images_dir) : N = n
    IMG_X = Vector(undef, N)
    IMG_Y = Vector{Vector{Int}}(undef, N)
    range_files = 1:N
    partition_size = ceil(Int, N / nt)
    @info "Reading $N files from $dir. Each Thread with $partition_size files"
    for idx_subset in Iterators.partition(range_files, partition_size)
        part = collect(idx_subset)
        subset_files = @view images_dir[part]
        vw_x = @view IMG_X[part]
        vw_y = @view IMG_Y[part]
        @dspawn begin
            @R subset_files
            @W vw_x
            @W vw_y
            tid = Threads.threadid()
            @debug "Thread $tid started reading $(length(part)) files"
            read_files(subset_files, dir, vw_x, vw_y)
            @debug "Thread $tid ended reading $(length(part)) files"
        end label = "$(part[begin]) : $(part[end])"
    end
    final_task = @dspawn @R(@view IMG_X[:]) label = "READ X"
    other_final = @dspawn @R(@view IMG_Y[:]) label = "READ Y"
    fetch(final_task)
    fetch(other_final)
    @info "Loading data from $dir DONE"
    (IMG_X, IMG_Y)
end

function rgb2hed(rgbimg)
    h, e, d = pycall_lock() do
        h, e, d = PythonCall.pyexec(@NamedTuple{h::Array{Float64,2}, e::Array{Float64,2}, d::Array{Float64,2}}, """
                      #import numpy as np
                      #import matplotlib.pyplot as plt
                      #from skimage import data
                      #from skimage.color import rgb2hed, hed2rgb, rgb2gray
                      #import skimage 
                      #print(skimage.__version__)
                      # Example IHC image
                      #ihc_rgb = data.immunohistochemistry()

                      # Separate the stains from the IHC image
                      ihc_hed = rgb2hed(img)

                      #print("#############")
                      #print(img[0:2,0:2,:])
                      #print(img[0:2,0:2,:].shape)

                      # Create an RGB image for each of the stains
                      null = np.zeros_like(ihc_hed[:, :, 0])
                      ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
                      ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
                      ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

                      #print(type(ihc_h))

                      #print((ihc_hed[:, :, 1])[0:2,0:2])

                      #h,e,d =  ihc_h,  ihc_e,  ihc_d

                      h,e,d =  rgb2gray(ihc_h),  rgb2gray(ihc_e),  rgb2gray(ihc_d)
                      #h,e,d = ihc_hed[:, :, 0], ihc_hed[:, :, 1], ihc_hed[:, :, 2]
                      """,
            Main,
            (img=float64.(permutedims(channelview(rgbimg), (2, 3, 1))),)
        )
        SImageND(N0f8.(clamp01nan.(h))), SImageND(N0f8.(clamp01nan.(e))), SImageND(N0f8.(clamp01nan.(d)))
    end
    h, e, d
end

function read_files(subset_files, dir, vw_x, vw_y)
    ys = [parse(Int, split(i, "_")[end][1]) for i in subset_files] # get label
    xs = [PNGFiles.load(joinpath(dir, i)) for i in subset_files] # load 3d img
    xs = [i[32:(32*2-1), 32:(32*2-1)] for i in xs] # https://github.com/basveeling/pcam # crop
    # imgs are still in rgb so make it hed 
    heds = rgb2hed.(xs)
    # HSV ? 
    hsvs = [HSV.(x) for x in xs]
    hsvs = [collect(channelview(float(x))) for x in hsvs]
    hsvs = [[x[1, :, :] / 360.0, x[2, :, :], x[3, :, :]] for x in hsvs]
    HSVS = []
    for (i, hsv) in enumerate(hsvs)
        push!(HSVS, [N0f8.(x) for x in hsv])
    end
    E = length(hsvs)
    HSVS = [[SImageND(HSVS[i][j]) for j in 1:3] for i in 1:E]
    # 
    Gray_imgs = [identity.(convert.(N0f8, Gray.(three_D_img))) for three_D_img in xs]
    xs = [collect(channelview(x)) for x in xs] # to [2D, 2D, 2D]
    E = length(xs)
    xs = [[SImageND(xs[i][j, :, :]) for j in 1:3] for i in 1:E]
    for (i, l) in enumerate(xs)
        push!(l, SImageND(Gray_imgs[i]))
    end
    Xs = [[obs..., hsv..., hed..., 0.1, -0.1, 0.5, -0.5, -1.0, -2.0, 2.0] for (obs, hsv, hed) in zip(xs, HSVS, heds)]
    Ys = [[l + 1] for l in ys]
    vw_x[:] = Xs
    vw_y[:] = Ys
end

# function get_python_path()
#     python_path = ENV["UTCGP_PYTHON"]
#     ENV["PYTHON"] = python_path
#     return python_path
# end

# function get_psb2_path()
#     dataset_path = ENV["UTCGP_PSB2_DATASET_PATH"]
#     return dataset_path
# end

# function get_nruns()
#     n_runs = ENV["UTCGP_NRUNS"]
#     n_runs = parse(Int, n_runs)
# end

# function get_unique_id()
#     return UUIDs.uuid4().value
# end

# function load_psb2_data(dataset_path::String, pb::String, n_train::Int, n_test::Int)
#     PROBLEM = pb
#     N_TRAIN = n_train
#     N_test = n_test

#     py"""
#     import psb2 
#     import numpy as np

#     (train_data, test_data) = psb2.fetch_examples(
#         $dataset_path, $PROBLEM, $N_TRAIN, $N_TEST, format="psb2", seed = 1
#     )
#     """
# end

# function make_rows(
#     in_keys::Vector{String},
#     in_casters::Vector,
#     out_keys::Vector{String},
#     out_casters::Vector,
#     extra_inputs::Vector,
#     df_in_python_memory::String)
#     X = []
#     Y = []
#     df = df_in_python_memory
#     for x in py"$$df"
#         ins = [caster(x[k]) for (caster, k) in zip(in_casters, in_keys)]
#         outs = [caster(x[k]) for (caster, k) in zip(out_casters, out_keys)]
#         push!(X, Any[ins..., extra_inputs...])
#         push!(Y, identity.([outs...]))
#     end
#     return X, Y
# end

function fix_all_output_nodes!(ut_genome::UTGenome)
    for (ith_out_node, output_node) in enumerate(ut_genome.output_nodes)
        to_node = output_node[2].highest_bound + 1 - ith_out_node
        set_node_element_value!(output_node[2],
            to_node)
        set_node_freeze_state(output_node[2])
        set_node_freeze_state(output_node[1])
        set_node_freeze_state(output_node[3])
        println("Output node at $ith_out_node: $(output_node.id) pointing to $to_node")
        println("Output Node material : $(node_to_vector(output_node))")
    end
end

# function args_parse()
#     s = ArgParseSettings()
#     @add_arg_table s begin
#         "--seed", "-s"
#         help = "Random seed"
#         arg_type = Int
#         required = true
#     end
#     parsed_args = parse_args(s)
#     return parsed_args
# end
