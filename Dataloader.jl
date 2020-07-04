# %% markdown
# # ImageList -> Main
# %% codecell
#export
using FileIO
using Images
using Serialization
using Random
using CUDAapi
using Plots
import GR
using Images
using CuArrays
using ImageView
using Statistics, Random
import ProgressMeter
using Distributions
using Zygote
gr()
# %% markdown
# fromFolder
# %% codecell
#export
# Helper function to add path to array
function add_path(cat::String)
    temp_dir = readdir(joinpath(path, cat))
    return [joinpath(path, cat, x) for x in temp_dir],
    fill(cat, size(temp_dir, 1))
end
# %% markdown
# ## Class Distribution
# %% codecell
#export
function classDistribution(y)
    """
    Function to plot class distribution to see if balanced or not.
    """
    labels = unique(y)
    cnts = [sum(y .== i) for i in labels]
    display(plot(cnts, seriestype = [:bar]))
    return cnts, maximum(cnts)
end
# %% codecell
#export
path = "/media/subhaditya/DATA/COSMO/Datasets/catDog/"
# %% codecell
#export
# Define number of threads
Threads.nthreads() = length(Sys.cpu_info())
# %% codecell
collect(1:10)
# %% codecell
#export

"""
Function to create an array of images and labels -> when the directory structure is as follows
- main
    - category1
        - file1...
    -category2
        - file1...
    ...
"""
function fromFolder(path::String, imageSize = 64::Int64)
    @info path, imageSize
    categories = readdir(path)
    total_files =
        collect(Iterators.flatten([add_path(x)[1] for x in categories]))
    total_categories =
        collect(Iterators.flatten([add_path(x)[2] for x in categories]))
    distrib, max_dis = classDistribution(total_categories)

    indices_repeat = indexin(unique(total_categories), total_categories)
    # oversample
    total_add = max_dis .- distrib # get the differences to oversample
    oversample = false
    if sum(total_add) > 100
        @info "Oversampling"
        images = zeros((
            imageSize,
            imageSize,
            3,
            size(max_dis * length(unique(total_categories)), 1),
        ))
        oversample = true
        oversample_index = length(y) - sum(total_add)# keep a track of indices from the back
    else
        @info "No need to oversample"
        images = zeros((imageSize, imageSize, 3, size(total_categories, 1)))
        oversample = false
    end

    Threads.@threads for idx in collect(1:size(total_files, 1))
        img = channelview(imresize(
            load(total_files[idx]),
            (imageSize, imageSize),
        ))
        img = convert(Array{Float64}, img)
        images[:, :, :, idx] = permutedims(img, (2, 3, 1))
        #         @info oversample
        if oversample == true

            if idx in indices_repeat
                labelrep = findfirst(x -> x == idx, indices_repeat) # index in the repeated list
                to_repeat = total_add[labelrep] # no of times to repeat
                total_categories = vcat(
                    total_categories,
                    fill(total_categories[indices_repeat[labelrep]], to_repeat),
                )
                Threads.@threads for idx2 in collect(oversample_index:to_repeat)
                    images[:, :, :, idx2] =
                        images[:, :, :, indices_repeat[labelrep]]

                end


            end
        end
    end


    @info "Done loading images"

    return images, total_categories



end

# %% codecell
#export
X, y = fromFolder(path, 64);
# %% codecell
size(X), size(y)
# %% markdown
# ## Splitting
# %% codecell
at = 0.7
n = length(y)
idx = shuffle(1:n)
train_idx = view(idx, 1:floor(Int, at * n));
test_idx = view(idx, (floor(Int, at * n)+1):n);
# %% codecell
ytrain, ytest = y[train_idx, :], y[test_idx, :]
Xtrain, Xtest = X[:, :, :, train_idx], X[:, :, :, test_idx]
@info length(ytrain), length(ytest)
@info length(Xtrain), length(Xtest)
# %% codecell
#export
function splitter(pct_split = 0.7::Float16)
    """
    Splits into train/test by pct_split%
    """
    n = length(y)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, pct_split * n))
    test_idx = view(idx, (floor(Int, pct_split * n)+1):n)
    ytrain, ytest = y[train_idx, :], y[test_idx, :]
    Xtrain, Xtest = X[:, :, :, train_idx], X[:, :, :, test_idx]
    return Xtrain, ytrain, Xtest, ytest
end
# %% codecell
#export
Xtrain, ytrain, Xtest, ytest = splitter(0.8);
# %% markdown
# ## Linear
# %% codecell
using Zygote
# %% codecell
W = rand(2, 5)
b = rand(2)
# %% codecell
Dense(x) = W * x .+ b
# %% codecell
function loss(x, y)
    ŷ = Dense(x)
    sum((y - ŷ) .^ 2)
end
# %% codecell
x, y = rand(5), rand(2)
# loss(x,y)
# %% codecell
α = 0.1
# %% codecell
x, y = rand(100), rand(2)
W = rand(2, 100)
b = rand(2)
# %% codecell
W = zeros(2, 100)
# %% codecell
for a in collect(1:50)
    gs = gradient(() -> loss(x, y), Params([W, b]))
    W̄ = gs[W]
    W .= α .* W̄
    ŷ = W * x .+ b
    @info sum((y - ŷ) .^ 2)
end
# %% markdown
# # Initialization
# - Zero Initialization: set all weights to 0
# - Normal Initialization: set all weights to random small numbers
# - Lecun Initialization: normalize variance
# - Xavier Intialization (glorot init)
# - Kaiming Initialization (he init)
# %% markdown
# ## Lecun
# It draws samples from a truncated normal distribution centered on 0 with stddev <- sqrt(1 / fan_in) where fan_in is the number of input units in the weight tensor..
# %% codecell
using Distributions
# %% codecell
#export
lecun_normal(fan_in) = return Distributions.Normal(0, sqrt(1 / fan_in))
# %% codecell
W = rand(lecun_normal(2), 2, 100)
b = rand(lecun_normal(2), 2)
# %% markdown
# ## Xavier Normal
# It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
# %% codecell
#export
xavier_normal(fan_in, fan_out) =
    return Distributions.Normal(0, sqrt(2 / (fan_in + fan_out)))
# %% codecell
W = rand(xavier_normal(2, 100), 2, 100)
b = rand(xavier_normal(2, 2), 2)
# %% markdown
# # Xavier Uniform
# It draws samples from a uniform distribution within -limit, limit where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
# %% codecell
#export
function xavier_uniform(fan_in, fan_out)
    limit = sqrt(6 / (fan_in + fan_out))
    return Distributions.Uniform(-limit, limit)
end
# %% codecell
W = rand(xavier_uniform(2, 100), 2, 100)
b = rand(xavier_uniform(2, 2), 2)
# %% markdown
# ## He Normal
# It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.
# %% codecell
#export
he_normal(fan_in) = return Distributions.Normal(0, sqrt(2 / (fan_in)))
# %% codecell
W = rand(he_normal(2), 2, 100)
b = rand(he_normal(2), 2)
# %% markdown
# # He Uniform
# It draws samples from a uniform distribution within -limit, limit where limit is sqrt(6 / fan_in) where fan_in is the number of input units in the weight tensor.
# %% codecell
#export
function he_uniform(fan_in)
    limit = sqrt(6 / (fan_in))
    return Distributions.Uniform(-limit, limit)
end
# %% codecell
W = rand(he_uniform(2), 2, 100)
b = rand(he_uniform(2), 2)
# %% markdown
# # Batching
# %% markdown
# ## One hot
# %% codecell
ytrain
# %% codecell
labels = unique(ytrain);
encodedlabels = Dict(labels .=> collect(1:length(labels)))
# %% markdown
## One cold
# %% codecell
ytrain
# %% codecell
function onecold(y_enc)
    labels = unique(y_enc)
    encodedlabels = Dict(labels .=> collect(1:length(labels)))
    global ytrain
    for a in keys(encodedlabels)
        ytrain = replace(ytrain, a=>encodedlabels[a])
    end
end
# %% codecell
onecold(ytrain)

# %% codecell
ytrain
# %% codecell
# export
function testGen(c::Channel)
    for n=1:4
        put!(c,n);
    end
    put!(c,"stop");
end
# %% codecell
test_yi = Channel(testGen);

# %%
take!(test_yi)

## Batch generator
# %%
#export
function datagen(c::Channel)
    global rep_len
    for n=1:bs:rep_len
        put!(c,n);
    end
    put!(c,"stop");
end

#%%
bs = 64
rep_len = length(ytest)
bunchData = Channel(datagen);
#%% markdown
## Actual batches
#%%
for i in collect(1:round(rep_len/bs)+1)
    current_index = take!(bunchData)
    try
        x_batch,y_batch = Xtest[:, :, :, current_index:current_index+bs-1],ytest[current_index:current_index+bs-1]
        @info size(x_batch),size(y_batch)
    catch e
        x_batch,y_batch = Xtest[:, :, :, rep_len-bs:rep_len],ytest[rep_len-bs:rep_len]
        @info size(x_batch),size(y_batch)
    end
end

#%%
## Training loop with SGD
bs = 64
rep_len = length(ytest)
bunchData = Channel(datagen);
W = rand(he_uniform(rep_len), rep_len, 100)
b = rand(he_uniform(rep_len), rep_len)

for i in collect(1:round(rep_len/bs)+1)
    current_index = take!(bunchData)
    try
        x_batch,y_batch = Xtest[:, :, :, current_index:current_index+bs-1],ytest[current_index:current_index+bs-1]
        @info size(x_batch),size(y_batch)
    catch e
        x_batch,y_batch = Xtest[:, :, :, rep_len-bs:rep_len],ytest[rep_len-bs:rep_len]
        @info size(x_batch),size(y_batch)
    end
end
