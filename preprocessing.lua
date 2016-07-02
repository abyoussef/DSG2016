require 'csvigo'
require 'image'
require 'xlua'
require 'nn'

local size = 32

local function LoadDataset(filename)
    print("Loading ", filename)
    local parsed = csvigo.load({path=filename, mode="query"})
    local dataset = parsed('all')
    local ndata = #dataset.Id
    local ret = {}
    ret.data = torch.Tensor(ndata, 3, size, size)
    ret.label = torch.IntTensor(ndata)
    ret.Id = torch.IntTensor(ndata)

    for k,v in ipairs(dataset.Id) do
        xlua.progress(k, ndata)
        ret.data[k] = image.scale(image.load('roof_images/' .. v .. '.jpg'), size, size)
        local label = tonumber(dataset.label[k])
        ret.label[k] = label
        ret.Id[k] = tonumber(dataset.Id[k])
    end

    print("Finished loading", filename)
    return ret
end

local function LoadAndAugmentDataset(filename)
    print("Loading", filename)
    local parsed = csvigo.load({path=filename, mode="query"})
    local dataset = parsed('all')
    local ndata = #dataset.Id
    local ret = {}
    ret.data = torch.Tensor(8 * ndata, 3, size, size)
    ret.label = torch.IntTensor(8 * ndata)

    for k,v in ipairs(dataset.Id) do
        xlua.progress(k, ndata)
        local i1 = image.scale(image.load('roof_images/' .. v .. '.jpg'), size, size)
        local i2 = image.hflip(i1)
        local i3 = image.vflip(i1)
        local i4 = image.vflip(i2)
        local label1 = tonumber(dataset.label[k])
        local label2 = label1

        if label1 == 1 then
            label2 = 2
        elseif label1 == 2 then
            label2 = 1
        end

        ret.data[8 * k - 7] = i1
        ret.label[8 * k - 7] = label1

        ret.data[8 * k - 6] = i2
        ret.label[8 * k - 6] = label1

        ret.data[8 * k - 5] = i3
        ret.label[8 * k - 5] = label1

        ret.data[8 * k - 4] = i4
        ret.label[8 * k - 4] = label1

        ret.data[8 * k - 3] = i1:transpose(2,3)
        ret.label[8 * k - 3] = label2

        ret.data[8 * k - 2] = i2:transpose(2,3)
        ret.label[8 * k - 2] = label2

        ret.data[8 * k - 1] = i3:transpose(2,3)
        ret.label[8 * k - 1] = label2

        ret.data[8 * k] = i4:transpose(2,3)
        ret.label[8 * k] = label2
    end

    print("Finished loading", filename)
    return ret
end

trainData = LoadAndAugmentDataset('id_train.csv')
local ntrain = trainData.data:size(1)
trainData.Id = nil
testData = LoadDataset('sample_submission4.csv')
local ntest = testData.data:size(1)
testData.label = nil
normalizationType = 'rgb'

if normalizationType == 'rgb' then
    mean = {}
    stdv = {}

    for i=1,3 do
        mean[i] = trainData.data[{ {}, {i}, {}, {} }]:mean()
        trainData.data[{ {}, {i}, {}, {} }]:add(-mean[i])

        stdv[i] = trainData.data[{ {}, {i}, {}, {} }]:std()
        trainData.data[{ {}, {i}, {}, {} }]:div(stdv[i])
    end

    for i = 1,3 do
        testData.data[{ {}, {i}, {}, {} }]:add(-mean[i])
        testData.data[{ {}, {i}, {}, {} }]:div(stdv[i])
    end

    torch.save('dsg_train_mean.t7', mean)
    torch.save('dsg_train_stdv.t7', stdv)
elseif normalizationType == 'yuv' then
    -- taken from: https://github.com/szagoruyko/cifar.torch/blob/master/provider.lua
    local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
    for i = 1,ntrain do
       xlua.progress(i, ntrain)
       -- rgb -> yuv
       local rgb = trainData.data[i]
       local yuv = image.rgb2yuv(rgb)
       -- normalize y locally:
       yuv[1] = normalization(yuv[{{1}}])
       trainData.data[i] = yuv
    end
    -- normalize u globally:
    local mean_u = trainData.data:select(2,2):mean()
    local std_u = trainData.data:select(2,2):std()
    trainData.data:select(2,2):add(-mean_u)
    trainData.data:select(2,2):div(std_u)
    -- normalize v globally:
    local mean_v = trainData.data:select(2,3):mean()
    local std_v = trainData.data:select(2,3):std()
    trainData.data:select(2,3):add(-mean_v)
    trainData.data:select(2,3):div(std_v)

    trainData.mean_u = mean_u
    trainData.std_u = std_u
    trainData.mean_v = mean_v
    trainData.std_v = std_v

    for i = 1,ntest do
      xlua.progress(i, ntest)
       -- rgb -> yuv
       local rgb = testData.data[i]
       local yuv = image.rgb2yuv(rgb)
       -- normalize y locally:
       yuv[{1}] = normalization(yuv[{{1}}])
       testData.data[i] = yuv
    end
    -- normalize u globally:
    testData.data:select(2,2):add(-mean_u)
    testData.data:select(2,2):div(std_u)
    -- normalize v globally:
    testData.data:select(2,3):add(-mean_v)
    testData.data:select(2,3):div(std_v)
end

torch.save('dsg_train.t7', trainData)
torch.save('dsg_test.t7', testData)
