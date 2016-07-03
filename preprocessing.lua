require 'csvigo'
require 'image'
require 'xlua'
require 'nn'
require 'unsup'
preprocessing = require 'preprocessing_nagadomi'

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

params = preprocessing.preprocessing(trainData.data)
preprocessing.preprocessing(testData.data, params)

torch.save('dsg_params.t7', params)
