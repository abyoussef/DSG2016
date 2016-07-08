require 'csvigo'
require 'image'
require 'xlua'
require 'nn'
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

    local cont = {0, 0, 0, 0}
    for i = 1,ndata do
        local cur = tonumber(dataset.label[i])
        cont[cur] = cont[cur] + 1
    end
    print(cont)

    local ret = {}
    local nret = 8 * (cont[1] + cont[2]) + 16 * cont[3] + 16 * cont[4]
    ret.data = torch.Tensor(nret, 3, size, size)
    ret.label = torch.IntTensor(nret)
    pos = 1

    for k,v in ipairs(dataset.Id) do
        xlua.progress(k, ndata)
        local i0 = image.load('roof_images/' .. v .. '.jpg')
        local aux = {}
        aux[1] = image.scale(i0, size, size)
        aux[2] = image.hflip(aux[1])
        aux[3] = image.vflip(aux[1])
        aux[4] = image.vflip(aux[2])
        aux[5] = aux[1]:transpose(2,3)
        aux[6] = aux[2]:transpose(2,3)
        aux[7] = aux[3]:transpose(2,3)
        aux[8] = aux[4]:transpose(2,3)
        local label1 = tonumber(dataset.label[k])
        local label2 = label1

        if label1 == 1 then
            label2 = 2
        elseif label1 == 2 then
            label2 = 1
        end

        for i = 1,8 do
            ret.data[pos] = aux[i]
            if i <= 4 then
                ret.label[pos] = label1
            else
                ret.label[pos] = label2
            end
            pos = pos + 1
        end

        if label1 == 3 or label1 == 4 then
            local h,w = i0:size(1),i0:size(2)
            local aux2 = {}

            if h > w then
                aux2[1] = image.crop(i0, 'tl', math.ceil(2 * h / 3), w)
            else
                aux2[1] = image.crop(i0, 'tl', h, math.ceil(2 * w / 3))
            end

            aux2[1] = image.scale(aux2[1], size, size)
            aux2[2] = image.hflip(aux2[1])
            aux2[3] = image.vflip(aux2[1])
            aux2[4] = image.vflip(aux2[2])
            aux2[5] = aux2[1]:transpose(2,3)
            aux2[6] = aux2[2]:transpose(2,3)
            aux2[7] = aux2[3]:transpose(2,3)
            aux2[8] = aux2[4]:transpose(2,3)

            for i = 1,8 do
                ret.data[pos] = aux2[i]
                ret.label[pos] = label1
                pos = pos + 1
            end
        end
    end

    print("Finished loading", filename)

    cont = {0, 0, 0, 0}
    for i = 1,pos - 1 do
        local cur =  ret.label[i]
        cont[cur] = cont[cur] + 1
    end
    print(cont)
    assert(pos - 1 == nret)

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

torch.save('dsg_train_augmented.t7', trainData)
torch.save('dsg_test.t7', testData)
torch.save('dsg_params.t7', params)
