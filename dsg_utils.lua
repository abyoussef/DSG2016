require 'torch'
require 'csvigo'
require 'image'
require 'nn'

local dsg_utils = {}
local size = 32

function dsg_utils.LoadDataset(filename)
    print("Loading dataset")
    local id_train = csvigo.load({path=filename, mode="query"})
    local dataset = id_train('all')
    local ndata = #dataset.Id
    dataset.data = torch.Tensor(ndata, 3, size, size)

    for k,v in ipairs(dataset.Id) do
        dataset.data[k] = image.scale(image.load('roof_images/' .. v .. '.jpg'), size, size)
        local label = tonumber(dataset.label[k])
        dataset.label[k] = torch.Tensor(1):fill(label)
    end

    merge_label = nn.Sequential()
                    :add(nn.JoinTable(1))
                    :add(nn.View(-1, 1))

    dataset.label = merge_label:forward(dataset.label)
    print("Finished loading dataset")
    return dataset
end

function dsg_utils.LoadAndAugmentDataset(filename)
    print("Loading dataset")
    local parsed = csvigo.load({path=filename, mode="query"})
    local dataset = parsed('all')
    local ndata = #dataset.Id
    local ret = {}
    ret.data = torch.Tensor(8 * ndata, 3, size, size)
    ret.label = torch.IntTensor(8 * ndata)

    for k,v in ipairs(dataset.Id) do
        local i1 = image.scale(image.load('roof_images/' .. v .. '.jpg'), size, size)
        local i2 = image.hflip(i1)
        local i3 = image.vflip(i1)
        local i4 = image.vflip(i2)
        local label1 = tonumber(dataset.label[k])
        local label2

        ret.data[8 * k - 7] = i1
        ret.label[8 * k - 7] = label1

        ret.data[8 * k - 6] = i2
        ret.label[8 * k - 6] = label1

        ret.data[8 * k - 5] = i3
        ret.label[8 * k - 5] = label1

        ret.data[8 * k - 4] = i4
        ret.label[8 * k - 4] = label1

        if label1 == 1 then
            label2 = 2
        elseif label1 == 2 then
            label2 = 1
        else
            label2 = label1
        end

        ret.data[8 * k - 3] = i1:transpose(2,3)
        ret.label[8 * k - 3] = label2

        ret.data[8 * k - 2] = i2:transpose(2,3)
        ret.label[8 * k - 2] = label2

        ret.data[8 * k - 1] = i3:transpose(2,3)
        ret.label[8 * k - 1] = label2

        ret.data[8 * k] = i4:transpose(2,3)
        ret.label[8 * k] = label2
    end

    print("Finished loading dataset")
    return ret
end

function dsg_utils.Normalize(trainset)
    mean = {}
    stdv = {}

    for i=1,3 do
        mean[i] = trainset.data[{ {}, {i}, {}, {} }]:mean()
        trainset.data[{ {}, {i}, {}, {} }]:add(-mean[i])

        stdv[i] = trainset.data[{ {}, {i}, {}, {} }]:std()
        trainset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
    end

    return mean,stdv
end

function dsg_utils.TrainNet(trainset, fnet, cuda_flag)
    setmetatable(trainset,
        {__index = function(t,i) return {t.data[i], t.label[i]} end}
    );
    function trainset:size()
        return self.data:size(1)
    end

    -- Create network
    net = fnet()

    -- Loss function
    criterion = nn.ClassNLLCriterion()

    -- Using CUDA
    if cuda_flag then
        net = net:cuda()
        criterion = criterion:cuda()
        trainset.data = trainset.data:cuda()
        trainset.label = trainset.label:cuda()
    end

    -- Train network
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.001
    trainer.maxIteration = 15
    trainer:train(trainset)

    return net
end

local function TestNet(net, testset, cuda_flag)
    local nTest = testset.data:size(1)

    -- Using CUDA

    if cuda_flag then
        testset.data = testset.data:cuda()
    end

    local correct = 0
    local correctByClass = {0,0,0,0}
    local totalByClass = {0,0,0,0}
    for i=1,nTest do
        local prediction = net:forward(testset.data[i])
        local confidences, indices = torch.sort(prediction, true) -- sort in descending order
        if indices[1] == testset.label[i] then
            correct = correct + 1
            correctByClass[ testset.label[i] ] = correctByClass[ testset.label[i] ] + 1
        end
        totalByClass[ testset.label[i] ] = totalByClass[ testset.label[i] ] + 1
    end

    local percentage = correct * 100.0 / nTest
    print(correct .. "/" .. nTest .. " " .. percentage)
    for i=1,4 do
        local percentagei = correctByClass[i] * 100.0 / totalByClass[i]
        print("class " .. i .. " : " .. correctByClass[i] .. "/" .. totalByClass[i] .. " " .. percentagei)
    end

    return percentage
end

local function AssignInterval(n, K, i)
    return n * (i - 1) / K + 1, n * i / K
end

function dsg_utils.KFoldedCV(trainset, K, fnet, cuda_flag)
    local nTrainset = trainset.data:size(1)
    local shuffledIndices = torch.randperm(nTrainset)
    local totalPercentage = 0
    local minPercentage = 100
    local maxPercentage = 0
    local net
    local mean
    local std

    for i=1,K do
        print("Iteration", i)
        local nTrain = 0
        local nValid = 0
        for j = 1,K do
            s,e = AssignInterval(nTrainset, K, j)
            if j == i then
                nValid = e - s + 1
            else
                nTrain = nTrain + e - s + 1
            end
        end
        --print("nTrain = " .. nTrain .. ", nValid = " .. nValid)

        local train = {}
        train.data = torch.Tensor(nTrain, 3, size, size)
        train.label = torch.IntTensor(nTrain)
        local valid = {}
        valid.data = torch.Tensor(nValid, 3, size, size)
        valid.label = torch.IntTensor(nValid)

        local posTrain = 0
        local posValid = 0
        for j = 1,K do
            s,e = AssignInterval(nTrainset, K, j)
            if j == i then
                for pos=s,e do
                    posValid = posValid + 1
                    valid.data[posValid] = trainset.data[posValid]
                    valid.label[posValid] = trainset.label[posValid]
                end
            else
                for pos=s,e do
                    posTrain = posTrain + 1
                    train.data[posTrain] = trainset.data[posTrain]
                    train.label[posTrain] = trainset.label[posTrain]
                end
            end
        end

        mean, std = dsg_utils.Normalize(train)
        net = dsg_utils.TrainNet(train, fnet, cuda_flag)

        for i = 1,3 do
            valid.data[{ {}, {i}, {}, {} }]:add(-mean[i])
            valid.data[{ {}, {i}, {}, {} }]:div(stdv[i])
        end

        local percentage = TestNet(net, valid, cuda_flag)
        totalPercentage = totalPercentage + percentage
        minPercentage = math.min(minPercentage, percentage)
        maxPercentage = math.max(maxPercentage, percentage)
    end

    print("Average percentage = " .. totalPercentage / K
        .. ", min percentage = " .. minPercentage
        .. ", max percantage = " .. maxPercentage)
    return net, mean, std
end

return dsg_utils
