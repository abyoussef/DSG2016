require 'torch'
require 'csvigo'
require 'image'
require 'nn'

local dsg_utils = {}

function dsg_utils.LoadDataset(filename)
    print("Loading dataset")
    local id_train = csvigo.load({path=filename, mode="query"})
    local dataset = id_train('all')
    local ndata = #dataset.Id
    dataset.data = torch.Tensor(ndata, 3, 32, 32)

    for k,v in ipairs(dataset.Id) do
        dataset.data[k] = image.scale(image.load('roof_images/' .. v .. '.jpg'), 32, 32)
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
    ret.data = torch.Tensor(4 * ndata, 3, 32, 32)
    ret.label = torch.IntTensor(4 * ndata)

    for k,v in ipairs(dataset.Id) do
        local i1 = image.scale(image.load('roof_images/' .. v .. '.jpg'), 32, 32)
        local i2 = image.hflip(i1)
        local i3 = image.vflip(i1)
        local i4 = image.vflip(i2)
        local label1 = tonumber(dataset.label[k])

        ret.data[4 * k - 3] = i1
        ret.label[4 * k - 3] = torch.Tensor(1):fill(label1)

        ret.data[4 * k - 2] = i2
        ret.label[4 * k - 2] = torch.Tensor(1):fill(label1)

        ret.data[4 * k - 1] = i3
        ret.label[4 * k - 1] = torch.Tensor(1):fill(label1)

        ret.data[4 * k] = i4
        ret.label[4 * k] = torch.Tensor(1):fill(label1)
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

function dsg_utils.TrainNet(trainset, cuda_flag)
    setmetatable(trainset,
        {__index = function(t,i) return {t.data[i], t.label[i]} end}
    );
    function trainset:size()
        return self.data:size(1)
    end

    -- Create network
    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3,6,5,5)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel
    net:add(nn.ReLU()) -- non-linearity
    net:add(nn.SpatialMaxPooling(2,2,2,2)) -- A max-pooling operation that look at 2x2 windows and finds the max
    net:add(nn.SpatialConvolution(6,16,5,5))
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(2,2,2,2))
    net:add(nn.View(16*5*5)) -- reshapes from a 3D tensor of 16x5x5 into a 1D tensor of 16*5*5
    net:add(nn.Linear(16*5*5,120)) -- fully connected layer
    net:add(nn.ReLU())
    net:add(nn.Linear(120,84))
    net:add(nn.ReLU())
    net:add(nn.Linear(84,4)) -- 10 is the number of outputs of the network
    net:add(nn.LogSoftMax()) -- converts the output to a log-probability. Useful for classification problems

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
    for i=1,nTest do
        local prediction = net:forward(testset.data[i])
        local confidences, indices = torch.sort(prediction, true) -- sort in descending order
        if indices[1] == testset.label[i] then
            correct = correct + 1
        end
    end
    local percentage = correct * 100.0 / nTest
    print(correct .. "/" .. nTest .. " " .. percentage)
    return percentage
end

local function AssignInterval(n, K, i)
    return n * (i - 1) / K + 1, n * i / K
end

function dsg_utils.KFoldedCV(trainset, K, cuda_flag)
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
        train.data = torch.Tensor(nTrain, 3, 32, 32)
        train.label = torch.IntTensor(nTrain)
        local valid = {}
        valid.data = torch.Tensor(nValid, 3, 32, 32)
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
        net = dsg_utils.TrainNet(train)

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
