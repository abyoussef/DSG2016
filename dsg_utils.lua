require 'torch'
require 'csvigo'
require 'image'
require 'nn'
require 'optim'
require 'xlua'
require 'cunn'
dsg_nets = require 'dsg_nets'

local dsg_utils = {}
local size = 32

function dsg_utils.PreprocessDataset(filename, output)
    print("Loading dataset")
    local parsed = csvigo.load({path=filename, mode="query"})
    local dataset = parsed('all')
    local ndata = #dataset.Id
    local ret = {}
    ret.data = torch.Tensor(ndata, 3, size, size)
    ret.label = torch.IntTensor(ndata)
    ret.Id = torch.IntTensor(ndata)

    for k,v in ipairs(dataset.Id) do
        ret.data[k] = image.scale(image.load('roof_images/' .. v .. '.jpg'), size, size)
        local label = tonumber(dataset.label[k])
        ret.label[k] = label
        ret.Id[k] = tonumber(dataset.Id[k])
    end

    print("Finished loading dataset")
    torch.save(output, ret)
end

function dsg_utils.PreprocessAndAugmentDataset(filename, output, format)
    print("Loading dataset")
    local parsed = csvigo.load({path=filename, mode="query"})
    local dataset = parsed('all')
    local ndata = #dataset.Id
    local ret = {}
    ret.data = torch.Tensor(8 * ndata, 3, size, size)
    ret.label = torch.IntTensor(8 * ndata)

    for k,v in ipairs(dataset.Id) do
        local i1 = image.scale(image.load('roof_images/' .. v .. '.jpg'), size, size)

        if format == 'yuv' then
            i1 = image.rgb2yuv(i1)
        end

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
    torch.save(output, ret)
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

function dsg_utils.TrainNet(trainset, fnet, w_init_name, cuda_flag)
    setmetatable(trainset,
        {__index = function(t,i) return {t.data[i], t.label[i]} end}
    );
    function trainset:size()
        return self.data:size(1)
    end

    -- Create network
    net = fnet()
    if w_init_name then
        dsg_nets.w_init(net, w_init_name)
    end
    net:training()

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

-- based on: https://github.com/szagoruyko/cifar.torch/blob/master/train.lua
function dsg_utils.TrainWithMinibatch(trainset, fnet, w_init_name, params)
    -- Create network
    net = fnet()
    if w_init_name then
        dsg_nets.w_init(net, w_init_name)
    end
    net:training()
    parameters, gradParameters = net:getParameters()

    -- Loss function
    criterion = nn.ClassNLLCriterion()

    -- Minibatch targets
    local targets = torch.FloatTensor(params.batchSize)

    -- Using CUDA
    if params.cuda then
        net = net:cuda()
        criterion = criterion:cuda()
        trainset.data = trainset.data:cuda()
        trainset.label = trainset.label:cuda()
        targets = targets:cuda()
    end

    local n_train = trainset.label:size(1)
    print("n_train = " .. n_train)
    optimState = {
      learningRate = 1,
      weightDecay = 0.0005,
      momentum = 0.9,
      learningRateDecay = 1e-7,
    }

    for epoch=1,params.nEpochs do
        print("epoch #" .. epoch .. "(of " .. params.nEpochs .. ") [batchSize = " .. params.batchSize .. "]")
        if epoch % params.epochLearningStep == 0 then
            optimState.learningRate = optimState.learningRate / 2
        end

        local indices = torch.randperm(n_train):long():split(params.batchSize)
        -- remove last element so that all the batches have equal size
        -- indices[#indices] = nil
        local totalerror = 0
        local err
        local time = sys.clock()
        local batch_pos = 1

        for t = 1, n_train, params.batchSize do
            xlua.progress(t, n_train)

            if (t + params.batchSize) >= n_train then
                break
            end

            local v = indices[batch_pos]
            batch_pos = batch_pos + 1

            local inputs = trainset.data:index(1,v)
            targets:copy(trainset.label:index(1,v))

            local feval = function(x)
              if x ~= parameters then parameters:copy(x) end
              gradParameters:zero()

              local outputs = net:forward(inputs)
              local f = criterion:forward(outputs, targets)
              local df_do = criterion:backward(outputs, targets)
              net:backward(inputs, df_do)

              return f, gradParameters
            end

            _, err = optim.sgd(feval, parameters, optimState)
            totalerror = totalerror + err
        end

        xlua.progress(n_train, n_train)

        time = sys.clock() - time
        print("time for epoch = " .. (time*1000) .. 'ms')
        print("Train error =", totalerror)

        if epoch % params.epochSaveStep == 0 then
            torch.save('/home/mario/Dropbox/DSG/' .. params.modelName .. '_epoch_' .. epoch .. '.net', net, 'ascii')
        end
    end

    return net
end

local function TestNet(net, testset, cuda_flag)
    local nTest = testset.data:size(1)

    -- Using CUDA
    if cuda_flag then
        testset.data = testset.data:cuda()
        testset.label = testset.label:cuda()
    end
    net:evaluate()

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
