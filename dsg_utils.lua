require 'torch'
require 'nn'
require 'optim'
require 'xlua'
require 'cunn'
dsg_nets = require 'dsg_nets'

local dsg_utils = {}
local size = 32

function dsg_utils.TrainNet(trainset, fnet, params)
    setmetatable(trainset,
        {__index = function(t,i) return {t.data[i], t.label[i]} end}
    );
    function trainset:size()
        return self.data:size(1)
    end

    -- Create network
    net = fnet()
    if params.init ~= 'none' then
        dsg_nets.w_init(net, params.init)
    end
    net:training()

    -- Loss function
    local criterion = nn.ClassNLLCriterion()

    if params.float then
        net = net:float()
        criterion = criterion:float()
        trainset.data = trainset.data:float()
    end

    -- Using CUDA
    if params.cuda then
        net = net:cuda()
        criterion = criterion:cuda()
        trainset.data = trainset.data:cuda()
        trainset.label = trainset.label:cuda()
    end

    -- Train network
    local trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.001
    trainer.maxIteration = 50
    trainer.learningRateDecay = 1e-7

    local tic
    local hookIteration = function(sgd, it, currentError)
        local time = torch.toc(tic)
        print("Iteration " .. it .. ", Time : " .. time)
        if it % 5 == 0 then
            torch.save(params.modelName .. '_iteration_' .. it .. '.net', sgd.module)
        end
        tic = torch.tic()
    end
    trainer.hookIteration = hookIteration

    tic = torch.tic()
    trainer:train(trainset)

    return net
end

local function TestWithMiniBatch(testset, net, params, epoch)
    local ntest = testset.data:size(1)
    local file = assert(io.open('/home/mario/Dropbox/DSG/' .. params.modelName .. '_submission_epoch_' .. epoch .. '.csv', "w"))
    file:write("Id,label\n")
    net:evaluate()

    for i=1,ntest,params.batchSize do
        xlua.progress(i, ntest)

        local bs = math.min(params.batchSize, ntest + 1 - i)
        local inputs = torch.FloatTensor(bs, 3, size, size)
        inputs:copy(testset.data:narrow(1, i, bs))
        if params.cuda then
            inputs = inputs:cuda()
        end

        local prediction = net:forward(inputs)
        prediction:exp()

        for j=1,bs do
            local confidences, indices = torch.sort(prediction[j], true) -- sort in descending order
            file:write(testset.Id[i + j - 1] .. "," .. indices[1] .. "\n")
        end
    end
    xlua.progress(ntest, ntest)
    file:close()
end

-- based on: https://github.com/szagoruyko/cifar.torch/blob/master/train.lua
function dsg_utils.TrainWithMinibatch(trainset, testset, fnet, params)
    -- Create network
    net = fnet()
    if params.init ~= 'none' then
        dsg_nets.w_init(net, params.init)
    end
    net:training()

    -- Loss function
    criterion = nn.ClassNLLCriterion()

    -- Minibatch targets
    local inputs = torch.FloatTensor(params.batchSize, 3, size, size)
    local targets = torch.FloatTensor(params.batchSize)

    -- Using CUDA
    if params.cuda then
        net = net:cuda()
        criterion = criterion:cuda()
        inputs = inputs:cuda()
        targets = targets:cuda()
    end

    parameters, gradParameters = net:getParameters()

    local n_train = trainset.label:size(1)
    print("n_train = " .. n_train)
    optimState = {
      learningRate = 0.0025,--0.5,
      weightDecay = 0.0001,
      momentum = 0.9,
      --learningRateDecay = 1e-7,
    }

    for epoch=1,params.nEpochs do
        print("epoch #" .. epoch .. "(of " .. params.nEpochs .. ") [batchSize = " .. params.batchSize .. "]")
        --if epoch % params.epochLearningStep == 0 then
        --    optimState.learningRate = optimState.learningRate / 2
        --end
        optimState.learningRate = 0.1 * math.pow(0.9, epoch - 1)

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

            inputs:copy(trainset.data:index(1,v))
            targets:copy(trainset.label:index(1,v))

            local feval = function(x)
              if x ~= parameters then parameters:copy(x) end
              gradParameters:zero()

              local outputs = net:forward(inputs)
              err = criterion:forward(outputs, targets)
              local df_do = criterion:backward(outputs, targets)
              net:backward(inputs, df_do)

              return f, gradParameters
            end

            optim.sgd(feval, parameters, optimState)
            totalerror = totalerror + err
        end

        xlua.progress(n_train, n_train)

        time = sys.clock() - time
        print("time for epoch = " .. time .. 's')
        print("Train error =", totalerror)

        if epoch % params.epochSaveStep == 0 then
            torch.save(params.modelName .. '_epoch_' .. epoch .. '.net', net)
            TestWithMiniBatch(testset, net, params, epoch)
            net:training()
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
