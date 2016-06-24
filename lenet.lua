require 'torch'
require 'csvigo'
require 'image'
require 'nn'

local cuda_flag = false

if cuda_flag then
    require 'cunn'
end

-- Load training set

local id_train = csvigo.load({path="id_train.csv", mode="query"})
trainset = id_train('all')
local ntrain = #trainset.Id
print("ntrain =", ntrain)

trainset.data = {}
for k,v in ipairs(trainset.Id) do
    trainset.data[k] = image.scale(image.load('roof_images/' .. v .. '.jpg'), 32, 32)
    local label = tonumber(trainset.label[k])
    trainset.label[k] = torch.Tensor(1):fill(label)
end

merge_data = nn.Sequential()
            :add(nn.JoinTable(1))
            :add(nn.View(-1, 3, 32, 32))

merge_label = nn.Sequential()
                :add(nn.JoinTable(1))
                :add(nn.View(-1, 1))

trainset.data  = merge_data:forward(trainset.data)
trainset.label  = merge_label:forward(trainset.label)

print("Finished loading training set")

--for i = 1,math.min(ntrain,5) do
--    local pos = math.random(ntrain)
--    image.display{image=trainset.data[pos], legend=trainset.label[pos]}
--end

setmetatable(trainset,
    {__index = function(t,i) return {t.data[i], t.label[i]} end}
);

function trainset:size()
    return self.data:size(1)
end

-- Normalize

mean = {}
stdv = {}

for i=1,3 do
    mean[i] = trainset.data[{ {}, {i}, {}, {} }]:mean()
    trainset.data[{ {}, {i}, {}, {} }]:add(-mean[i])

    stdv[i] = trainset.data[{ {}, {i}, {}, {} }]:std()
    trainset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
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

-- Train the neural network

print("Training")
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 10
trainer:train(trainset)

-- Load test set

local id_test = csvigo.load({path="sample_submission4.csv", mode="query"})
testset = id_test('all')
local ntest = #testset.Id
print("ntest =", ntest)

testset.data = {}
for k,v in ipairs(testset.Id) do
    testset.data[k] = image.scale(image.load('roof_images/' .. v .. '.jpg'), 32, 32)
    --local label = tonumber(testset.label[k])
    --testset.label[k] = torch.Tensor(1):fill(label)
end

testset.data = merge_data:forward(testset.data)

print("Finished loading test set")

-- Using CUDA

if cuda_flag then
    testset.data = testset.data:cuda()
end

-- Test the network

print("Testing")

for i = 1,3 do
    testset.data[{ {}, {i}, {}, {} }]:add(-mean[i])
    testset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end

classes = {"North-South", "East-West", "Flat roof" , "Other"}

rtest = math.random(ntest)
predicted = net:forward(testset.data[rtest])
predicted:exp() -- convert log-probability to probability

for i = 1,predicted:size(1) do
    print(classes[i], predicted[i])
end

image.display(testset.data[rtest])

local filename = 'submission.csv'
local file = assert(io.open(filename, "w"))
file:write("Id,label\n")

for i=1,testset.data:size(1) do
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true) -- sort in descending order

    file:write(testset.Id[i] .. "," .. indices[1] .. "\n")
end

file:close()