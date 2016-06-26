require 'nn'

local dsg_nets = {}

function dsg_nets.Lenet()
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
    return net
end

return dsg_nets
