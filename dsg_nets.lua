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

-- http://torch.ch/blog/2015/07/30/cifar.html
function dsg_nets.VggBNDrop()
    local vgg = nn.Sequential()
    -- building block
    local function ConvBNReLU(nInputPlane, nOutputPlane)
      vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
      --vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
      vgg:add(nn.ReLU(true))
      return vgg
    end
    -- Will use "ceil" MaxPooling because we want to save as much feature space as we can
    local MaxPooling = nn.SpatialMaxPooling

    ConvBNReLU(3,64):add(nn.Dropout(0.3))
    ConvBNReLU(64,64)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(64,128):add(nn.Dropout(0.4))
    ConvBNReLU(128,128)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(128,256):add(nn.Dropout(0.4))
    ConvBNReLU(256,256):add(nn.Dropout(0.4))
    ConvBNReLU(256,256)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(256,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    vgg:add(nn.View(512))
    vgg:add(nn.Dropout(0.5))
    vgg:add(nn.Linear(512,512))
    --vgg:add(nn.BatchNormalization(512))
    vgg:add(nn.ReLU(true))
    vgg:add(nn.Dropout(0.5))
    vgg:add(nn.Linear(512,4))
    vgg:add(nn.LogSoftMax())
    return vgg
end

-- http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
function dsg_nets.KerasNet()
    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3,32,3,3))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(2,2))
    net:add(nn.SpatialConvolution(32,32,3,3))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(2,2))
    net:add(nn.SpatialConvolution(32,64,3,3))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(2,2))
    net:add(nn.View(64*2*2))
    net:add(nn.Dropout(0.5))
    net:add(nn.Linear(64*2*2,64))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.5))
    net:add(nn.Linear(64,4))
    net:add(nn.LogSoftMax())
    return net
end

local function w_init_heuristic(fan_in, fan_out)
    return math.sqrt(1 / (3 * fan_in))
end

local function w_init_xavier(fan_in, fan_out)
    return math.sqrt(2 / (fan_in + fan_out))
end

local function w_init_xavier_caffe(fan_in, fan_out)
    return math.sqrt(1 / fan_in)
end

local function w_init_kaiming(fan_in, fan_out)
    return math.sqrt(4 / (fan_in + fan_out))
end

-- http://arxiv.org/pdf/1502.01852v1.pdf
local function w_init_msr(fan_in, fan_out)
    return math.sqrt(2 / fan_out)
end

-- https://github.com/e-lab/torch-toolbox/tree/master/Weight-init
function dsg_nets.w_init(net, arg)
    local method = nInputPlane
    if arg == 'heuristic' then
        method = w_init_heuristic
    elseif arg == 'xavier' then
        method = w_init_xavier
    elseif arg == 'xavier_caffe' then
        method = w_init_xavier_caffe
    elseif arg =='kaiming' then
        method = w_init_kaiming
    elseif arg == 'msr' then
        method = w_init_msr
    else
        assert(false)
    end

    for k,v in pairs(net:findModules('nn.SpatialConvolution')) do
        local fan_in = v.kW * v.kH * v.nInputPlane
        local fan_out = v.kW * v.kH * v.nOutputPlane
        local n = v.kW * v.kH * v.nOutputPlane
        v:reset(method(fan_in, fan_out))

        if v.bias then
            v.bias:zero()
        end
    end
end

return dsg_nets
