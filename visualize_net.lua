require 'nn'
require 'image'
preprocessing = require 'preprocessing_nagadomi'

cmd = torch.CmdLine()
cmd:option('-modelName', 'model', 'name of the model')
cmd:option('-cuda', false, 'if true cast to cuda')
cmd:option('-float', false, 'if true cas to float')

opt = cmd:parse(arg or {})

size = 32
id = 169711900---3935637
path = 'visualization/'

img = image.load('roof_images/' .. id .. '.jpg')
image.save('visualization/0.jpg', img)

print('Original dimensions : (' .. img:size(1) .. ', ' .. img:size(2) .. ')')

img = image.scale(img, size, size)
image.save(path .. '0_resized.jpg', img)

if opt.float then
    img = img:float()
end

if opt.cuda then
    require 'cunn'
    img = img:cuda()
end

--params = torch.load('dsg_params.t7')
--preprocessing.preprocessing(img, params)

net = torch.load(opt.modelName .. '.net')
net:evaluate()

local function CheckModule(name)
    if name == 'nn.SpatialConvolution' then return true end
    if name == 'nn.ReLU' then return true end
    if name == 'nn.SpatialMaxPooling' then return true end
    return false
end

for i = 1,net:size() do
    local module = net:get(i)
    local name = torch.typename(module)
    img = module:forward(img)

    if CheckModule(name) then
        print(torch.typename(module))

        if name:sub(1,3) == 'nn.' then
            name = name:sub(4, name:len())
        end

        if img:dim() == 3 then
            local nw = math.min(30, img:size(1))
            local nh = math.ceil(img:size(1) / nw)
            local moduleOut = torch.Tensor(nh * img:size(2), nw * img:size(3))
            local moduleWeight
            local saveWeights = false
            if name == 'SpatialConvolution' and module.weight:size(2) == 3 then
                saveWeights = true
                moduleWeight = torch.Tensor(module.weight:size(2), nh * module.weight:size(3), nw * module.weight:size(4))
            end
            print(i, img:size(1), img:size(2), img:size(3))

            for j = 1,img:size(1) do
                local row = math.ceil(j / nw)
                local col = j - (row - 1) * nw

                local aux = moduleOut:narrow(1, 1 + (row - 1) * img:size(2), img:size(2))
                                :narrow(2, 1 + (col - 1) * img:size(3), img:size(3))
                aux:copy(img[j])

                if saveWeights then
                    aux = moduleWeight:narrow(2, 1 + (row - 1) * module.weight:size(3), module.weight:size(3))
                                    :narrow(3, 1 + (col - 1) * module.weight:size(4), module.weight:size(4))
                    aux:copy(module.weight[j])
                end
            end

            image.save(path .. i .. '_' .. name .. '_out' .. '.jpg', moduleOut)
            if saveWeights then
                image.save(path .. i .. '_' .. name .. '_weight' .. '.jpg', moduleWeight)
            end
        end
    end
end

print(img:exp())
