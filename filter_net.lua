require 'csvigo'
require 'image'
require 'cunn'
preprocessing = require 'preprocessing_nagadomi'

cmd = torch.CmdLine()
cmd:addTime()
cmd:option('-modelName', 'model', 'name of the model')
cmd:option('-cuda', false, 'if true train with minibatches')
cmd:option('-float', false, 'if true cast to float')

opt = cmd:parse(arg or {})

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

-- Load extra set
extraset = LoadDataset('id_supplementary.csv')
local nextra = extraset.data:size(1)

local params = torch.load('dsg_params.t7')
preprocessing.preprocessing(extraset.data, params)

-- Load model
net = torch.load(opt.modelName .. '.net')
net:evaluate()

if opt.float then
    extraset.data = extraset.data:float()
end

print("Evaluating")

local file = assert(io.open('id_extra.csv', "w"))
file:write("Id,label\n")

for i = 1,nextra do
	local cur = extraset.data[i]
	-- Using CUDA
	if opt.cuda then
		cur = cur:cuda()
	end

    local prediction = net:forward(cur)
    prediction:exp()
    local confidences, indices = torch.sort(prediction, true) -- sort in descending order

    if confidences[1] > 0.9 then
	    file:write(extraset.Id[i] .. "," .. indices[1] .. "\n")
	end
end

file:close()
print("Evaluating finished")
