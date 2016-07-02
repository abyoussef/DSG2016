require 'nn'
require 'image'

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

mean = torch.load('dsg_train_mean.t7')
stdv = torch.load('dsg_train_stdv.t7')

for i = 1,3 do
    img[{ {i}, {}, {} }]:add(-mean[i])
    img[{ {i}, {}, {} }]:div(stdv[i])
end

net = torch.load(opt.modelName .. '.net')
net:evaluate()

for i = 1,net:size() do
    module = net:get(i)
    img = module:forward(img)

    if torch.typename(module) ~= 'nn.Dropout' then
        if img:dim() == 3 then
            for j = 1,img:size(1) do
                image.save(path .. i .. '_' .. j .. '.jpg', img[j])
            end
        end
    end
end

print(img:exp())
