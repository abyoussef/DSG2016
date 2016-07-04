require 'torchx'
require 'csvigo'

local path = "roof_images/"
local files = paths.indexdir(path, 'jpg', true)
local ids = {}
print(files:size())

local parsed = csvigo.load({path='id_train.csv', mode="query"})
local dataset = parsed('all')
ntrain = #dataset.Id

for i = 1,ntrain do
	cur = dataset.Id[i]
	if ids[cur] then
		print('repeated train', cur)
	end
	ids[cur] = 1
end

parsed = csvigo.load({path='sample_submission4.csv', mode="query"})
dataset = parsed('all')
ntest = #dataset.Id

for i = 1,ntest do
	cur = dataset.Id[i]
	if ids[cur] then
		print('repeated test', cur)
	end
	ids[cur] = 1
end

local file = assert(io.open('id_supplementary.csv', "w"))
file:write("Id,label\n")
cont = 0

for i = 1,files:size() do
	local filename = files:filename(i)
	filename = filename:sub(path:len() + 1, filename:len() - 4)
	if ids[filename] then
		cont = cont + 1
	else
		file:write(filename .. "," .. 4 .. "\n")
		ids[filename] = 1
	end
end

print(ntrain, ntest, cont)
file:close()