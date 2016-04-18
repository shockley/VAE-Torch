-- Joost van Amersfoort - <joost@joo.st>
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'


nngraph.setDebug(false)

local VAE = require 'VAE'
require 'KLDCriterion'
require 'GaussianCriterion'
require 'Sampler'

--For loading data files
require 'load'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Variational Autoencoder')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-continuous',1,'continuous or binary data')
cmd:option('-max_epochs',30,'number of full passes through the training data')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-hidden',400,'hidden_layer_size')
cmd:option('-latent',20,'latent_variable_size')
cmd:option('-batch_size',100,'batch_size')
cmd:option('-save_to','save/protos.t7','save the model to')
cmd:option('-init_from','','initialize network parameters from checkpoint at this path')
cmd:option('-skip_train',0,'skip training')
cmd:option('-gen_data','datasets/gen_data.hdf5','reconstructed data')
cmd:text()

opt = cmd:parse(arg)

if opt.continuous == 1 then
    opt.continuous = true
else
    opt.continuous = false
end
print(opt)


function buildModel(opt)
    protos = {}
    local encoder = VAE.get_encoder(opt.input_size, opt.hidden, opt.latent)
    local decoder = VAE.get_decoder(opt.input_size, opt.hidden, opt.latent, opt.continuous)
    local input = nn.Identity()()
    local mean, log_var = encoder(input):split(2)
    local z = nn.Sampler()({mean, log_var})
    local reconstruction, reconstruction_var, model
    if opt.continuous then
        reconstruction, reconstruction_var = decoder(z):split(2)
        model = nn.gModule({input},{reconstruction, reconstruction_var, mean, log_var})
        criterion = nn.GaussianCriterion()
    else
        reconstruction = decoder(z)
        model = nn.gModule({input},{reconstruction, mean, log_var})
        criterion = nn.BCECriterion()
        criterion.sizeAverage = false
    end
    protos.model = model
    protos.criterion = criterion
    protos.KLD = nn.KLDCriterion()
    return protos
end

function computationGraph(model, data)
    --Some code to draw computational graph
    -- dummy_x = torch.rand(10)
    local rg = torch.Tensor({1}):long()
    local dummy_x = data.train:index(1,rg)
    model:forward(dummy_x)
    --Uncomment to get structure of the Variational Autoencoder
    graph.dot(model.fg, 'Variational Autoencoder', 'VA')
end

function ftrain(protos, data, opt)
    local parameters, gradients = protos.model:getParameters()
    local config = {
        learningRate = 0.001
    }

    local state = {}

    epoch = 0
    -- while true do

    while epoch < opt.max_epochs do
        epoch = epoch + 1
        local lowerbound = 0
        local tic = torch.tic()

        --local shuffle = torch.randperm(data:size(1))

        -- This batch creation is inspired by szagoruyko CIFAR example.
        local indices = torch.randperm(data.train:size(1)):long():split(opt.batch_size)
        indices[#indices] = nil
        local N = #indices * opt.batch_size

        local tic = torch.tic()
        for t,v in ipairs(indices) do
            xlua.progress(t, #indices)

            local inputs = data.train:index(1,v)
            --inputs = inputs:cuda()
            --print(inputs)
            local opfunc = function(x)
                if x ~= parameters then
                    parameters:copy(x)
                end

                protos.model:zeroGradParameters()
                local reconstruction, reconstruction_var, mean, log_var
                if opt.continuous then
                    reconstruction, reconstruction_var, mean, log_var = unpack(protos.model:forward(inputs))
                    reconstruction = {reconstruction, reconstruction_var}
                else
                    reconstruction, mean, log_var = unpack(protos.model:forward(inputs))
                end

                local err = protos.criterion:forward(reconstruction, inputs)
                local df_dw = protos.criterion:backward(reconstruction, inputs)

                local KLDerr = protos.KLD:forward(mean, log_var)
                local dKLD_dmu, dKLD_dlog_var = unpack(protos.KLD:backward(mean, log_var))

                if opt.continuous then
                    error_grads = {df_dw[1], df_dw[2], dKLD_dmu, dKLD_dlog_var}
                else
                    error_grads = {df_dw, dKLD_dmu, dKLD_dlog_var}
                end

                protos.model:backward(inputs, error_grads)

                local batchlowerbound = err + KLDerr

                return batchlowerbound, gradients
            end

            x, batchlowerbound = optim.adam(opfunc, parameters, config, state)

            lowerbound = lowerbound + batchlowerbound[1]
        end

        print("Epoch: " .. epoch .. " Lowerbound: " .. lowerbound/N .. " time: " .. torch.toc(tic)) 

        if lowerboundlist then
            lowerboundlist = torch.cat(lowerboundlist,torch.Tensor(1,1):fill(lowerbound/N),1)
        else
            lowerboundlist = torch.Tensor(1,1):fill(lowerbound/N)
        end

        if epoch % 2 == 0 then
            --torch.save('save/parameters.t7', parameters)
            feval(protos, data, opt)
            torch.save(opt.save_to, protos)
            torch.save('save/state.t7', state)
            torch.save('save/lowerbound.t7', torch.Tensor(lowerboundlist))
        end
    end
end

function feval(protos, data, opt)
    -- print on the test dataset
    local lowerbound = 0
    local err = 0
    local kldiv = 0
    local gen_data = torch.zeros(data.test:size()):double()
    if opt.gpuid >=0 then
        gen_data = gen_data:cuda()
    end
    print(gen_data:size())
    local indices =  torch.range(1, data.test:size(1)):long():split(opt.batch_size)
    indices[#indices] = nil
    local N = #indices * opt.batch_size

    local tic = torch.tic()
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)
        local inputs = data.test:index(1,v)
        --model:zeroGradParameters()
        protos.model:evaluate()
        local reconstruction, reconstruction_var, mean, log_var
        if opt.continuous then
            reconstruction, reconstruction_var, mean, log_var = unpack(protos.model:forward(inputs))
            gen_data:indexCopy(1,v,reconstruction)
            reconstruction = {reconstruction, reconstruction_var}
        else
            reconstruction, mean, log_var = unpack(protos.model:forward(inputs))
            gen_data:indexCopy(1,v,reconstruction)
        end
        
        

        local batch_err = protos.criterion:forward(reconstruction, inputs)

        local batch_KLDerr = protos.KLD:forward(mean, log_var)

        local batchlowerbound = batch_err + batch_KLDerr

        lowerbound = lowerbound + batchlowerbound
        err = err + batch_err
        kldiv = kldiv + batch_KLDerr
    end
    -- local myFile = hdf5.open(opt.gen_data, 'w')
    -- myFile:write('gen_data', gen_data:double())
    -- myFile:close()
    -- myFile = hdf5.open(opt.gen_data, 'r')
    -- local gen_data = myFile:read('gen_data')
    -- myFile:close()
    print("Test Lowerbound: " .. lowerbound/N .. " Error: " .. err/N .. " kldiv: " .. kldiv/N .. " time: " .. torch.toc(tic))

    print("print some data")

    --local shuffle = torch.randperm(gen_data:size(1))
    local i = torch.random(1, gen_data:size(1))
    local image = require 'image'
    local org_image, gen_image
    if opt.continuous then
        org_image = image.scale(data.test[i]:double():view(28,20), 280, 200)
        gen_image = image.scale(gen_data[i]:double():view(28,20), 280, 200)
    else
        org_image = image.scale(data.test[i]:double():view(28,28), 280, 200)
        gen_image = image.scale(gen_data[i]:double():view(28,28), 280, 200)
    end
    image.display(org_image)
    image.display(gen_image)
end


if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(1)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

data = load(opt.continuous)
opt.input_size = data.train:size(2)
--itorch.image(data.train[1])
--print(data.train[1]:view(28,28))



torch.manualSeed(opt.seed)

local protos
if string.len(opt.init_from) > 0 then
    print("Loading Warm Start..." .. opt.init_from)
    protos = torch.load(opt.init_from)
else
    protos = buildModel(opt)
end

if opt.gpuid >= 0 then
    data.train = data.train:cuda()
    data.test = data.test:cuda()
    protos.criterion = protos.criterion:cuda()
    protos.KLD = protos.KLD:cuda()
    protos.model = protos.model:cuda()
end

if opt.skip_train == 0 then
    print("Begin Training...")
    ftrain(protos, data, opt)
end
--print("Begin Testing...")
--feval(protos, data.test, opt)





