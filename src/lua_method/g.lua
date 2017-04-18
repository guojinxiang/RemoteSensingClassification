require 'torch'
require 'nn'
require 'optim'
matio = require 'matio'
-- load all arrays from file
fullset = matio.load('trainset.mat')
testset = matio.load('testset.mat')

trainset = {
    size = 800,
    data = fullset.data[{ { 1, 800 } }]:double(),
    label = fullset.label[{ { 1, 800 } }]
}

validationset = {
    size = 200,
    data = fullset.data[{ { 801, 1000 } }]:double(),
    label = fullset.label[{ { 801, 1000 } }]
}

testset = {
    size = 22,
    data = testset.data:double()[{ { 1, 22 } }],
    label = testset.label[{ { 1, 22 } }]
}

trainset.data = trainset.data - trainset.data:mean()
validationset.data = validationset.data - validationset.data:mean()



model = nn.Sequential()
model:add(nn.Reshape(1, 25, 25))
model:add(nn.MulConstant(1 / 256.0 * 3.2))
model:add(nn.SpatialConvolutionMM(1, 20, 5, 5, 1, 1, 0, 0))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
model:add(nn.SpatialConvolutionMM(20, 50, 5, 5, 1, 1, 0, 0))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
model:add(nn.Reshape(4 * 4 * 50))
model:add(nn.Linear(4 * 4 * 50, 500))
model:add(nn.ReLU())
model:add(nn.Linear(500, 2))
model:add(nn.LogSoftMax())

model = require('weight-init')(model, 'xavier')

criterion = nn.ClassNLLCriterion()


sgd_params = {
    learningRate = 1e-2,
    learningRateDecay = 1e-4,
    weightDecay = 1e-3,
    momentum = 1e-4
}

x, dl_dx = model:getParameters()

step = function(batch_size)
    local current_loss = 0
    local count = 0
    local shuffle = torch.randperm(trainset.size)
    batch_size = batch_size or 200
    for t = 1, trainset.size, batch_size do
        -- setup inputs and targets for this mini-batch  
        local size = math.min(t + batch_size - 1, trainset.size) - t
        local inputs = torch.Tensor(size, 25, 25) --:cuda()
        local targets = torch.Tensor(size) --:cuda()
        for i = 1, size do
            local input = trainset.data[shuffle[i + t]]
            local target = trainset.label[shuffle[i + t]]
            -- if target == 0 then target = 10 end  
            inputs[i] = input
            targets[i] = target
        end
        targets:add(1)
        local feval = function(x_new)
            -- reset data  
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()

            -- perform mini-batch gradient descent  
            local loss = criterion:forward(model:forward(inputs), targets)
            model:backward(inputs, criterion:backward(model.output, targets))

            return loss, dl_dx
        end

        _, fs = optim.sgd(feval, x, sgd_params)

        -- fs is a table containing value of the loss function  
        -- (just 1 value for the SGD optimization)  
        count = count + 1
        current_loss = current_loss + fs[1]
    end

    -- normalize loss  
    return current_loss / count
end

eval = function(dataset, batch_size)
    local count = 0
    batch_size = batch_size or 200

    for i = 1, dataset.size, batch_size do
        local size = math.min(i + batch_size - 1, dataset.size) - i
        local inputs = dataset.data[{ { i, i + size - 1 } }] --:cuda()
        local targets = dataset.label[{ { i, i + size - 1 } }]:long() --:cuda()
        local outputs = model:forward(inputs)
        local _, indices = torch.max(outputs, 2)
        indices:add(-1)
        local guessed_right = indices:eq(targets):sum()
        count = count + guessed_right
    end

    return count / dataset.size
end

max_iters = 100

do
    local last_accuracy = 0
    local decreasing = 0
    local threshold = 1 -- how many deacreasing epochs we allow  
    for i = 1, max_iters do
        local loss = step()
        print(string.format('Epoch: %d Current loss: %4f', i, loss))
        local accuracy = eval(validationset)
        print(string.format('Accuracy on the validation set: %4f', accuracy))
        if accuracy < last_accuracy then
            if decreasing > threshold then break end
            decreasing = decreasing + 1
        else
            decreasing = 0
        end
        last_accuracy = accuracy
    end
end


accuracy = eval(testset)
print(string.format('Accuracy on the test set: %4f', accuracy))  

