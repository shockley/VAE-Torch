require 'hdf5'
local LinearVA, parent = torch.class('nn.LinearVA', 'nn.Linear')

--Custom reset function
function LinearVA:__init(inputSize, outputSize)
    parent.__init(self, inputSize, outputSize)
end

function LinearVA:reset()
    sigmaInit = 0.01
    self.weight:normal(0, 0.01)
    self.bias:normal(0, 0.01)
end

function LinearVA:updateOutput(input)
    if input:dim() == 1 then
        self.output:resize(self.bias:size(1))
        self.output:copy(self.bias)
        self.output:addmv(1, self.weight, input)
    elseif input:dim() == 2 then
        local nframe = input:size(1)
        local nunit = self.bias:size(1)

        self.output:resize(nframe, nunit)
        self.output:zero():addr(1, input.new(nframe):fill(1), self.bias)
        self.output:addmm(1, input, self.weight:t())
    else
        error('input must be vector or matrix')
    end

    return self.output
end

function LinearVA:updateGradInput(input, gradOutput)
    if self.gradInput then

        local nElement = self.gradInput:nElement()
        self.gradInput:resizeAs(input)
        if self.gradInput:nElement() ~= nElement then
            self.gradInput:zero()
        end
        if input:dim() == 1 then
            self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
        elseif input:dim() == 2 then
            self.gradInput:addmm(0, 1, gradOutput, self.weight)
        end

        return self.gradInput
    end
end

function LinearVA:accGradParameters(input, gradOutput, scale)
    scale = scale or 1

    if input:dim() == 1 then
        self.gradWeight:addr(scale, gradOutput, input)
        self.gradBias:add(scale, gradOutput)
    elseif input:dim() == 2 then
        local nframe = input:size(1)
        local nunit = self.bias:size(1)

        -- print(torch.norm(gradOutput), torch.norm(input))
        self.gradWeight:addmm(scale, gradOutput:t(), -input)
        self.gradBias:addmv(scale, gradOutput:t(), input.new(nframe):fill(1))
        -- print(torch.norm(self.gradWeight))
        -- print(torch.norm(self.gradBias))
        -- io.read()
    end

end
