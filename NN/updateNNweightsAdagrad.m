function [nn, histData] = updateNNweightsAdagrad(nn, dW, dBiases, histData)
% updateNNweightsAdagrad - update weights using Adagrad

% INPUTS:
% nn: neural network structure, see manual for details
% dW: 1 x N cell array where N is the number of layers. Each
% cell contains the weight gradients for the corresponding layer 
% dBiases: same as above but each layer contains the gradients for the
% biases
% histData: structure which contains the previous updates for weights,  sum
% of past gradients, sum of squares of past gradients, sum of squares of past updates


% OUTPUTS:
% nn: neural network structure, see manual for details
% histData: structure which contains the previous updates for weights,  sum
% of past gradients, sum of squares of past gradients, sum of squares of past updates


lr = nn.trParams.lrParams.lr;

epsilon = nn.trParams.adagrad.epsilon;

for i = 1 : nn.noLayers
        
    gradW = dW{i};
    gradBiases = dBiases{i};
    weights = nn.W{i};
    biases = nn.biases{i};
    
    histGradW = histData.sum_gradW_sq{i};
    histGradBiases = histData.sum_gradBiases_sq{i};
    
    if nn.weightConstraints.weightPenaltyL1 > 0 || nn.weightConstraints.weightPenaltyL2 > 0
       
        gradW = applyWeightConstraints(gradW, weights, nn.weightConstraints.weightPenaltyL1, nn.weightConstraints.weightPenaltyL2);
   
    end
      
    histGradW = histGradW + gradW .^ 2;
    histGradBiases = histGradBiases + gradBiases .^ 2;
    
    adaptedLR_w = lr ./ sqrt(histGradW + epsilon);
    adaptedLR_biases = lr ./ sqrt(histGradBiases + epsilon);
    
    updateW = -adaptedLR_w .* gradW;
    
    updateBiases = -adaptedLR_biases .* gradBiases;
        
       
    newW = weights + updateW;
  
    if nn.weightConstraints.maxNormConstraint > 0
    
        newW = applyMaxNormRegularisation(newW, nn.weightConstraints.maxNormConstraint);
        updateW = newW - weights; % some weight vectors may have been rescaled
    end
    
    nn.W{i} = newW;
    nn.biases{i} = biases + updateBiases;
    
    histData.sum_gradW_sq{i} = histGradW;
    histData.sum_gradBiases_sq{i} = histGradBiases;
    
    histData.vW{i} = updateW;
    histData.vBiases{i} = updateBiases;
    
end





        

