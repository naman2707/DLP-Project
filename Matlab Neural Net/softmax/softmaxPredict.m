function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.


W = theta(:);
inputSize = softmaxModel.inputSize;
numClasses = softmaxModel.numClasses;
W(end+1:end+numClasses) = zeros(numClasses,1);
[fI,f0,~,nx] = factoryLayeredNet([ inputSize, numClasses],[],SoftMax,@FunctKLDivergence,0,2,0,.05);
[~, XI, ~] = forwardBackwardValidate(0, W, nx, data, zeros(numClasses,size(data,2)), fI, f0);
Y = XI{1}.XI(end-numClasses+1:end,:);
[ ~, pred ] = max( Y, [], 1 );



% ---------------------------------------------------------------------

end

