function [activation] = feedForwardAutoencoder(theta, hiddenSize, visibleSize, data)

% theta: trained weights from the autoencoder
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
  
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the activation of the hidden layer for the Sparse Autoencoder.
[fI,f0,~,nx] = factoryLayeredNet([visibleSize,hiddenSize,visibleSize],[],[],[],0,2,0,.5);
hiddenIndex = visibleSize + hiddenSize + (1:hiddenSize);
[~, XI, ~] = forwardBackwardValidate(0, theta, nx, data, data, fI, f0);
activation = XI{1}.XI(hiddenIndex,:);


% maxInput = 1000; % for memory problems
% 
% cost = 0;
% grad = zeros(size(theta));
% [fI,f0,~,nx] = factoryLayeredNet([visibleSize,hiddenSize,visibleSize],[],[],[],lambda/2,2,beta,sparsityParam);
% % for i = 1:length(fI)
% %     fI{i}
% % end
% activation = zeros(size(data,1),0);
% for i = 1:ceil(size(data,2)/maxInput)
%     index = (i-1)*maxInput+1:min(size(data,2),i*maxInput);
%     [~, XI, ~] = forwardBackwardValidate(0, theta, nx, data(:,index), data(:,index), fI, f0);
%     activation = [activation,XI{1}.XI(hiddenIndex,:)];
%     cost = cost + length(index)*cost0/size(data,2);
%     grad = grad + length(index)*grad0/size(data,2);
% end

%-------------------------------------------------------------------

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
