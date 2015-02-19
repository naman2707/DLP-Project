function [ W, stat ] = optimize(fnn,W,X,Y,batch,tmax,eta0,epsilon,gamma,mu,delta,alpha,beta)
    ddW0 = ones(size(W));
    r = zeros(size(W));
    eta = eta0/4;
    i = 1;
    
    stat = [];
    while i<=tmax% && eta > epsilon
        k = randperm(size(X,2));
        k = k(1:batch);
        %k = k(1:floor(exp(log(batch)*i/tmax)));
        %k = k(1:min(length(k),ceil(batch*exp(-log(batch)*(tmax-i)/tmax))));
        X0 = X(:,k);
        Y0 = Y(:,k);
        
        [ cost, dW, ddW ] = fnn(W,X0,Y0);
        ddW0 = (1-gamma) * ddW0 + gamma * ddW.^2;
        C = 1./(ddW0+mu^2).^(1/2);
        CdW = C .* dW;
        %CdW = 1./(sqrt(ddW0)./dW+mu);
        
        W = W - eta * CdW;
        r = (1-delta) * r + delta * CdW;
        eta = min(eta, eta0);
        eta = eta + alpha * eta * (beta * norm(r)-eta);
        
        if norm(r)>1
            break
        end
        i = i + 1;
        if mod(i,2^8) == 0
            fprintf('%d, ',i/2^8);
        end
        stat(end+1,:) = [cost,length(k),norm(W),norm(dW),norm(ddW0),norm(C),eta,norm(r)]';
    end
    fprintf('\n',i);
end

