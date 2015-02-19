classdef SoftMax < SimpleFunct
    
    properties
    end
    
    methods
        function Y = f(obj, X)
            m = size(X,1);
            Xm = max(X,[],1);
            Y = exp(X - repmat(Xm,m,1));
            Y = Y ./ repmat(sum(Y,1),m,1);
        end
        function dX = df(obj, d, X, dY, c)
            if nargin < 5
                c = 1;
            end
            m = size(X,1);
            X = X - repmat(max(X,[],1),m,1);
            eX = exp(X);
            eXc = (-eX).^c;
            seX = repmat(sum(eX,1),m,1);
            if d == 1
                dX = ((seX-eX).^c-eXc) .* dY;
                for i = 1:m
                    dX(i,:) = dX(i,:)+sum(eXc.*dY,1);
                end
                seX2 = seX.^-2;
                dX = dX.*(eX.*seX2).^c;
            elseif d == 2
%                 dX = seX.^c .* dY
%                 seX3 = seX.^-3
%                 for i = 1:m
%                     dX(i,:) = dX(i,:)-sum(eXc.*dY,1)
%                 end
%                 dX = dX.*(eX-2*seX3.*eX.*eX).^c
                dX = -seX.^c .* dY;
                for i = 1:m
                    dX(i,:) = dX(i,:)-sum(eXc.*dY,1);
                end
                seX3 = seX.^-3;
                dX = dX.*((2.*eX-seX).*eX.*seX3).^c;
            end
            assert(d>0, 'Error: SoftMax derivative d<=0');
            assert(d<=2, 'Error: SoftMax derivative d>2');
        end
    end
    
end