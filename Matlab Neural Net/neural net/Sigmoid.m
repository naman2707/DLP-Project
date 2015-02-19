classdef Sigmoid < SimpleFunct
    
    properties
    end
    
    methods
        function Y = f(obj, X)
            Y = 1 ./ ( 1 + exp(-X) );
        end
        function dX = df(obj, d, X, dY, c)
            if nargin < 5
                c = 1;
            end
%             assert(sum(X(:))~=0,'Error: Sigmoid X=0');
%             assert(sum(dY(:))~=0,'Error: Sigmoid dY=0');
            Y0 = obj.f(X);
            Y1 = ( 1 - Y0 ) .* Y0;
            if d == 1
                dX = Y1;
            elseif d == 2
                dX = (1 - 2 * Y0) .* Y1;
            else
                dX = zeros(size(dY));
            end
            dX = dX.^c .* dY;
            assert(d>0, 'Error: Sigmoid derivative d<=0');
            assert(d<=2, 'Error: Sigmoid derivative d>2');
        end
    end
    
end